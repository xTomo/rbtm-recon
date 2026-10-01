"""Шумоподавление полной вариацией (TV) объёма срезов после FBP — быстрый градиентный метод FGP на GPU и CPU.

Решается та же задача, что у ``skimage.restoration.denoise_tv_chambolle`` (ROF, изотропная TV):
min_u ½‖u − f‖² + weight·Σ|∇u|. TV сохраняет края: гладкие области выравниваются, перепады остаются резкими. На
шумных сканах robotom (af443cef, 79d1ba3e) лёгкий гаусс проекций σ1 + TV 3D дают края как у σ≈1 при шуме как у
σ≈3 (разбор 01.10.2026), без итерационной реконструкции.

Метод — FGP (Beck, Teboulle 2009, ускоренный градиентный по двойственной переменной p, |p| ≤ 1):
u = f + weight·∇ᵀr; q = r − ∇u / (L·weight); p⁺ = q / max(1, |q|); t⁺ = (1 + √(1 + 4t²))/2;
r = p⁺ + (t − 1)/t⁺·(p⁺ − p); результат f + weight·∇ᵀp. L = 4·ndim, ∇ — разности вперёд (0 на последнем отсчёте),
∇ᵀ — сопряжённый (как ``d`` в skimage). Число итераций фиксировано: превью и задача делают одно и то же. На af443cef
50 итераций FGP ≈ 200 итераций метода Шамболя из skimage (в среднем 0,03σ шума от предельного решения).

GPU: два слитых ядра на итерацию; в памяти f, u, p и r по осям — 8 объёмов float32 (32 байта на воксель).
Большой объём — ``denoise_volume``: плитки по y, x с ореолом ``HALO`` (все z сразу; ореол по z — у вызывающего,
см. ``DenoiseWriter``). 3D-массив с ``ndim=2`` — стопка независимых 2D-срезов.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

METHODS = ('tv',)
DEFAULT_ITERATIONS = 50
ITER_RANGE = (5, 500)
#: Сила — вес TV в долях σ шума среза (до TV, после сглаживания проекций); по умолчанию 2.
STRENGTH_RANGE = (0.1, 10.0)
DEFAULT_STRENGTH = 2.0
#: Ореол (вокселей) плитки и порции по z: на af443cef при 12 разница с обработкой целиком — 0,025σ шума.
HALO = 12
#: Байт на воксель плитки на GPU (8 объёмов float32) и доля свободной памяти под плитку.
BYTES_PER_VOXEL = 32
GPU_FRACTION = 0.5

DEFAULTS: Dict[str, Any] = {'method': None, 'strength': DEFAULT_STRENGTH, 'weight': None,
                            'iterations': DEFAULT_ITERATIONS}

_KERNELS: dict = {}

_SRC = r'''
#define LOOP for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; \
                  i += (long long)blockDim.x * gridDim.x)
extern "C" __global__ void tv_primal(const float* f, const float* rz, const float* ry, const float* rx, float* u,
                                     const int nz, const int ny, const int nx, const float lam) {
    const long long n = (long long)nz * ny * nx, ys = nx, zs = (long long)ny * nx;
    LOOP {
        const int x = (int)(i % nx), y = (int)((i / nx) % ny), z = (int)(i / zs);
        float d = -(rz[i] + ry[i] + rx[i]);
        if (z > 0) d += rz[i - zs];
        if (y > 0) d += ry[i - ys];
        if (x > 0) d += rx[i - 1];
        u[i] = f[i] + lam * d;
    }
}
extern "C" __global__ void tv_dual(const float* u, float* pz, float* py, float* px, float* rz, float* ry, float* rx,
                                   const int nz, const int ny, const int nx, const float s, const float beta,
                                   const int usez) {
    const long long n = (long long)nz * ny * nx, ys = nx, zs = (long long)ny * nx;
    LOOP {
        const int x = (int)(i % nx), y = (int)((i / nx) % ny), z = (int)(i / zs);
        const float o = u[i];
        const float gz = (usez && z < nz - 1) ? u[i + zs] - o : 0.0f;
        const float gy = (y < ny - 1) ? u[i + ys] - o : 0.0f;
        const float gx = (x < nx - 1) ? u[i + 1] - o : 0.0f;
        const float qz = rz[i] - s * gz, qy = ry[i] - s * gy, qx = rx[i] - s * gx;
        const float m = fmaxf(1.0f, sqrtf(qz * qz + qy * qy + qx * qx));
        const float az = qz / m, ay = qy / m, ax = qx / m;
        rz[i] = az + beta * (az - pz[i]);
        ry[i] = ay + beta * (ay - py[i]);
        rx[i] = ax + beta * (ax - px[i]);
        pz[i] = az; py[i] = ay; px[i] = ax;
    }
}
'''


def _kernels():
    if not _KERNELS:
        import cupy as cp  # noqa: WPS433
        mod = cp.RawModule(code=_SRC)
        _KERNELS['primal'] = mod.get_function('tv_primal')
        _KERNELS['dual'] = mod.get_function('tv_dual')
    return _KERNELS['primal'], _KERNELS['dual']


def _adjoint_np(rz, ry, rx):
    """∇ᵀr (как ``d`` в skimage): −Σr + r, сдвинутые на 1 вперёд по своей оси."""
    d = -(rz + ry + rx)
    d[1:] += rz[:-1]
    d[:, 1:] += ry[:, :-1]
    d[:, :, 1:] += rx[:, :, :-1]
    return d


def denoise(image, weight: float, iterations: int = DEFAULT_ITERATIONS, ndim: Optional[int] = None, xp=None):
    """TV-шумоподавление массива 2D (y, x) или 3D (z, y, x) → float32 той же формы на xp.

    weight — вес TV в единицах значений (> 0); iterations — число итераций FGP; ndim — по скольким осям связывать
    (по умолчанию image.ndim; 2 у 3D-массива — каждый срез отдельно)."""
    from .gpu import get_xp  # noqa: WPS433
    xp = xp or get_xp()
    weight = float(weight)
    if not (math.isfinite(weight) and weight > 0):
        raise ValueError('tv: weight должен быть > 0, получено {}'.format(weight))
    nd = int(ndim or image.ndim)
    if image.ndim not in (2, 3) or nd not in (2, 3) or nd > image.ndim:
        raise ValueError('tv: ожидается 2D или 3D массив и ndim ≤ его размерности, получено {} и ndim={}'.format(
            getattr(image, 'shape', None), ndim))
    f = xp.ascontiguousarray(xp.asarray(image, dtype=xp.float32))
    f3 = f if f.ndim == 3 else f[None]
    nz, ny, nx = f3.shape
    s = 1.0 / (4.0 * nd * weight)
    pz, py, px, rz, ry, rx = (xp.zeros_like(f3) for _ in range(6))
    u = xp.empty_like(f3)
    t = 1.0
    gpu = xp is not np
    if gpu:
        primal, dual = _kernels()
        threads = 256
        blocks = int(min(65535 * 4, (f3.size + threads - 1) // threads))
        dims = (np.int32(nz), np.int32(ny), np.int32(nx))
    for _ in range(int(iterations)):
        tn = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        beta = (t - 1.0) / tn
        if gpu:
            primal((blocks,), (threads,), (f3, rz, ry, rx, u) + dims + (np.float32(weight),))
            dual((blocks,), (threads,), (u, pz, py, px, rz, ry, rx) + dims +
                 (np.float32(s), np.float32(beta), np.int32(nd == 3)))
        else:
            np.add(f3, weight * _adjoint_np(rz, ry, rx), out=u)
            gz = np.zeros_like(u)
            if nd == 3:
                gz[:-1] = u[1:] - u[:-1]
            gy = np.zeros_like(u)
            gy[:, :-1] = u[:, 1:] - u[:, :-1]
            gx = np.zeros_like(u)
            gx[:, :, :-1] = u[:, :, 1:] - u[:, :, :-1]
            qz, qy, qx = rz - s * gz, ry - s * gy, rx - s * gx
            m = np.maximum(1.0, np.sqrt(qz * qz + qy * qy + qx * qx))
            az, ay, ax = qz / m, qy / m, qx / m
            rz[...] = az + beta * (az - pz)
            ry[...] = ay + beta * (ay - py)
            rx[...] = ax + beta * (ax - px)
            pz[...], py[...], px[...] = az, ay, ax
        t = tn
    if gpu:
        primal((blocks,), (threads,), (f3, pz, py, px, u) + dims + (np.float32(weight),))
    else:
        np.add(f3, weight * _adjoint_np(pz, py, px), out=u)
    return u if image.ndim == 3 else u[0]


# --- параметры (блок рецепта denoise) ------------------------------------------------------------------------

def default_block() -> Dict[str, Any]:
    """Блок рецепта ``denoise`` по умолчанию — выключено."""
    return dict(DEFAULTS)


def resolve(block: Optional[Dict[str, Any]], need_weight: bool = True) -> Optional[Dict[str, Any]]:
    """Нормализованные параметры или None (выключено: нет блока или method пустой). need_weight — weight
    обязателен (задача); в превью вес задаётся силой и шумом фрагмента."""
    if not block or not block.get('method'):
        return None
    if block['method'] not in METHODS:
        raise ValueError('denoise.method: одно из {}, получено {!r}'.format(', '.join(METHODS), block['method']))
    p = dict(DEFAULTS)
    p.update({k: v for k, v in block.items() if k in DEFAULTS and v is not None})
    strength = float(p['strength'])
    if not (math.isfinite(strength) and STRENGTH_RANGE[0] <= strength <= STRENGTH_RANGE[1]):
        raise ValueError('denoise.strength: {} вне [{}, {}]'.format(strength, *STRENGTH_RANGE))
    it = p['iterations']
    if isinstance(it, bool) or not isinstance(it, (int, float)) or int(it) != it or \
            not ITER_RANGE[0] <= int(it) <= ITER_RANGE[1]:
        raise ValueError('denoise.iterations: целое в [{}, {}], получено {!r}'.format(*ITER_RANGE, it))
    w = p['weight']
    if w is not None:
        w = float(w)
        if not (math.isfinite(w) and w > 0):
            raise ValueError('denoise.weight должен быть > 0: {!r}'.format(p['weight']))
    elif need_weight:
        raise ValueError('denoise.weight не задан')
    return {'method': p['method'], 'strength': strength, 'weight': w, 'iterations': int(it)}


# --- большой объём: плитки по y, x ---------------------------------------------------------------------------

def _tile_side(nz: int, budget_voxels: int) -> int:
    """Сторона плитки (с ореолом) — чтобы nz·side² ≤ бюджет, не меньше 64 + 2·HALO."""
    return max(64 + 2 * HALO, int(math.sqrt(max(1, budget_voxels) / max(1, nz))))


def gpu_budget_voxels(xp) -> int:
    """Вокселей плитки, помещающихся в GPU_FRACTION свободной памяти (CPU — без ограничения на практике)."""
    from . import gpu as gpu_mod  # noqa: WPS433
    if not gpu_mod.is_gpu(xp):
        return 1 << 40
    gpu_mod.free_memory()
    info = gpu_mod.mem_info()
    free = info[0] if info else 1 << 30
    return int(GPU_FRACTION * free // BYTES_PER_VOXEL)


def denoise_volume(vol: np.ndarray, weight: float, iterations: int = DEFAULT_ITERATIONS, xp=None,
                   budget_voxels: Optional[int] = None) -> np.ndarray:
    """TV 3D объёма (z, y, x) numpy → numpy float32: плитки по y, x с ореолом HALO (все z плитки сразу), ядро плитки
    берётся из результата с ореолом. Совпадает с обработкой целиком до ~0,03σ шума на краях плиток."""
    from .gpu import get_xp, to_numpy  # noqa: WPS433
    xp = xp or get_xp()
    nz, ny, nx = vol.shape
    budget = budget_voxels if budget_voxels is not None else gpu_budget_voxels(xp)
    side = _tile_side(nz, budget)
    if side >= max(ny, nx):
        return np.asarray(to_numpy(denoise(vol, weight, iterations, xp=xp)), dtype='float32')
    core = side - 2 * HALO
    out = np.empty((nz, ny, nx), dtype='float32')
    for y0 in range(0, ny, core):
        for x0 in range(0, nx, core):
            y1, x1 = min(ny, y0 + core), min(nx, x0 + core)
            a0, b0 = max(0, y0 - HALO), max(0, x0 - HALO)
            a1, b1 = min(ny, y1 + HALO), min(nx, x1 + HALO)
            res = denoise(np.ascontiguousarray(vol[:, a0:a1, b0:b1]), weight, iterations, xp=xp)
            out[:, y0:y1, x0:x1] = to_numpy(res[:, y0 - a0:y1 - a0, x0 - b0:x1 - b0])
            del res
    return out


class DenoiseWriter:
    """Обёртка писателя объёма (интерфейс ``outputs.VolumeWriter``: write/close/abort): копит срезы и отдаёт дальше
    порциями по ``chunk`` срезов после TV 3D по срезам порции ± HALO (в пределах объёма). Срезы приходят строго по
    порядку; памяти — chunk + 3·HALO срезов в RAM. mask (ny, nx) bool — после TV вне маски 0 (круг среза)."""

    def __init__(self, inner, params: Dict[str, Any], nz: int, chunk: int = 64, xp=None, on_chunk=None,
                 mask: Optional[np.ndarray] = None):
        self.inner = inner
        self.weight = float(params['weight'])
        self.iterations = int(params['iterations'])
        self.nz = int(nz)
        self.chunk = max(1, int(chunk))
        self.xp = xp
        self.on_chunk = on_chunk          # (z0, срезы после TV) — для выборки статистики
        self.mask = mask
        self._buf: List[np.ndarray] = []  # срезы [self._b0, self._b0 + len)
        self._b0 = 0
        self._emitted = 0
        self.seconds = 0.0
        self.voxel_iterations = 0         # воксели порций с ореолом по z × итерации (без ореола плиток)

    def write(self, z0: int, slab: np.ndarray) -> None:
        if int(z0) != self._b0 + len(self._buf):
            raise ValueError('DenoiseWriter: запись не по порядку: z0={}, ожидалось {}'.format(
                z0, self._b0 + len(self._buf)))
        self._buf.extend(np.asarray(s, dtype='float32') for s in slab)
        while self._b0 + len(self._buf) >= min(self.nz, self._emitted + self.chunk + HALO) and self._emitted < self.nz:
            self._emit(min(self.nz, self._emitted + self.chunk))

    def _emit(self, e1: int) -> None:
        import time  # noqa: WPS433
        e0 = self._emitted
        i0 = max(self._b0, e0 - HALO)
        i1 = min(self._b0 + len(self._buf), e1 + HALO)
        block = np.stack(self._buf[i0 - self._b0:i1 - self._b0])
        t0 = time.time()
        res = denoise_volume(block, self.weight, self.iterations, xp=self.xp)[e0 - i0:e1 - i0]
        self.voxel_iterations += int(block.size) * self.iterations
        if self.mask is not None:
            res = np.where(self.mask[None], res, np.float32(0))
        self.seconds += time.time() - t0
        self.inner.write(e0, res)
        if self.on_chunk is not None:
            self.on_chunk(e0, res)
        self._emitted = e1
        keep = max(self._b0, e1 - HALO)                     # нужны как ореол следующей порции
        self._buf = self._buf[keep - self._b0:]
        self._b0 = keep

    def close(self):
        while self._emitted < min(self.nz, self._b0 + len(self._buf)):
            self._emit(min(self.nz, self._b0 + len(self._buf), self._emitted + self.chunk))
        self._buf = []
        return self.inner.close()

    def abort(self) -> None:
        self._buf = []
        self.inner.abort()


def chunk_slices(ny: int, nx: int, ram_bytes: int = 4 << 30) -> int:
    """Срезов в порции DenoiseWriter: буфер (порция + 3·HALO) не больше ram_bytes, от 16 до 256."""
    per = max(1, ny * nx * 4)
    return int(min(256, max(16, ram_bytes // per - 3 * HALO)))
