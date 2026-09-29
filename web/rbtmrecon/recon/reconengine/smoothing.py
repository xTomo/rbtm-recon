"""Сглаживание проекций гауссом и деблюринг тем же ядром — одним фильтром проекций перед FBP.

Задумка (как в «Реконструкции Марса»): каждую проекцию сгладить 2D-гауссом σ по осям детектора (u — столбцы,
v — строки; по углу — нет), восстановить, затем обратить внесённое размытие тем же ядром (Винер или нерезкая маска).

Параллельный пучок: 2D-гаусс σ проекций ≡ 3D-гаусс σ объекта (гаусс изотропен). Сглаживание → FBP → 3D-деблюр
линейны и коммутируют, а 3D-фильтр, радиально симметричный в (x, y), по теореме о центральном сечении равен 2D-фильтру
проекций с той же передаточной функцией T(k_u, k_v). Поэтому вся цепочка — одна свёртка каждой проекции перед FBP
(после колец: нелинейные шаги остаются до неё), без прохода по объёму и без перекрытия слоёв по z:

- H(k) — спектр дискретного 2D-гаусса σ (выборка exp(−x²/2σ²) на ±ceil(4σ), нормированная на 1 — как
  ``scipy.ndimage.gaussian_filter`` и ``gauss_kernel`` «Марса»), разделимый: H = h(k_v)·h(k_u);
- ``wiener``: T = H² / (H² + β·L²), L — спектр дискретного лапласиана (4 − 2cos 2πk_v − 2cos 2πk_u), как у
  ``skimage.restoration.wiener`` с reg=None; β = balance (0,02 — рабочая точка «Марса»);
- ``unsharp``: T = H·(1 + a·(1 − H)) — как ``skimage.filters.unsharp_mask`` с радиусом σ, a = amount;
- ``none``: T = H — только сглаживание.

Дискретные формы изотропны лишь приближённо (на высоких частотах), поэтому равенство «фильтр проекций ≡ 3D-фильтр
объёма» тоже приближённое — на порядки точнее, чем видно глазом (тест ``test_projection_filter_equals_volume_filter``).

Фильтр применяется как свёртка с ядром t(v, u) = F⁻¹[T], обрезанным до (2h + 1)² (h — ``halo_rows``: хвост за ±h
меньше ``TAIL`` массы |t|) и нормированным на сумму 1 (постоянная составляющая не меняется). Конечное ядро делает
результат слоя с ореолом ±h строк точно равным обработке всего кропа, а превью строки — срезу задачи.
За краями блока (края кропа по v, края детектора по u) — отражение.
"""
from __future__ import annotations

import functools
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

DEBLUR_METHODS = ('wiener', 'unsharp', 'none')
DEFAULTS: Dict[str, Any] = {'sigma': None, 'deblur': 'wiener', 'balance': 0.02, 'amount': 1.5}
SIGMA_RANGE = (0.5, 3.0)
BALANCE_RANGE = (1e-4, 1.0)
AMOUNT_RANGE = (0.0, 5.0)
#: доля массы |t| за пределами ±h — отбрасывается (ядро нормируется заново). Ядро Винера звенит (резкий срез
#: спектра): при σ = 1,5 h = 16 при 3e-3 против 25 при 1e-4 — разница в результате ~0,3 % от размаха ядра
TAIL = 3e-3
#: байт на порцию углов в apply (комплексный спектр порции)
CHUNK_BYTES = 256 * 1024 * 1024


def default_block() -> Dict[str, Any]:
    """Блок рецепта ``smoothing`` по умолчанию — выключено."""
    return dict(DEFAULTS)


def _in(name: str, v: float, rng: Tuple[float, float]) -> float:
    v = float(v)
    if not (math.isfinite(v) and rng[0] <= v <= rng[1]):
        raise ValueError('smoothing.{}: {} вне [{}, {}]'.format(name, v, rng[0], rng[1]))
    return v


def resolve(block: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Нормализованные параметры фильтра или None (выключено: нет блока или sigma пустая/0)."""
    if not block or block.get('sigma') in (None, 0, 0.0):
        return None
    p = dict(DEFAULTS)
    p.update({k: v for k, v in block.items() if k in DEFAULTS and v is not None})
    method = p['deblur']
    if method not in DEBLUR_METHODS:
        raise ValueError('smoothing.deblur: одно из {}, получено {!r}'.format(', '.join(DEBLUR_METHODS), method))
    return {'sigma': _in('sigma', p['sigma'], SIGMA_RANGE), 'deblur': method,
            'balance': _in('balance', p['balance'], BALANCE_RANGE),
            'amount': _in('amount', p['amount'], AMOUNT_RANGE)}


def _gauss_spectrum_1d(n: int, sigma: float, real: bool) -> np.ndarray:
    """Спектр (rfft или fft) дискретного 1D-гаусса σ, выборка на ±ceil(4σ), сумма 1, центр в нуле."""
    r = int(math.ceil(4 * sigma))
    x = np.arange(-r, r + 1)
    g = np.exp(-x * x / (2.0 * sigma * sigma))
    g /= g.sum()
    a = np.zeros(n)
    a[x % n] = g
    return (np.fft.rfft(a) if real else np.fft.fft(a)).real


def transfer(n: int, p: Dict[str, Any]) -> np.ndarray:
    """Передаточная функция T на сетке n × n (раскладка rfft2: (n, n//2 + 1)), float64."""
    h = _gauss_spectrum_1d(n, p['sigma'], False)[:, None] * _gauss_spectrum_1d(n, p['sigma'], True)[None, :]
    if p['deblur'] == 'wiener':
        kv = np.fft.fftfreq(n)[:, None]
        ku = np.fft.rfftfreq(n)[None, :]
        lap = 4.0 - 2.0 * np.cos(2 * math.pi * kv) - 2.0 * np.cos(2 * math.pi * ku)
        h2 = h * h
        return h2 / (h2 + p['balance'] * lap * lap)
    if p['deblur'] == 'unsharp':
        return h * (1.0 + p['amount'] * (1.0 - h))
    return h


def _key(p: Dict[str, Any]) -> Tuple:
    return (round(p['sigma'], 6), p['deblur'], round(p['balance'], 9), round(p['amount'], 6))


@functools.lru_cache(maxsize=32)
def _kernel_cached(key: Tuple) -> np.ndarray:
    p = {'sigma': key[0], 'deblur': key[1], 'balance': key[2], 'amount': key[3]}
    n = 1 << int(math.ceil(math.log2(max(128, 32 * math.ceil(4 * p['sigma']) + 32))))
    t = np.fft.irfft2(transfer(n, p), s=(n, n))
    t = np.fft.fftshift(t)
    c = n // 2
    mass = np.abs(t)
    total = mass.sum()
    # h — наименьший радиус (квадрат (2h+1)², ядро изотропно), вне которого масса < TAIL
    h = 1
    while h < c - 1:
        inside = mass[c - h:c + h + 1, c - h:c + h + 1].sum()
        if total - inside < TAIL * total:
            break
        h += 1
    k = t[c - h:c + h + 1, c - h:c + h + 1]
    k = k / k.sum()
    k.flags.writeable = False
    return k.astype(np.float64)


def kernel(p: Dict[str, Any]) -> np.ndarray:
    """Ядро свёртки t (2h+1, 2h+1), float64, сумма 1 (только чтение)."""
    return _kernel_cached(_key(p))


def halo_rows(p: Optional[Dict[str, Any]]) -> int:
    """Радиус ядра h: сколько соседних строк (и столбцов) нужно с каждой стороны; 0 — фильтр выключен."""
    if not p:
        return 0
    return (kernel(p).shape[0] - 1) // 2


def _fast_len(n: int) -> int:
    """Ближайшая длина ≥ n вида 2^a·3^b·5^c (быстрая для FFT)."""
    best = 1 << int(math.ceil(math.log2(max(1, n))))
    f5 = 1
    while f5 < best:
        f3 = f5
        while f3 < best:
            f2 = f3
            while f2 < n:
                f2 *= 2
            best = min(best, f2)
            f3 *= 3
        f5 *= 5
    return best


def _pad_reflect(xp, a, pad_v: int, pad_u: int):
    """Отражение по осям (1, 2) массива (c, s, w); допускает отступ больше размера (повторным отражением)."""
    while pad_v > 0 or pad_u > 0:
        pv = min(pad_v, a.shape[1] - 1) if a.shape[1] > 1 else 0
        pu = min(pad_u, a.shape[2] - 1) if a.shape[2] > 1 else 0
        if pv == 0 and pu == 0:
            # одна строка/столбец — отражать нечего, повторяем край
            return xp.pad(a, ((0, 0), (pad_v, pad_v), (pad_u, pad_u)), mode='edge')
        a = xp.pad(a, ((0, 0), (pv, pv), (pu, pu)), mode='reflect')
        pad_v -= pv
        pad_u -= pu
    return a


def apply(sino, p: Optional[Dict[str, Any]], keep: Optional[Tuple[int, int]] = None, xp=None):
    """Фильтр слоя синограмм ``sino`` (s, n, w): строки детектора × углы × столбцы → строки ``keep`` [a, b) той же
    раскладки (по умолчанию все). p=None — вернуть строки keep как есть.

    Каждая проекция (срез [:, i, :]) сворачивается с ядром ``kernel(p)``; за краями блока — отражение. Строки keep
    точны (совпадают с обработкой всего кропа), если в блоке есть по ``halo_rows(p)`` строк с каждой стороны от них
    или блок доходит до края кропа (там и так отражение)."""
    from .gpu import get_xp
    xp = xp or get_xp()
    s, n, w = sino.shape
    a, b = keep if keep is not None else (0, s)
    if not (0 <= a < b <= s):
        raise ValueError('keep=({}, {}) вне [0, {})'.format(a, b, s))
    if not p:
        return sino[a:b]
    k = kernel(p)
    h = (k.shape[0] - 1) // 2
    # строки блока, от которых зависят строки keep
    r0, r1 = max(0, a - h), min(s, b + h)
    sv = r1 - r0
    nv, nu = _fast_len(sv + 2 * h), _fast_len(w + 2 * h)
    # ядро с центром в начале координат (циклическая раскладка) → спектр
    kg = np.zeros((nv, nu), dtype=np.float64)
    idx = np.arange(-h, h + 1)
    kg[np.ix_(idx % nv, idx % nu)] = k
    kf = xp.asarray(np.fft.rfft2(kg).astype(np.complex64))

    out = xp.empty((b - a, n, w), dtype=xp.float32)
    chunk = max(1, int(CHUNK_BYTES // (nv * (nu // 2 + 1) * 8 * 3)))
    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        part = xp.ascontiguousarray(xp.swapaxes(xp.asarray(sino[r0:r1, c0:c1, :], dtype=xp.float32), 0, 1))
        part = _pad_reflect(xp, part, h, h)                       # (c, sv + 2h, w + 2h)
        spec = xp.fft.rfft2(part, s=(nv, nu), axes=(1, 2))
        spec *= kf
        res = xp.fft.irfft2(spec, s=(nv, nu), axes=(1, 2))
        del spec
        # строка i блока — индекс h + i в дополненном массиве; keep начинается со строки a − r0 блока
        v0 = h + (a - r0)
        out[:, c0:c1, :] = xp.swapaxes(res[:, v0:v0 + (b - a), h:h + w], 0, 1)
        del res, part
    return out
