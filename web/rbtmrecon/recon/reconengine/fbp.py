"""Восстановление срезов по синограммам (параллельный пучок).

Синограмма строки — (n_углов, w), ось вращения в столбце (w−1)/2 (после axis.align_rows).
Бэкенды:
- 'astra' — ``tomo.recon.astra_utils.astra_recon_2d_parallel(sino, angles, [['FBP_CUDA']])``, как в
  ``tomotools4.recon_2d_parallel`` (те же соглашения об углах и ориентации);
- 'cpu'   — FBP на numpy по формулам ``skimage.transform.iradon`` (тот же ramp-фильтр Kak–Slaney с дополнением
  до степени двойки ≥ 64 и 2·w, линейная интерполяция при обратной проекции, множитель π/(2·n_углов)),
  но с центром детектора и среза в (w−1)/2, как у astra. Сам iradon не подходит: у него центр — w//2, и при
  чётной ширине ось смещалась бы на 0,5 пикселя. Для нечётной ширины результат совпадает с
  ``iradon(sino.T, theta=angles, filter_name='ramp', circle=False, output_size=w)`` (проверяется тестом).
  Ориентация: пиксель (строка r, столбец c) среза — точка (X, Y) = (c − (w−1)/2, r − (w−1)/2), проекция точки
  на детектор при угле θ — t = X·cos θ − Y·sin θ (соглашение skimage.radon; совпадает с геометрией astra
  ``parallel`` при строке 0 объёма сверху — проверить на сервере сравнением с 'astra').
Результат делится на pixel_size (мм) → коэффициент ослабления в 1/мм, как сейчас.

Фрагмент среза (``region`` = (x0, y0, x1, y1) в пикселях среза w×w, полуоткрытый): обратная проекция идёт только
по его пикселям — стоимость ∝ площади (фрагмент 384² вместо 3216² — в десятки раз дешевле), фильтрация синограммы та
же. 'cpu' — те же формулы по координатам пикселей фрагмента (поэлементно те же числа, что обрезка полного среза);
'astra' — ``create_vol_geom`` с окном (``astra_window``): в astra пиксель (строка i, столбец j) объёма с окном
[min_x, max_x] × [min_y, max_y] и шагом 1 имеет центр (min_x + j + 0,5, max_y − i − 0,5), а окно полного среза по
умолчанию — ±w/2; окно фрагмента (x0 − w/2, x1 − w/2, w/2 − y1, w/2 − y0) даёт те же центры пикселей, что у
пикселей [y0, y1) × [x0, x1) полного среза, и тот же шаг (масштаб FBP_CUDA не меняется) — проверить на сервере
сравнением с обрезкой полного среза.

Выбор углов (``select_angles``):
- 'first_180'   — углы с (a − a.min) < 180 (текущее поведение);
- 'full_halves' — наибольшее кратное 180° число полуоборотов k от начала скана (для 0–360° — все углы).
  Каждый полуоборот восстанавливается отдельно (обычный 180°-скан, масштаб FBP корректен при любой нормировке
  бэкенда), результаты усредняются, т.е. сумма делится на k: значения не зависят от числа полуоборотов.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ANGLE_MODES = ('first_180', 'full_halves')
BACKENDS = ('auto', 'astra', 'cpu')
_HALF_TURN = 180.0


def _check_mode(mode: str) -> None:
    if mode not in ANGLE_MODES:
        raise ValueError('неизвестный режим углов: {} (допустимы {})'.format(mode, ', '.join(ANGLE_MODES)))


def _angle_step(a: np.ndarray) -> float:
    """Типичный шаг по углу: медиана положительных разностей уникальных углов (0, если угол один)."""
    uniq = np.unique(np.asarray(a, dtype='float64'))
    d = np.diff(uniq)
    d = d[d > 1e-6]
    return float(np.median(d)) if d.size else 0.0


def halves_count(angles_deg: np.ndarray, mode: str = 'first_180') -> int:
    """Число полуоборотов, которое покрывают выбранные углы (1 для first_180).

    Для 'full_halves' — floor((размах + шаг) / 180), не меньше 1: углы 0..359.5 с шагом 0.5 → 2, 0..199.5 → 1."""
    _check_mode(mode)
    if mode == 'first_180':
        return 1
    a = np.asarray(angles_deg, dtype='float64')
    if a.size == 0:
        return 1
    span = float(a.max() - a.min()) + _angle_step(a)
    return max(1, int(math.floor(span / _HALF_TURN + 1e-3)))


def select_angles(angles_deg: np.ndarray, mode: str = 'first_180') -> np.ndarray:
    """Булева маска выбранных углов: (a − a.min) < 180·k, k = halves_count(angles_deg, mode).

    Сравнение ведётся в типе входного массива, как в старом коде (``(data_angles − data_angles.min()) < 180``)."""
    k = halves_count(angles_deg, mode)
    a = np.asarray(angles_deg)
    if a.size == 0:
        return np.zeros(0, dtype=bool)
    return (a - a.min()) < _HALF_TURN * k


def _half_groups(angles_deg: np.ndarray, mode: str):
    """Индексы (в исходном массиве) выбранных углов, разбитые по полуоборотам."""
    a = np.asarray(angles_deg)
    mask = select_angles(a, mode)
    k = halves_count(a, mode)
    idx = np.where(mask)[0]
    if k == 1:
        return [idx]
    rel = np.asarray(a[idx], dtype='float64') - float(np.asarray(a, dtype='float64').min())
    g = np.clip(np.floor(rel / _HALF_TURN).astype(int), 0, k - 1)
    groups = [idx[g == i] for i in range(k)]
    return [grp for grp in groups if grp.size]


# --- бэкенды ----------------------------------------------------------------------------------------------

def _astra_utils():
    from tomo.recon import astra_utils  # noqa: WPS433 — ленивый импорт (astra, GPU)
    return astra_utils


def astra_available() -> bool:
    """Импортируется ли ``tomo.recon.astra_utils`` (astra и её зависимости установлены)."""
    try:
        _astra_utils()
    except Exception:  # noqa: BLE001 — нет astra/tomopy, сломанная установка
        return False
    return True


def resolve_backend(backend: str = 'auto') -> str:
    """'auto' → 'astra', если импорт tomo.recon.astra_utils успешен и выбран GPU (gpu.is_gpu()), иначе 'cpu'."""
    if backend not in BACKENDS:
        raise ValueError('неизвестный бэкенд FBP: {} (допустимы {})'.format(backend, ', '.join(BACKENDS)))
    if backend != 'auto':
        return backend
    from . import gpu  # noqa: WPS433
    return 'astra' if gpu.is_gpu() and astra_available() else 'cpu'


def ramp_filter(size: int) -> np.ndarray:
    """Ramp-фильтр в частотной области, как ``skimage.transform.radon_transform._get_fourier_filter(size, 'ramp')``."""
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int), np.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2
    return 2 * np.real(np.fft.fft(f))


Region = Tuple[int, int, int, int]


def check_region(region: Optional[Sequence[int]], width: int) -> Optional[Region]:
    """Фрагмент среза w×w: None или весь срез → None; иначе (x0, y0, x1, y1) c 0 ≤ x0 < x1 ≤ w, 0 ≤ y0 < y1 ≤ w
    (иначе ValueError)."""
    if region is None:
        return None
    x0, y0, x1, y1 = (int(v) for v in region)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= width):
        raise ValueError('region {},{},{},{} вне среза {}×{}'.format(x0, y0, x1, y1, width, width))
    if (x0, y0, x1, y1) == (0, 0, width, width):
        return None
    return x0, y0, x1, y1


def astra_window(width: int, region: Region) -> Tuple[float, float, float, float]:
    """(min_x, max_x, min_y, max_y) для ``astra.create_vol_geom(y1 − y0, x1 − x0, ...)``: пиксели фрагмента region
    совпадают с пикселями [y0, y1) × [x0, x1) полного среза w×w (окно по умолчанию ±w/2; строка 0 — сверху, у max_y).
    Весь срез → (−w/2, w/2, −w/2, w/2)."""
    x0, y0, x1, y1 = region
    half = width / 2.0
    return x0 - half, x1 - half, half - y1, half - y0


def _fbp_cpu(sinos: np.ndarray, angles_deg: np.ndarray, region: Optional[Region] = None) -> np.ndarray:
    """FBP на numpy: sinos (s, n, w) → (s, w, w) float32 в единицах «ослабление на пиксель»; с region —
    (s, y1 − y0, x1 − x0): обратная проекция только в пикселях фрагмента."""
    s, n, w = sinos.shape
    size = max(64, int(2 ** np.ceil(np.log2(2 * w))))
    padded = np.zeros((s, n, size), dtype='float64')
    padded[:, :, :w] = sinos
    filtered = np.real(np.fft.ifft(np.fft.fft(padded, axis=-1) * ramp_filter(size), axis=-1))[:, :, :w]
    filtered = np.ascontiguousarray(filtered, dtype='float32')
    x0, y0, x1, y1 = region if region is not None else (0, 0, w, w)
    c = (w - 1) / 2.0
    xx = (np.arange(x0, x1, dtype='float64') - c)[None, :]
    yy = (np.arange(y0, y1, dtype='float64') - c)[:, None]
    rec = np.zeros((s, (y1 - y0) * (x1 - x0)), dtype='float32')
    for j, th in enumerate(np.deg2rad(np.asarray(angles_deg, dtype='float64'))):
        t = (xx * math.cos(th) - yy * math.sin(th) + c).ravel()   # индекс детектора
        valid = (t >= 0) & (t <= w - 1)
        i0 = np.clip(np.floor(t).astype(np.int64), 0, max(w - 2, 0))
        frac = (t - i0).astype('float32')
        frac[~valid] = 0
        wgt0 = (1 - frac) * valid
        wgt1 = frac
        col = filtered[:, j, :]
        rec += col[:, i0] * wgt0 + col[:, np.minimum(i0 + 1, w - 1)] * wgt1
    rec *= np.float32(np.pi / (2 * n))
    return rec.reshape(s, y1 - y0, x1 - x0)


def _astra_fbp_window(au, sino: np.ndarray, angles_deg: np.ndarray, region: Region) -> np.ndarray:
    """FBP_CUDA одной синограммы (n, w) только во фрагменте region — как ``astra_recon_2d_parallel`` (та же
    геометрия проекций, шаг детектора 1), но объём — окно ``astra_window``."""
    astra = au.astra
    w = sino.shape[-1]
    x0, y0, x1, y1 = region
    proj_geom = au.build_proj_geometry_parallel_2d(w, angles_deg, 1.0)
    vol_geom = astra.create_vol_geom(y1 - y0, x1 - x0, *astra_window(w, region))
    sino_id = astra.data2d.create('-sino', proj_geom, data=sino)
    rec_id = astra.data2d.create('-vol', vol_geom)
    try:
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sino_id
        cfg['option'] = {}
        alg_id = astra.algorithm.create(cfg)
        try:
            astra.algorithm.run(alg_id, 1)
        finally:
            astra.algorithm.delete(alg_id)
        return np.asarray(astra.data2d.get(rec_id), dtype='float32')
    finally:
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sino_id)


def _fbp_astra(sinos: np.ndarray, angles_deg: np.ndarray, region: Optional[Region] = None) -> np.ndarray:
    au = _astra_utils()
    ang = np.asarray(angles_deg, dtype='float64')
    if region is not None:
        return np.stack([_astra_fbp_window(au, np.ascontiguousarray(sino, dtype='float32'), ang, region)
                         for sino in sinos])
    out = [np.asarray(au.astra_recon_2d_parallel(np.ascontiguousarray(sino, dtype='float32'), ang,
                                                 [['FBP_CUDA']]), dtype='float32')
           for sino in sinos]
    return np.stack(out)


def recon_slice(sino: np.ndarray, angles_deg: np.ndarray, pixel_size: float,
                backend: str = 'auto', angle_mode: str = 'first_180',
                region: Optional[Sequence[int]] = None) -> np.ndarray:
    """Срез (w, w) float32 по синограмме (n, w). backend: 'auto' (astra, если доступна, иначе cpu), 'astra', 'cpu'.
    region — только фрагмент (см. recon_rows)."""
    from .gpu import to_numpy  # noqa: WPS433
    sino = to_numpy(sino)
    if sino.ndim != 2:
        raise ValueError('ожидается синограмма (n, w), получено {}'.format(sino.shape))
    return recon_rows(sino[None], angles_deg, pixel_size, backend=backend, angle_mode=angle_mode, region=region)[0]


def recon_rows(sino_rows: np.ndarray, angles_deg: np.ndarray, pixel_size: float,
               backend: str = 'auto', angle_mode: str = 'first_180',
               region: Optional[Sequence[int]] = None) -> np.ndarray:
    """Слой срезов: sino_rows (s, n, w) → (s, w, w) float32. region = (x0, y0, x1, y1) в пикселях среза w×w —
    восстановить только этот фрагмент: (s, y1 − y0, x1 − x0), те же значения, что обрезка полного среза (см.
    модуль); вне среза — ValueError."""
    from .gpu import to_numpy  # noqa: WPS433
    sino_rows = to_numpy(sino_rows)
    angles = np.asarray(angles_deg)
    if sino_rows.ndim != 3 or sino_rows.shape[1] != angles.shape[0]:
        raise ValueError('ожидается слой синограмм (s, n, w) с n = {} углов, получено {}'.format(
            angles.shape[0], sino_rows.shape))
    if not pixel_size or pixel_size <= 0:
        raise ValueError('pixel_size должен быть > 0, получено {}'.format(pixel_size))
    s, _, w = sino_rows.shape
    reg = check_region(region, w)
    be = resolve_backend(backend)
    groups = _half_groups(angles, angle_mode)
    if not groups:
        raise ValueError('нет углов для реконструкции')
    fbp = _fbp_astra if be == 'astra' else _fbp_cpu
    x0, y0, x1, y1 = reg if reg is not None else (0, 0, w, w)
    out = np.zeros((s, y1 - y0, x1 - x0), dtype='float32')
    for idx in groups:
        out += fbp(np.asarray(sino_rows[:, idx, :], dtype='float32'), angles[idx], reg)
    out /= np.float32(len(groups) * float(pixel_size))
    return out
