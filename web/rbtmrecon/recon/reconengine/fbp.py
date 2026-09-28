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

Выбор углов (``select_angles``):
- 'first_180'   — углы с (a − a.min) < 180 (текущее поведение);
- 'full_halves' — наибольшее кратное 180° число полуоборотов k от начала скана (для 0–360° — все углы).
  Каждый полуоборот восстанавливается отдельно (обычный 180°-скан, масштаб FBP корректен при любой нормировке
  бэкенда), результаты усредняются, т.е. сумма делится на k: значения не зависят от числа полуоборотов.
"""
from __future__ import annotations

import logging
import math

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


def _fbp_cpu(sinos: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """FBP на numpy: sinos (s, n, w) → (s, w, w) float32 в единицах «ослабление на пиксель»."""
    s, n, w = sinos.shape
    size = max(64, int(2 ** np.ceil(np.log2(2 * w))))
    padded = np.zeros((s, n, size), dtype='float64')
    padded[:, :, :w] = sinos
    filtered = np.real(np.fft.ifft(np.fft.fft(padded, axis=-1) * ramp_filter(size), axis=-1))[:, :, :w]
    filtered = np.ascontiguousarray(filtered, dtype='float32')
    c = (w - 1) / 2.0
    coord = np.arange(w, dtype='float64') - c
    xx = coord[None, :]
    yy = coord[:, None]
    rec = np.zeros((s, w * w), dtype='float32')
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
    return rec.reshape(s, w, w)


def _fbp_astra(sinos: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    au = _astra_utils()
    ang = np.asarray(angles_deg, dtype='float64')
    out = [np.asarray(au.astra_recon_2d_parallel(np.ascontiguousarray(sino, dtype='float32'), ang,
                                                 [['FBP_CUDA']]), dtype='float32')
           for sino in sinos]
    return np.stack(out)


def recon_slice(sino: np.ndarray, angles_deg: np.ndarray, pixel_size: float,
                backend: str = 'auto', angle_mode: str = 'first_180') -> np.ndarray:
    """Срез (w, w) float32 по синограмме (n, w). backend: 'auto' (astra, если доступна, иначе cpu), 'astra', 'cpu'."""
    from .gpu import to_numpy  # noqa: WPS433
    sino = to_numpy(sino)
    if sino.ndim != 2:
        raise ValueError('ожидается синограмма (n, w), получено {}'.format(sino.shape))
    return recon_rows(sino[None], angles_deg, pixel_size, backend=backend, angle_mode=angle_mode)[0]


def recon_rows(sino_rows: np.ndarray, angles_deg: np.ndarray, pixel_size: float,
               backend: str = 'auto', angle_mode: str = 'first_180') -> np.ndarray:
    """Слой срезов: sino_rows (s, n, w) → (s, w, w) float32."""
    from .gpu import to_numpy  # noqa: WPS433
    sino_rows = to_numpy(sino_rows)
    angles = np.asarray(angles_deg)
    if sino_rows.ndim != 3 or sino_rows.shape[1] != angles.shape[0]:
        raise ValueError('ожидается слой синограмм (s, n, w) с n = {} углов, получено {}'.format(
            angles.shape[0], sino_rows.shape))
    if not pixel_size or pixel_size <= 0:
        raise ValueError('pixel_size должен быть > 0, получено {}'.format(pixel_size))
    be = resolve_backend(backend)
    groups = _half_groups(angles, angle_mode)
    if not groups:
        raise ValueError('нет углов для реконструкции')
    fbp = _fbp_astra if be == 'astra' else _fbp_cpu
    s, _, w = sino_rows.shape
    out = np.zeros((s, w, w), dtype='float32')
    for idx in groups:
        out += fbp(np.asarray(sino_rows[:, idx, :], dtype='float32'), angles[idx])
    out /= np.float32(len(groups) * float(pixel_size))
    return out
