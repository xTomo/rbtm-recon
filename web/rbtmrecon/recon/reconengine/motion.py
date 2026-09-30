"""Смещение образца во время съёмки: оценка по самим проекциям и компенсация сдвигом кадров.

Если образец неподвижен, центр масс проекции (−ln T) вдоль линии, перпендикулярной оси, меняется с углом строго по
синусоиде a + b·cos θ + c·sin θ; a — положение оси на этой линии. Сдвиг образца по горизонтали на dx(t) добавляет к
центру масс dx(t) — одинаково на всех высотах. Сбой угла даёт другое: отклонение пропорционально амплитуде синусоиды
полосы и зависит от её фазы, то есть различается по высоте. Поэтому:

- по полосам строк (линии ⟂ оси с наклоном оси, на кадрах, уменьшенных в ``bin`` раз) считается центр масс, из него
  вычитается синусоида, подогнанная по углам; полосы почти без вещества отбрасываются;
- медиана остатков по полосам — оценка сдвига кадра; согласие полос (доля общей части) отличает сдвиг образца от
  сбоя угла и прочих несогласованностей;
- кадры сразу после периодической вставки (``skip_after_insert``) в оценку не входят: у части камер после возврата
  образца несколько кадров искажены инерцией детектора, центр масс там скачет, а изображение не сдвигается
  (проверено по контрольным кадрам: сдвиг на вставках < 0,1 px); оценка сглаживается по времени (``smooth_frames``)
  и через эти кадры проводится непрерывно;
- из сглаженной оценки снова вычитается синусоида: постоянная часть сдвига неотличима от положения оси (её берёт
  авто-ось), первая гармоника — от положения объекта (сдвигает его в срезе, резкость не меняет).

Компенсация — сдвиг кадра по x на −dx (``Prepared.frame_sx``), вместе со сдвигами после вставок. После неё авто-ось
считается заново. Оценка проверена на синтетике с заданным дрейфом (ошибка 0,1–0,2 px при размахе 13 px) и на
реальных сканах: af443cef (сдвиг до 13 px, центр оси −3 px), 79d1ba3e (±2 px, резкость +2…13 %), 524efd6e (сбой угла
на вставках — полосы не согласованы, компенсация не включается).
"""
from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from . import gpu, preprocess
from .model import CropData, check_cancel

logger = logging.getLogger(__name__)

MODES = ('auto', 'on', 'off')
DEFAULTS: Dict[str, Any] = {
    'bin': 4,                  # уменьшение кадров для оценки
    'bands': 16,               # полос строк
    'band_rows': 7,            # строк (уменьшенных) в полосе
    'keep_mass': 0.25,         # полоса учитывается, если в ней ≥ этой доли медианной «массы»
    'skip_after_insert': 2,    # data-кадров после каждой периодической вставки не учитывать
    'smooth_frames': 2.0,      # σ сглаживания по времени, кадров
    'min_rms': 0.75,           # СКО сдвига, px детектора, начиная с которого смещение считается найденным
    'min_common': 0.5,         # доля общей для полос части остатка, начиная с которой это сдвиг образца
}
STATUSES = ('detected', 'none', 'inconsistent', 'no_object')
_CHUNK = 16


@dataclasses.dataclass
class Estimate:
    """Оценка смещения. Массивы — по data-кадрам в порядке съёмки (как ``Prepared.idx``)."""
    fnums: np.ndarray          # frame_number кадров
    angles: np.ndarray         # их углы, градусы
    dx: np.ndarray             # сглаженный сдвиг образца по x, px детектора (компенсация: frame_sx − dx)
    dx_raw: np.ndarray         # медиана остатков по полосам до сглаживания; NaN — кадр не учитывался
    rms: float                 # СКО dx
    ptp: float                 # размах dx
    discrepancy: float         # медиана |остаток полосы − медиана|, px — расхождение полос
    common: float              # доля общей для полос части остатка (1 — все полосы одинаковы)
    bands: int                 # полос учтено
    status: str                # detected | none | inconsistent | no_object

    def summary(self) -> Dict[str, Any]:
        return {'status': self.status, 'rms': round(self.rms, 3), 'ptp': round(self.ptp, 3),
                'discrepancy': round(self.discrepancy, 3), 'common': round(self.common, 3), 'bands': self.bands}

    def to_dict(self) -> Dict[str, Any]:
        d = self.summary()
        d.update(fnums=[int(v) for v in self.fnums], angles=[round(float(v), 4) for v in self.angles],
                 dx=[round(float(v), 3) for v in self.dx],
                 dx_raw=[None if not np.isfinite(v) else round(float(v), 3) for v in self.dx_raw])
        return d


def params(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = dict(DEFAULTS)
    if overrides:
        unknown = set(overrides) - set(DEFAULTS)
        if unknown:
            raise ValueError('motion: неизвестные параметры {}'.format(sorted(unknown)))
        p.update(overrides)
    return p


# --- профили полос ---------------------------------------------------------------------------------------------

def _bin2(a: np.ndarray, b: int) -> np.ndarray:
    h, w = a.shape[-2] // b * b, a.shape[-1] // b * b
    a = np.asarray(a, dtype='float32')[..., :h, :w]
    return a.reshape(a.shape[:-2] + (h // b, b, w // b, b)).mean(axis=(-3, -1))


def bin_dark_empty(de: preprocess.DarkEmpty, b: int) -> preprocess.DarkEmpty:
    """Опорные кадры, уменьшенные в b раз усреднением (как кадры в :func:`band_profiles`)."""
    return preprocess.DarkEmpty(_bin2(de.dark, b), _bin2(de.initial_empty, b), de.initial_empty_fnumber,
                                [_bin2(e, b) for e in de.periodic_empties], list(de.periodic_empty_fnumbers))


def band_lines(hb: int, wb: int, tilt_deg: float, n_bands: int, band_rows: int) -> Tuple[np.ndarray, np.ndarray]:
    """Координаты (y, x) линий ⟂ оси с наклоном tilt_deg: (K, D, wb) каждая; и y центров полос на столбце wb/2.
    Полосы размещены так, чтобы линии с наклоном не выходили за кадр."""
    half = band_rows // 2
    t = math.tan(math.radians(tilt_deg))
    reach = abs(t) * (wb - 1) / 2.0
    lo, hi = half + reach + 1, hb - 2 - half - reach
    k = 0 if hi <= lo else max(1, min(int(n_bands), int(hi - lo) // max(1, band_rows)))
    ys = np.linspace(lo, hi, k) if k else np.zeros(0)
    xs = np.arange(wb, dtype='float64')
    xc = (wb - 1) / 2.0
    yy = ys[:, None, None] + np.arange(-half, half + 1)[None, :, None] - (xs - xc)[None, None, :] * t
    xx = np.broadcast_to(xs, yy.shape)
    return np.stack([yy, xx]), ys


def band_profiles(crop: CropData, idx: Sequence[int], fnums: Sequence[int], de: preprocess.DarkEmpty,
                  frame_sy: np.ndarray, frame_sx: np.ndarray, tilt_deg: float,
                  p: Optional[Dict[str, Any]] = None, xp=None, cancel=None) -> Tuple[np.ndarray, np.ndarray]:
    """Профили −ln T вдоль линий ⟂ оси для каждого data-кадра: (n, K, wb) float32 и y центров полос (уменьшенные
    координаты кропа). Кадры уменьшаются в bin раз усреднением сырых отсчётов (на GPU, если есть), нормируются
    уменьшенными опорными кадрами (без медианы 3×3), сдвигаются на frame_sy/frame_sx (сдвиги после вставок)."""
    import scipy.ndimage as ndi  # noqa: WPS433

    p = params(p)
    h0, w0 = crop.frames.shape[1], crop.frames.shape[2]
    b = effective_bin(h0, w0, int(p['bin']))
    xp = xp or gpu.get_xp()
    nd = gpu.ndimage(xp)
    h, w = crop.frames.shape[1], crop.frames.shape[2]
    hb, wb = h // b, w // b
    de_b = bin_dark_empty(de, b)
    coords, ys = band_lines(hb, wb, tilt_deg, p['bands'], p['band_rows'])
    K, D = coords.shape[1], coords.shape[2]
    flat = [coords[0].ravel(), coords[1].ravel()]
    idx = np.asarray(idx, dtype=np.int64)
    fnums = np.asarray(fnums)
    out = np.empty((len(idx), K, wb), dtype='float32')
    for a in range(0, len(idx), _CHUNK):
        check_cancel(cancel)
        sl = slice(a, min(len(idx), a + _CHUNK))
        raw = xp.asarray(np.asarray(crop.frames[idx[sl], :hb * b, :wb * b]))
        n = raw.shape[0]
        small = raw.reshape(n, hb, b, wb, b).astype(xp.float32).mean(axis=(2, 4))
        norm = preprocess.normalize_slab(small, fnums[sl], de_b, xp=xp, median3=False)
        for i in range(n):
            sy, sx = float(frame_sy[a + i]) / b, float(frame_sx[a + i]) / b
            if abs(sy) >= 1e-6 or abs(sx) >= 1e-6:
                norm[i] = nd.shift(norm[i], [sy, sx], order=1, mode='nearest')
        norm = gpu.to_numpy(norm)
        for i in range(n):
            out[a + i] = ndi.map_coordinates(norm[i], flat, order=1, mode='nearest').reshape(K, D, wb).mean(axis=1)
    return out, ys


def effective_bin(height: int, width: int, b: int) -> int:
    """Уменьшение не больше заданного и такое, чтобы в уменьшенном кадре осталось ≥ 256 столбцов и ≥ 128 строк
    (маленький кроп — без уменьшения)."""
    return max(1, min(int(b), int(width) // 256, int(height) // 128))


# --- оценка ------------------------------------------------------------------------------------------------------

def _excluded(fnums: np.ndarray, insert_fnums: Sequence[int], skip: int) -> np.ndarray:
    """Маска data-кадров, идущих первыми skip после каждой периодической вставки (порядок — по frame_number)."""
    ex = np.zeros(len(fnums), dtype=bool)
    if skip <= 0:
        return ex
    order = np.argsort(fnums, kind='stable')
    fs = fnums[order]
    for f0 in insert_fnums:
        after = np.nonzero(fs > f0)[0][:skip]
        ex[order[after]] = True
    return ex


def _sinusoid(angles_deg: np.ndarray) -> np.ndarray:
    th = np.radians(np.asarray(angles_deg, dtype='float64'))
    return np.stack([np.ones_like(th), np.cos(th), np.sin(th)], 1)


def _smooth(values: np.ndarray, weights: np.ndarray, sigma: float) -> np.ndarray:
    """Гауссово сглаживание по порядку кадров с весами (нулевой вес — значение восстанавливается по соседям)."""
    import scipy.ndimage as ndi  # noqa: WPS433
    v = np.where(weights > 0, np.nan_to_num(values), 0.0)
    if sigma <= 0:
        return np.where(weights > 0, v, np.interp(np.arange(len(v)), np.nonzero(weights > 0)[0], v[weights > 0]))
    num = ndi.gaussian_filter1d(v * weights, sigma, mode='nearest')
    den = ndi.gaussian_filter1d(weights.astype('float64'), sigma, mode='nearest')
    return num / np.maximum(den, 1e-12)


def estimate(profiles: np.ndarray, angles: Sequence[float], fnums: Sequence[int], insert_fnums: Sequence[int],
             p: Optional[Dict[str, Any]] = None, bin_used: Optional[int] = None) -> Estimate:
    """Оценка смещения по профилям полос (:func:`band_profiles`; bin_used — их уменьшение, см.
    :func:`effective_bin`, по умолчанию p['bin']). Кадры — в порядке съёмки."""
    p = params(p)
    b = float(bin_used if bin_used is not None else p['bin'])
    angles = np.asarray(angles, dtype='float64')
    fnums = np.asarray(fnums)
    n = len(angles)
    zeros = np.zeros(n)
    s = np.clip(np.asarray(profiles, dtype='float64'), 0, None)
    if s.ndim != 3 or s.shape[1] == 0:
        return Estimate(fnums, angles, zeros, zeros * np.nan, 0.0, 0.0, 0.0, 0.0, 0, 'no_object')
    mass = s.sum(axis=2)                                           # (n, K)
    mean_mass = mass.mean(axis=0)
    keep = mean_mass > p['keep_mass'] * np.median(mean_mass) if mean_mass.size else np.zeros(0, bool)
    if keep.sum() < 3 or n < 8:
        return Estimate(fnums, angles, zeros, zeros * np.nan, 0.0, 0.0, 0.0, 0.0, int(keep.sum()), 'no_object')
    xs = np.arange(s.shape[2], dtype='float64')
    cm = (s[:, keep, :] * xs).sum(axis=2) / np.maximum(mass[:, keep], 1e-12)    # (n, Kk)
    use = ~_excluded(fnums, insert_fnums, int(p['skip_after_insert']))
    A = _sinusoid(angles)
    coef, *_ = np.linalg.lstsq(A[use], cm[use], rcond=None)
    r = (cm - A @ coef) * b                                         # остатки полос, px детектора
    dx_raw = np.median(r, axis=1)
    dev = np.abs(r[use] - dx_raw[use, None])
    discrepancy = float(np.median(dev))
    spread = float(np.median(np.abs(r[use])))
    common = float(1.0 - discrepancy / spread) if spread > 1e-9 else 1.0
    order = np.argsort(fnums, kind='stable')                       # сглаживание — по времени
    sm = np.empty(n)
    sm[order] = _smooth(dx_raw[order], use[order].astype('float64'), float(p['smooth_frames']))
    c2, *_ = np.linalg.lstsq(A, sm, rcond=None)
    dx = sm - A @ c2
    rms, ptp = float(np.std(dx)), float(np.ptp(dx))
    level = float(np.sqrt(np.mean(r[use] ** 2)))                    # общий остаток полос (сбой угла — большой)
    if max(rms, level) < p['min_rms']:
        status = 'none'
    elif common < p['min_common']:
        status = 'inconsistent'
    else:
        status = 'detected'
    raw = np.where(use, dx_raw, np.nan)
    logger.info('смещение образца: %s, СКО %.2f px, размах %.1f px, согласие полос %.0f%% (%d полос)',
                status, rms, ptp, 100 * common, int(keep.sum()))
    return Estimate(fnums, angles, dx, raw, rms, ptp, discrepancy, common, int(keep.sum()), status)


def decide(est: Optional[Estimate], mode: str, rotated: bool = False) -> Tuple[bool, str]:
    """Компенсировать ли смещение: (да/нет, пояснение для человека). rotated — проверка контрольных кадров нашла
    сбой угла (тогда остаток — не сдвиг, и авто не включается)."""
    if mode not in MODES:
        raise ValueError('motion.mode: ожидается одно из {}, получено {!r}'.format(', '.join(MODES), mode))
    if mode == 'off':
        return False, 'компенсация выключена'
    if est is None:
        return False, 'смещение не измерялось'
    if est.status == 'no_object':
        return False, 'в кадре мало вещества — смещение не измерить'
    size = 'СКО {:.1f} px, размах {:.1f} px'.format(est.rms, est.ptp)
    if mode == 'on':
        return True, 'компенсация включена вручную ({})'.format(size)
    if rotated:
        return False, 'на вставках сбой угла — остаток не компенсируется сдвигом ({})'.format(size)
    if est.status == 'none':
        return False, 'смещения нет ({})'.format(size)
    if est.status == 'inconsistent':
        return False, ('отклонение различается по высоте (согласие {:.0%}) — это не сдвиг образца, возможно, сбой '
                       'угла ({})').format(est.common, size)
    return True, 'образец смещался ({}) — компенсируется'.format(size)


def from_block(block: Dict[str, Any], fnums: Sequence[int]) -> Optional[np.ndarray]:
    """dx из блока рецепта, если он задан и относится к тем же кадрам; иначе None. Несовпадение кадров — ValueError."""
    dx, fn = block.get('dx'), block.get('fnums')
    if dx is None:
        return None
    fnums = [int(v) for v in fnums]
    if fn is None or [int(v) for v in fn] != fnums:
        raise ValueError('motion.dx задан для других кадров ({} против {} data-кадров скана)'.format(
            len(fn or []), len(fnums)))
    return np.asarray(dx, dtype='float64')
