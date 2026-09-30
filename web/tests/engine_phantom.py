"""Синтетика для тестов движка реконструкции (reconengine).

Объект — набор 3D-гауссовых пятен; их параллельные проекции считаются аналитически, поэтому ось вращения
можно поставить в любое (дробное) место детектора и с любым наклоном без интерполяционных ошибок генератора.

Геометрия (координаты полного кадра детектора: столбец x, строка y, 0 — слева/сверху):
- ось проходит через точку (center_x, y_ref) с наклоном tilt_deg: столбец оси на строке y равен
  center_x + tan(tilt)·(y − y_ref) (как ``model.Axis``);
- точка объекта (xi, eta, zeta): zeta — вдоль оси (растёт вниз по строкам), (xi, eta) — плоскость среза;
- при угле θ точка проецируется в p = O + zeta·a + s·e1, s = xi·cos θ − eta·sin θ,
  a = (sin τ, cos τ), e1 = (cos τ, −sin τ) в координатах (x, y), τ = tilt;
- срез, восстановленный FBP (ось в столбце (w−1)/2), в пикселе (строка r, столбец c) равен
  μ(xi = c − (w−1)/2, eta = r − (w−1)/2) — соглашение skimage.transform.radon/iradon.

Значения проекций — линейные интегралы ослабления в единицах «μ на пиксель × пиксель».
"""
from __future__ import annotations

import dataclasses
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from reconengine.model import (MODE_DARK, MODE_DATA, MODE_DATA_CHECK, MODE_EMPTY, CropData, Overview, ROI,
                               ScanInfo)

SQRT_2PI = math.sqrt(2.0 * math.pi)


@dataclasses.dataclass
class Blob:
    """Гауссово пятно: центр (xi, eta, zeta), ширина sigma (пикс.), amp — μ в центре (1/пикс.)."""
    xi: float
    eta: float
    zeta: float
    sigma: float
    amp: float

    def line_integral(self, d2: np.ndarray) -> np.ndarray:
        """Проекция на детектор как функция квадрата расстояния до проекции центра."""
        return self.amp * self.sigma * SQRT_2PI * np.exp(-d2 / (2 * self.sigma ** 2))

    def density(self, r2: np.ndarray) -> np.ndarray:
        return self.amp * np.exp(-r2 / (2 * self.sigma ** 2))


@dataclasses.dataclass
class Ball:
    """Однородный шар с резкой границей: центр (xi, eta, zeta), радиус radius (пикс.), mu (1/пикс.)."""
    xi: float
    eta: float
    zeta: float
    radius: float
    mu: float

    def line_integral(self, d2: np.ndarray) -> np.ndarray:
        return 2 * self.mu * np.sqrt(np.maximum(self.radius ** 2 - d2, 0.0))

    def density(self, r2: np.ndarray) -> np.ndarray:
        return np.where(r2 <= self.radius ** 2, self.mu, 0.0)


def sharp_objects() -> list:
    """Шары с резкими краями и пятна — для фазовой корреляции (ей нужны высокие частоты)."""
    return [
        Ball(-4.0, 3.0, 0.0, 12.0, 0.02),
        Ball(9.0, -6.0, -10.0, 5.0, 0.05),
        Ball(-12.0, -8.0, 9.0, 4.0, 0.06),
        Ball(6.0, 12.0, 14.0, 3.5, 0.07),
        Blob(15.0, 5.0, -16.0, 2.5, 0.05),
        Blob(-6.0, -15.0, 18.0, 3.0, 0.05),
    ]


def make_blobs(seed: int = 0, n: int = 10, r_max: float = 24.0, z_range: Tuple[float, float] = (-20.0, 20.0),
               sigma_range: Tuple[float, float] = (1.8, 4.0), amp_range: Tuple[float, float] = (0.02, 0.06)
               ) -> List[Blob]:
    """Случайные пятна внутри цилиндра радиуса r_max (с запасом 3σ), высоты z_range."""
    rng = np.random.default_rng(seed)
    blobs = []
    while len(blobs) < n:
        sigma = float(rng.uniform(*sigma_range))
        r_lim = r_max - 3 * sigma
        xi, eta = rng.uniform(-r_lim, r_lim, size=2)
        if xi * xi + eta * eta > r_lim * r_lim:
            continue
        zeta = float(rng.uniform(*z_range))
        blobs.append(Blob(float(xi), float(eta), zeta, sigma, float(rng.uniform(*amp_range))))
    return blobs


def asymmetric_blobs() -> List[Blob]:
    """Детерминированный несимметричный объект (для поиска оси и центра): крупное тело + мелкие детали."""
    return [
        Blob(-6.0, 4.0, 0.0, 9.0, 0.020),
        Blob(10.0, -7.0, -8.0, 2.5, 0.060),
        Blob(-14.0, -9.0, 6.0, 2.0, 0.050),
        Blob(4.0, 15.0, 12.0, 3.0, 0.045),
        Blob(17.0, 6.0, -14.0, 2.2, 0.055),
        Blob(-3.0, -17.0, 16.0, 2.8, 0.040),
        Blob(0.0, 0.0, -20.0, 3.5, 0.035),
        Blob(-18.0, 8.0, -4.0, 1.8, 0.070),
    ]


def project(blobs: Sequence[Blob], angles_deg: Sequence[float], height: int, width: int,
            center_x: float, y_ref: float, tilt_deg: float = 0.0,
            offsets: Optional[np.ndarray] = None) -> np.ndarray:
    """Аналитические проекции (k, height, width) float64.

    offsets — (k, 2) сдвиги (dy, dx) проекции в плоскости детектора (имитация смещения образца)."""
    angles = np.deg2rad(np.asarray(angles_deg, dtype='float64'))
    tau = math.radians(tilt_deg)
    ax, ay = math.sin(tau), math.cos(tau)
    ex, ey = math.cos(tau), -math.sin(tau)
    yy, xx = np.mgrid[0:height, 0:width].astype('float64')
    out = np.zeros((len(angles), height, width), dtype='float64')
    for i, th in enumerate(angles):
        dy, dx = (0.0, 0.0) if offsets is None else (float(offsets[i][0]), float(offsets[i][1]))
        c, s_ = math.cos(th), math.sin(th)
        for b in blobs:
            s = b.xi * c - b.eta * s_
            px = center_x + b.zeta * ax + s * ex + dx
            py = y_ref + b.zeta * ay + s * ey + dy
            d2 = (xx - px) ** 2 + (yy - py) ** 2
            out[i] += b.line_integral(d2)
    return out


def sinogram(blobs: Sequence[Blob], angles_deg: Sequence[float], width: int, zeta: float = 0.0,
             center: Optional[float] = None) -> np.ndarray:
    """Синограмма среза zeta (k, width), ось в столбце center (по умолчанию (width−1)/2), наклона нет."""
    c0 = (width - 1) / 2.0 if center is None else center
    return project(blobs, angles_deg, 1, width, c0, -zeta)[:, 0, :]


def slice_truth(blobs: Sequence[Blob], width: int, zeta: float = 0.0) -> np.ndarray:
    """Истинный срез (width, width): μ(xi = c − (w−1)/2, eta = r − (w−1)/2)."""
    c0 = (width - 1) / 2.0
    eta, xi = np.mgrid[0:width, 0:width].astype('float64') - c0
    out = np.zeros((width, width), dtype='float64')
    for b in blobs:
        out += b.density((xi - b.xi) ** 2 + (eta - b.eta) ** 2 + (zeta - b.zeta) ** 2)
    return out


def flat_fields(height: int, width: int, dark_level: float = 100.0, empty_level: float = 3000.0,
                seed: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Неоднородные dark и empty (плавный градиент + фиксированный шум пикселей), float64."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype('float64')
    dark = dark_level * (1 + 0.1 * rng.standard_normal((height, width)))
    empty = empty_level * (1 + 0.2 * np.sin(xx / max(width, 1) * 3) * np.cos(yy / max(height, 1) * 2))
    return dark, dark + empty


def to_counts(p: np.ndarray, dark: np.ndarray, empty: np.ndarray, gain: float = 1.0,
              rng: Optional[np.random.Generator] = None, noise: float = 0.0) -> np.ndarray:
    """Отсчёты детектора I = D + gain·(E − D)·exp(−p) (+ гауссов шум noise·√I), uint16."""
    i = dark + gain * (empty - dark) * np.exp(-p)
    if rng is not None and noise > 0:
        i = i + noise * np.sqrt(np.maximum(i, 0)) * rng.standard_normal(i.shape)
    return np.clip(np.rint(i), 0, 65535).astype('uint16')


def bin_frames(frames: np.ndarray, b: int) -> np.ndarray:
    """Среднее b×b по последним двум осям (края, не кратные b, отбрасываются), float32."""
    frames = np.asarray(frames, dtype='float32')
    h, w = frames.shape[-2] // b * b, frames.shape[-1] // b * b
    f = frames[..., :h, :w]
    f = f.reshape(f.shape[:-2] + (h // b, b, w // b, b))
    return f.mean(axis=(-3, -1)).astype('float32')


# --- timeline и сведения о скане ---------------------------------------------------------------------------

@dataclasses.dataclass
class Timeline:
    modes: np.ndarray            # (N,) uint8
    angles: np.ndarray           # (N,) float32
    segments: np.ndarray         # (N,) int — номер сегмента (для advanced), −1 для dark
    frame_numbers: np.ndarray    # (N,) int64

    def as_frame_dicts(self) -> List[dict]:
        """Формат helpers.build_advanced_timeline (для helpers.make_v2_file)."""
        names = {MODE_DARK: 'dark', MODE_EMPTY: 'empty', MODE_DATA: 'data', MODE_DATA_CHECK: 'data_check'}
        return [{'mode': names[int(m)], 'angle': float(a), 'segment': int(s)}
                for m, a, s in zip(self.modes, self.angles, self.segments)]


def simple_timeline(angles: Sequence[float], n_dark: int = 3, n_empty: int = 4) -> Timeline:
    """dark × n_dark, empty × n_empty, data по углам."""
    modes = [MODE_DARK] * n_dark + [MODE_EMPTY] * n_empty + [MODE_DATA] * len(angles)
    ang = [0.0] * (n_dark + n_empty) + list(angles)
    seg = [-1] * n_dark + [0] * (n_empty + len(angles))
    n = len(modes)
    return Timeline(np.array(modes, 'uint8'), np.array(ang, 'float32'), np.array(seg, 'int64'),
                    np.arange(n, dtype='int64'))


def advanced_timeline(angles: Sequence[float], n_segments: int, series_length: int = 3,
                      n_dark: int = 3) -> Timeline:
    """Как драйвер: dark, начальная empty-серия, data сегмента 0, затем для k = 1..n_segments−1:
    empty-серия, data_check при угле последнего data, data сегмента k. Углы делятся на сегменты поровну."""
    angles = list(angles)
    per = int(math.ceil(len(angles) / n_segments))
    modes, ang, seg = [], [], []
    modes += [MODE_DARK] * n_dark
    ang += [0.0] * n_dark
    seg += [-1] * n_dark
    last_angle = 0.0
    for k in range(n_segments):
        chunk = angles[k * per:(k + 1) * per]
        modes += [MODE_EMPTY] * series_length
        ang += [0.0] * series_length
        seg += [k] * series_length
        if k > 0:
            modes.append(MODE_DATA_CHECK)
            ang.append(last_angle)
            seg.append(k)
        modes += [MODE_DATA] * len(chunk)
        ang += chunk
        seg += [k] * len(chunk)
        last_angle = chunk[-1]
    n = len(modes)
    return Timeline(np.array(modes, 'uint8'), np.array(ang, 'float32'), np.array(seg, 'int64'),
                    np.arange(n, dtype='int64'))


def make_scan_info(tl: Timeline, height: int, width: int, is_advanced: bool, series_length: int = 0,
                   empty_period: int = 0) -> ScanInfo:
    modes = tl.modes
    return ScanInfo(
        path='synthetic.h5', exp_id='synthetic', n_frames=len(modes), height=height, width=width,
        dtype='uint16', chunk_frames=4, compression='gzip', shuffle=False, fast_path=True,
        angles=tl.angles.astype('float64'), modes=modes, frame_numbers=tl.frame_numbers,
        dark_idx=np.where(modes == MODE_DARK)[0], empty_idx=np.where(modes == MODE_EMPTY)[0],
        data_idx=np.where(modes == MODE_DATA)[0], check_idx=np.where(modes == MODE_DATA_CHECK)[0],
        is_advanced=is_advanced, series_length=series_length, empty_period=empty_period,
        metadata={}, fingerprint='synthetic')


def make_crop(frames_full: np.ndarray, roi: ROI) -> CropData:
    """CropData из полных кадров (N, H, W) uint16 (в памяти, без memmap)."""
    roi.validate(frames_full.shape[1], frames_full.shape[2])
    fr = np.ascontiguousarray(frames_full[:, roi.y0:roi.y1, roi.x0:roi.x1])
    return CropData(roi=roi, frames=fr, path='', fingerprint='synthetic')


@dataclasses.dataclass
class SyntheticScan:
    """Полный синтетический скан: кадры в порядке timeline, истинная геометрия и объект."""
    scan: ScanInfo
    frames: np.ndarray           # (N, H, W) uint16
    blobs: List[Blob]
    dark: np.ndarray
    empty: np.ndarray
    center_x: float
    y_ref: float
    tilt_deg: float
    projections: np.ndarray      # (N, H, W) float64 — линейные интегралы (0 для dark/empty)


def make_synthetic_scan(angles: Sequence[float], height: int = 64, width: int = 96, center_x: float = 44.3,
                        y_ref: float = 31.5, tilt_deg: float = 0.0, blobs: Optional[List[Blob]] = None,
                        advanced: bool = False, n_segments: int = 3, series_length: int = 3,
                        segment_offsets: Optional[Sequence[Tuple[float, float]]] = None,
                        segment_angle_offsets: Optional[Sequence[float]] = None,
                        drift: float = 0.0, noise: float = 0.0, seed: int = 0,
                        frame_dx: Optional[Sequence[float]] = None) -> SyntheticScan:
    """Скан с аналитическими проекциями.

    advanced: периодические empty-серии и data_check; segment_offsets[k] = (dy, dx) — абсолютный сдвиг образца
    в плоскости детектора в сегменте k (сегмент 0 — (0, 0)); segment_angle_offsets[k] — сбой угла в сегменте k,
    градусы: data и data_check сегмента сняты под углом «записанный + сбой» (поворот образца во время вставки,
    которого счётчик мотора не видит); drift — относительный дрейф яркости источника за весь скан (линейно по
    frame_number, для проверки интерполяции empty); frame_dx — сдвиг образца по x для каждого кадра timeline
    (действует на data и data_check; имитация смещения образца во время съёмки)."""
    blobs = asymmetric_blobs() if blobs is None else blobs
    if advanced:
        tl = advanced_timeline(angles, n_segments, series_length=series_length)
    else:
        tl = simple_timeline(angles)
    n = len(tl.modes)
    offsets = np.zeros((n, 2))
    if advanced and segment_offsets is not None:
        for i in range(n):
            if tl.modes[i] in (MODE_DATA, MODE_DATA_CHECK):
                offsets[i] = segment_offsets[int(tl.segments[i])]
    obj = (tl.modes == MODE_DATA) | (tl.modes == MODE_DATA_CHECK)
    if frame_dx is not None:
        offsets[:, 1] += np.where(obj, np.asarray(frame_dx, dtype='float64'), 0.0)
    p = np.zeros((n, height, width))
    idx = np.where(obj)[0]
    true_angles = tl.angles.astype('float64')
    if advanced and segment_angle_offsets is not None:
        true_angles = true_angles + np.array([segment_angle_offsets[int(s)] if s >= 0 else 0.0 for s in tl.segments])
    p[idx] = project(blobs, true_angles[idx], height, width, center_x, y_ref, tilt_deg, offsets[idx])
    dark, empty = flat_fields(height, width, seed=seed + 1)
    rng = np.random.default_rng(seed) if noise > 0 else None
    frames = np.empty((n, height, width), dtype='uint16')
    for i in range(n):
        gain = 1.0 + drift * i / max(n - 1, 1)
        if tl.modes[i] == MODE_DARK:
            frames[i] = to_counts(np.full((height, width), np.inf), dark, empty, rng=rng, noise=noise)
        else:
            frames[i] = to_counts(p[i], dark, empty, gain=gain, rng=rng, noise=noise)
    scan = make_scan_info(tl, height, width, advanced, series_length if advanced else 0,
                          empty_period=0)
    return SyntheticScan(scan, frames, list(blobs), dark, empty, center_x, y_ref, tilt_deg, p)


def make_overview(ss: SyntheticScan, sample_idx: Sequence[int], b: int = 2) -> Overview:
    """Обзор с биннингом b по синтетическому скану."""
    sc = ss.scan
    dark = np.median(bin_frames(ss.frames[sc.dark_idx], b), axis=0).astype('float32')
    empty = np.median(bin_frames(ss.frames[sc.empty_idx[:3]], b), axis=0).astype('float32')
    sample_idx = np.asarray(sample_idx)
    return Overview(bin=b, dark=dark, empty=empty, samples=bin_frames(ss.frames[sample_idx], b),
                    sample_idx=sample_idx, sample_angles=sc.angles[sample_idx].astype('float64'),
                    full_height=sc.height, full_width=sc.width)
