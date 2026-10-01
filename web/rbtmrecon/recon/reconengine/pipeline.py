"""Реконструкция по рецепту: конвейер по слоям строк кропа.

Порядок операций — как в ноутбуке ``reconstructor4.py``:
    нормировка (−ln T, safe_median) → коррекция смещения образца (advanced) → выравнивание оси
    (``transform_image``) → подавление колец → FBP по выбранным углам → запись объёма.
Отличие — данные не держатся целиком: из кропа (memmap uint16) берётся слой из ``slab_rows`` выходных строк
плюс запас сверху и снизу, где запас покрывает наклон оси (``axis.margin_rows``), вертикальный сдвиг образца
и затухание сплайн-префильтров (+8 строк). Всё, что зависит от соседних строк (медиана 3×3, сдвиг, поворот),
на выходных строках слоя совпадает с обработкой полного кропа; кольца и FBP считаются по каждой строке отдельно.

Кольца, как и в ноутбуке, подавляются по всем data-кадрам, углы для FBP выбираются уже после
(``fbp.recon_rows``).

Сглаживание проекций с деблюром (блок рецепта ``smoothing``, модуль :mod:`smoothing`) — линейный фильтр проекций
после колец и перед FBP; он связывает соседние строки: выходной строке нужны ±h строк после колец (h —
``smoothing.halo_rows``). Поэтому при включённом сглаживании слой [a, b) выравнивается и чистится от колец с ореолом
(a − h, b + h) ∩ [0, H) (ореол входит в запас входных строк), затем ``smoothing.apply`` оставляет строки [a, b) — они
точно совпадают с фильтром по всему кропу (за краями кропа — отражение). Выключено — слой как раньше, без ореола.

Шумоподавление объёма (блок ``denoise``, модуль :mod:`tv`) — TV 3D после FBP: срезы идут в ``tv.DenoiseWriter``,
который копит их и отдаёт писателю объёма порциями после TV по порции ± ``tv.HALO`` срезов (на GPU плитками по
y, x). Выборка статистики объёма — после TV.
"""
from __future__ import annotations

import dataclasses
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import __version__ as ENGINE_VERSION
from . import axis as axis_mod
from . import data, fbp, gpu, outputs, preprocess, rings, smoothing, tv
from . import motion as motion_mod
from . import recipe as recipe_mod
from .model import Axis, CropData, ProgressFn, ScanInfo, check_cancel, no_progress

logger = logging.getLogger(__name__)

#: Дополнительный запас строк на затухание префильтра сплайна при сдвиге образца (см. axis.align_rows).
_SHIFT_EXTRA_ROWS = 8
#: Модель памяти GPU слоя в единицах «строка × все кадры × float32» (n·w·4 байт):
#: на входную строку — слой float32, копия (медиана или сдвиг) и маска; на выходную — выровненная синограмма;
#: кольца и FBP идут кусками по _RING_CHUNK строк, remove_all_stripe держит ~8 копий куска. Со сглаживанием
#: входных и выровненных строк на 2h больше (ореол); фильтр — после колец: выровненные строки, результат (s строк)
#: и рабочие порции (_SMOOTH_WORK_CHUNKS · smoothing.CHUNK_BYTES — спектр, обратное преобразование, кусок с
#: дополнением, рабочая память cuFFT).
_GPU_MEM_FRACTION = 0.6
_IN_COPIES, _OUT_COPIES, _RING_COPIES = 2.25, 1.0, 8.0
_SMOOTH_WORK_CHUNKS = 2
_RING_CHUNK = 16
_CPU_SLAB_ROWS = 16
_MIN_SLAB_ROWS, _MAX_SLAB_ROWS = 4, 256
#: Доли общего прогресса по стадиям.
_P_CROP, _P_PREPARE = 0.3, 0.35


def _sub_progress(progress: ProgressFn, a: float, b: float) -> ProgressFn:
    def inner(frac: float, stage: str) -> None:
        progress(a + (b - a) * min(max(float(frac), 0.0), 1.0), stage)
    return inner


# --- выбор кадров ------------------------------------------------------------------------------------------

def data_frames(scan: ScanInfo) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """data-кадры (без data_check) в порядке съёмки: (индексы timeline, углы float64, frame_numbers)."""
    idx = np.asarray(scan.data_idx, dtype=np.int64)
    fnums = np.asarray(scan.frame_numbers)[idx]
    order = np.argsort(fnums, kind='stable')
    idx = idx[order]
    return idx, np.asarray(scan.angles, dtype='float64')[idx], np.asarray(scan.frame_numbers)[idx].astype(np.int64)


def pair_0_180(angles_deg: np.ndarray, tol: Optional[float] = None) -> Tuple[int, int]:
    """Первая пара позиций с разностью углов 180° (как ``tomotools4.get_angles_at_180_deg``): позиции в
    angles_deg. Допуск — половина минимального шага, но не меньше 1e-3°. Нет пары — ValueError."""
    angles = np.asarray(angles_deg, dtype='float64')
    if tol is None:
        uniq = np.unique(angles)
        min_step = float(np.min(np.diff(uniq))) if uniq.size > 1 else 0.0
        tol = max(min_step / 2.0, 1e-3)
    t = np.subtract.outer(angles, angles) % 360.0
    for p0, p180 in np.argwhere(np.abs(t - 180.0) <= tol):
        if p0 < p180:
            return int(p0), int(p180)
    raise ValueError(
        'нет пары кадров с разностью углов 180° (допуск {:.4f}°; углы {:.2f}..{:.2f}°, кадров {}): '
        'для оси нужна съёмка минимум на 180°'.format(tol, float(angles.min()), float(angles.max()), angles.size))


# --- подготовка --------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Prepared:
    """Всё, что нужно конвейеру слоёв, кроме самих данных кропа."""
    idx: np.ndarray                  # индексы timeline data-кадров (порядок съёмки)
    angles: np.ndarray               # их углы, градусы
    fnums: np.ndarray                # их frame_numbers
    de: preprocess.DarkEmpty         # опорные кадры по всему кропу
    frame_sy: np.ndarray             # сдвиг образца для каждого data-кадра (накопленный), строки
    frame_sx: np.ndarray             # то же, столбцы
    shifts: Optional[Dict[str, List[float]]]   # измеренные/заданные сдвиги checkpoint-ов (для рецепта)
    axis: Axis
    shift_x: float
    alfa: float
    warnings: List[str]
    motion: Optional[Dict[str, Any]] = None    # блок motion рецепта, как применён (dx — если компенсирован)


def _normalization_de(scan: ScanInfo, de: preprocess.DarkEmpty, mode: str) -> preprocess.DarkEmpty:
    """'standard' — единый (начальный) empty; 'timeline'/'auto' — с периодическими, если они есть.

    Для advanced-скана в режиме 'standard' берётся только начальная серия (ноутбук в этом режиме брал медиану
    всех empty, включая периодические); по умолчанию ('auto') advanced нормируется с интерполяцией, как в ноутбуке."""
    if mode == 'standard' or (mode == 'auto' and not scan.is_advanced):
        return dataclasses.replace(de, periodic_empties=[], periodic_empty_fnumbers=[])
    return de


def _rows_de(de: preprocess.DarkEmpty, r0: int, r1: int) -> preprocess.DarkEmpty:
    return dataclasses.replace(
        de, dark=de.dark[r0:r1], initial_empty=de.initial_empty[r0:r1],
        periodic_empties=[e[r0:r1] for e in de.periodic_empties])


def _frame_shifts(fnums: np.ndarray, periodic_fnums, sy, sx, warnings: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Накопленный сдвиг для каждого кадра по номеру его сегмента (как apply_repositioning_correction)."""
    if len(sy) == 0:
        z = np.zeros(len(fnums), dtype='float64')
        return z, z.copy()
    cum_y, cum_x = preprocess.cumulative_shifts(np.asarray(sy, dtype='float64'), np.asarray(sx, dtype='float64'))
    seg = np.asarray(preprocess.segment_index(fnums, periodic_fnums), dtype=np.int64)
    if seg.size and seg.max() >= len(cum_y):
        warnings.append('сегментов больше, чем измеренных сдвигов ({} > {}): для лишних взят последний'.format(
            int(seg.max()), len(cum_y) - 1))
        seg = np.minimum(seg, len(cum_y) - 1)
    return np.asarray(cum_y)[seg], np.asarray(cum_x)[seg]


def _apply_shifts(slab, sy: np.ndarray, sx: np.ndarray, xp):
    """Сдвинуть каждый кадр слоя (n, s, w) на (sy[i], sx[i]) — order=3, 'nearest', как в ноутбуке."""
    nd = gpu.ndimage(xp)
    for i in np.nonzero((np.abs(sy) >= 1e-6) | (np.abs(sx) >= 1e-6))[0]:
        slab[i] = nd.shift(slab[i], [float(sy[i]), float(sx[i])], order=3, mode='nearest')
    return slab


def drop_bad_checkpoint_shifts(sy: np.ndarray, sx: np.ndarray, checks) -> Tuple[np.ndarray, np.ndarray]:
    """Сдвиги вставок, где проверка контрольных кадров нашла поворот или изменение образца, — 0: корреляция пары там
    меряет не сдвиг (см. preprocess.check_checkpoints), и её «сдвиг» в десятки пикселей только портит данные."""
    sy, sx = np.array(sy, dtype='float64'), np.array(sx, dtype='float64')
    for c in checks:
        k = int(c['k'])
        if c.get('status') in ('rotated', 'changed') and k < len(sy):
            sy[k] = sx[k] = 0.0
    return sy, sx


def checks_warnings(checks) -> List[str]:
    """Предупреждения по проверке контрольных кадров: по вставкам и итоговое (пусто — всё в порядке)."""
    out = [c['message'] for c in checks if c.get('status') in ('rotated', 'changed') and c.get('message')]
    summary = preprocess.checks_summary(checks)
    return ([summary] + out) if summary else []


def prepare(scan: ScanInfo, crop: CropData, r: recipe_mod.Recipe, xp=None) -> Prepared:
    """Опорные кадры, сдвиги образца, ось. Ось из рецепта используется как есть; если её нет — авто по паре
    0°/180° на нормированных (и скорректированных по сдвигу) кадрах."""
    xp = xp or gpu.get_xp()
    warnings: List[str] = []
    idx, angles, fnums = data_frames(scan)
    if idx.size == 0:
        raise ValueError('{}: в скане нет data-кадров'.format(scan.exp_id))

    de_full = preprocess.dark_empty_from_crop(scan, crop, skip_first=r.empty_skip_first)
    de = _normalization_de(scan, de_full, r.normalization)

    shifts = None
    sy = sx = np.zeros(0)
    checks: List[Dict[str, Any]] = []
    if scan.is_advanced and de_full.periodic_empty_fnumbers:
        # проверка контрольных кадров — всегда (диагностика данных), даже если коррекция сдвигов выключена
        checks = preprocess.check_checkpoints(scan, crop, de_full)
    if r.repositioning.get('enabled') and scan.is_advanced and de_full.periodic_empty_fnumbers:
        given = r.repositioning.get('shifts')
        if given:
            sy, sx = np.asarray(given['sy'], dtype='float64'), np.asarray(given['sx'], dtype='float64')
        else:
            _, sy, sx = preprocess.repositioning_shifts(scan, crop, de_full)
            sy, sx = np.asarray(sy, dtype='float64'), np.asarray(sx, dtype='float64')
            sy, sx = drop_bad_checkpoint_shifts(sy, sx, checks)
        if np.isnan(sy).any() or np.isnan(sx).any():
            warnings.append('сдвиг образца измерен не на всех checkpoint-ах: неизмеренные приняты за 0')
        shifts = {'sy': [float(v) for v in sy], 'sx': [float(v) for v in sx]}
    warnings.extend(checks_warnings(checks))
    frame_sy, frame_sx = _frame_shifts(fnums, de_full.periodic_empty_fnumbers, sy, sx, warnings)

    def auto(fsx):
        p0, p180 = pair_0_180(angles)
        pair = np.asarray(crop.frames[[int(idx[p0]), int(idx[p180])]])
        norm = preprocess.normalize_slab(pair, fnums[[p0, p180]], de, xp=xp)
        norm = _apply_shifts(norm, frame_sy[[p0, p180]], fsx[[p0, p180]], xp)
        a = axis_mod.auto_axis(gpu.to_numpy(norm[0]), gpu.to_numpy(norm[1]), crop.roi)
        logger.info('авто-ось: center_x=%.2f на y=%.1f, наклон %.4f°', a.center_x, a.y_ref, a.tilt_deg)
        return a

    ax = r.axis
    frame_sx, motion_used = apply_motion(r.motion, scan, crop, idx, angles, fnums, de, de_full, frame_sy, frame_sx,
                                         checks, tilt=lambda: (ax or auto(frame_sx)).tilt_deg, xp=xp,
                                         warnings=warnings)
    if ax is None:
        ax = auto(frame_sx)
    shift_x, alfa = axis_mod.to_crop_params(ax, crop.roi)
    return Prepared(idx=idx, angles=angles, fnums=fnums, de=de, frame_sy=frame_sy, frame_sx=frame_sx,
                    shifts=shifts, axis=ax, shift_x=shift_x, alfa=alfa, warnings=warnings, motion=motion_used)


def rotated_checks(checks) -> bool:
    """Проверка контрольных кадров нашла сбой угла хотя бы на одной вставке."""
    return any(c.get('status') == 'rotated' for c in (checks or []))


def apply_motion(block: Dict[str, Any], scan: ScanInfo, crop: CropData, idx, angles, fnums, de, de_full,
                 frame_sy: np.ndarray, frame_sx: np.ndarray, checks, tilt, xp=None,
                 warnings: Optional[List[str]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Компенсация смещения образца по блоку рецепта ``motion``: frame_sx − dx. Решение студии (``applied``)
    выполняется как есть (dx — из рецепта); без него — оценка :func:`motion.estimate` и :func:`motion.decide`.
    tilt — функция, возвращающая наклон оси (линии полос ⟂ оси); вызывается, только если нужна оценка.
    Возвращает (frame_sx, блок motion как применён)."""
    mode = block.get('mode', 'off')
    if mode == 'off':
        return frame_sx, recipe_mod.motion_block('off', applied=False)
    applied = block.get('applied')
    if applied is not None:
        if not applied:
            return frame_sx, recipe_mod.motion_block(mode, applied=False, summary=block.get('summary'))
        dx = motion_mod.from_block(block, fnums)
        logger.info('смещение образца: компенсация по рецепту (СКО %.2f px)', float(np.std(dx)))
        return frame_sx - dx, recipe_mod.motion_block(mode, True, dx, fnums, block.get('summary'))
    profiles, _ys = motion_mod.band_profiles(crop, idx, fnums, de, frame_sy, frame_sx, tilt(), xp=xp)
    b = motion_mod.effective_bin(crop.frames.shape[1], crop.frames.shape[2], motion_mod.DEFAULTS['bin'])
    est = motion_mod.estimate(profiles, angles, fnums, de_full.periodic_empty_fnumbers, bin_used=b)
    ok, why = motion_mod.decide(est, mode, rotated_checks(checks))
    if warnings is not None and est.status == 'inconsistent':
        warnings.append('смещение образца: ' + why)
    if not ok:
        return frame_sx, recipe_mod.motion_block(mode, False, summary=est.summary())
    return frame_sx - est.dx, recipe_mod.motion_block(mode, True, est.dx, fnums, est.summary())


# --- слои --------------------------------------------------------------------------------------------------

def margin(prep: Prepared, width: int) -> int:
    """Запас входных строк с каждой стороны слоя."""
    max_sy = float(np.max(np.abs(prep.frame_sy))) if prep.frame_sy.size else 0.0
    extra = _SHIFT_EXTRA_ROWS if max_sy > 0 else 0
    return axis_mod.margin_rows(prep.alfa, width) + int(math.ceil(max_sy)) + extra


def slab_rows_for_memory(free_bytes: int, n_frames: int, width: int, margin_rows: int, halo: int = 0) -> int:
    """Выходных строк слоя, помещающихся в free_bytes по модели памяти (см. _IN_COPIES и др.). halo — ореол
    сглаживания h (0 — выключено): входных и выровненных строк на 2h больше; фильтр идёт после выравнивания и колец
    (входной слой уже отпущен) и держит выровненные строки, результат (s строк) и рабочие порции — второе
    ограничение."""
    unit = max(1, n_frames * width * 4)
    budget = free_bytes * _GPU_MEM_FRACTION / unit
    rows = ((budget - _IN_COPIES * 2 * (margin_rows + halo) - _OUT_COPIES * 2 * halo - _RING_COPIES * _RING_CHUNK)
            / (_IN_COPIES + _OUT_COPIES))
    if halo > 0:
        work = _SMOOTH_WORK_CHUNKS * smoothing.CHUNK_BYTES / unit
        rows = min(rows, (budget - _OUT_COPIES * 2 * halo - work) / (_OUT_COPIES + 1.0))
    return int(min(_MAX_SLAB_ROWS, max(_MIN_SLAB_ROWS, rows)))


def auto_slab_rows(n_frames: int, width: int, margin_rows: int, xp=None, halo: int = 0) -> int:
    """Число выходных строк слоя по свободной памяти GPU (на CPU — фиксированное)."""
    xp = xp or gpu.get_xp()
    if not gpu.is_gpu(xp):
        return _CPU_SLAB_ROWS
    info = gpu.mem_info()
    if info is None:
        return _CPU_SLAB_ROWS
    return slab_rows_for_memory(info[0], n_frames, width, margin_rows, halo)


def halo_range(out_rows: Tuple[int, int], halo: int, height: int) -> Tuple[int, int]:
    """Строки кропа, нужные сглаживанию для выходных строк out_rows = [r0, r1): (r0 − h, r1 + h) ∩ [0, height)."""
    return max(0, int(out_rows[0]) - int(halo)), min(int(height), int(out_rows[1]) + int(halo))


def _is_gpu_oom(exc: BaseException) -> bool:
    return type(exc).__name__ == 'OutOfMemoryError'      # cupy.cuda.memory.OutOfMemoryError без импорта cupy


def process_slab(crop: CropData, prep: Prepared, out_rows: Tuple[int, int], m: int, ring_params,
                 xp=None) -> Any:
    """Синограммы (s, n, w) выходных строк кропа out_rows = [r0, r1): нормировка, сдвиг образца,
    выравнивание оси, кольца (кусками по _RING_CHUNK строк — память колец не растёт со слоем).
    Возвращает xp-массив."""
    xp = xp or gpu.get_xp()
    h = crop.roi.height
    r0, r1 = out_rows
    in0, in1 = max(0, r0 - m), min(h, r1 + m)
    raw = np.asarray(crop.frames[prep.idx, in0:in1, :])
    slab = preprocess.normalize_slab(raw, prep.fnums, _rows_de(prep.de, in0, in1), xp=xp)
    del raw
    slab = _apply_shifts(slab, prep.frame_sy, prep.frame_sx, xp)
    aligned = axis_mod.align_rows(slab, in0, (r0, r1), prep.shift_x, prep.alfa, h, xp=xp)
    del slab
    sino = xp.ascontiguousarray(xp.swapaxes(aligned, 0, 1))
    del aligned
    if ring_params:
        for c in range(0, sino.shape[0], _RING_CHUNK):
            sino[c:c + _RING_CHUNK] = rings.apply(sino[c:c + _RING_CHUNK], ring_params, xp=xp, frame_dx=prep.frame_sx)
    return sino


def smoothed_slab(crop: CropData, prep: Prepared, out_rows: Tuple[int, int], m: int, ring_params, smooth_params,
                  xp=None) -> Any:
    """Синограммы (s, n, w) строк кропа out_rows = [a, b) после колец и сглаживания: ``process_slab`` для строк с
    ореолом ``halo_range`` (кольца — на всех), затем ``smoothing.apply`` → строки [a, b). Совпадают с фильтром по
    всему кропу (см. модуль). smooth_params=None — ``process_slab`` с кольцами."""
    xp = xp or gpu.get_xp()
    a, b = out_rows
    ea, eb = halo_range(out_rows, smoothing.halo_rows(smooth_params), crop.roi.height)
    sino = process_slab(crop, prep, (ea, eb), m, ring_params, xp=xp)
    if not smooth_params:
        return sino
    out = smoothing.apply(sino, smooth_params, keep=(a - ea, b - ea), xp=xp)
    del sino
    return out


def output_window(r: recipe_mod.Recipe) -> Tuple[Tuple[int, int, int, int], Optional[np.ndarray]]:
    """Окно среза (y0, y1, x0, x1) в пикселях среза w×w и маска круга в этом окне (или None)."""
    w = r.fov.width
    roi = r.recon.get('xy_roi') or {'kind': None}
    kind = roi.get('kind')
    if kind == 'rect':
        return (int(roi['y0']), int(roi['y1']), int(roi['x0']), int(roi['x1'])), None
    if kind == 'circle':
        cx, cy, rad = float(roi['cx']), float(roi['cy']), float(roi['r'])
        y0, y1 = max(0, int(math.floor(cy - rad))), min(w, int(math.ceil(cy + rad)) + 1)
        x0, x1 = max(0, int(math.floor(cx - rad))), min(w, int(math.ceil(cx + rad)) + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        return (y0, y1, x0, x1), (yy - cy) ** 2 + (xx - cx) ** 2 <= rad ** 2
    return (0, w, 0, w), None


def output_shape(r: recipe_mod.Recipe) -> Tuple[int, int, int]:
    (y0, y1, x0, x1), _ = output_window(r)
    z0, z1 = int(r.recon['slices'][0]), int(r.recon['slices'][1])
    return z1 - z0, y1 - y0, x1 - x0


def ring_rows_total(slices: Tuple[int, int], rows: int, halo: int, height: int) -> int:
    """Сколько строк пройдут выравнивание и кольца при слоях по rows выходных строк для строк кропа
    slices = [c0, c1): со сглаживанием каждый слой берётся с ореолом ±halo (``halo_range``)."""
    c0, c1 = int(slices[0]), int(slices[1])
    rows = max(1, int(rows))
    return sum(eb - ea for ea, eb in (halo_range((a, min(a + rows, c1)), halo, height) for a in range(c0, c1, rows)))


def estimate(r: recipe_mod.Recipe, scan: ScanInfo) -> Dict[str, Any]:
    """Размеры без вычислений: кроп, объём, копии с биннингом (байты). Со сглаживанием — ореол ``halo_rows`` и во
    сколько раз строк через выравнивание и кольца больше, чем срезов (``ring_rows_factor``, грубо: слой — по
    свободной памяти GPU сейчас, на CPU — _CPU_SLAB_ROWS; запас — по наклону оси рецепта, без сдвигов образца)."""
    shape = output_shape(r)
    n_data = int(len(scan.data_idx))
    full = int(np.prod(shape)) * 4
    binned = {int(b): int(np.prod([s // b for s in shape])) * 4 for b in r.outputs.get('binning', [])}
    hs = smoothing.halo_rows(smoothing.resolve(r.smoothing))
    factor = 1.0
    if hs:
        m = axis_mod.margin_rows(-r.axis.tilt_deg if r.axis is not None else 0.0, r.fov.width)
        rows = auto_slab_rows(n_data, r.fov.width, m, halo=hs)
        c0, c1 = int(r.recon['slices'][0]) - r.fov.y0, int(r.recon['slices'][1]) - r.fov.y0
        factor = ring_rows_total((c0, c1), rows, hs, r.fov.height) / float(max(1, c1 - c0))
    return {
        'crop_bytes': int(scan.n_frames) * r.fov.height * r.fov.width * np.dtype(scan.dtype).itemsize,
        'volume_shape': list(shape), 'volume_bytes': full, 'binned_bytes': binned,
        'n_data_frames': n_data,
        'n_angles_used': int(fbp.select_angles(data_frames(scan)[1], r.recon['angles']).sum()),
        'halo_rows': hs, 'ring_rows_factor': factor,
        'denoise_voxel_iterations': _denoise_work(r, shape),
    }


def _denoise_work(r: recipe_mod.Recipe, shape) -> int:
    """Воксель-итераций TV с учётом ореолов (по z — порции tv.chunk_slices, по y, x — грубо +10 %); 0 — выключено."""
    dn = tv.resolve(r.denoise, need_weight=False)
    if not dn:
        return 0
    nz, ny, nx = (int(s) for s in shape)
    chunk = tv.chunk_slices(ny, nx)
    z_factor = min(nz, chunk + 2 * tv.HALO) / float(min(nz, chunk))
    return int(nz * ny * nx * dn['iterations'] * z_factor * 1.1)


# --- запуск ------------------------------------------------------------------------------------------------

@dataclasses.dataclass
class RunResult:
    recipe: recipe_mod.Recipe        # рецепт с заполненными осью и сдвигами (воспроизводит запуск)
    result: Dict[str, Any]           # документ result.json
    out_dir: str


def resolved_recipe(r: recipe_mod.Recipe, prep: Prepared) -> recipe_mod.Recipe:
    """Копия рецепта с фактически использованными осью и сдвигами образца."""
    d = recipe_mod.to_dict(r)
    d['axis'] = prep.axis.to_dict()
    if prep.shifts is not None:
        d['repositioning'] = dict(d['repositioning'], shifts=prep.shifts)
    if prep.motion is not None:
        d['motion'] = prep.motion
    return recipe_mod.from_dict(d)


def run_recipe(r: recipe_mod.Recipe, scan_path: str, out_dir: str, cache_dir: str, *,
               name: Optional[str] = None, progress: ProgressFn = no_progress, cancel=None,
               backend: str = 'auto', slab_rows: Optional[int] = None, workers: int = 8,
               scan: Optional[ScanInfo] = None) -> RunResult:
    """Реконструкция по рецепту. Пишет в out_dir объём (Amira raw + .hx, копии с биннингом), ``recipe.json``
    (с фактическими осью и сдвигами) и ``result.json``. При отмене (``cancel.is_set()`` → model.Cancelled) и
    ошибке созданные файлы объёма удаляются. ``name`` — имя образца для файлов (по умолчанию exp_id)."""
    t_start = time.time()
    timings: Dict[str, float] = {}
    xp = gpu.get_xp()
    scan = scan or data.open_scan(scan_path, r.input.get('exp_id'))
    recipe_mod.validate(r, scan.height, scan.width)
    warnings: List[str] = []
    fp = r.input.get('fingerprint')
    if fp and fp != scan.fingerprint:
        warnings.append('отпечаток файла скана не совпадает с рецептом: файл изменился после настройки')
    if r.pixel_size.get('source') == 'default':
        warnings.append('размер пикселя взят по умолчанию ({} мм)'.format(r.pixel_size['value_mm']))
    if not r.outputs.get('full', True):
        warnings.append('outputs.full=false пока не поддерживается: полный объём записан')

    crop = data.CropLoader(scan, cache_dir).load(r.fov, progress=_sub_progress(progress, 0.0, _P_CROP),
                                                 cancel=cancel, workers=workers)
    timings['crop_s'] = time.time() - t_start

    t0 = time.time()
    check_cancel(cancel)
    progress(_P_CROP, 'prepare')
    prep = prepare(scan, crop, r, xp=xp)
    warnings.extend(prep.warnings)
    resolved = resolved_recipe(r, prep)
    timings['prepare_s'] = time.time() - t0

    t0 = time.time()
    w = r.fov.width
    m = margin(prep, w)
    sp = smoothing.resolve(r.smoothing)
    hs = smoothing.halo_rows(sp)
    dn = tv.resolve(r.denoise)
    rows = int(slab_rows or auto_slab_rows(len(prep.idx), w, m, xp, halo=hs))
    z0, z1 = int(r.recon['slices'][0]), int(r.recon['slices'][1])
    c0, c1 = z0 - r.fov.y0, z1 - r.fov.y0                       # строки кропа
    (wy0, wy1, wx0, wx1), circle = output_window(r)
    ring_params = rings.resolve(r.rings.get('preset', 'medium'), r.rings.get('params'), version=r.rings.get('version', 1))
    pixel_size = float(r.pixel_size['value_mm'])
    shape = output_shape(r)
    base = name or scan.exp_id
    logger.info('run_recipe %s: срезы [%d, %d), слой %d строк, запас %d, ореол сглаживания %d, объём %s, '
                'бэкенд FBP %s', scan.exp_id, z0, z1, rows, m, hs, shape, fbp.resolve_backend(backend))
    timings['slab_rows'] = rows
    timings['n_angles'] = int(fbp.select_angles(prep.angles, r.recon['angles']).sum())   # для оценки времени
    if sp:
        timings['halo_rows'] = hs

    writer = outputs.VolumeWriter(out_dir, base, shape, pixel_size, binning=r.outputs.get('binning', [4]))
    samples: List[np.ndarray] = []
    sink = writer
    if dn:
        def keep_sample(z0: int, res: np.ndarray) -> None:
            samples.append(res[res.shape[0] // 2, ::4, ::4].copy())
        sink = tv.DenoiseWriter(writer, dn, nz=shape[0], chunk=tv.chunk_slices(shape[1], shape[2]), xp=xp,
                                on_chunk=keep_sample, mask=circle)
        logger.info('run_recipe %s: TV 3D, вес %.4g (%.2gσ), %d итераций, порции по %d срезов', scan.exp_id,
                    dn['weight'], dn['strength'], dn['iterations'], sink.chunk)
    try:
        a = c0
        while a < c1:
            check_cancel(cancel)
            b = min(a + rows, c1)
            try:
                if sp:
                    # кольца на слое с ореолом (на всех строках), затем фильтр → готовые к FBP строки [a, b)
                    sino = smoothed_slab(crop, prep, (a, b), m, ring_params, sp, xp=xp)
                else:
                    sino = process_slab(crop, prep, (a, b), m, None, xp=xp)
            except Exception as exc:  # noqa: BLE001 — нехватка памяти GPU: слой вдвое меньше и заново
                if not _is_gpu_oom(exc) or rows <= _MIN_SLAB_ROWS:
                    raise
                gpu.free_memory()
                rows = max(_MIN_SLAB_ROWS, rows // 2)
                logger.warning('нехватка памяти GPU: слой уменьшен до %d строк', rows)
                continue
            # кольца и FBP кусками: память колец и объём среза в RAM не растут со слоем
            for c in range(0, b - a, _RING_CHUNK):
                part = sino[c:c + _RING_CHUNK] if sp else rings.apply(sino[c:c + _RING_CHUNK], ring_params, xp=xp,
                                                                      frame_dx=prep.frame_sx)
                rec = fbp.recon_rows(part, prep.angles, pixel_size, backend=backend, angle_mode=r.recon['angles'])
                del part
                rec = rec[:, wy0:wy1, wx0:wx1]
                if circle is not None:
                    rec = np.where(circle[None], rec, np.float32(0))
                sink.write(a - c0 + c, rec)
                if c == 0 and not dn:
                    samples.append(rec[rec.shape[0] // 2, ::4, ::4].copy())
            del sino
            progress(_P_PREPARE + (1 - _P_PREPARE) * (b - c0) / (c1 - c0), 'recon')
            a = b
        files = sink.close()
    except BaseException:
        sink.abort()
        raise
    finally:
        if gpu.is_gpu(xp):
            gpu.free_memory()
    timings['recon_s'] = time.time() - t0
    if dn:
        timings['denoise_s'] = round(sink.seconds, 3)
    timings['total_s'] = time.time() - t_start

    rd = recipe_mod.to_dict(resolved)
    sha = recipe_mod.sha256(resolved)
    stats = outputs.volume_stats(np.concatenate([s.ravel() for s in samples]) if samples else np.zeros(0))
    gpu_name = gpu.device_name()
    doc = outputs.result_document(rd, sha, files, shape, pixel_size, stats, timings, warnings,
                                  ENGINE_VERSION, gpu_name)
    recipe_mod.save(resolved, os.path.join(out_dir, 'recipe.json'))
    outputs.write_json(os.path.join(out_dir, 'result.json'), doc)
    progress(1.0, 'done')
    return RunResult(recipe=resolved, result=doc, out_dir=str(out_dir))

