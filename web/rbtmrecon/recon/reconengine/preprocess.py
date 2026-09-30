"""Предобработка: огибающая и авто-ROI по обзору, dark/empty по кропу, нормировка слоями, репозиционирование.

Нормировка повторяет ``tomotools4.normalize_projections`` / ``normalize_projections_with_timeline``:
    empty и data после вычитания dark обрезаются снизу до 1, d = log(empty) − log(data),
    затем ``safe_median`` (медиана 3×3 для выбросов), затем clip(d, 0, ∞).
Для advanced-экспериментов empty для кадра интерполируется по frame_number между начальной и
периодическими empty-сериями (семантика ``tomotools4._interpolate_empty`` после исправления: у начальной серии
свой ``initial_empty_fnumber``).

safe_median — медиана 3×3 только в плоскости кадра (``median_filter(size=(1, 3, 3))``, граница 'reflect'); значение
заменяется медианой, если |медиана − d| > 0,1·|d|. У слоя строк медиана крайних строк слоя считается с отражением,
а в полном кадре — по соседним строкам, поэтому ПЕРВАЯ и ПОСЛЕДНЯЯ строки слоя неточны: конвейер подаёт слой
с запасом строк и отбрасывает края (кроме настоящих краёв кропа, где поведение совпадает с полным кадром).

Порог объекта (огибающая, авто-ROI, углы за пределами ROI): фон — медиана изображения −ln T, шум —
MAD·1.4826; маска — значения больше фон + k_sigma·шум; столбец (строка) «занят», если в нём доля маски > 2 %.

Контрольные кадры (advanced, ``check_checkpoints``): после каждой периодической вставки снимается data_check под тем
же углом, что последний data-кадр до неё. ``repositioning_shifts`` меряет по этой паре только сдвиг; поворот образца
или стола (скан 524efd6e от 15.09.2026: после вставок на 74,5°, 99,5° и 124,5° образец отставал на 3,3°, 2,8° и 14°,
счётчик мотора этого не видел) такой сдвиг не описывает — корреляция выдаёт ложные «сдвиги» в десятки пикселей.
Проверка сравнивает расхождение пары с расхождением соседних data-кадров (масштаб одного шага угла) и при большом
расхождении ищет, с каким более ранним углом контрольный кадр совпадает лучше.
"""
from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .model import CropData, Overview, ROI, ScanInfo

logger = logging.getLogger(__name__)

MAD_SCALE = 1.4826          # MAD → σ для нормального шума
OCCUPIED_FRAC = 0.02        # доля маски, при которой столбец/строка считается занятой объектом
ANGLE_MATCH_TOL = 0.5       # градусы: допуск совпадения угла data и data_check (как в tomotools4)
#: проверка контрольных кадров (check_checkpoints): бининг кадров, глубина поиска угла назад, пороги.
#: Калибровка (15.09 и 28.09): у исправных вставок rms(пара)/rms(соседние кадры) = 0,4…1,9; у повернувшихся — 3,9…9,5,
#: а rms с кадром лучшего угла в 3–4 раза меньше, чем с кадром того же угла
CHECK_BIN = 4
CHECK_SEARCH_DEG = 30.0
CHECK_COARSE_DEG = 2.0
CHECK_SUSPECT_RATIO = 2.5
CHECK_ROTATED_RATIO = 0.5
CHECK_ROTATED_MIN_DEG = 1.0
_MEDIAN_BLOCK_BYTES = 256 * 1024 * 1024


# --- обзор: огибающая, авто-ROI ----------------------------------------------------------------------------

def _minus_log_t(frames: np.ndarray, dark: np.ndarray, empty: np.ndarray, eps: float) -> np.ndarray:
    """−ln T, T = (I − D) / (E − D), T ≥ eps. Пиксели без пучка (E − D ≤ 1 отсчёта) считаются прозрачными."""
    dark = np.asarray(dark, dtype='float32')
    den = np.asarray(empty, dtype='float32') - dark
    valid = den > 1
    t = (np.asarray(frames, dtype='float32') - dark) / np.where(valid, den, 1).astype('float32')
    t = np.where(valid, t, np.float32(1))
    return -np.log(np.maximum(t, np.float32(eps)))


def envelope(ov: Overview, eps: float = 1e-3) -> np.ndarray:
    """max по выборке углов от −ln T, T = (I − D) / (E − D), T обрезается снизу до eps. float32 (h, w)."""
    samples = np.asarray(ov.samples, dtype='float32')
    if samples.ndim != 3 or samples.shape[0] == 0:
        raise ValueError('в обзоре нет кадров выборки')
    env = None
    for frame in samples:
        p = _minus_log_t(frame, ov.dark, ov.empty, eps)
        env = p if env is None else np.maximum(env, p)
    return env.astype('float32')


def _object_mask(img: np.ndarray, k_sigma: float) -> np.ndarray:
    """Маска объекта: img > медиана + k_sigma·MAD·1.4826 (шум не меньше 1e-6, чтобы работало и без шума)."""
    img = np.asarray(img, dtype='float64')
    bg = float(np.median(img))
    noise = max(float(np.median(np.abs(img - bg))) * MAD_SCALE, 1e-6)
    return img > bg + k_sigma * noise


def _occupied_span(frac: np.ndarray) -> Optional[Tuple[int, int]]:
    idx = np.where(frac > OCCUPIED_FRAC)[0]
    if idx.size == 0:
        return None
    return int(idx[0]), int(idx[-1]) + 1


def suggest_roi(env: np.ndarray, bin: int, full_height: int, full_width: int,
                margin_frac: float = 0.03, k_sigma: float = 6.0) -> ROI:
    """Предложить ROI по огибающей: маска объекта = env > фон + k_sigma·шум (MAD); столбцы/строки, где доля маски
    заметна, расширяются на margin_frac ширины/высоты кадра и обрезаются по кадру. Координаты — полного кадра.
    Если объект не найден — весь кадр."""
    mask = _object_mask(env, k_sigma)
    cols = _occupied_span(mask.mean(axis=0))
    rows = _occupied_span(mask.mean(axis=1))
    if cols is None or rows is None:
        logger.warning('suggest_roi: объект не найден — весь кадр')
        return ROI(0, full_width, 0, full_height)
    mx = int(round(margin_frac * full_width))
    my = int(round(margin_frac * full_height))
    x0 = max(0, cols[0] * bin - mx)
    x1 = min(full_width, cols[1] * bin + mx)
    y0 = max(0, rows[0] * bin - my)
    y1 = min(full_height, rows[1] * bin + my)
    return ROI(x0, x1, y0, y1)


def angles_outside(ov: Overview, roi: ROI, k_sigma: float = 6.0) -> List[float]:
    """Углы выборки, на которых маска объекта (−ln T по отдельному кадру) выходит за столбцы [x0, x1).

    Учитываются только строки ROI (объект выше/ниже ROI на эти срезы не влияет); столбец обзора занят, если доля
    маски в нём > 2 %, и лежит вне ROI, если его центр (в пикселях полного кадра) вне [x0, x1)."""
    b = int(ov.bin)
    samples = np.asarray(ov.samples, dtype='float32')
    h = samples.shape[1]
    yb0 = min(max(roi.y0 // b, 0), h - 1)
    yb1 = min(max(int(math.ceil(roi.y1 / b)), yb0 + 1), h)
    out = []
    for frame, angle in zip(samples, np.asarray(ov.sample_angles, dtype='float64')):
        mask = _object_mask(_minus_log_t(frame, ov.dark, ov.empty, 1e-3), k_sigma)
        occupied = np.where(mask[yb0:yb1].mean(axis=0) > OCCUPIED_FRAC)[0]
        centers = (occupied + 0.5) * b
        if np.any(centers < roi.x0) or np.any(centers >= roi.x1):
            out.append(float(angle))
    return out


# --- dark / empty по кропу ---------------------------------------------------------------------------------

@dataclasses.dataclass
class DarkEmpty:
    """Опорные кадры по кропу (строки rows кропа, если заданы при расчёте). float32."""
    dark: np.ndarray                               # (h, w)
    initial_empty: np.ndarray                      # (h, w), dark вычтен
    initial_empty_fnumber: int
    periodic_empties: List[np.ndarray]             # K × (h, w), dark вычтен (только advanced)
    periodic_empty_fnumbers: List[int]


def _median_frames(frames: np.ndarray, idx: np.ndarray, r0: int, r1: int,
                   subtract: Optional[np.ndarray] = None) -> np.ndarray:
    """Медиана по кадрам idx строк [r0, r1) (float32), блоками строк, чтобы не держать всё в памяти.
    subtract (h, w) вычитается из каждого кадра ДО медианы (как в hdf5_v2.load_tomo_data_advanced_v2)."""
    w = frames.shape[2]
    out = np.empty((r1 - r0, w), dtype='float32')
    block = max(1, _MEDIAN_BLOCK_BYTES // max(1, len(idx) * w * 4))
    for a in range(r0, r1, block):
        b = min(r1, a + block)
        chunk = np.asarray(frames[idx, a:b], dtype='float32')
        if subtract is not None:
            chunk = chunk - subtract[a - r0:b - r0]
        out[a - r0:b - r0] = np.median(chunk, axis=0).astype('float32')
    return out


def series_tail(idx: np.ndarray, skip_first: int) -> np.ndarray:
    """Кадры серии без первых skip_first, но не меньше двух (серия из одного кадра — как есть)."""
    idx = np.asarray(idx)
    return idx[min(max(0, int(skip_first)), max(0, len(idx) - 2)):]


def dark_empty_from_crop(scan: ScanInfo, crop: CropData, rows: Optional[Tuple[int, int]] = None,
                         skip_first: int = 0) -> DarkEmpty:
    """Медианы dark, начальной empty-серии и периодических empty-серий по кропу (строки rows кропа, если заданы).

    Разбиение empty на серии — как в ``hdf5_v2.load_tomo_data_advanced_v2`` (первые series_length — начальная,
    далее по series_length подряд), frame_number серии — номер её первого кадра.

    skip_first — сколько первых кадров каждой серии не брать в медиану (:func:`series_tail`, остаётся не меньше двух).
    У части камер первые кадры после ухода образца из пучка несут «тень» объекта (инерция детектора, 0,1–0,3 %) и
    темнее остальных на ~1 %; в начальной серии — прогрев трубки. 0 — все кадры (рецепты, записанные до появления
    параметра).

    Advanced: dark вычитается из каждого empty-кадра, потом медиана (как load_tomo_data_advanced_v2).
    Не-advanced: все empty — одна начальная серия, медиана, потом вычитание dark (как load_tomo_data_v2).
    Нет dark-кадров — dark нулевой (предупреждение); нет empty — ValueError."""
    frames = crop.frames
    if frames.ndim != 3 or frames.shape[0] != scan.n_frames:
        raise ValueError('кроп {} не соответствует скану из {} кадров'.format(frames.shape, scan.n_frames))
    h, w = frames.shape[1], frames.shape[2]
    r0, r1 = (0, h) if rows is None else (int(rows[0]), int(rows[1]))
    if not (0 <= r0 < r1 <= h):
        raise ValueError('строки [{}, {}) вне кропа высотой {}'.format(r0, r1, h))

    dark_idx = np.asarray(scan.dark_idx, dtype=np.int64)
    if dark_idx.size:
        dark = _median_frames(frames, dark_idx, r0, r1)
    else:
        logger.warning('dark-кадров нет — dark нулевой')
        dark = np.zeros((r1 - r0, w), dtype='float32')

    empty_idx = np.asarray(scan.empty_idx, dtype=np.int64)
    if empty_idx.size == 0:
        raise ValueError('нет empty-кадров — нормировка невозможна')
    fnums = np.asarray(scan.frame_numbers)
    order = np.argsort(fnums[empty_idx], kind='stable')
    empty_idx = empty_idx[order]
    empty_fn = fnums[empty_idx]

    if not scan.is_advanced:
        initial = _median_frames(frames, series_tail(empty_idx, skip_first), r0, r1) - dark
        return DarkEmpty(dark, initial.astype('float32'), int(empty_fn[0]), [], [])

    sl = int(scan.series_length)
    if sl <= 0:
        raise ValueError('advanced-скан без series_length — разбиение empty на серии невозможно')
    initial = _median_frames(frames, series_tail(empty_idx[:sl], skip_first), r0, r1, subtract=dark)
    periodic, periodic_fn = [], []
    remaining = empty_idx[sl:]
    remaining_fn = empty_fn[sl:]
    for k in range(len(remaining) // sl):
        periodic.append(_median_frames(frames, series_tail(remaining[k * sl:(k + 1) * sl], skip_first), r0, r1,
                                       subtract=dark))
        periodic_fn.append(int(remaining_fn[k * sl]))
    if len(remaining) % sl:
        logger.warning('empty-кадров после начальной серии %d — не кратно series_length=%d, хвост %d отброшен',
                       len(remaining), sl, len(remaining) % sl)
    return DarkEmpty(dark, initial, int(empty_fn[0]), periodic, periodic_fn)


def _clip1(e: np.ndarray) -> np.ndarray:
    e = np.asarray(e, dtype='float32').copy()
    e[e < 1] = 1
    return e


def _interp_plan(de: DarkEmpty, frame_numbers: Sequence[int]):
    """Для каждого кадра (i0, i1, w): empty = (1 − w)·E[i0] + w·E[i1], E = [initial] + periodic (как
    tomotools4._interpolate_empty)."""
    all_fn = [int(de.initial_empty_fnumber)] + [int(fn) for fn in de.periodic_empty_fnumbers]
    last = len(all_fn) - 1
    plan = []
    for fn in frame_numbers:
        fn = int(fn)
        if last == 0 or fn <= all_fn[0]:
            plan.append((0, 0, 0.0))
            continue
        if fn >= all_fn[-1]:
            plan.append((last, last, 0.0))
            continue
        idx = int(np.searchsorted(np.asarray(all_fn), fn, side='right')) - 1
        idx = min(max(idx, 0), last - 1)
        fn0, fn1 = all_fn[idx], all_fn[idx + 1]
        w = float(fn - fn0) / float(fn1 - fn0) if fn1 != fn0 else 0.0
        plan.append((idx, idx + 1, w))
    return plan


def empty_for_frame(de: DarkEmpty, frame_number: int) -> np.ndarray:
    """Empty для кадра: линейная интерполяция по frame_number между соседними сериями; до первой периодической —
    между начальной и первой; после последней — последняя; без периодических — начальная.
    Как tomotools4._interpolate_empty: серии и результат обрезаются снизу до 1. float32 (h, w), новый массив."""
    i0, i1, w = _interp_plan(de, [frame_number])[0]
    empties = [de.initial_empty] + list(de.periodic_empties)
    e0 = _clip1(empties[i0])
    if i0 == i1:
        return e0
    e = ((1.0 - w) * e0 + w * _clip1(empties[i1])).astype('float32')
    e[e < 1] = 1
    return e


def _normalize(frames, frame_numbers: Sequence[int], de: DarkEmpty, xp, median3: bool, clip0: bool):
    from .gpu import get_xp, ndimage  # noqa: WPS433
    xp = xp or get_xp()
    td = xp.asarray(frames)
    if td.ndim != 3:
        raise ValueError('ожидается слой кадров (n, s, w), получено {}'.format(td.shape))
    n = td.shape[0]
    if len(frame_numbers) != n:
        raise ValueError('frame_numbers: {} значений на {} кадров'.format(len(frame_numbers), n))
    if tuple(td.shape[1:]) != tuple(np.shape(de.dark)):
        raise ValueError('слой {} не совпадает с dark/empty {}'.format(td.shape[1:], np.shape(de.dark)))
    td = td.astype(xp.float32)
    td -= xp.asarray(de.dark, dtype=xp.float32)
    td[td < 1] = 1
    xp.log(td, out=td)

    if not de.periodic_empties:
        log_e = xp.log(xp.asarray(_clip1(de.initial_empty)))
        xp.subtract(log_e[None], td, out=td)
    else:
        stack = xp.asarray(np.stack([_clip1(e) for e in [de.initial_empty] + list(de.periodic_empties)]))
        plan = _interp_plan(de, frame_numbers)
        step = 64
        for a in range(0, n, step):
            part = plan[a:a + step]
            i0 = xp.asarray(np.array([p[0] for p in part], dtype=np.int64))
            i1 = xp.asarray(np.array([p[1] for p in part], dtype=np.int64))
            wts = np.array([p[2] for p in part], dtype='float64')
            # (1.0 − w) считается в float64 и приводится к float32 — ровно как Python-скаляр в tomotools4
            ca = xp.asarray((1.0 - wts).astype('float32'))[:, None, None]
            cb = xp.asarray(wts.astype('float32'))[:, None, None]
            e = ca * stack[i0] + cb * stack[i1]
            e[e < 1] = 1
            xp.log(e, out=e)
            xp.subtract(e, td[a:a + step], out=td[a:a + step])

    if median3:
        m = ndimage(xp).median_filter(td, size=(1, 3, 3))
        mask = xp.abs(m - td) > 0.1 * xp.abs(td)
        td[mask] = m[mask]
    if clip0:
        td[td < 0] = 0
    return td


def normalize_slab(frames, frame_numbers: Sequence[int], de: DarkEmpty, xp=None, median3: bool = True):
    """Нормировать слой кадров (n, s, w) uint16 → xp.float32 −ln(T) как в tomotools4 (см. модуль).
    de должен быть посчитан для тех же строк. Работает на xp (cupy или numpy).

    Крайние строки слоя неточны из-за медианы 3×3 (см. модуль) — подавайте слой с запасом строк."""
    return _normalize(frames, frame_numbers, de, xp, median3, clip0=True)


# --- репозиционирование ------------------------------------------------------------------------------------

def _find_matching_data_frame(angle: float, data_angles: np.ndarray, data_numbers: np.ndarray,
                              segment_end_fnumber: int, tol: float = ANGLE_MATCH_TOL) -> Optional[int]:
    """Порт ``tomotools4._find_matching_data_frame``: ПОСЛЕДНИЙ data-кадр до segment_end_fnumber с углом в допуске
    (по кругу 360°). None — кадров до segment_end_fnumber нет; ValueError — ни один угол не в допуске."""
    mask = data_numbers < segment_end_fnumber
    if not mask.any():
        return None
    candidates = np.where(mask)[0]
    diffs = np.abs(data_angles[candidates] - angle) % 360
    diffs = np.minimum(diffs, 360 - diffs)
    in_tol = np.where(diffs <= tol)[0]
    if len(in_tol) == 0:
        raise ValueError('Не найден data-кадр с углом {:.2f}° (допуск {:.2f}°) среди кадров до frame_number={}; '
                         'ближайший угол отличается на {:.2f}°'.format(angle, tol, segment_end_fnumber,
                                                                       float(diffs.min())))
    return int(candidates[in_tol[-1]])


def repositioning_shifts(scan: ScanInfo, crop: CropData, de: DarkEmpty) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Сдвиги образца после каждой периодической вставки (advanced): (углы checkpoint-ов, sy, sx), K элементов.
    Порт ``tomotools4.measure_repositioning_shifts`` (debug=False): пара «первый data_check после вставки k ↔
    последний data-кадр с тем же углом до вставки», субпиксельная взаимная корреляция нормированных кадров.
    Checkpoint без пары — сдвиг 0 и предупреждение в лог. Не-advanced — пустые массивы.

    de — по всему кропу (rows=None). Оба кадра пары нормируются одним empty = periodic_empties[k]
    (нормировка normalize_slab на numpy, но без обрезки отрицательных значений — как в оригинале).
    Знак — как у ``phase_cross_correlation(reference=data, moving=data_check)``: ndi.shift(data_check, s) ≈ data.

    ОТЛИЧИЕ от tomotools4: ``phase_cross_correlation(..., normalization=None)`` — обычная взаимная корреляция
    вместо фазовой (normalization='phase' по умолчанию). Фазовая нормировка выравнивает спектр и раздувает
    высокочастотный шум (квантование, шум общего для пары empty, артефакты safe_median): на гладком объекте с
    шумом она ошибается на 0,5–1,9 px при дробных сдвигах, а без нормировки — ≤ 0,1 px (синтетика в
    test_engine_preprocess.py). tomotools4.measure_repositioning_shifts не менялся."""
    from skimage.registration import phase_cross_correlation  # noqa: WPS433

    empty_result = (np.array([], dtype='float32'), np.array([], dtype='float64'), np.array([], dtype='float64'))
    if not scan.is_advanced or not de.periodic_empties:
        if scan.is_advanced:
            logger.warning('No periodic empties found — no checkpoints to measure')
        return empty_result
    frames = crop.frames
    if tuple(np.shape(de.dark)) != tuple(frames.shape[1:]):
        raise ValueError('dark/empty {} посчитаны не по всему кропу {}'.format(np.shape(de.dark), frames.shape[1:]))

    fnums = np.asarray(scan.frame_numbers)
    angles = np.asarray(scan.angles)
    d_idx = np.asarray(scan.data_idx, dtype=np.int64)
    d_idx = d_idx[np.argsort(fnums[d_idx], kind='stable')]
    c_idx = np.asarray(scan.check_idx, dtype=np.int64)
    c_idx = c_idx[np.argsort(fnums[c_idx], kind='stable')]
    d_fn, d_ang = fnums[d_idx], angles[d_idx]
    c_fn, c_ang = fnums[c_idx], angles[c_idx]

    k_total = len(de.periodic_empties)
    cp_angles = np.zeros(k_total, dtype='float32')
    sy = np.zeros(k_total, dtype='float64')
    sx = np.zeros(k_total, dtype='float64')
    if d_idx.size == 0:
        logger.warning('repositioning_shifts: нет data-кадров')
        return cp_angles, sy, sx
    pf = [int(v) for v in de.periodic_empty_fnumbers]
    for k in range(k_total):
        next_fn = pf[k + 1] if k + 1 < k_total else int(d_fn[-1]) + 1
        fn_start = pf[k]
        dc = np.where((c_fn >= fn_start) & (c_fn < next_fn))[0]
        if dc.size == 0:
            logger.warning('Checkpoint %d: no data_check frames found, skipping', k)
            continue
        dc0 = int(dc[0])
        dc_angle = float(c_ang[dc0])
        cp_angles[k] = dc_angle
        try:
            j = _find_matching_data_frame(dc_angle, d_ang, d_fn, fn_start)
        except ValueError as exc:
            logger.warning('Checkpoint %d: %s; skipping', k, exc)
            continue
        if j is None:
            logger.warning('Checkpoint %d: no matching data frame at angle %.2f, skipping', k, dc_angle)
            continue
        de_k = DarkEmpty(de.dark, de.periodic_empties[k], 0, [], [])
        pair = np.stack([np.asarray(frames[d_idx[j]]), np.asarray(frames[c_idx[dc0]])])
        norm = _normalize(pair, [0, 0], de_k, np, median3=True, clip0=False)
        shift, _err, _phase = phase_cross_correlation(norm[0], norm[1], upsample_factor=10, normalization=None)
        sy[k], sx[k] = float(shift[0]), float(shift[1])
        logger.info('Checkpoint %d, angle=%.2f: shift_y=%.3f, shift_x=%.3f', k, dc_angle, sy[k], sx[k])
    return cp_angles, sy, sx


def _binned_norm(scan: ScanInfo, crop: CropData, de: DarkEmpty, idx: Sequence[int], b: int = CHECK_BIN) -> np.ndarray:
    """Кадры idx (индексы timeline) → −ln T с интерполяцией empty по frame_number, уменьшенные ×b (среднее), (n, h, w)."""
    fn = np.asarray(scan.frame_numbers)[list(idx)]
    out = []
    for a in range(0, len(idx), 8):
        part = np.asarray(crop.frames[[int(i) for i in idx[a:a + 8]]])
        nm = _normalize(part, fn[a:a + 8], de, np, median3=False, clip0=False)
        n, h, w = nm.shape
        h2, w2 = h // b * b, w // b * b
        out.append(nm[:, :h2, :w2].reshape(n, h2 // b, b, w2 // b, b).mean(axis=(2, 4)))
    return np.concatenate(out) if out else np.zeros((0, 1, 1), dtype='float32')


def _pair_rms(ref: np.ndarray, mov: np.ndarray) -> float:
    """rms разности ref и mov после совмещения сдвигом (взаимная корреляция), без полос у краёв."""
    from scipy import ndimage  # noqa: WPS433
    from skimage.registration import phase_cross_correlation  # noqa: WPS433
    s, _, _ = phase_cross_correlation(ref, mov, upsample_factor=4, normalization=None)
    moved = ndimage.shift(mov, s, order=1, mode='nearest')
    m = int(math.ceil(float(np.abs(s).max()))) + 4
    if 2 * m >= min(ref.shape):
        return float('inf')
    return float(np.sqrt(((ref[m:-m, m:-m] - moved[m:-m, m:-m]) ** 2).mean()))


def check_checkpoints(scan: ScanInfo, crop: CropData, de: DarkEmpty) -> List[Dict[str, Any]]:
    """Проверка контрольных кадров advanced-скана: по одному результату на периодическую вставку k.

    Поля: k, angle (угол data_check), status — 'ok' | 'rotated' (кадр после вставки совпадает с кадром более раннего
    угла: образец или стол провернулись назад, счётчик мотора этого не видит) | 'changed' (кадр отличается сильнее
    шага угла, но более ранний угол не подходит: образец сместился нежёстко или провернулся вперёд) | 'skipped';
    rms_same, rms_step (соседние data-кадры — масштаб одного шага угла), ratio, best_angle, offset_deg (best − angle),
    rms_best, message (для предупреждения, по-русски; у 'ok' — пусто).

    de — по всему кропу. Кадры уменьшаются ×CHECK_BIN; угол ищется только назад (кадров после вставки для сравнения
    нет — они сняты уже со сбоем), сначала через CHECK_COARSE_DEG, затем по всем кадрам около лучшего. Поиск идёт
    только для подозрительных вставок (ratio > CHECK_SUSPECT_RATIO): у исправного скана проверка — 3 кадра на вставку."""
    out: List[Dict[str, Any]] = []
    if not scan.is_advanced or not de.periodic_empties:
        return out
    # бининг: ×CHECK_BIN для рабочих кропов (сотни–тысячи пикселей), меньше — для маленьких
    b = CHECK_BIN if min(crop.frames.shape[1:]) >= 64 * CHECK_BIN else max(1, min(crop.frames.shape[1:]) // 64)
    fnums = np.asarray(scan.frame_numbers)
    angles = np.asarray(scan.angles, dtype='float64')
    d_idx = np.asarray(scan.data_idx, dtype=np.int64)
    d_idx = d_idx[np.argsort(fnums[d_idx], kind='stable')]
    c_idx = np.asarray(scan.check_idx, dtype=np.int64)
    c_idx = c_idx[np.argsort(fnums[c_idx], kind='stable')]
    d_fn, d_ang = fnums[d_idx], angles[d_idx]
    c_fn = fnums[c_idx]
    pf = [int(v) for v in de.periodic_empty_fnumbers]
    k_total = len(pf)
    for k in range(k_total):
        res: Dict[str, Any] = {'k': k, 'angle': None, 'status': 'skipped', 'rms_same': None, 'rms_step': None,
                               'ratio': None, 'best_angle': None, 'offset_deg': None, 'rms_best': None, 'message': ''}
        out.append(res)
        next_fn = pf[k + 1] if k + 1 < k_total else int(d_fn[-1]) + 1
        dc = np.where((c_fn >= pf[k]) & (c_fn < next_fn))[0]
        if dc.size == 0 or d_idx.size < 2:
            continue
        c = int(c_idx[int(dc[0])])
        th = float(angles[c])
        res['angle'] = th
        try:
            j = _find_matching_data_frame(th, d_ang, d_fn, pf[k])
        except ValueError:
            j = None
        if j is None or j == 0:
            continue
        before = np.where(d_fn < pf[k])[0]                       # data-кадры до вставки (позиции в d_idx)
        st = _binned_norm(scan, crop, de, [c, int(d_idx[j]), int(d_idx[j - 1])], b)
        r_same, r_step = _pair_rms(st[1], st[0]), _pair_rms(st[1], st[2])
        ratio = r_same / r_step if r_step > 0 else float('inf')
        res.update(status='ok', rms_same=r_same, rms_step=r_step, ratio=ratio, best_angle=th, offset_deg=0.0,
                   rms_best=r_same)
        if not ratio > CHECK_SUSPECT_RATIO:
            continue
        # поиск назад: грубо через CHECK_COARSE_DEG, затем все кадры около лучшего
        diff = th - d_ang[before]
        cand = before[(diff >= 0) & (diff <= CHECK_SEARCH_DEG)]
        step = float(np.median(np.abs(np.diff(d_ang[before])))) if before.size > 1 else 0.5
        stride = max(1, int(round(CHECK_COARSE_DEG / max(step, 1e-6))))
        coarse = cand[::-1][::stride][::-1]
        rms_of: Dict[int, float] = {int(j): r_same}
        for pos, img in zip(coarse, _binned_norm(scan, crop, de, [int(d_idx[q]) for q in coarse], b)):
            rms_of[int(pos)] = _pair_rms(img, st[0])
        best = min(rms_of, key=rms_of.get)
        near = [int(q) for q in cand if abs(d_ang[q] - d_ang[best]) <= CHECK_COARSE_DEG + 1e-6 and int(q) not in rms_of]
        for pos, img in zip(near, _binned_norm(scan, crop, de, [int(d_idx[q]) for q in near], b)):
            rms_of[pos] = _pair_rms(img, st[0])
        best = min(rms_of, key=rms_of.get)
        best_angle = float(d_ang[best])
        # уточнение параболой по соседним кадрам (если есть оба)
        nb = [q for q in (best - 1, best + 1) if q in rms_of]
        if len(nb) == 2:
            y0, y1, y2 = rms_of[best - 1], rms_of[best], rms_of[best + 1]
            den = y0 - 2 * y1 + y2
            if den > 0:
                best_angle += 0.5 * (y0 - y2) / den * float(d_ang[best + 1] - d_ang[best])
        r_best = rms_of[best]
        off = best_angle - th
        res.update(best_angle=best_angle, offset_deg=off, rms_best=r_best)
        if off <= -CHECK_ROTATED_MIN_DEG and r_best < CHECK_ROTATED_RATIO * r_same:
            res['status'] = 'rotated'
            res['message'] = ('после вставки {} ({:.1f}°) образец повернулся назад примерно на {:.1f}°: кадр после вставки '
                              'совпадает с кадром {:.1f}° (расхождение {:.3f} против {:.3f} с кадром того же угла)'
                              ).format(k + 1, th, -off, best_angle, r_best, r_same)
        else:
            res['status'] = 'changed'
            res['message'] = ('после вставки {} ({:.1f}°) кадр под тем же углом отличается в {:.1f} раза сильнее, чем '
                              'соседние кадры: образец сместился, изменился или провернулся'
                              ).format(k + 1, th, ratio)
        logger.warning('Checkpoint %d: %s', k, res['message'])
    return out


def checks_summary(checks: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Итоговое предупреждение по проверке контрольных кадров или None (все в порядке)."""
    bad = [c for c in checks if c.get('status') in ('rotated', 'changed')]
    if not bad:
        return None
    total = sum(c['offset_deg'] for c in bad if c['status'] == 'rotated')
    tail = ' (накопленный сбой угла ≈ {:.1f}°)'.format(total) if total else ''
    return ('контрольные кадры: после {} из {} вставок данные сняты не под записанным углом{} — сдвиги образца по этим '
            'вставкам не применяются, авто-ось и срез по всем углам ненадёжны; проверьте крепление образца и удержание '
            'поворотного стола').format(len(bad), len(checks), tail)


def segment_index(frame_numbers: Sequence[int], periodic_empty_fnumbers: Sequence[int]) -> np.ndarray:
    """Номер сегмента для каждого кадра: 0 до первой периодической вставки, k после k-й
    (кадр с frame_number > fnumber k-й серии — как в tomotools4.apply_repositioning_correction)."""
    pf = np.sort(np.asarray(periodic_empty_fnumbers, dtype=np.int64))
    return np.searchsorted(pf, np.asarray(frame_numbers, dtype=np.int64), side='left').astype(np.int64)


def cumulative_shifts(sy: np.ndarray, sx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Абсолютные сдвиги сегментов 0..K: cum[0] = 0, cum[k] = sum(shifts[:k]); NaN трактуется как 0."""
    sy = np.asarray(sy, dtype='float64')
    sx = np.asarray(sx, dtype='float64')
    nan_mask = np.isnan(sy) | np.isnan(sx)
    for k in np.where(nan_mask)[0]:
        logger.error('cumulative_shifts: сдвиг checkpoint-а %d не измерен (NaN); принимаем его за 0, '
                     'но сдвиги сегментов после %d ненадёжны', k, k)
    sy = np.where(np.isnan(sy), 0.0, sy)
    sx = np.where(np.isnan(sx), 0.0, sx)
    return np.concatenate(([0.0], np.cumsum(sy))), np.concatenate(([0.0], np.cumsum(sx)))
