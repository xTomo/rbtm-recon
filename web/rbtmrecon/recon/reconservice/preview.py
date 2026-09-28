"""Вычисления превью по кропу сессии: срез, авто-ось, перебор центра, вид 0°−180°, кольца, сдвиги образца.

Строится на тех же функциях движка, что и ``pipeline.run_recipe`` (``preprocess.normalize_slab``,
``pipeline._apply_shifts``, ``axis.align_rows``, ``rings.apply``, ``fbp.recon_rows``), поэтому превью совпадает со
срезом итоговой реконструкции при тех же параметрах (проверяется тестом). Кэш полосы строк — см. ``sessions``.

Контекст (``Context``) создаётся после загрузки кропа (``build_context``: опорные кадры по кропу и сдвиги образца —
как ``pipeline.prepare``, но без оси) и живёт в ``Session.extra['ctx']``. Методы вызываются под
``Session.compute_lock`` (на GPU сессии одновременно считается один запрос), сам контекст не потокобезопасен.
``check`` — проверка «запрос не устарел» (``arbiter.Ticket.check``), вызывается между стадиями.

Срез строки кропа r при оси (shift_x, alfa) — то же, что ``pipeline.process_slab`` для слоя из одной строки:
нормированные и сдвинутые по образцу строки [r − m, r + 1 + m) (m = ``pipeline.margin``) → ``axis.align_rows`` →
``rings.apply`` → ``fbp.recon_rows``. Отличие одно: нормированные строки берутся из полосы (кэша), посчитанной для
более широкого диапазона строк, а края слоя (медиана 3×3, сплайн-префильтр сдвига образца) у полосы и у слоя
run_recipe — в разных местах; за запасом их влияние затухает (см. pipeline), расхождение ~1e-5 от размаха.

Полоса (``Band``) — нормированные и сдвинутые по образцу строки [b0, b1) всех data-кадров, (n, s, w) float32.
Высота — запас ``axis.margin_rows`` под текущий наклон + BAND_TILT_RESERVE_DEG и BAND_RESERVE_ROWS + запас под
вертикальный сдвиг образца. Смена центра, и наклона в пределах резерва, полосу не пересчитывает; строка вне
полосы или больший наклон — пересчёт. На сервере (n = 400–800, ширина ~3200) строка полосы — 5–10 МБ, полоса при
наклоне до ~0,5° — 40–60 строк, т.е. сотни МБ: она держится на GPU, если занимает не больше BAND_GPU_FRACTION
свободной памяти, иначе — в RAM (на GPU для каждого запроса переносится только нужный кусок строк). Нормировка
идёт кусками кадров под свободную память и при нехватке памяти GPU повторяется с полосой в RAM.

Перебор центра — как ``axis.center_scan``: строка выравнивается по базовой оси и чистится от колец, для кандидата c
сдвигается на (центр кропа − c) в частотной области (``axis.fourier_shift_rows``: сплайн сглаживал шум сильнее при
полуцелых сдвигах, и метрика — и глаз — выбирали полуцелые центры), восстанавливается; метрика (по умолчанию
``grad`` — энергия градиента сглаженного фрагмента) — по показываемым фрагментам ``region``, по умолчанию — квадрат
с наибольшей энергией краёв (``structured_region``): внутри однородного образца сведений о центре нет.

Смена только центра при просмотре среза идёт быстрым путём (``Context.corrected_row``): готовая строка после колец
сдвигается в частотной области, без нового выравнивания и колец; ``exact`` — полный путь.
"""
from __future__ import annotations

import dataclasses
import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from reconengine import axis as axis_mod
from reconengine import fbp, gpu, pipeline, preprocess, rings
from reconengine import recipe as recipe_mod
from reconengine.model import Axis, CropData, ProgressFn, ROI, ScanInfo, check_cancel, no_progress

logger = logging.getLogger(__name__)

#: Резерв полосы: наклон сверх текущего и строки сверх запаса, при которых полоса не пересчитывается.
BAND_TILT_RESERVE_DEG = 0.25
BAND_RESERVE_ROWS = 4
#: Полоса держится на GPU, если занимает не больше этой доли свободной памяти (остальное — выравнивание, FBP).
BAND_GPU_FRACTION = 0.4
#: Доля свободной памяти GPU на рабочие копии нормировки куска кадров (слой, медиана, маска — ~3 копии).
_WORK_GPU_FRACTION = 0.2
_NORM_COPIES = 3
_OOM_RETRIES = 3

DEFAULT_RINGS = 'medium'
DEFAULT_ANGLES = 'first_180'
#: Сторона фрагмента перебора центра по умолчанию (центральный квадрат) и предел числа кандидатов.
SCAN_REGION_PX = 256
#: Быстрый путь смены центра (сдвиг готовой строки) — при разнице сдвигов кропа не больше стольких пикселей.
FAST_SHIFT_MAX_PX = 16.0
SCAN_MAX_N = 25

Check = Callable[[], None]
#: Область среза (x0, y0, x1, y1) в пикселях среза w×w, полуоткрытая.
Region = Tuple[int, int, int, int]


def _no_check() -> None:
    """Проверка «запрос не устарел» по умолчанию — ничего не делает."""


def check_region(region: Optional[Region], width: int) -> Region:
    """Область среза w×w: None — весь срез; иначе 0 ≤ x0 < x1 ≤ w, 0 ≤ y0 < y1 ≤ w (иначе ValueError)."""
    if region is None:
        return 0, 0, width, width
    x0, y0, x1, y1 = (int(v) for v in region)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= width):
        raise ValueError('region {},{},{},{} вне среза {}×{}'.format(x0, y0, x1, y1, width, width))
    return x0, y0, x1, y1


def structured_region(img: np.ndarray, side: int = SCAN_REGION_PX) -> Region:
    """Квадрат стороны min(side, w) внутри вписанного круга, где больше всего краёв (энергия градиента среза,
    сглаженного и уменьшенного в 4 раза): фрагмент перебора центра по умолчанию. Внутри однородного образца
    сведений о центре нет — фрагмент должен захватывать края."""
    import scipy.ndimage as ndi  # noqa: WPS433
    a = np.asarray(img, dtype='float32')
    w = a.shape[0]
    s = min(int(side), w)
    f = 4 if w >= 512 else 1                  # на реальных срезах (~3000 px) — считать по уменьшенному
    small = a[:w // f * f, :w // f * f].reshape(w // f, f, w // f, f).mean(axis=(1, 3)) if f > 1 else a
    gy, gx = np.gradient(ndi.gaussian_filter(small, 1.0))
    e = (gx * gx + gy * gy) * axis_mod._circle_mask(small)
    k = max(1, s // f)
    box = ndi.uniform_filter(e, size=k, mode='constant')
    n = small.shape[0]
    lo, hi = k // 2, n - (k - k // 2)
    if hi < lo:
        return central_region(w, side)
    sub = box[lo:hi + 1, lo:hi + 1]
    iy, ix = np.unravel_index(int(np.argmax(sub)), sub.shape)
    x0 = min(max(0, ix * f), w - s)
    y0 = min(max(0, iy * f), w - s)
    return x0, y0, x0 + s, y0 + s


def central_region(width: int, side: int = SCAN_REGION_PX) -> Region:
    """Центральный квадрат стороны min(side, w)."""
    s = min(int(side), int(width))
    a = (width - s) // 2
    return a, a, a + s, a + s


def _check_angles(mode: str) -> str:
    if mode not in fbp.ANGLE_MODES:
        raise ValueError('неизвестный режим углов: {} (допустимы {})'.format(mode, ', '.join(fbp.ANGLE_MODES)))
    return mode


def pooled_window(arrays) -> Tuple[float, float]:
    """Общее окно квантования: персентили 0,1 и 99,9 всех значений."""
    v = np.concatenate([np.asarray(a, dtype='float32').ravel() for a in arrays])
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, [0.1, 99.9])
    return float(lo), float(hi) if hi > lo else float(lo) + 1.0


def fragment_metrics(frags: List[np.ndarray], metric: str) -> np.ndarray:
    """Метрики фрагментов, как в axis.center_scan: для 'entropy' диапазон гистограммы общий (персентили 0,1…99,9
    объединённых значений внутри вписанных кругов)."""
    value_range = None
    if metric == 'entropy' and frags:
        mask = axis_mod._circle_mask(frags[0])
        pooled = np.concatenate([f[mask] for f in frags])
        value_range = tuple(np.percentile(pooled, [0.1, 99.9]))
    return np.array([axis_mod.center_metric(f, metric, value_range) for f in frags], dtype='float64')


@dataclasses.dataclass
class Band:
    """Нормированные и сдвинутые по образцу строки кропа [rows[0], rows[1]) всех data-кадров."""
    rows: Tuple[int, int]
    good: Tuple[int, int]            # строки, где полоса совпадает с обработкой всего кропа (без краёв)
    data: Any                        # (n, s, w) float32: xp-массив (on_gpu) или numpy
    on_gpu: bool
    seconds: float                   # время расчёта


class Context:
    """Всё для превью по загруженному кропу: опорные кадры, сдвиги образца, текущая ось, кэши."""

    def __init__(self, scan: ScanInfo, crop: CropData, prep: pipeline.Prepared, pixel_size_mm: float,
                 checkpoints: Optional[Dict[str, List[float]]] = None, xp=None):
        self.scan = scan
        self.crop = crop
        self.roi: ROI = crop.roi
        self.prep = prep
        self.pixel_size = float(pixel_size_mm)
        self.checkpoints = checkpoints            # {'angles', 'sy', 'sx'} измеренные (advanced) или None
        self.xp = xp or gpu.get_xp()
        self.axis: Optional[Axis] = None          # текущая ось сессии (авто или заданная через axis/tilt)
        self.warnings: List[str] = list(prep.warnings)
        self.timings: Dict[str, float] = {}       # последнего среза (для оценки времени задачи)
        self._band: Optional[Band] = None
        self._band_row_s: Optional[float] = None  # секунд нормировки на строку полосы
        self._pair = None                         # (img0, img180) нормированные кадры пары 0°/180°, numpy
        self._pair_pos: Optional[Tuple[int, int]] = None
        self._row = None                          # (ключ, полоса, выровненная строка)
        self._corrected = None                    # выровненная строка после колец (см. corrected_row)

    def release(self) -> None:
        """Отпустить кэши (закрытие сессии, новая загрузка)."""
        self._band = None
        self._pair = None
        self._row = None
        self._corrected = None

    # --- строки и ось --------------------------------------------------------------------------------------

    def crop_row(self, row: int) -> int:
        """Строка детектора → строка кропа (ValueError вне ROI)."""
        r = int(row) - self.roi.y0
        if not 0 <= r < self.roi.height:
            raise ValueError('строка {} вне ROI [{}, {})'.format(row, self.roi.y0, self.roi.y1))
        return r

    def pair(self) -> Tuple[np.ndarray, np.ndarray]:
        """Нормированные (и сдвинутые по образцу) кадры кропа при ~0° и ~180° — как в pipeline.prepare."""
        if self._pair is None:
            p = self.prep
            p0, p180 = pipeline.pair_0_180(p.angles)
            raw = np.asarray(self.crop.frames[[int(p.idx[p0]), int(p.idx[p180])]])
            norm = preprocess.normalize_slab(raw, p.fnums[[p0, p180]], p.de, xp=self.xp)
            norm = pipeline._apply_shifts(norm, p.frame_sy[[p0, p180]], p.frame_sx[[p0, p180]], self.xp)
            self._pair = (np.asarray(gpu.to_numpy(norm[0])), np.asarray(gpu.to_numpy(norm[1])))
            self._pair_pos = (p0, p180)
        return self._pair

    def auto_axis(self, check: Check = _no_check) -> Dict[str, Any]:
        """Авто-ось по паре 0°/180° (как pipeline.prepare без оси в рецепте); становится текущей осью сессии."""
        t0 = time.time()
        img0, img180 = self.pair()
        check()
        ax = axis_mod.auto_axis(img0, img180, self.roi)
        self.axis = ax
        shift_x, alfa = axis_mod.to_crop_params(ax, self.roi)
        p0, p180 = self._pair_pos
        return {'axis': ax.to_dict(), 'shift_x': shift_x, 'alfa': alfa,
                'pair': {'angles': [float(self.prep.angles[p0]), float(self.prep.angles[p180])],
                         'indices': [int(self.prep.idx[p0]), int(self.prep.idx[p180])]},
                'seconds': round(time.time() - t0, 3)}

    def current_axis(self, check: Check = _no_check) -> Axis:
        if self.axis is None:
            self.auto_axis(check)
        return self.axis

    def resolve_axis(self, row: int, center: Optional[float] = None, tilt: Optional[float] = None,
                     check: Check = _no_check) -> Axis:
        """Ось запроса: center — столбец оси на строке детектора row, tilt — наклон, градусы. Недостающее
        берётся из текущей оси сессии (авто-ось считается при первом запросе)."""
        if center is None or tilt is None:
            base = self.current_axis(check)
            tilt = base.tilt_deg if tilt is None else tilt
            center = base.center_at(row) if center is None else center
        return Axis(center_x=float(center), y_ref=float(row), tilt_deg=float(tilt), method='manual')

    # --- полоса --------------------------------------------------------------------------------------------

    def _edge_rows(self) -> int:
        """Запас сверх axis.margin_rows (вертикальный сдвиг образца и префильтр сплайна) — как в pipeline.margin."""
        w = self.roi.width
        return pipeline.margin(self.prep, w) - axis_mod.margin_rows(self.prep.alfa, w)

    def _need(self, r: int, alfa: float) -> Tuple[int, int]:
        """Строки, которые align_rows берёт для выходной строки r при наклоне alfa."""
        ma = axis_mod.margin_rows(alfa, self.roi.width)
        return max(0, r - ma), min(self.roi.height, r + 1 + ma)

    def band(self, r: int, alfa: float, check: Check = _no_check) -> Tuple[Band, bool]:
        """Полоса, покрывающая строку кропа r при наклоне alfa: (полоса, взята ли из кэша)."""
        need = self._need(r, alfa)
        b = self._band
        if b is not None and b.good[0] <= need[0] and need[1] <= b.good[1]:
            return b, True
        self._band = self._row = None
        if gpu.is_gpu(self.xp):
            gpu.free_memory()
        h, w = self.roi.height, self.roi.width
        g = self._edge_rows()
        mb = axis_mod.margin_rows(abs(alfa) + BAND_TILT_RESERVE_DEG, w) + BAND_RESERVE_ROWS + g
        b0, b1 = max(0, r - mb), min(h, r + 1 + mb)
        t0 = time.time()
        arr, on_gpu = self._normalize_rows(b0, b1, check)
        dt = time.time() - t0
        good = (b0 + g if b0 > 0 else 0, b1 - g if b1 < h else h)
        self._band = Band((b0, b1), good, arr, on_gpu, dt)
        self._band_row_s = dt / (b1 - b0)
        logger.info('полоса превью: строки кропа [%d, %d), %.0f МБ, %s, %.2f с', b0, b1,
                    arr.nbytes / 2 ** 20, 'GPU' if on_gpu else 'RAM', dt)
        return self._band, False

    def _normalize_rows(self, b0: int, b1: int, check: Check) -> Tuple[Any, bool]:
        """Нормировка и сдвиг образца строк [b0, b1) всех data-кадров кусками кадров (результат покадровый, поэтому
        совпадает с нормировкой слоя целиком). (массив, на GPU ли)."""
        p = self.prep
        xp = self.xp
        n, s, w = len(p.idx), b1 - b0, self.roi.width
        de = pipeline._rows_de(p.de, b0, b1)
        frame_bytes = s * w * 4
        on_gpu, step = False, n
        if gpu.is_gpu(xp):
            info = gpu.mem_info()
            free = info[0] if info else 0
            on_gpu = n * frame_bytes <= BAND_GPU_FRACTION * free
            step = int(max(1, min(n, _WORK_GPU_FRACTION * free // (frame_bytes * _NORM_COPIES))))
        for attempt in range(_OOM_RETRIES + 1):
            try:
                store = xp if on_gpu else np
                out = store.empty((n, s, w), dtype=store.float32)
                for a in range(0, n, step):
                    check()
                    e = min(n, a + step)
                    raw = np.asarray(self.crop.frames[p.idx[a:e], b0:b1, :])
                    part = preprocess.normalize_slab(raw, p.fnums[a:e], de, xp=xp)
                    del raw
                    part = pipeline._apply_shifts(part, p.frame_sy[a:e], p.frame_sx[a:e], xp)
                    out[a:e] = part if on_gpu or not gpu.is_gpu(xp) else gpu.to_numpy(part)
                    del part
                return out, on_gpu
            except Exception as exc:  # noqa: BLE001 — нехватка памяти GPU: полоса в RAM, куски меньше
                if not pipeline._is_gpu_oom(exc) or attempt == _OOM_RETRIES:
                    raise
                out = None
                gpu.free_memory()
                on_gpu, step = False, max(1, step // 4)
                logger.warning('нехватка памяти GPU при нормировке полосы: полоса в RAM, кусок %d кадров', step)
        raise AssertionError('недостижимо')

    def aligned_row(self, row: int, ax: Axis, check: Check = _no_check) -> Tuple[Any, Dict[str, float]]:
        """Синограмма строки детектора row, выровненная по оси ax: xp (n, w) — как pipeline.process_slab до колец."""
        r = self.crop_row(row)
        shift_x, alfa = axis_mod.to_crop_params(ax, self.roi)
        band, cached = self.band(r, alfa, check)
        t = {'band_s': 0.0 if cached else round(band.seconds, 4), 'band_cached': cached}
        key = (r, shift_x, alfa)
        if self._row is not None and self._row[0] == key and self._row[1] is band:
            t['align_s'] = 0.0
            return self._row[2], t
        check()
        t0 = time.time()
        h, w = self.roi.height, self.roi.width
        m = axis_mod.margin_rows(alfa, w) + self._edge_rows()      # = pipeline.margin при этом наклоне
        in0, in1 = max(0, r - m), min(h, r + 1 + m)
        b0 = band.rows[0]
        sub = band.data[:, in0 - b0:in1 - b0, :]
        aligned = axis_mod.align_rows(sub, in0, (r, r + 1), shift_x, alfa, h, xp=self.xp)
        sino = self.xp.ascontiguousarray(aligned[:, 0, :])
        del aligned
        t['align_s'] = round(time.time() - t0, 4)
        self._row = (key, band, sino)
        return sino, t

    # --- срез, кольца, перебор центра ------------------------------------------------------------------------

    def _recon(self, sino, params, angle_mode: str) -> np.ndarray:
        """Срез (w, w) по выровненной строке (n, w): кольца, FBP — как в run_recipe."""
        s = rings.apply(sino[None], params, xp=self.xp)
        return fbp.recon_rows(s, self.prep.angles, self.pixel_size, angle_mode=angle_mode)[0]

    def corrected_row(self, row: int, ax: Axis, preset: str = DEFAULT_RINGS, exact: bool = False,
                      check: Check = _no_check) -> Tuple[Any, Dict[str, Any]]:
        """Выровненная строка после колец, xp (n, w).

        Смена только центра (та же строка, наклон и пресет колец) без ``exact`` не выравнивает и не чистит кольца
        заново: строка, посчитанная при прежнем центре, сдвигается по x в частотной области
        (``axis.fourier_shift_rows``) на разницу сдвигов кропа · cos(наклона) — на GPU ~1 с → ~0,2 с на срез при
        перетаскивании центра. Приближение: поворот после сдвига уводит его и по y на δ·sin(наклона) (при наклоне
        1° и сдвиге 2 px — 0,03 px), а кольца чистились до сдвига (полосы сдвигаются вместе со строкой).
        ``exact`` — полный путь, как ``pipeline.process_slab``."""
        r = self.crop_row(row)
        shift_x, alfa = axis_mod.to_crop_params(ax, self.roi)
        c = self._corrected
        if (not exact and c is not None and c['key'] == (r, alfa, preset)
                and abs(shift_x - c['shift_x']) <= FAST_SHIFT_MAX_PX):
            d = (shift_x - c['shift_x']) * math.cos(math.radians(alfa))
            t: Dict[str, Any] = {'band_s': 0.0, 'band_cached': True, 'align_s': 0.0, 'rings_s': 0.0,
                                 'fast_shift_px': round(d, 4)}
            if d == 0:
                return c['row'], t
            t0 = time.time()
            s = axis_mod.fourier_shift_rows(c['row'], d, self.xp)
            t['shift_s'] = round(time.time() - t0, 4)
            return s, t
        sino, t = self.aligned_row(row, ax, check)
        check()
        t0 = time.time()
        s = rings.apply(sino[None], rings.resolve(preset), xp=self.xp)[0]
        t['rings_s'] = round(time.time() - t0, 4)
        self._corrected = {'key': (r, alfa, preset), 'shift_x': shift_x, 'row': s}
        return s, t

    def slice(self, row: int, ax: Axis, preset: str = DEFAULT_RINGS, angle_mode: str = DEFAULT_ANGLES,
              region: Optional[Region] = None, check: Check = _no_check, exact: bool = False
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Срез строки детектора row (float32, область region среза w×w) и сведения для X-Meta.
        Без ``exact`` смена только центра идёт быстрым путём (см. ``corrected_row``)."""
        t_start = time.time()
        rings.resolve(preset)                      # неизвестный пресет — ValueError до вычислений
        _check_angles(angle_mode)
        x0, y0, x1, y1 = check_region(region, self.roi.width)
        s, t = self.corrected_row(row, ax, preset, exact, check)
        check()
        t0 = time.time()
        rec = fbp.recon_rows(s[None], self.prep.angles, self.pixel_size, angle_mode=angle_mode)[0]
        t['fbp_s'] = round(time.time() - t0, 4)
        img = np.ascontiguousarray(rec[y0:y1, x0:x1])
        t['total_s'] = round(time.time() - t_start, 4)
        self.timings = dict(t)
        meta = {'row': int(row), 'axis': ax.to_dict(), 'rings': preset, 'angles': angle_mode,
                'n_angles': int(fbp.select_angles(self.prep.angles, angle_mode).sum()),
                'region': [x0, y0, x1, y1], 'exact': 'fast_shift_px' not in t, 'timings': t}
        return img, meta

    def rings_preview(self, row: int, ax: Axis, preset: str = DEFAULT_RINGS, angle_mode: str = DEFAULT_ANGLES,
                      region: Optional[Region] = None, check: Check = _no_check
                      ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """(2, th, tw): срез без колец и с пресетом preset."""
        params = rings.resolve(preset)
        _check_angles(angle_mode)
        x0, y0, x1, y1 = check_region(region, self.roi.width)
        sino, t = self.aligned_row(row, ax, check)
        out = []
        for prm in (None, params):
            check()
            out.append(self._recon(sino, prm, angle_mode)[y0:y1, x0:x1])
        meta = {'row': int(row), 'axis': ax.to_dict(), 'rings': preset, 'params': params, 'angles': angle_mode,
                'region': [x0, y0, x1, y1], 'timings': t}
        return np.stack(out), meta

    def center_scan(self, row: int, ax: Axis, step: float = 1.0, n: int = 9, metric: str = 'grad',
                    region: Optional[Region] = None, preset: str = DEFAULT_RINGS,
                    angle_mode: str = DEFAULT_ANGLES, check: Check = _no_check
                    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Фрагменты среза (n, th, tw) при центрах ax.center_at(row) + (i − n//2)·step и метрики (см. модуль)."""
        if metric not in axis_mod.METRICS:
            raise ValueError('неизвестная метрика: {} (допустимы {})'.format(metric, ', '.join(axis_mod.METRICS)))
        if not 1 <= int(n) <= SCAN_MAX_N:
            raise ValueError('n = {} вне [1, {}]'.format(n, SCAN_MAX_N))
        if not (math.isfinite(step) and step > 0):
            raise ValueError('step должен быть > 0, получено {}'.format(step))
        rings.resolve(preset)
        _check_angles(angle_mode)
        w = self.roi.width
        if region is not None:
            check_region(region, w)
        t_start = time.time()
        base, t = self.corrected_row(row, ax, preset, exact=True, check=check)
        offsets = (np.arange(int(n)) - int(n) // 2) * float(step)

        def recon(off: float) -> np.ndarray:
            # как axis.center_scan: центр кропа + off ↔ сдвиг строки на −off (в частотной области)
            s = axis_mod.fourier_shift_rows(base, -float(off), self.xp)
            return fbp.recon_rows(s[None], self.prep.angles, self.pixel_size, angle_mode=angle_mode)[0]

        check()
        rec0 = recon(0.0)
        x0, y0, x1, y1 = check_region(region if region is not None else structured_region(rec0), w)
        frags = []
        for off in offsets:
            check()
            rec = rec0 if off == 0 else recon(off)
            frags.append(np.ascontiguousarray(rec[y0:y1, x0:x1], dtype='float32'))
        check()
        metrics = fragment_metrics(frags, metric)
        centers = ax.center_at(row) + offsets
        best = int(np.argmin(metrics))
        t['total_s'] = round(time.time() - t_start, 4)
        meta = {'row': int(row), 'axis': ax.to_dict(), 'step': float(step), 'metric': metric,
                'centers': [float(c) for c in centers], 'metrics': [float(v) for v in metrics],
                'best': float(centers[best]), 'best_index': best, 'rings': preset, 'angles': angle_mode,
                'region': [x0, y0, x1, y1], 'timings': t}
        return np.stack(frags), meta

    def diff(self, ax: Axis) -> np.ndarray:
        """axis.diff_view пары 0°/180° при оси ax: (h, w) float32."""
        img0, img180 = self.pair()
        shift_x, alfa = axis_mod.to_crop_params(ax, self.roi)
        return np.asarray(axis_mod.diff_view(img0, img180, shift_x, alfa), dtype='float32')

    # --- сведения ------------------------------------------------------------------------------------------

    def repositioning_info(self) -> Dict[str, Any]:
        """Применимость коррекции сдвига образца, checkpoint-ы, накопленные сдвиги сегментов, предупреждения."""
        p = self.prep
        periodic = list(p.de.periodic_empty_fnumbers) if p.de.periodic_empties else []
        cp = self.checkpoints
        out: Dict[str, Any] = {'advanced': bool(self.scan.is_advanced),
                               'applicable': bool(self.scan.is_advanced and cp is not None),
                               'checkpoints': [], 'cumulative': None, 'periodic_empty_fnumbers': periodic,
                               'warnings': list(self.warnings)}
        if cp is not None:
            out['checkpoints'] = [{'angle': float(a), 'sy': float(y), 'sx': float(x)}
                                  for a, y, x in zip(cp['angles'], cp['sy'], cp['sx'])]
            cy, cx = preprocess.cumulative_shifts(np.asarray(cp['sy']), np.asarray(cp['sx']))
            out['cumulative'] = {'sy': [float(v) for v in cy], 'sx': [float(v) for v in cx]}
            out['max_shift'] = {'sy': float(np.max(np.abs(p.frame_sy))) if p.frame_sy.size else 0.0,
                                'sx': float(np.max(np.abs(p.frame_sx))) if p.frame_sx.size else 0.0}
        elif self.scan.is_advanced:
            out['warnings'].append('нет периодических empty-серий — сдвиг образца не измеряется')
        return out

    def seconds_per_slice(self) -> Optional[float]:
        """Грубая оценка секунд на срез задачи по последнему превью: нормировка строки + выравнивание + кольца + FBP."""
        t = self.timings
        if not t or self._band_row_s is None:
            return None
        return float(self._band_row_s + t.get('align_s', 0.0) + t.get('rings_s', 0.0) + t.get('fbp_s', 0.0))


def build_context(scan: ScanInfo, crop: CropData, pixel_size_mm: float, progress: ProgressFn = no_progress,
                  cancel=None, normalization: str = 'auto', xp=None) -> Context:
    """Опорные кадры по кропу и сдвиги образца (advanced) — как pipeline.prepare с включённым repositioning, но без
    оси (ось задаётся в каждом запросе превью) и с углами checkpoint-ов для /repositioning."""
    xp = xp or gpu.get_xp()
    warnings: List[str] = []
    idx, angles, fnums = pipeline.data_frames(scan)
    if idx.size == 0:
        raise ValueError('{}: в скане нет data-кадров'.format(scan.exp_id))
    progress(0.0, 'dark_empty')
    de_full = preprocess.dark_empty_from_crop(scan, crop)
    check_cancel(cancel)
    de = pipeline._normalization_de(scan, de_full, normalization)

    checkpoints = None
    shifts = None
    sy = sx = np.zeros(0)
    if scan.is_advanced and de_full.periodic_empty_fnumbers:
        progress(0.5, 'repositioning')
        cp_angles, sy, sx = preprocess.repositioning_shifts(scan, crop, de_full)
        sy, sx = np.asarray(sy, dtype='float64'), np.asarray(sx, dtype='float64')
        if np.isnan(sy).any() or np.isnan(sx).any():
            warnings.append('сдвиг образца измерен не на всех checkpoint-ах: неизмеренные приняты за 0')
        shifts = {'sy': [float(v) for v in sy], 'sx': [float(v) for v in sx]}
        checkpoints = dict(shifts, angles=[float(a) for a in cp_angles])
        check_cancel(cancel)
    frame_sy, frame_sx = pipeline._frame_shifts(fnums, de_full.periodic_empty_fnumbers, sy, sx, warnings)
    roi = crop.roi
    # ось-заглушка (центр кропа, без наклона): Prepared требует ось, превью подставляет ось запроса
    ax0 = axis_mod.from_crop_params(0.0, 0.0, roi, method='placeholder')
    prep = pipeline.Prepared(idx=idx, angles=angles, fnums=fnums, de=de, frame_sy=frame_sy, frame_sx=frame_sx,
                             shifts=shifts, axis=ax0, shift_x=0.0, alfa=0.0, warnings=warnings)
    progress(1.0, 'ready')
    return Context(scan, crop, prep, pixel_size_mm, checkpoints, xp=xp)


def estimate(r: recipe_mod.Recipe, scan: ScanInfo, ctx: Optional[Context] = None,
             rate: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """pipeline.estimate + оценка времени. rate — скорость последних выполненных задач (``JobService.recent_rate``):
    секунд на срез · ширина² · угол и секунд подготовки; без неё — по замерам превью (секунд на срез,
    масштабированных на ширину рецепта как w²; превью считает строку с запасом, поэтому оценка завышена);
    без того и другого — time: None."""
    est = pipeline.estimate(r, scan)
    n_slices = int(est['volume_shape'][0])
    w = r.fov.width
    if rate:
        sps = rate['recon_s_per_slice_px2_angle'] * w * w * est['n_angles_used']
        est['time'] = {'s_per_slice': sps, 'n_slices': n_slices, 'recon_s': sps * n_slices,
                       'prepare_s': rate.get('prepare_s'), 'source': 'jobs', 'jobs': rate.get('jobs')}
        return est
    sps = ctx.seconds_per_slice() if ctx is not None else None
    if sps is None:
        est['time'] = None
        return est
    scale = (w / float(ctx.roi.width)) ** 2
    est['time'] = {'s_per_slice': sps * scale, 'n_slices': n_slices, 'recon_s': sps * scale * n_slices,
                   'source': 'preview'}
    return est
