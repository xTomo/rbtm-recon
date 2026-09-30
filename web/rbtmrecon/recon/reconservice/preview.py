"""Вычисления превью по кропу сессии: срез, авто-ось, перебор центра, вид 0°−180°, кольца, сглаживание, сравнение
вариантов на фрагменте, сдвиги образца.

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

Сглаживание проекций с деблюром (``smoothing``: линейный фильтр после колец и перед FBP) связывает соседние строки:
срез строки r — фильтр по строкам r ± h после колец (h — ``smoothing.halo_rows``), и только строка r идёт в FBP.
Поэтому со сглаживанием превью держит блок (``corrected_block``): выровненные (порциями по
``axis.BATCH_MAX_OUT_ROWS`` строк — батч быстрее) и очищенные от колец (порциями по BLOCK_RING_CHUNK) строки
r ± hb, hb = max(h, ореол σ = SMOOTH_RESERVE_SIGMA), ключ — (строка, наклон, пресет) и сдвиг кропа. Смена σ и метода
в этих пределах — только фильтр по блоку и FBP; смена центра — сдвиг блока в частотной области (как строки);
смена пресета колец или строки — кольца на блоке заново (дорого: секунды на GPU). Полоса под блок расширяется на
ореол один раз (с тем же запасом). Строки за краями кропа фильтр отражает — как задача, поэтому срез совпадает со
срезом run_recipe (тест). Без сглаживания путь прежний (одна строка).

Фрагмент (``region``) восстанавливается FBP только его пикселей (``fbp.recon_rows(region=...)``) — в десятки раз
дешевле полного среза. Сравнение вариантов (``compare``): кольца — раз на пресет, выравнивание — раз на запрос (блок
под наибольший ореол), каждый вариант — фильтр блока и FBP фрагмента; метрики — шум (``noise_level``) и резкость
(энергия градиента относительно первого варианта). Сравнивать варианты сглаживания честно при равном шуме.
"""
from __future__ import annotations

import dataclasses
import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from reconengine import axis as axis_mod
from reconengine import fbp, gpu, motion, pipeline, preprocess, rings, smoothing
from reconengine import recipe as recipe_mod
from reconengine.model import Axis, Cancelled, CropData, ProgressFn, ROI, ScanInfo, check_cancel, no_progress

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
#: Блок строк со сглаживанием строится с ореолом не меньше, чем у σ = SMOOTH_RESERVE_SIGMA (Винер по умолчанию):
#: σ до него (и метод) меняются без нового выравнивания и колец. Кольца на блоке — порциями по BLOCK_RING_CHUNK
#: строк, сдвиг блока в частотной области — по BLOCK_SHIFT_CHUNK (память).
SMOOTH_RESERVE_SIGMA = 2.0
BLOCK_RING_CHUNK = 16
BLOCK_SHIFT_CHUNK = 8
#: Сравнение вариантов: предел числа вариантов и сторона фрагмента по умолчанию.
COMPARE_MAX_VARIANTS = 8
COMPARE_REGION_PX = 384

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


def reserve_halo() -> int:
    """Ореол блока строк «с запасом»: halo_rows при σ = SMOOTH_RESERVE_SIGMA и прочих параметрах по умолчанию."""
    return smoothing.halo_rows(smoothing.resolve({'sigma': SMOOTH_RESERVE_SIGMA}))


def resolve_smoothing(block: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """``smoothing.resolve`` для параметров запроса (None, словарь; ошибка — ValueError)."""
    if block is not None and not isinstance(block, dict):
        raise ValueError('smoothing: ожидается объект {sigma, deblur, balance, amount} или null')
    return smoothing.resolve(block)


def compare_variants(variants) -> List[Dict[str, Any]]:
    """Варианты сравнения ``[{rings, smoothing}]`` → нормализованные ``{rings: пресет, smoothing: параметры | None}``:
    1…COMPARE_MAX_VARIANTS, rings по умолчанию DEFAULT_RINGS; неизвестный пресет, неверные параметры — ValueError."""
    if not isinstance(variants, (list, tuple)) or not 1 <= len(variants) <= COMPARE_MAX_VARIANTS:
        raise ValueError('variants: нужен список из 1…{} вариантов {{rings, smoothing}}'.format(COMPARE_MAX_VARIANTS))
    out = []
    for i, v in enumerate(variants):
        v = {} if v is None else v
        if not isinstance(v, dict):
            raise ValueError('variants[{}]: ожидается объект {{rings, smoothing}}, получено {!r}'.format(i, v))
        preset = v.get('rings') or DEFAULT_RINGS
        try:
            rings.resolve(preset)
            sp = resolve_smoothing(v.get('smoothing'))
        except ValueError as exc:
            raise ValueError('variants[{}]: {}'.format(i, exc)) from None
        out.append({'rings': preset, 'smoothing': sp})
    return out


def noise_level(img: np.ndarray) -> float:
    """Оценка шума фрагмента (те же единицы, 1/мм): 1,4826·MAD 5-точечного лапласиана / √20 — у белого шума σ
    дисперсия лапласиана 20σ², а MAD устойчива к краям объекта."""
    a = np.asarray(img, dtype='float64')
    if a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    lap = 4 * a[1:-1, 1:-1] - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    return float(1.4826 * np.median(np.abs(lap - np.median(lap))) / math.sqrt(20.0))


def compare_metrics(frags: List[np.ndarray]) -> List[Dict[str, Any]]:
    """Метрики фрагментов сравнения: noise (``noise_level``) и sharpness — энергия градиента (как
    ``axis.center_metric('grad')``, но «больше = резче»), делённая на значение у первого фрагмента (None, если
    там 0)."""
    energy = [-axis_mod.center_metric(f, 'grad') for f in frags]
    e0 = energy[0] if energy else 0.0
    return [{'noise': noise_level(f), 'sharpness': (float(e / e0) if e0 > 0 else None)}
            for f, e in zip(frags, energy)]


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
        self.axis: Optional[Axis] = None          # текущая ось сессии (авто или заданная через axis/tilt, axis/set)
        self.warnings: List[str] = list(prep.warnings)
        self.timings: Dict[str, float] = {}       # последнего среза
        self._full_timings: Dict[str, float] = {}  # последнего среза полным путём (выравнивание + кольца) — для оценки
        self.prepare_s: Optional[float] = None    # секунд build_context — задача повторяет ту же подготовку
        self._band: Optional[Band] = None
        self._band_row_s: Optional[float] = None  # секунд нормировки на строку полосы
        self._pair = None                         # (img0, img180) нормированные кадры пары 0°/180°, numpy
        self._auto: Optional[Axis] = None         # авто-ось по текущей паре (кэш; сбрасывается вместе с парой)
        self._pair_pos: Optional[Tuple[int, int]] = None
        self._row = None                          # (ключ, полоса, выровненная строка)
        self._corrected = None                    # выровненная строка после колец (см. corrected_row)
        self._block = None                        # блок строк после колец для сглаживания (см. corrected_block)
        # смещение образца (motion): оценка, режим, применено ли; сдвиги кадров без компенсации смещения
        self.empty_skip: int = 0                  # сколько первых кадров серий empty не брали (рецепт)
        self.motion_est: Optional[motion.Estimate] = None
        self.motion_mode: str = 'off'
        self.motion_applied: bool = False
        self.motion_message: str = 'смещение образца не измерялось'
        self._base_frame_sx = np.asarray(prep.frame_sx, dtype='float64').copy()

    def release(self) -> None:
        """Отпустить кэши (закрытие сессии, новая загрузка)."""
        self._band = None
        self._pair = None
        self._auto = None
        self._row = None
        self._corrected = None
        self._block = None

    # --- смещение образца ----------------------------------------------------------------------------------

    def init_motion(self, est: Optional[motion.Estimate], mode: str) -> Dict[str, Any]:
        """Оценка смещения (по сдвигам кадров без компенсации) и начальный режим."""
        self.motion_est = est
        self._base_frame_sx = np.asarray(self.prep.frame_sx, dtype='float64').copy()
        self.motion_applied = False
        return self.set_motion(mode)

    def _rotated(self) -> bool:
        return pipeline.rotated_checks((self.checkpoints or {}).get('checks'))

    def set_motion(self, mode: str) -> Dict[str, Any]:
        """Режим компенсации смещения: 'auto' | 'on' | 'off' (motion.decide). Если решение меняется — сдвиги кадров
        пересчитываются, кэши полос и пары 0°/180° сбрасываются, авто-ось сессии забывается (найдётся заново по
        сдвинутой паре); ось, заданная вручную, остаётся."""
        ok, why = motion.decide(self.motion_est, mode, self._rotated())
        self.motion_mode, self.motion_message = mode, why
        if ok != self.motion_applied:
            self.motion_applied = ok
            fsx = self._base_frame_sx - self.motion_est.dx if ok else self._base_frame_sx
            self.prep = dataclasses.replace(self.prep, frame_sx=np.asarray(fsx, dtype='float64'))
            self.release()
            if self.axis is not None and self.axis.method == 'auto':
                self.axis = None
        return self.motion_info(full=False)

    def motion_info(self, full: bool = True) -> Dict[str, Any]:
        """Для студии: режим, применено ли, пояснение; сводка оценки; full — ещё и dx по кадрам (для графика)."""
        est = self.motion_est
        out: Dict[str, Any] = {'mode': self.motion_mode, 'applied': self.motion_applied,
                               'message': self.motion_message, 'measured': est is not None}
        if est is not None:
            out.update(est.to_dict() if full else est.summary())
        return out

    def recipe_motion(self) -> Dict[str, Any]:
        """Блок motion рецепта по состоянию сессии: решение студии и сдвиги (если компенсировано)."""
        est = self.motion_est
        if self.motion_mode == 'off':
            return recipe_mod.motion_block('off', applied=False)
        if est is None:
            return recipe_mod.motion_block(self.motion_mode)            # задача оценит сама
        if not self.motion_applied:
            return recipe_mod.motion_block(self.motion_mode, applied=False, summary=est.summary())
        return recipe_mod.motion_block(self.motion_mode, True, est.dx, est.fnums, est.summary())

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
        ax = self.auto_axis_value(check)
        self.axis = ax
        shift_x, alfa = axis_mod.to_crop_params(ax, self.roi)
        p0, p180 = self._pair_pos
        return {'axis': ax.to_dict(), 'shift_x': shift_x, 'alfa': alfa,
                'pair': {'angles': [float(self.prep.angles[p0]), float(self.prep.angles[p180])],
                         'indices': [int(self.prep.idx[p0]), int(self.prep.idx[p180])]},
                'seconds': round(time.time() - t0, 3)}

    def auto_axis_value(self, check: Check = _no_check) -> Axis:
        """Авто-ось по текущей паре 0°/180° (с кэшем), не меняя ось сессии."""
        if self._auto is None:
            img0, img180 = self.pair()
            check()
            self._auto = axis_mod.auto_axis(img0, img180, self.roi)
        return self._auto

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

    def _need(self, r: int, alfa: float, halo: int = 0) -> Tuple[int, int]:
        """Строки, которые align_rows берёт для выходных строк r ± halo при наклоне alfa."""
        ma = axis_mod.margin_rows(alfa, self.roi.width)
        return max(0, r - halo - ma), min(self.roi.height, r + 1 + halo + ma)

    def band(self, r: int, alfa: float, check: Check = _no_check, halo: int = 0) -> Tuple[Band, bool]:
        """Полоса, покрывающая строки кропа r ± halo при наклоне alfa: (полоса, взята ли из кэша). Полоса под
        ореол сглаживания строится с запасом до ``reserve_halo`` строк."""
        need = self._need(r, alfa, halo)
        b = self._band
        if b is not None and b.good[0] <= need[0] and need[1] <= b.good[1]:
            return b, True
        self._band = self._row = self._block = None
        if gpu.is_gpu(self.xp):
            gpu.free_memory()
        h, w = self.roi.height, self.roi.width
        g = self._edge_rows()
        hb = max(halo, reserve_halo()) if halo > 0 else 0
        mb = axis_mod.margin_rows(abs(alfa) + BAND_TILT_RESERVE_DEG, w) + BAND_RESERVE_ROWS + g + hb
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

    def _aligned_block(self, r: int, shift_x: float, alfa: float, rows: Tuple[int, int],
                       check: Check = _no_check) -> Tuple[Any, Dict[str, Any]]:
        """Строки кропа rows = [r0, r1) (вокруг строки r), выровненные по оси: xp (s, n, w), как
        pipeline.process_slab до колец. Выравнивание — порциями по BATCH_MAX_OUT_ROWS выходных строк (батч на слой
        быстрее покадрового), вход каждой порции — её строки ± запас pipeline.margin из полосы."""
        halo = max(r - rows[0], rows[1] - 1 - r)
        band, cached = self.band(r, alfa, check, halo)
        t: Dict[str, Any] = {'band_s': 0.0 if cached else round(band.seconds, 4), 'band_cached': cached}
        xp = self.xp
        t0 = time.time()
        h, w = self.roi.height, self.roi.width
        m = axis_mod.margin_rows(alfa, w) + self._edge_rows()      # = pipeline.margin при этом наклоне
        b0 = band.rows[0]
        r0, r1 = rows
        out = xp.empty((r1 - r0, band.data.shape[0], w), dtype=xp.float32)
        for c0 in range(r0, r1, axis_mod.BATCH_MAX_OUT_ROWS):
            check()
            c1 = min(r1, c0 + axis_mod.BATCH_MAX_OUT_ROWS)
            in0, in1 = max(0, c0 - m), min(h, c1 + m)
            aligned = axis_mod.align_rows(band.data[:, in0 - b0:in1 - b0, :], in0, (c0, c1), shift_x, alfa, h, xp=xp)
            out[c0 - r0:c1 - r0] = xp.swapaxes(aligned, 0, 1)
            del aligned
        t['align_s'] = round(time.time() - t0, 4)
        return out, t

    def _rings_block(self, block, preset: str, check: Check = _no_check) -> Any:
        """Кольца на блоке (s, n, w) порциями по BLOCK_RING_CHUNK строк — на месте (по строкам независимо)."""
        params = rings.resolve(preset)
        if params:
            for c in range(0, block.shape[0], BLOCK_RING_CHUNK):
                check()
                block[c:c + BLOCK_RING_CHUNK] = rings.apply(block[c:c + BLOCK_RING_CHUNK], params, xp=self.xp)
        return block

    def _cached_block(self, r: int, shift_x: float, alfa: float, preset: str, rows: Tuple[int, int],
                      exact: bool) -> Optional[Tuple[Any, Dict[str, Any]]]:
        """Строки rows из кэша блока (тот же ключ, строки покрыты) или None. Другой сдвиг кропа в пределах
        FAST_SHIFT_MAX_PX без exact — быстрый путь: блок сдвигается по x в частотной области (см. corrected_row)."""
        c = self._block
        if (c is None or c['key'] != (r, alfa, preset) or not c['rows'][0] <= rows[0] <= rows[1] <= c['rows'][1]):
            return None
        sub = c['data'][rows[0] - c['rows'][0]:rows[1] - c['rows'][0]]
        t: Dict[str, Any] = {'band_s': 0.0, 'band_cached': True, 'align_s': 0.0, 'rings_s': 0.0,
                             'block_rows': c['rows'][1] - c['rows'][0]}
        if shift_x == c['shift_x']:
            return sub, t
        if exact or abs(shift_x - c['shift_x']) > FAST_SHIFT_MAX_PX:
            return None
        d = (shift_x - c['shift_x']) * math.cos(math.radians(alfa))
        t0 = time.time()
        out = self.xp.empty_like(sub)
        for k in range(0, sub.shape[0], BLOCK_SHIFT_CHUNK):
            out[k:k + BLOCK_SHIFT_CHUNK] = axis_mod.fourier_shift_rows(sub[k:k + BLOCK_SHIFT_CHUNK], d, self.xp)
        t['fast_shift_px'] = round(d, 4)
        t['shift_s'] = round(time.time() - t0, 4)
        return out, t

    def corrected_block(self, row: int, ax: Axis, preset: str = DEFAULT_RINGS, halo: int = 0, exact: bool = False,
                        check: Check = _no_check) -> Tuple[Any, int, Dict[str, Any]]:
        """Выровненные и очищенные от колец строки детектора row ± halo (в пределах кропа): (xp (s, n, w), индекс
        строки row в блоке, времена). Кэш — один блок (строка, наклон, пресет, сдвиг кропа), строится с ореолом
        max(halo, ``reserve_halo``); смена только центра без ``exact`` — быстрый путь (сдвиг блока)."""
        r = self.crop_row(row)
        shift_x, alfa = axis_mod.to_crop_params(ax, self.roi)
        rows = pipeline.halo_range((r, r + 1), halo, self.roi.height)
        got = self._cached_block(r, shift_x, alfa, preset, rows, exact)
        if got is not None:
            return got[0], r - rows[0], got[1]
        rings.resolve(preset)
        full = pipeline.halo_range((r, r + 1), max(halo, reserve_halo()) if halo > 0 else 0, self.roi.height)
        self._block = None                        # память прежнего блока — до выравнивания нового
        block, t = self._aligned_block(r, shift_x, alfa, full, check)
        t0 = time.time()
        self._rings_block(block, preset, check)
        t['rings_s'] = round(time.time() - t0, 4)
        t['block_rows'] = full[1] - full[0]
        self._block = {'key': (r, alfa, preset), 'shift_x': shift_x, 'rows': full, 'data': block}
        return block[rows[0] - full[0]:rows[1] - full[0]], r - rows[0], t

    # --- срез, кольца, перебор центра ------------------------------------------------------------------------

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

    def _fbp(self, rows, angle_mode: str, region: Region) -> np.ndarray:
        """FBP строк (s, n, w) только во фрагменте region (весь срез — полный FBP): (s, th, tw) float32."""
        return fbp.recon_rows(rows, self.prep.angles, self.pixel_size, angle_mode=angle_mode, region=region)

    def slice(self, row: int, ax: Axis, preset: str = DEFAULT_RINGS, angle_mode: str = DEFAULT_ANGLES,
              region: Optional[Region] = None, check: Check = _no_check, exact: bool = False,
              smooth: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Срез строки детектора row (float32, область region среза w×w — восстанавливается только она) и сведения
        для X-Meta. Без ``exact`` смена только центра идёт быстрым путём (см. ``corrected_row``). smooth — блок
        сглаживания (как в рецепте; None или sigma пустая — выключено): фильтр по блоку строк row ± h
        (``corrected_block``), в FBP — строка row."""
        t_start = time.time()
        rings.resolve(preset)                      # неизвестный пресет — ValueError до вычислений
        _check_angles(angle_mode)
        sp = resolve_smoothing(smooth)
        w = self.roi.width
        x0, y0, x1, y1 = check_region(region, w)
        if sp is None:
            s, t = self.corrected_row(row, ax, preset, exact, check)
            line = s[None]
            t.update(smooth_s=0.0, block_rows=1)
        else:
            block, i, t = self.corrected_block(row, ax, preset, smoothing.halo_rows(sp), exact, check)
            check()
            t0 = time.time()
            line = smoothing.apply(block, sp, keep=(i, i + 1), xp=self.xp)
            t['smooth_s'] = round(time.time() - t0, 4)
        check()
        t0 = time.time()
        img = np.ascontiguousarray(self._fbp(line, angle_mode, (x0, y0, x1, y1))[0])
        t['fbp_s'] = round(time.time() - t0, 4)
        t['fbp_px'] = (x1 - x0) * (y1 - y0)
        t['total_s'] = round(time.time() - t_start, 4)
        self.timings = dict(t)
        if t.get('align_s', 0.0) > 0 and 'rings_s' in t:
            self._full_timings = dict(t)
        meta = {'row': int(row), 'axis': ax.to_dict(), 'rings': preset, 'angles': angle_mode,
                'n_angles': int(fbp.select_angles(self.prep.angles, angle_mode).sum()),
                'region': [x0, y0, x1, y1], 'smoothing': sp, 'exact': 'fast_shift_px' not in t, 'timings': t}
        return img, meta

    def rings_preview(self, row: int, ax: Axis, preset: str = DEFAULT_RINGS, angle_mode: str = DEFAULT_ANGLES,
                      region: Optional[Region] = None, check: Check = _no_check
                      ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """(2, th, tw): срез без колец и с пресетом preset (region — по умолчанию весь срез) — частный случай
        ``compare``."""
        params = rings.resolve(preset)
        _check_angles(angle_mode)
        reg = check_region(region, self.roi.width)
        frags, meta = self.compare(row, ax, [{'rings': 'off'}, {'rings': preset}], region=reg,
                                   angle_mode=angle_mode, check=check, metrics=False)
        meta = {'row': int(row), 'axis': ax.to_dict(), 'rings': preset, 'params': params, 'angles': angle_mode,
                'region': meta['region'], 'timings': meta['timings']}
        return frags, meta

    def compare(self, row: int, ax: Axis, variants, region: Optional[Region] = None, size: int = COMPARE_REGION_PX,
                angle_mode: str = DEFAULT_ANGLES, check: Check = _no_check, metrics: bool = True
                ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Сравнение вариантов ``[{rings, smoothing}]`` на одном фрагменте: (k, th, tw) float32 и сведения для X-Meta
        (варианты нормализованы, метрики — ``compare_metrics``).

        Выравнивание — один блок строк row ± (наибольший ореол вариантов); кольца — раз на пресет (на строках под
        наибольший ореол его вариантов); каждый вариант — фильтр блока и FBP только фрагмента. region=None —
        ``structured_region`` стороны size по полному срезу первого варианта. Кэш блока (``corrected_block``)
        используется при точном совпадении оси и остаётся с последним посчитанным пресетом колец. metrics=False —
        без метрик (``metrics: null``; на полном срезе они стоят секунды CPU)."""
        t_start = time.time()
        vs = compare_variants(variants)
        _check_angles(angle_mode)
        w, h = self.roi.width, self.roi.height
        reg = check_region(region, w) if region is not None else None
        r = self.crop_row(row)
        shift_x, alfa = axis_mod.to_crop_params(ax, self.roi)
        halos = [smoothing.halo_rows(v['smoothing']) for v in vs]
        t: Dict[str, Any] = {'band_s': 0.0, 'band_cached': True, 'align_s': 0.0, 'rings_s': 0.0, 'smooth_s': 0.0,
                             'fbp_s': 0.0, 'block_rows': 0, 'ring_rows': 0}
        lines: List[Any] = [None] * len(vs)
        aligned, arows = None, None
        for preset in dict.fromkeys(v['rings'] for v in vs):          # пресеты в порядке появления
            idx = [j for j, v in enumerate(vs) if v['rings'] == preset]
            rows = pipeline.halo_range((r, r + 1), max(halos[j] for j in idx), h)
            got = self._cached_block(r, shift_x, alfa, preset, rows, exact=True)
            if got is not None:
                block = got[0]
            else:
                if aligned is None:
                    arows = pipeline.halo_range((r, r + 1), max(halos), h)
                    aligned, ta = self._aligned_block(r, shift_x, alfa, arows, check)
                    t.update(band_s=ta['band_s'], band_cached=ta['band_cached'], align_s=ta['align_s'],
                             block_rows=arows[1] - arows[0])
                block = aligned[rows[0] - arows[0]:rows[1] - arows[0]]
                if rings.resolve(preset):
                    t0 = time.time()
                    block = self._rings_block(block.copy(), preset, check)
                    t['rings_s'] = round(t['rings_s'] + time.time() - t0, 4)
                    t['ring_rows'] += rows[1] - rows[0]
                    self._block = {'key': (r, alfa, preset), 'shift_x': shift_x, 'rows': rows, 'data': block}
            i = r - rows[0]
            for j in idx:
                check()
                t0 = time.time()
                lines[j] = smoothing.apply(block, vs[j]['smoothing'], keep=(i, i + 1), xp=self.xp)
                t['smooth_s'] = round(t['smooth_s'] + time.time() - t0, 4)
            del block
        del aligned
        check()
        t0 = time.time()
        full0 = None
        if reg is None:
            full0 = self._fbp(lines[0], angle_mode, (0, 0, w, w))[0]
            reg = structured_region(full0, side=size)
        x0, y0, x1, y1 = reg
        frags = []
        for j, line in enumerate(lines):
            check()
            frag = full0[y0:y1, x0:x1] if j == 0 and full0 is not None else self._fbp(line, angle_mode, reg)[0]
            frags.append(np.ascontiguousarray(frag, dtype='float32'))
        t['fbp_s'] = round(time.time() - t0, 4)
        values = compare_metrics(frags) if metrics else None
        t['total_s'] = round(time.time() - t_start, 4)
        meta = {'row': int(row), 'axis': ax.to_dict(), 'angles': angle_mode,
                'n_angles': int(fbp.select_angles(self.prep.angles, angle_mode).sum()),
                'region': [x0, y0, x1, y1], 'variants': vs, 'metrics': values, 'timings': t,
                'downsample': 1}                  # binary.array_response заменит при уменьшении
        return np.stack(frags), meta

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

        def recon(off: float, reg: Region) -> np.ndarray:
            # как axis.center_scan: центр кропа + off ↔ сдвиг строки на −off (в частотной области)
            s = axis_mod.fourier_shift_rows(base, -float(off), self.xp)
            return self._fbp(s[None], angle_mode, reg)[0]

        check()
        if region is None:                         # фрагмент с краями — по полному срезу при текущем центре
            rec0 = recon(0.0, (0, 0, w, w))
            x0, y0, x1, y1 = check_region(structured_region(rec0), w)
            rec0 = rec0[y0:y1, x0:x1]
        else:
            x0, y0, x1, y1 = check_region(region, w)
            rec0 = None
        frags = []
        for off in offsets:
            check()
            rec = rec0 if off == 0 and rec0 is not None else recon(off, (x0, y0, x1, y1))
            frags.append(np.ascontiguousarray(rec, dtype='float32'))
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
            # проверка контрольных кадров (preprocess.check_checkpoints): статус, сбой угла, расхождения
            out['checks'] = list(cp.get('checks') or [])
            cy, cx = preprocess.cumulative_shifts(np.asarray(cp['sy']), np.asarray(cp['sx']))
            out['cumulative'] = {'sy': [float(v) for v in cy], 'sx': [float(v) for v in cx]}
            base_sx = self._base_frame_sx
            out['max_shift'] = {'sy': float(np.max(np.abs(p.frame_sy))) if p.frame_sy.size else 0.0,
                                'sx': float(np.max(np.abs(base_sx))) if base_sx.size else 0.0}
        elif self.scan.is_advanced:
            out['warnings'].append('нет периодических empty-серий — сдвиг образца не измеряется')
        return out

    def seconds_per_slice(self, ring_rows_factor: float = 1.0) -> Optional[float]:
        """Грубая оценка секунд на срез задачи по превью полным путём: нормировка строки + выравнивание + кольца
        (+ фильтр сглаживания) на строку + FBP. Быстрый путь (сдвиг готовой строки) выравнивание и кольца не
        повторяет — по нему оценка занижена (на реальном скане 0,19 с/срез против 0,30 в задаче); полный путь
        считает строку одну — с запасом. Блок строк (сглаживание) — делится на его строки; ring_rows_factor — во
        сколько раз строк через выравнивание и кольца в задаче больше срезов (ореол, ``pipeline.estimate``). FBP
        фрагмента пересчитывается на весь срез по площади (грубо)."""
        t = self._full_timings or self.timings
        if not t or self._band_row_s is None:
            return None
        k = max(1, int(t.get('block_rows', 1)))
        f = float(ring_rows_factor)
        per_row = (t.get('align_s', 0.0) + t.get('rings_s', 0.0) + t.get('smooth_s', 0.0)) / k
        fbp_s = t.get('fbp_s', 0.0) * self.roi.width ** 2 / max(1, int(t.get('fbp_px', self.roi.width ** 2)))
        return float((self._band_row_s + per_row) * f + fbp_s)


def build_context(scan: ScanInfo, crop: CropData, pixel_size_mm: float, progress: ProgressFn = no_progress,
                  cancel=None, normalization: str = 'auto', xp=None, motion_mode: str = 'auto',
                  empty_skip: int = recipe_mod.EMPTY_SKIP_DEFAULT) -> Context:
    """Опорные кадры по кропу и сдвиги образца (advanced) — как pipeline.prepare с включённым repositioning, но без
    оси (ось задаётся в каждом запросе превью) и с углами checkpoint-ов для /repositioning. Затем оценка смещения
    образца во время съёмки (reconengine.motion, по наклону авто-оси) и режим компенсации motion_mode;
    empty_skip — первые кадры серий empty, не берущиеся в медиану."""
    xp = xp or gpu.get_xp()
    t_start = time.time()
    warnings: List[str] = []
    idx, angles, fnums = pipeline.data_frames(scan)
    if idx.size == 0:
        raise ValueError('{}: в скане нет data-кадров'.format(scan.exp_id))
    progress(0.0, 'dark_empty')
    de_full = preprocess.dark_empty_from_crop(scan, crop, skip_first=empty_skip)
    check_cancel(cancel)
    de = pipeline._normalization_de(scan, de_full, normalization)

    checkpoints = None
    shifts = None
    sy = sx = np.zeros(0)
    if scan.is_advanced and de_full.periodic_empty_fnumbers:
        progress(0.4, 'repositioning')
        cp_angles, sy, sx = preprocess.repositioning_shifts(scan, crop, de_full)
        check_cancel(cancel)
        checks = preprocess.check_checkpoints(scan, crop, de_full)
        sy, sx = pipeline.drop_bad_checkpoint_shifts(sy, sx, checks)
        warnings.extend(pipeline.checks_warnings(checks))
        if np.isnan(sy).any() or np.isnan(sx).any():
            warnings.append('сдвиг образца измерен не на всех checkpoint-ах: неизмеренные приняты за 0')
        shifts = {'sy': [float(v) for v in sy], 'sx': [float(v) for v in sx]}
        checkpoints = dict(shifts, angles=[float(a) for a in cp_angles], checks=checks)
        check_cancel(cancel)
    frame_sy, frame_sx = pipeline._frame_shifts(fnums, de_full.periodic_empty_fnumbers, sy, sx, warnings)
    roi = crop.roi
    # ось-заглушка (центр кропа, без наклона): Prepared требует ось, превью подставляет ось запроса
    ax0 = axis_mod.from_crop_params(0.0, 0.0, roi, method='placeholder')
    prep = pipeline.Prepared(idx=idx, angles=angles, fnums=fnums, de=de, frame_sy=frame_sy, frame_sx=frame_sx,
                             shifts=shifts, axis=ax0, shift_x=0.0, alfa=0.0, warnings=warnings)
    ctx = Context(scan, crop, prep, pixel_size_mm, checkpoints, xp=xp)
    ctx.empty_skip = int(empty_skip)
    progress(0.7, 'motion')
    est = None
    try:
        tilt = ctx.auto_axis_value().tilt_deg      # ось сессии не задаётся: её находит axis/auto (из кэша)
        check_cancel(cancel)
        profiles, _ = motion.band_profiles(crop, idx, fnums, de, frame_sy, frame_sx, tilt, xp=xp, cancel=cancel)
        b = motion.effective_bin(crop.frames.shape[1], crop.frames.shape[2], motion.DEFAULTS['bin'])
        est = motion.estimate(profiles, angles, fnums, de_full.periodic_empty_fnumbers, bin_used=b)
    except Cancelled:
        raise
    except Exception as exc:  # noqa: BLE001 — без оценки смещения студия работает как раньше
        logger.exception('оценка смещения образца не удалась')
        ctx.warnings.append('смещение образца не оценено: {}: {}'.format(type(exc).__name__, exc))
    ctx.init_motion(est, motion_mode)
    ctx.prepare_s = round(time.time() - t_start, 3)
    progress(1.0, 'ready')
    return ctx


def estimate(r: recipe_mod.Recipe, scan: ScanInfo, ctx: Optional[Context] = None,
             rate: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """pipeline.estimate + оценка времени. rate — скорость последних выполненных задач (``JobService.recent_rate``):
    секунд на срез · ширина² · угол и секунд подготовки; без неё — по замерам превью (``seconds_per_slice`` с
    ореолом сглаживания рецепта — ``ring_rows_factor``, масштабированных на ширину рецепта как w²) и подготовки
    сессии (``prepare_s`` — задача повторяет опорные кадры и сдвиги образца); без того и другого — time: None."""
    est = pipeline.estimate(r, scan)
    n_slices = int(est['volume_shape'][0])
    w = r.fov.width
    if rate:
        sps = rate['recon_s_per_slice_px2_angle'] * w * w * est['n_angles_used']
        est['time'] = {'s_per_slice': sps, 'n_slices': n_slices, 'recon_s': sps * n_slices,
                       'prepare_s': rate.get('prepare_s'), 'source': 'jobs', 'jobs': rate.get('jobs')}
        return est
    sps = ctx.seconds_per_slice(est.get('ring_rows_factor', 1.0)) if ctx is not None else None
    if sps is None:
        est['time'] = None
        return est
    scale = (w / float(ctx.roi.width)) ** 2
    est['time'] = {'s_per_slice': sps * scale, 'n_slices': n_slices, 'recon_s': sps * scale * n_slices,
                   'prepare_s': ctx.prepare_s, 'source': 'preview'}
    return est
