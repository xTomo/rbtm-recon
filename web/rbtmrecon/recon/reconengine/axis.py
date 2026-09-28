"""Ось вращения.

В рецепте ось хранится в координатах детектора (``model.Axis``: столбец ``center_x`` на строке ``y_ref``,
наклон ``tilt_deg``), чтобы правка ROI её не портила. Для кропа ось переводится в параметры
``(shift_x, alfa)`` старой функции ``tomotools4.transform_image`` (сдвиг по x, затем поворот вокруг центра
кропа ``((h−1)/2, (w−1)/2)``, order=3, mode='nearest'): после преобразования ось вертикальна и проходит через
столбец ``(w−1)/2`` кропа — на этом держится и старый ноутбук, и FBP без смещения центра.

Геометрия (кроп h×w, центр (cy, cx) = ((h−1)/2, (w−1)/2), координаты массива: строка y вниз, столбец x вправо):

- ``shift(im, [0, shift_x])``: S(y, x) = im(y, x − shift_x) — содержимое уезжает вправо на shift_x;
- ``rotate(S, alfa, reshape=False)`` (scipy/cupyx, axes=(1, 0)) — выход o берёт вход в точке
  ``(y, x) = R·(o − c) + c``, ``R = [[cos α, sin α], [−sin α, cos α]]``; на экране (строка 0 сверху) положительный
  alfa поворачивает картинку ПРОТИВ часовой стрелки (точка справа от центра уходит вверх — проверено тестом);
- выходной столбец cx берёт вход на прямой ``x = cx − shift_x − tan α · (y − cy)`` — это и есть ось в кропе.

Отсюда в координатах детектора (кроп начинается в (roi.y0, roi.x0)):

    tilt_deg = −alfa,  center_x (на строке y_ref = roi.y0 + cy) = roi.x0 + cx − shift_x,
    shift_x = roi.x0 + cx − axis.center_at(roi.y0 + cy),  alfa = −axis.tilt_deg.

Выравнивание слоя строк (``align_rows``) — то же преобразование, но только для нужных выходных строк: из-за
наклона выходная строка берёт данные из входных строк в пределах ``margin_rows``. Реализация повторяет два шага
transform_image (сдвиг по x кубическим сплайном, затем поворот ``affine_transform`` с той же матрицей и смещением,
пересчитанным к центру ПОЛНОГО кропа), поэтому совпадает с transform_image на полном кропе до ошибок округления
и влияния границы слоя на префильтр сплайна (затухает как 0,27^d по расстоянию d от края слоя; при запасе
``extra`` = 8 строк — ~1e-5 от перепада яркости).
"""
from __future__ import annotations

import logging
import math
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .model import Axis, ROI

logger = logging.getLogger(__name__)


# --- геометрия ---------------------------------------------------------------------------------------------

def _crop_center(height: int, width: int) -> Tuple[float, float]:
    return (height - 1) / 2.0, (width - 1) / 2.0


def to_crop_params(axis: Axis, roi: ROI) -> Tuple[float, float]:
    """(shift_x, alfa) для transform_image на кропе roi, эквивалентные оси axis."""
    cy, cx = _crop_center(roi.height, roi.width)
    shift_x = roi.x0 + cx - axis.center_at(roi.y0 + cy)
    return float(shift_x), float(-axis.tilt_deg)


def from_crop_params(shift_x: float, alfa: float, roi: ROI, method: str = 'auto') -> Axis:
    """Обратное преобразование: Axis в координатах детектора (y_ref — строка центра кропа)."""
    cy, cx = _crop_center(roi.height, roi.width)
    return Axis(center_x=float(roi.x0 + cx - shift_x), y_ref=float(roi.y0 + cy), tilt_deg=float(-alfa),
                method=method)


def rotation_matrix(alfa: float) -> np.ndarray:
    """Матрица ``scipy.ndimage.rotate`` (axes=(1, 0)) для угла alfa в градусах: вход = R·(выход − c) + c."""
    from scipy import special  # noqa: WPS433 — cosdg/sindg точны на кратных 90°, как в scipy.rotate
    c, s = float(special.cosdg(alfa)), float(special.sindg(alfa))
    return np.array([[c, s], [-s, c]])


#: До стольких выходных строк align_rows выравнивает слой одним вызовом (превью), больше — покадрово (задача).
BATCH_MAX_OUT_ROWS = 8


def margin_rows(alfa: float, width: int, extra: int = 8) -> int:
    """Запас входных строк сверху и снизу для align_rows при повороте на alfa: ceil(width/2·|tan alfa|) + extra."""
    return int(math.ceil(width / 2.0 * abs(math.tan(math.radians(alfa))))) + int(extra)


# --- преобразования изображений ----------------------------------------------------------------------------

def _backend(xp):
    from .gpu import get_xp, ndimage  # noqa: WPS433
    xp = xp or get_xp()
    return xp, ndimage(xp)


def _as_float(a, xp):
    a = xp.asarray(a)
    if a.dtype.kind != 'f':
        a = a.astype(xp.float32)
    return a


def transform_image(im, shift_x: float, alfa: float, xp=None):
    """Порт ``tomotools4.transform_image`` на xp: сдвиг по x на shift_x (order=3, 'nearest'), затем поворот на
    alfa градусов вокруг центра массива (order=3, reshape=False, 'nearest'). Возвращает xp-массив."""
    xp, nd = _backend(xp)
    a = _as_float(im, xp)
    a = nd.shift(a, [0, shift_x], order=3, mode='nearest')
    return nd.rotate(a, alfa, order=3, reshape=False, mode='nearest')


def align_rows(frames, in_row0: int, out_rows: Tuple[int, int], shift_x: float, alfa: float,
               crop_height: int, xp=None):
    """Выровнять слой: frames (n, s_in, w) — строки кропа [in_row0, in_row0 + s_in); вернуть (n, s_out, w) для
    строк кропа out_rows = [r0, r1), совпадающих с transform_image на полном кропе высоты crop_height
    (в пределах интерполяции). Требует, чтобы вход покрывал out_rows ± margin_rows (иначе ValueError).

    Два шага, как в transform_image: сдвиг по x, затем ``affine_transform`` с матрицей поворота и смещением
    относительно центра полного кропа ((crop_height−1)/2, (w−1)/2). При alfa = 0 поворот пропускается.

    До ``BATCH_MAX_OUT_ROWS`` выходных строк (превью) оба шага — одним вызовом на слой (n, s, w): по оси кадров
    преобразование тождественно (координаты целые, а сплайн, построенный префильтром и по этой оси, в целых узлах
    воспроизводит данные точно) — результат тот же, что покадрово, без n запусков ядер (на GPU покадровый цикл по
    360 кадрам — 0,7 с на строку превью, батч — 1,5 мс). Для слоёв задачи — покадрово (батч там медленнее, см.
    код). Склеивать кадры в (n·s, w) нельзя: cupyx.spline_filter1d выбирает блок 2^ceil(log2(длина оси / 32))
    потоков, и при длине оси > 32768 запуск ядра падает (CUDA_ERROR_INVALID_VALUE)."""
    xp, nd = _backend(xp)
    frames = _as_float(frames, xp)
    if frames.ndim != 3:
        raise ValueError('ожидается слой кадров (n, s, w), получено {}'.format(frames.shape))
    n, s_in, w = frames.shape
    r0, r1 = int(out_rows[0]), int(out_rows[1])
    in_row0 = int(in_row0)
    if not (0 <= r0 < r1 <= crop_height):
        raise ValueError('выходные строки [{}, {}) вне кропа высотой {}'.format(r0, r1, crop_height))
    if in_row0 < 0 or in_row0 + s_in > crop_height:
        raise ValueError('входные строки [{}, {}) вне кропа высотой {}'.format(in_row0, in_row0 + s_in,
                                                                              crop_height))
    m = margin_rows(alfa, w)
    need0, need1 = max(0, r0 - m), min(crop_height, r1 + m)
    if in_row0 > need0 or in_row0 + s_in < need1:
        raise ValueError('вход [{}, {}) не покрывает строки [{}, {}) с запасом {} (нужно [{}, {}))'.format(
            in_row0, in_row0 + s_in, r0, r1, m, need0, need1))

    rot = rotation_matrix(alfa)
    c = np.array(_crop_center(crop_height, w))
    offset = rot @ (np.array([r0, 0.0]) - c) + c - np.array([in_row0, 0.0])
    if r1 - r0 <= BATCH_MAX_OUT_ROWS:
        # несколько выходных строк (превью): один вызов на слой вместо n запусков ядер
        shifted = nd.shift(frames, [0, 0, shift_x], order=3, mode='nearest')
        if alfa == 0:
            return xp.ascontiguousarray(shifted[:, r0 - in_row0:r1 - in_row0, :])
        matrix = np.eye(3)
        matrix[1:, 1:] = rot
        out = nd.affine_transform(shifted, xp.asarray(matrix), offset=[0.0, float(offset[0]), float(offset[1])],
                                  output_shape=(n, r1 - r0, w), order=3, mode='nearest')
        return out.astype(frames.dtype, copy=False)
    # слои задачи: покадрово — префильтр сплайна по оси кадров (шаг по памяти s·w) и лишние проходы по всему
    # слою делали батч на GPU в 2,4 раза медленнее (замер: 360 кадров × 257 строк × 3216)
    out = xp.empty((n, r1 - r0, w), dtype=frames.dtype)
    matrix = xp.asarray(rot)
    for i in range(n):
        s = nd.shift(frames[i], [0, shift_x], order=3, mode='nearest')
        if alfa == 0:
            out[i] = s[r0 - in_row0:r1 - in_row0]
        else:
            out[i] = nd.affine_transform(s, matrix, offset=[float(offset[0]), float(offset[1])],
                                         output_shape=(r1 - r0, w), order=3, mode='nearest')
    return out


# --- авто-ось ----------------------------------------------------------------------------------------------

AUTO_AXIS_SMOOTH_SIGMA = 1.5


def auto_axis(img0: np.ndarray, img180: np.ndarray, roi: ROI,
              smooth_sigma: float = AUTO_AXIS_SMOOTH_SIGMA) -> Axis:
    """Авто-ось по нормированным кадрам кропа при ~0° и ~180° (img180 НЕ отражён): поиск (shift, alfa)
    как ``tomotools4.find_axis_correction`` (Powell от начального приближения по X центра масс, целевая —
    ‖T(im0, s, a) − T(flip(im180), −s, −a)‖²), результат переводится в Axis.

    Два отличия от старого кода, оба из-за шума реальных кадров:
    - кадры перед сравнением сглаживаются гауссом ``smooth_sigma`` px (0 — без сглаживания). Иначе целевую
      определяет то, как интерполяция дробного сдвига сглаживает шум: на однородном образце она изрезана с
      периодом меньше пикселя, и Powell останавливается в случайном локальном минимуме (на реальном скане
      изрезанность падает в ~17 раз, ответ перестаёт зависеть от старта);
    - сумма квадратов считается в float64: у старой float32-суммы Powell останавливался раньше минимума.
    Кадры нормируются на L2-норму. Преобразования считаются на выбранном бэкенде (cupy на GPU)."""
    import scipy.ndimage as ndi  # noqa: WPS433
    import scipy.optimize as optimize  # noqa: WPS433

    from .gpu import to_numpy  # noqa: WPS433

    a0 = np.asarray(to_numpy(img0), dtype='float32')
    a1 = np.fliplr(np.asarray(to_numpy(img180), dtype='float32'))
    if a0.shape != a1.shape or a0.shape != (roi.height, roi.width):
        raise ValueError('кадры {} и {} не совпадают с кропом {}×{}'.format(
            a0.shape, a1.shape, roi.height, roi.width))
    if smooth_sigma and smooth_sigma > 0:
        a0 = ndi.gaussian_filter(a0, smooth_sigma, mode='nearest')
        a1 = ndi.gaussian_filter(a1, smooth_sigma, mode='nearest')
    im0 = a0 / (a0.astype('float64') ** 2).sum() ** 0.5
    im1 = a1 / (a1.astype('float64') ** 2).sum() ** 0.5
    cm0 = ndi.center_of_mass(im0)
    cm1 = ndi.center_of_mass(im1)
    initial_shift = (float(cm1[1]) - float(cm0[1])) / 2

    xp, _ = _backend(None)
    g0, g1 = xp.asarray(im0, dtype=xp.float32), xp.asarray(im1, dtype=xp.float32)

    def _objective(shift_angle):
        s, a = shift_angle
        diff = transform_image(g0, s, a, xp) - transform_image(g1, -s, -a, xp)
        return float(xp.sum(diff * diff, dtype=xp.float64))

    result = optimize.minimize(_objective, np.array([initial_shift, 0.0]), method='Powell')
    shift_x, alfa = (float(v) for v in result.x)
    logger.info('auto_axis: shift_x=%.3f alfa=%.4f (оценок целевой: %s)', shift_x, alfa, result.nfev)
    return from_crop_params(shift_x, alfa, roi, method='auto')


# --- перебор центра ----------------------------------------------------------------------------------------

METRICS = ('grad', 'entropy', 'tv')


def _circle_mask(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = (min(h, w) - 1) / 2.0
    yy, xx = np.ogrid[0:h, 0:w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def fourier_shift_rows(a, d: float, xp=None):
    """Сдвиг вдоль последней оси на d пикселей в частотной области: содержимое уезжает вправо при d > 0, как
    ``ndimage.shift(a, [..., d])``. В отличие от сплайна, не сглаживает шум по-разному при разных дробных d: при
    переборе центра сплайн-сдвиг на полпикселя делал срез глаже, и метрики (и глаз) выбирали полуцелые центры
    (на реальном скане — ложный минимум в 1,5 px от оси). Края продолжаются крайними значениями (дополнение до
    степени двойки ≥ 2w поровну с обеих сторон), так что сдвиг на несколько пикселей не заворачивает строку."""
    from .gpu import get_xp  # noqa: WPS433
    xp = xp or get_xp()
    a = xp.asarray(a, dtype=xp.float32)
    if d == 0:
        return a.copy()
    w = a.shape[-1]
    size = 1 << int(math.ceil(math.log2(2 * w)))
    left = (size - w) // 2
    pad = [(0, 0)] * (a.ndim - 1) + [(left, size - w - left)]
    f = xp.fft.rfft(xp.pad(a, pad, mode='edge'), axis=-1)
    k = xp.fft.rfftfreq(size)
    f *= xp.exp(-2j * math.pi * float(d) * k)
    return xp.fft.irfft(f, n=size, axis=-1)[..., left:left + w].astype(xp.float32)


def center_metric(slice_img: np.ndarray, kind: str = 'grad',
                  value_range: Optional[Tuple[float, float]] = None) -> float:
    """Метрика качества среза внутри вписанного круга, меньше = лучше.

    'grad' — минус средняя энергия градиента среза, сглаженного гауссом σ=1,5 (резче края — больше энергия):
    сглаживание убирает вклад шума, иначе на однородном шумном образце метрику определяет шум, а не геометрия.
    Фрагмент должен содержать края (внутри однородного образца сведений о центре нет).
    'entropy' — энтропия гистограммы (256 бинов) значений в диапазоне value_range (по умолчанию — процентили
    0,1…99,9 самого среза; значения за пределами прижимаются к краям). При переборе центра диапазон должен быть
    общим для всех срезов (см. center_scan), иначе сравнение нечестно.
    'tv' — полная вариация (сумма модуля градиента): неверный центр добавляет дуги и двоения краёв, поэтому TV
    минимальна при верном центре (проверено на синтетике) — возвращается как есть."""
    img = np.asarray(slice_img, dtype='float64')
    mask = _circle_mask(img)
    if kind == 'grad':
        import scipy.ndimage as ndi  # noqa: WPS433
        gy, gx = np.gradient(ndi.gaussian_filter(img, 1.5))
        return -float((gx * gx + gy * gy)[mask].mean())
    if kind == 'entropy':
        v = img[mask]
        lo, hi = value_range if value_range is not None else np.percentile(v, [0.1, 99.9])
        lo, hi = float(lo), float(hi)
        if not hi > lo:
            return 0.0
        hist, _ = np.histogram(np.clip(v, lo, hi), bins=256, range=(lo, hi))
        p = hist[hist > 0] / float(hist.sum())
        return float(-(p * np.log(p)).sum())
    if kind == 'tv':
        gy, gx = np.gradient(img)
        return float(np.sqrt(gx ** 2 + gy ** 2)[mask].sum())
    raise ValueError('неизвестная метрика: {} (допустимы {})'.format(kind, ', '.join(METRICS)))


def center_scan(sino_row: np.ndarray, angles_deg: np.ndarray, centers: Sequence[float], crop_center: float,
                pixel_size: float, recon_fn: Callable, metric: str = 'grad'
                ) -> Tuple[List[np.ndarray], np.ndarray]:
    """Перебор центра для одной строки. sino_row (n, w) — строка, уже выровненная по текущей оси (ось в
    crop_center = (w−1)/2); для центра c строка сдвигается на (crop_center − c) по x (``fourier_shift_rows`` —
    без разного сглаживания шума при разных дробных сдвигах) и восстанавливается
    recon_fn(sino, angles, pixel_size) → (w, w). Возвращает (срезы, метрики).

    Для 'entropy' диапазон гистограммы общий для всех срезов (процентили 0,1…99,9 объединённых значений)."""
    from .gpu import to_numpy  # noqa: WPS433

    if metric not in METRICS:
        raise ValueError('неизвестная метрика: {} (допустимы {})'.format(metric, ', '.join(METRICS)))
    sino = np.asarray(to_numpy(sino_row), dtype='float32')
    slices = []
    for c in centers:
        d = float(crop_center) - float(c)
        s = np.asarray(fourier_shift_rows(sino, d, np))
        slices.append(np.asarray(to_numpy(recon_fn(s, angles_deg, pixel_size)), dtype='float32'))
    value_range = None
    if metric == 'entropy' and slices:
        mask = _circle_mask(slices[0])
        pooled = np.concatenate([s[mask] for s in slices])
        value_range = tuple(np.percentile(pooled, [0.1, 99.9]))
    metrics = np.array([center_metric(s, metric, value_range) for s in slices], dtype='float64')
    return slices, metrics


def tilt_from_centers(y_top: float, c_top: float, y_bottom: float, c_bottom: float, method: str = 'tilt') -> Axis:
    """Ось по центрам на двух строках детектора: наклон tan(tilt) = (c_bottom − c_top) / (y_bottom − y_top),
    y_ref — середина между строками, center_x — середина между центрами."""
    dy = float(y_bottom) - float(y_top)
    if dy == 0:
        raise ValueError('строки для наклона совпадают: {}'.format(y_top))
    tilt = math.degrees(math.atan((float(c_bottom) - float(c_top)) / dy))
    return Axis(center_x=(float(c_top) + float(c_bottom)) / 2.0, y_ref=(float(y_top) + float(y_bottom)) / 2.0,
                tilt_deg=tilt, method=method)


def diff_view(img0: np.ndarray, img180: np.ndarray, shift_x: float, alfa: float) -> np.ndarray:
    """Вспомогательный вид: T(img0, s, a) − flip(T(img180, s, a)) (как «Показать совмещение» в ноутбуке)."""
    from .gpu import to_numpy  # noqa: WPS433
    xp, _ = _backend(None)
    t0 = transform_image(to_numpy(img0), shift_x, alfa, xp)
    t1 = transform_image(to_numpy(img180), shift_x, alfa, xp)
    return np.asarray(to_numpy(t0 - xp.flip(t1, axis=1)))
