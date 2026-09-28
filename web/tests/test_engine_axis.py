"""reconengine.axis: геометрия оси (детектор ↔ кроп), выравнивание слоя, авто-ось, перебор центра."""
import functools
import math

import numpy as np
import pytest

import engine_phantom as ph
import tomotools4 as t4
from reconengine import axis as ax, fbp, gpu, preprocess as pp
from reconengine.model import Axis, ROI

H, W = 96, 128
TRUE_AXIS = Axis(center_x=61.3, y_ref=47.5, tilt_deg=0.7)
ANGLES = np.arange(0, 180, 2.0)
# оба ROI целиком содержат объект (x ≈ 27..96, y ≈ 18..77), ось в них не по центру
ROIS = (ROI(18, 108, 12, 84), ROI(10, 122, 14, 90))
CPU_FBP = functools.partial(fbp.recon_slice, backend='cpu')


@pytest.fixture(autouse=True)
def _cpu_backend(monkeypatch):
    monkeypatch.setenv('RECON_ENGINE_CPU', '1')
    gpu.reset_backend()
    yield
    gpu.reset_backend()


@pytest.fixture(scope='module')
def projections():
    """Проекции (без dark/empty) несимметричного объекта с известной наклонной осью; углы ANGLES + 180°."""
    angles = np.concatenate([ANGLES, [180.0]])
    p = ph.project(ph.asymmetric_blobs(), angles, H, W, TRUE_AXIS.center_x, TRUE_AXIS.y_ref, TRUE_AXIS.tilt_deg)
    return p.astype('float32'), angles


def _crop(a, roi):
    return a[..., roi.y0:roi.y1, roi.x0:roi.x1]


# --- геометрия ---------------------------------------------------------------------------------------------

def test_transform_image_directions_match_tomotools():
    """Сдвиг +s — вправо; поворот +alfa — против часовой на экране (точка справа от центра уходит вверх)."""
    im = np.zeros((21, 31), dtype='float32')
    im[10, 25] = 1.0
    moved = gpu.to_numpy(ax.transform_image(im, 2.0, 0.0))
    assert np.unravel_index(np.argmax(moved), moved.shape) == (10, 27)
    rot = gpu.to_numpy(ax.transform_image(im, 0.0, 10.0))
    assert np.unravel_index(np.argmax(rot), rot.shape) == (8, 25)   # 10·sin 10° ≈ 1,7 строки вверх
    rng = np.random.default_rng(0)
    img = rng.random((24, 40)).astype('float32')
    np.testing.assert_array_equal(gpu.to_numpy(ax.transform_image(img, 1.3, -0.8)), t4.transform_image(img, 1.3, -0.8))


@pytest.mark.parametrize('axis', [Axis(100.0, 50.0, 0.0), Axis(812.25, 1700.0, 0.35), Axis(40.0, 10.0, -1.2)])
@pytest.mark.parametrize('roi', [ROI(0, 200, 0, 100), ROI(37, 181, 23, 90), ROI(700, 1100, 1500, 1901)])
def test_crop_params_roundtrip(axis, roi):
    shift_x, alfa = ax.to_crop_params(axis, roi)
    back = ax.from_crop_params(shift_x, alfa, roi, method='manual')
    assert back.method == 'manual'
    assert back.y_ref == pytest.approx(roi.y0 + (roi.height - 1) / 2)
    assert back.tilt_deg == pytest.approx(axis.tilt_deg)
    for y in (0.0, roi.y0, roi.y1, 3000.0):
        assert back.center_at(y) == pytest.approx(axis.center_at(y), abs=1e-9)
    assert ax.to_crop_params(back, roi) == pytest.approx((shift_x, alfa), abs=1e-12)


def test_crop_params_formula():
    roi = ROI(40, 140, 0, 100)
    assert ax.to_crop_params(Axis(100.0, 13.0, 0.0), roi) == pytest.approx((40 + 49.5 - 100.0, 0.0))
    # наклон: tilt = −alfa, центр берётся на строке центра кропа
    axis = Axis(100.0, 0.0, 1.0)
    s, a = ax.to_crop_params(axis, roi)
    assert a == pytest.approx(-1.0)
    assert s == pytest.approx(40 + 49.5 - (100.0 + math.tan(math.radians(1.0)) * 49.5))


def test_margin_rows():
    assert ax.margin_rows(0.0, 100) == 8
    assert ax.margin_rows(1.0, 2000) == math.ceil(1000 * math.tan(math.radians(1.0))) + 8
    assert ax.margin_rows(-1.0, 2000, extra=0) == math.ceil(1000 * math.tan(math.radians(1.0)))


# --- выравнивание слоя -------------------------------------------------------------------------------------

@pytest.mark.parametrize('rows', [(0, 12), (30, 45), (60, 72)])
def test_align_rows_matches_transform_image(projections, rows):
    p, _ = projections
    roi = ROIS[0]
    frames = _crop(p[:4], roi)
    shift_x, alfa = ax.to_crop_params(TRUE_AXIS, roi)
    m = ax.margin_rows(alfa, roi.width)
    i0, i1 = max(0, rows[0] - m), min(roi.height, rows[1] + m)
    out = ax.align_rows(frames[:, i0:i1], i0, rows, shift_x, alfa, roi.height)
    assert out.shape == (4, rows[1] - rows[0], roi.width) and out.dtype == np.float32
    for k in range(4):
        ref = t4.transform_image(frames[k], shift_x, alfa)[rows[0]:rows[1]]
        assert np.abs(out[k] - ref).max() < 1e-5 * np.abs(ref).max()


def test_align_rows_zero_angle_is_shift_only(projections):
    p, _ = projections
    frames = _crop(p[:2], ROIS[1])
    out = ax.align_rows(frames[:, 10:50], 10, (20, 40), 2.5, 0.0, ROIS[1].height)
    for k in range(2):
        ref = t4.transform_image(frames[k], 2.5, 0.0)[20:40]
        assert np.abs(out[k] - ref).max() < 1e-6


def test_align_rows_interpolates_frame_by_frame(projections, monkeypatch):
    """Интерполяции получают отдельные кадры слоя (s_in, w), а не склеенный (n·s_in, w): у cupyx.spline_filter1d
    блок потоков растёт с длиной оси, и при длине > 32768 (сотни кадров × десятки строк) запуск ядра падает."""
    import scipy.ndimage as ndi
    shapes = []

    class Spy:
        def __getattr__(self, name):
            fn = getattr(ndi, name)

            def wrapper(a, *args, **kwargs):
                shapes.append(np.shape(a))
                return fn(a, *args, **kwargs)
            return wrapper

    monkeypatch.setattr(ax, '_backend', lambda xp: (np, Spy()))
    p, _ = projections
    roi = ROIS[0]
    frames = _crop(p[:3], roi)
    shift_x, alfa = ax.to_crop_params(TRUE_AXIS, roi)
    ax.align_rows(frames, 0, (10, 20), shift_x, alfa, roi.height)
    assert shapes and all(s == (roi.height, roi.width) for s in shapes)


def test_align_rows_requires_margin(projections):
    p, _ = projections
    roi = ROIS[0]
    frames = _crop(p[:2], roi)
    shift_x, alfa = ax.to_crop_params(TRUE_AXIS, roi)
    m = ax.margin_rows(alfa, roi.width)
    with pytest.raises(ValueError, match='запас'):
        ax.align_rows(frames[:, 30 - m + 1:50 + m], 30 - m + 1, (30, 50), shift_x, alfa, roi.height)
    with pytest.raises(ValueError, match='запас'):
        ax.align_rows(frames[:, 30 - m:50 + m - 1], 30 - m, (30, 50), shift_x, alfa, roi.height)
    with pytest.raises(ValueError):
        ax.align_rows(frames, 0, (60, roi.height + 1), shift_x, alfa, roi.height)
    with pytest.raises(ValueError):
        ax.align_rows(frames[0], 0, (10, 20), shift_x, alfa, roi.height)
    # у настоящего края кропа запас не нужен
    ax.align_rows(frames[:, :20 + m], 0, (0, 20), shift_x, alfa, roi.height)


# --- инвариантность оси к ROI ------------------------------------------------------------------------------

@pytest.mark.parametrize('roi', ROIS)
def test_same_axis_centers_any_roi(projections, roi):
    """Одна и та же Axis для разных ROI: после выравнивания кадры 0° и 180° зеркальны относительно центра кропа,
    а FBP строки резче всего при центре (w−1)/2 и совпадает с истинным срезом."""
    p, angles = projections
    crop = _crop(p, roi)
    shift_x, alfa = ax.to_crop_params(TRUE_AXIS, roi)
    i0, i180 = 0, len(angles) - 1

    good = ax.diff_view(crop[i0], crop[i180], shift_x, alfa)
    bad = ax.diff_view(crop[i0], crop[i180], shift_x + 1.0, alfa)
    inner = (slice(6, -6), slice(6, -6))
    assert np.sqrt((good[inner] ** 2).mean()) < 0.01 * np.sqrt((bad[inner] ** 2).mean())

    row = roi.height // 2
    m = ax.margin_rows(alfa, roi.width)
    slab = ax.align_rows(crop[:len(ANGLES), row - m:row + 1 + m], row - m, (row, row + 1), shift_x, alfa,
                         roi.height)
    sino = slab[:, 0, :]
    cc = (roi.width - 1) / 2
    centers = cc + np.arange(-2.0, 2.01, 0.25)
    slices, metrics = ax.center_scan(sino, ANGLES, centers, cc, 1.0, CPU_FBP)
    assert centers[np.argmin(metrics)] == pytest.approx(cc, abs=0.25)

    # строка row выровненного кропа — срез на высоте zeta истинного объекта
    cy = (roi.height - 1) / 2
    y_det = roi.y0 + cy + math.cos(math.radians(alfa)) * (row - cy)
    zeta = (y_det - TRUE_AXIS.y_ref) / math.cos(math.radians(TRUE_AXIS.tilt_deg))
    truth = ph.slice_truth(ph.asymmetric_blobs(), roi.width, zeta)
    rec = slices[len(centers) // 2]
    assert np.corrcoef(rec.ravel(), truth.ravel())[0, 1] > 0.97


# --- авто-ось ----------------------------------------------------------------------------------------------

def test_auto_axis_finds_known_axis_on_normalized_frames():
    """Полный путь: отсчёты с шумом → dark/empty → normalize_slab → auto_axis по 0° и 180°.

    Точность наклона определяется высотой объекта в кропе и шумом (шум через интерполяцию смещает минимум),
    поэтому объект здесь вытянут почти на всю высоту кропа."""
    axis = Axis(center_x=61.3, y_ref=63.5, tilt_deg=0.7)
    blobs = ph.make_blobs(seed=1, n=16, r_max=28, z_range=(-45, 45), sigma_range=(2.5, 4.0))
    ss = ph.make_synthetic_scan([0.0, 180.0], height=128, width=W, center_x=axis.center_x, y_ref=axis.y_ref,
                                tilt_deg=axis.tilt_deg, blobs=blobs, noise=0.3, seed=4)
    roi = ROI(20, 116, 8, 120)          # ось не в центре кропа (центр кропа — 67,5)
    crop = ph.make_crop(ss.frames, roi)
    de = pp.dark_empty_from_crop(ss.scan, crop)
    idx = ss.scan.data_idx
    norm = pp.normalize_slab(crop.frames[idx], ss.scan.frame_numbers[idx], de, xp=np)
    found = ax.auto_axis(norm[0], norm[1], roi)
    assert found.method == 'auto'
    assert found.center_at(axis.y_ref) == pytest.approx(axis.center_x, abs=0.25)
    assert found.tilt_deg == pytest.approx(axis.tilt_deg, abs=0.02)


def test_auto_axis_matches_find_axis_correction(projections):
    """Без сглаживания на тех же кадрах — те же (shift, alfa), что у tomotools4.find_axis_correction."""
    p, angles = projections
    roi = ROIS[1]
    crop = _crop(p, roi)
    s_old, a_old = t4.find_axis_correction(np.stack([crop[0], crop[-1]]), np.array([0.0, 180.0]))
    found = ax.auto_axis(crop[0], crop[-1], roi, smooth_sigma=0)
    s_new, a_new = ax.to_crop_params(found, roi)
    # целевая отдаётся Powell как float (в старом коде — float32), траектории чуть расходятся
    assert s_new == pytest.approx(s_old, abs=1e-4)
    assert a_new == pytest.approx(a_old, abs=1e-4)
    assert found.center_at(TRUE_AXIS.y_ref) == pytest.approx(TRUE_AXIS.center_x, abs=0.01)
    assert found.tilt_deg == pytest.approx(TRUE_AXIS.tilt_deg, abs=0.005)


def test_auto_axis_rejects_wrong_shape(projections):
    p, _ = projections
    with pytest.raises(ValueError):
        ax.auto_axis(p[0], p[-1], ROIS[0])


# --- перебор центра, метрики, наклон -----------------------------------------------------------------------

@pytest.mark.parametrize('metric', ['entropy', 'tv'])
def test_center_scan_minimum_at_true_center(metric):
    w = 96
    true_c = 46.3
    sino = ph.sinogram(ph.asymmetric_blobs(), ANGLES, w, zeta=2.0, center=true_c)
    step = 0.25
    centers = np.arange(true_c - 3, true_c + 3.01, step)
    slices, metrics = ax.center_scan(sino, ANGLES, centers, (w - 1) / 2, 1.0, CPU_FBP, metric=metric)
    assert len(slices) == len(centers) and slices[0].shape == (w, w)
    assert centers[np.argmin(metrics)] == pytest.approx(true_c, abs=step)


def test_center_metric_edge_cases():
    assert ax.center_metric(np.ones((16, 16))) == 0.0
    with pytest.raises(ValueError):
        ax.center_metric(np.ones((16, 16)), kind='sharpness')
    with pytest.raises(ValueError):
        ax.center_scan(np.ones((4, 16)), np.arange(4.0), [7.5], 7.5, 1.0, CPU_FBP, metric='sharpness')


def test_tilt_from_centers():
    axis = ax.tilt_from_centers(100.0, 50.0, 300.0, 52.0)
    assert axis.method == 'tilt'
    assert axis.y_ref == 200.0 and axis.center_x == 51.0
    assert axis.tilt_deg == pytest.approx(math.degrees(math.atan(2.0 / 200.0)))
    assert axis.center_at(100.0) == pytest.approx(50.0)
    assert axis.center_at(300.0) == pytest.approx(52.0)
    # порядок строк не важен
    swapped = ax.tilt_from_centers(300.0, 52.0, 100.0, 50.0)
    assert swapped.tilt_deg == pytest.approx(axis.tilt_deg)
    # согласовано с Axis: восстанавливает исходную ось
    back = ax.tilt_from_centers(10.0, TRUE_AXIS.center_at(10.0), 90.0, TRUE_AXIS.center_at(90.0))
    assert back.tilt_deg == pytest.approx(TRUE_AXIS.tilt_deg)
    assert back.center_at(TRUE_AXIS.y_ref) == pytest.approx(TRUE_AXIS.center_x)
    with pytest.raises(ValueError):
        ax.tilt_from_centers(10.0, 1.0, 10.0, 2.0)


def test_diff_view_matches_notebook(projections):
    p, _ = projections
    crop = _crop(p, ROIS[0])
    dv = ax.diff_view(crop[0], crop[-1], 3.0, 0.4)
    ref = t4.transform_image(crop[0], 3.0, 0.4) - np.fliplr(t4.transform_image(crop[-1], 3.0, 0.4))
    assert isinstance(dv, np.ndarray)
    np.testing.assert_allclose(dv, ref, atol=1e-7)


def test_auto_axis_smoothing_is_not_worse_on_noisy_frames():
    """Сглаживание перед сравнением 0°/180° не ухудшает точность на зашумлённых кадрах с известной осью."""
    axis = Axis(center_x=61.3, y_ref=63.5, tilt_deg=0.7)
    errs = {0: [], ax.AUTO_AXIS_SMOOTH_SIGMA: []}
    for seed in range(4):
        blobs = ph.make_blobs(seed=seed + 10, n=16, r_max=28, z_range=(-45, 45), sigma_range=(2.5, 4.0))
        ss = ph.make_synthetic_scan([0.0, 180.0], height=128, width=W, center_x=axis.center_x, y_ref=axis.y_ref,
                                    tilt_deg=axis.tilt_deg, blobs=blobs, noise=0.5, seed=seed)
        roi = ROI(20, 116, 8, 120)
        crop = ph.make_crop(ss.frames, roi)
        de = pp.dark_empty_from_crop(ss.scan, crop)
        idx = ss.scan.data_idx
        norm = pp.normalize_slab(crop.frames[idx], ss.scan.frame_numbers[idx], de, xp=np)
        for sigma in errs:
            found = ax.auto_axis(norm[0], norm[1], roi, smooth_sigma=sigma)
            errs[sigma].append(abs(found.center_at(axis.y_ref) - axis.center_x))
    assert max(errs[ax.AUTO_AXIS_SMOOTH_SIGMA]) < 0.25
    assert np.mean(errs[ax.AUTO_AXIS_SMOOTH_SIGMA]) <= np.mean(errs[0]) + 0.02
