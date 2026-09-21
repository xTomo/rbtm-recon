"""Тесты алгоритмов tomotools4 на синтетических данных (без GPU)."""
import numpy as np
import pytest
import scipy.ndimage as ndi

import tomotools4 as t4

H = W = 64
EMPTY_LEVEL = 3000.0


def make_object(seed=3):
    """Синтетический объект с деталями (для кросс-корреляции)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    obj = np.zeros((H, W), dtype='float32')
    obj += 0.8 * np.exp(-(((yy - 30) ** 2 + (xx - 28) ** 2) / 60.0))
    obj += 0.5 * np.exp(-(((yy - 20) ** 2 + (xx - 40) ** 2) / 20.0))
    obj[10:14, 15:45] += 0.4
    obj[40:44, 20:30] += 0.6
    obj += 0.02 * rng.random((H, W))
    return ndi.gaussian_filter(obj, 1.0).astype('float32')


def to_counts(obj):
    """Переводит оптическую плотность в отсчёты детектора."""
    return (EMPTY_LEVEL * np.exp(-obj)).astype('float32')


def normalize(counts):
    d = np.log(EMPTY_LEVEL) - np.log(counts)
    d[d < 0] = 0
    return d.astype('float32')


def rms(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


def _flat(value=EMPTY_LEVEL):
    return np.full((H, W), value, dtype='float32')


def build_adv(segment_objects, true_shifts, angles_per_segment=(0., 30., 60., 90.),
              first_periodic_fn=100, segment_stride=100):
    """Собирает AdvancedTomoData с K checkpoint-ами.

    segment_objects : список из K+1 изображений объекта (сегмент 0 — референс)
    true_shifts     : список K сдвигов [dy, dx] (сегмент k+1 относительно k)
    """
    n_per = len(angles_per_segment)
    data_images, data_angles, data_numbers = [], [], []
    for seg, obj in enumerate(segment_objects):
        base = seg * segment_stride + 10
        for j, a in enumerate(angles_per_segment):
            data_images.append(to_counts(obj))
            data_angles.append(a)
            data_numbers.append(base + j)

    periodic_fnumbers, dc_images, dc_angles, dc_numbers = [], [], [], []
    for k in range(len(true_shifts)):
        fn = first_periodic_fn + k * segment_stride
        periodic_fnumbers.append(fn)
        # data_check снимается сразу после k-й вставки, при угле последнего
        # data-кадра предыдущего сегмента
        dc_images.append(to_counts(segment_objects[k + 1]))
        dc_angles.append(angles_per_segment[-1])
        dc_numbers.append(fn + 5)

    return t4.AdvancedTomoData(
        dark_image=np.zeros((H, W), 'float32'),
        initial_empty=_flat(),
        initial_empty_fnumber=0,
        periodic_empties=[_flat() for _ in true_shifts],
        periodic_empty_fnumbers=periodic_fnumbers,
        data_images=np.stack(data_images).astype('float32'),
        data_angles=np.array(data_angles, dtype='float32'),
        data_numbers=np.array(data_numbers, dtype='int64'),
        data_check_images=np.stack(dc_images).astype('float32') if dc_images
        else np.empty((0, H, W), 'float32'),
        data_check_angles=np.array(dc_angles, dtype='float32'),
        data_check_numbers=np.array(dc_numbers, dtype='int64'),
        series_length=3,
    ), n_per


# =============================================================================
# 4b — знак сдвига (round-trip)
# =============================================================================

def test_phase_cross_correlation_convention():
    """Контроль соглашения skimage: ndi.shift(moving, s) ≈ reference."""
    from skimage.registration import phase_cross_correlation

    obj = make_object()
    moved = ndi.shift(obj, [1.5, -2.0], order=3, mode='nearest')
    s, _, _ = phase_cross_correlation(obj, moved, upsample_factor=10)
    assert np.allclose(s, [-1.5, 2.0], atol=0.05)
    assert rms(ndi.shift(moved, s, order=3, mode='nearest'), obj) < 0.01


def test_repositioning_roundtrip_reduces_error():
    """measure → apply должен уменьшать расхождение сегмента 1 с референсом."""
    obj = make_object()
    true_shift = [1.5, -2.0]
    obj_moved = ndi.shift(obj, true_shift, order=3, mode='nearest').astype('float32')

    adv, n_per = build_adv([obj, obj_moved], [true_shift])

    angles, sy, sx = t4.measure_repositioning_shifts(adv, 0, W, 0, H, debug=False)
    assert len(sy) == 1
    # Измеренный сдвиг направлен обратно приложенному (соглашение skimage)
    assert sy[0] < 0 and sx[0] > 0

    norm = np.stack([normalize(f) for f in adv.data_images]).astype('float32')
    target = np.clip(obj, 0, None)
    before = rms(norm[n_per], target)

    t4.apply_repositioning_correction(norm, adv.data_numbers, adv, sy, sx)
    after = rms(norm[n_per], target)

    assert after < before, f'коррекция ухудшила совмещение: {before} -> {after}'
    assert after < 0.5 * before
    # Сегмент 0 (референс) не трогаем
    assert rms(norm[0], target) == pytest.approx(0.0, abs=1e-6)


# =============================================================================
# 4f — поиск data-кадра с тем же углом
# =============================================================================

def test_find_matching_data_frame_takes_last_match():
    """При нескольких оборотах берём последний кадр с нужным углом."""
    angles = np.array([0., 90., 180., 270., 0., 90.], dtype='float32')
    numbers = np.array([10, 11, 12, 13, 14, 15], dtype='int64')

    idx = t4._find_matching_data_frame(90.0, angles, numbers,
                                       segment_end_fnumber=16)
    assert idx == 5

    # если ограничить историю первым оборотом — вернётся кадр первого оборота
    idx = t4._find_matching_data_frame(90.0, angles, numbers,
                                       segment_end_fnumber=14)
    assert idx == 1


def test_find_matching_data_frame_wraps_around_360():
    angles = np.array([0., 90., 359.9], dtype='float32')
    numbers = np.array([1, 2, 3], dtype='int64')
    assert t4._find_matching_data_frame(0.0, angles, numbers, 10) == 2


def test_find_matching_data_frame_raises_when_no_angle_in_tolerance():
    angles = np.array([0., 90., 180.], dtype='float32')
    numbers = np.array([1, 2, 3], dtype='int64')
    with pytest.raises(ValueError, match='допуск'):
        t4._find_matching_data_frame(45.0, angles, numbers, 10)


def test_find_matching_data_frame_returns_none_without_candidates():
    angles = np.array([0., 90.], dtype='float32')
    numbers = np.array([100, 101], dtype='int64')
    assert t4._find_matching_data_frame(0.0, angles, numbers, 10) is None


# =============================================================================
# 4a — накопительные сдвиги сегментов
# =============================================================================

def test_apply_repositioning_uses_cumulative_shifts():
    """Сдвиг сегмента m = cumsum(shifts[:m]), а не shifts[m-1]."""
    obj = make_object()
    adv, n_per = build_adv([obj, obj, obj], [[1.0, 0.0], [1.0, 0.0]])

    norm = np.stack([normalize(f) for f in adv.data_images]).astype('float32')
    shifts_y = np.array([2.0, 3.0])
    shifts_x = np.array([-1.0, 0.5])

    t4.apply_repositioning_correction(norm, adv.data_numbers, adv,
                                      shifts_y, shifts_x)

    src = normalize(to_counts(obj))
    expected_seg1 = ndi.shift(src, [2.0, -1.0], order=3, mode='nearest')
    expected_seg2 = ndi.shift(src, [5.0, -0.5], order=3, mode='nearest')

    assert np.allclose(norm[0], src, atol=1e-5)                   # сегмент 0
    assert np.allclose(norm[n_per], expected_seg1, atol=1e-5)     # сегмент 1
    assert np.allclose(norm[2 * n_per], expected_seg2, atol=1e-5)  # сегмент 2


def test_repositioning_roundtrip_two_checkpoints():
    """Два последовательных репозиционирования: коррекция возвращает референс."""
    obj = make_object()
    d1 = [1.5, -2.0]
    d2 = [-1.0, 1.5]
    obj1 = ndi.shift(obj, d1, order=3, mode='nearest').astype('float32')
    obj2 = ndi.shift(obj1, d2, order=3, mode='nearest').astype('float32')

    adv, n_per = build_adv([obj, obj1, obj2], [d1, d2])
    angles, sy, sx = t4.measure_repositioning_shifts(adv, 0, W, 0, H, debug=False)
    assert len(sy) == 2

    norm = np.stack([normalize(f) for f in adv.data_images]).astype('float32')
    target = np.clip(obj, 0, None)
    before_1 = rms(norm[n_per], target)
    before_2 = rms(norm[2 * n_per], target)

    t4.apply_repositioning_correction(norm, adv.data_numbers, adv, sy, sx)

    assert rms(norm[n_per], target) < before_1
    # Ключевая проверка: второй сегмент смещён на d1+d2, одного shifts[1] мало
    assert rms(norm[2 * n_per], target) < before_2


def test_apply_repositioning_no_shifts_is_noop():
    obj = make_object()
    adv, _ = build_adv([obj], [])
    norm = np.stack([normalize(f) for f in adv.data_images]).astype('float32')
    before = norm.copy()
    t4.apply_repositioning_correction(norm, adv.data_numbers, adv,
                                      np.array([]), np.array([]))
    assert np.array_equal(norm, before)


# =============================================================================
# 4c — интерполяция empty между сериями
# =============================================================================

def _adv_for_interpolation(initial_value, initial_fn, periodic_values, periodic_fns):
    def const(v):
        return np.full((4, 5), float(v), dtype='float32')

    return t4.AdvancedTomoData(
        dark_image=np.zeros((4, 5), 'float32'),
        initial_empty=const(initial_value),
        initial_empty_fnumber=initial_fn,
        periodic_empties=[const(v) for v in periodic_values],
        periodic_empty_fnumbers=list(periodic_fns),
        data_images=np.empty((0, 4, 5), 'float32'),
        data_angles=np.empty((0,), 'float32'),
        data_numbers=np.empty((0,), 'int64'),
        data_check_images=np.empty((0, 4, 5), 'float32'),
        data_check_angles=np.empty((0,), 'float32'),
        data_check_numbers=np.empty((0,), 'int64'),
        series_length=2,
    )


@pytest.mark.parametrize('frame_number, expected', [
    (2, 100.0),      # ровно начальная серия
    (0, 100.0),      # до начальной серии — константа
    (50, 149.0),     # линейно между fn=2 (100) и fn=100 (200)
    (100, 200.0),    # ровно первая periodic
    (150, 300.0),    # линейно между fn=100 (200) и fn=200 (400)
    (200, 400.0),    # ровно последняя periodic
    (250, 400.0),    # после последней — константа
])
def test_interpolate_empty(frame_number, expected):
    adv = _adv_for_interpolation(100, 2, [200, 400], [100, 200])
    res = t4._interpolate_empty(adv, frame_number, 0, 5, 0, 4)
    assert res.shape == (4, 5)
    assert res == pytest.approx(np.full((4, 5), expected), abs=0.51)


def test_interpolate_empty_uses_initial_fnumber():
    """Ветка idx == 0 должна интерполировать, а не возвращать initial всегда."""
    adv = _adv_for_interpolation(100, 2, [200], [100])
    near_initial = t4._interpolate_empty(adv, 10, 0, 5, 0, 4)[0, 0]
    near_periodic = t4._interpolate_empty(adv, 90, 0, 5, 0, 4)[0, 0]
    assert 100.0 < near_initial < near_periodic < 200.0


def test_interpolate_empty_without_periodic():
    adv = _adv_for_interpolation(123, 5, [], [])
    for fn in (0, 5, 1000):
        assert t4._interpolate_empty(adv, fn, 0, 5, 0, 4) == pytest.approx(
            np.full((4, 5), 123.0))


def test_interpolate_empty_respects_roi():
    adv = _adv_for_interpolation(100, 2, [200], [100])
    res = t4._interpolate_empty(adv, 50, 1, 4, 0, 2)
    assert res.shape == (2, 3)


# =============================================================================
# 4d — поиск пар 0°/180° с допуском
# =============================================================================

def test_get_angles_at_180_deg_half_degree_step():
    angles = np.arange(0, 360, 0.5, dtype='float32')
    p0, p180 = t4.get_angles_at_180_deg(angles)
    assert len(p0) == len(p180) > 0
    for a, b in zip(p0, p180):
        assert abs((float(angles[b]) - float(angles[a])) % 360 - 180.0) < 0.26


def test_get_angles_at_180_deg_tolerates_float32_noise():
    """Шаг 0.1° в float32 не представим точно — сравнение == 0 не работает."""
    angles = np.arange(0, 360, 0.1, dtype='float32')
    exact = np.argwhere(
        np.abs(np.subtract.outer(angles, angles) % 360 - 180) % 360 == 0)
    p0, _ = t4.get_angles_at_180_deg(angles)
    assert len(p0) > 0
    # именно эти пары строгое равенство и теряло
    assert len(p0) >= len(exact) // 2


def test_get_angles_at_180_deg_raises_without_pairs():
    angles = np.linspace(0, 120, 241).astype('float32')
    with pytest.raises(ValueError, match='180'):
        t4.get_angles_at_180_deg(angles)


# =============================================================================
# 4e — начальное приближение сдвига оси из X-компоненты центра масс
# =============================================================================

def _axis_pair(s_true, height=48, width=64):
    """Готовит кадры 0°/180° с известным сдвигом оси вращения s_true.

    Целевая функция find_axis_correction обнуляется при
    im1 == shift(im0, [0, 2 * s_true]).
    """
    yy, xx = np.mgrid[0:height, 0:width]
    obj = (0.9 * np.exp(-(((yy - 18) ** 2 + (xx - 26) ** 2) / 40.))
           + 0.5 * np.exp(-(((yy - 30) ** 2 + (xx - 38) ** 2) / 15.)))
    im0 = ndi.gaussian_filter(obj, 1.0).astype('float32')
    im1 = ndi.shift(im0, [0, 2 * s_true], order=3, mode='nearest').astype('float32')
    images = np.stack([im0, np.fliplr(im1)]).astype('float32')
    return images, np.array([0., 180.], dtype='float32')


def test_transform_image_shifts_along_x():
    """transform_image(im, s, 0) сдвигает по столбцам (X), не по строкам."""
    im = np.zeros((16, 20), dtype='float32')
    im[8, 10] = 1.0
    moved = t4.transform_image(im, 3.0, 0.0)
    peak = np.unravel_index(np.argmax(moved), moved.shape)
    assert peak == (8, 13)


@pytest.mark.parametrize('s_true', [3.0, -2.5])
def test_find_axis_correction_recovers_known_shift(s_true):
    images, angles = _axis_pair(s_true)
    shift_x, alfa = t4.find_axis_correction(images, angles)
    assert shift_x == pytest.approx(s_true, abs=0.05)
    assert alfa == pytest.approx(0.0, abs=0.01)


def test_initial_shift_uses_x_component():
    """Y-компонента центра масс здесь тождественно ~0 — приближение бесполезно."""
    images, _ = _axis_pair(3.0)
    im0 = images[0]
    im1 = np.fliplr(images[1])
    n0 = im0 / (im0 ** 2).sum() ** 0.5
    n1 = im1 / (im1 ** 2).sum() ** 0.5
    cm0 = ndi.center_of_mass(n0)
    cm1 = ndi.center_of_mass(n1)

    assert abs(cm0[0] - cm1[0]) < 1e-3                      # по Y разницы нет
    assert (cm1[1] - cm0[1]) / 2 == pytest.approx(3.0, abs=0.05)


# =============================================================================
# 4g — save_amira, show_frames_with_border, series_length
# =============================================================================

def test_amira_raw_name_sanitizes_spaces():
    assert t4.amira_raw_name('обр 1 2', (3, 4, 5), 1) == 'обр_1_2.3_4_5.1.raw'


def test_save_amira_hx_points_to_existing_raw(tmp_path):
    vol = np.arange(2 * 3 * 4, dtype='float32').reshape(2, 3, 4)
    t4.save_amira(vol, str(tmp_path), 'образец 1', reshape=1, pixel_size=0.01)

    raw = tmp_path / 'образец_1.2_3_4.1.raw'
    hx = tmp_path / 'tomo.образец_1.1.hx'
    assert raw.exists()
    assert hx.exists()
    assert raw.name in hx.read_text(encoding='utf8')


def test_save_amira_keeps_existing_raw(tmp_path):
    """Файл, уже записанный persistent_array, не переписывается."""
    vol = np.zeros((2, 3, 4), dtype='float32')
    raw = tmp_path / 'sample.2_3_4.1.raw'
    raw.write_bytes(b'x' * 10)

    t4.save_amira(vol, str(tmp_path), 'sample', reshape=1)

    assert raw.read_bytes() == b'x' * 10
    assert raw.name in (tmp_path / 'tomo.sample.1.hx').read_text(encoding='utf8')


def test_save_amira_writes_binned_raw(tmp_path):
    vol = np.ones((4, 4, 4), dtype='float32')
    t4.save_amira(vol, str(tmp_path), 'sample', reshape=2)
    raw = tmp_path / 'sample.2_2_2.2.raw'
    assert raw.exists()
    assert raw.stat().st_size == 2 * 2 * 2 * 4
    assert raw.name in (tmp_path / 'tomo.sample.2.hx').read_text(encoding='utf8')


def test_show_frames_with_border_does_not_mutate_input(monkeypatch):
    import pylab as plt

    monkeypatch.setattr(plt, 'show', lambda *a, **k: None)
    data = np.full((3, 8, 8), 0.5, dtype='float32')
    before = data.copy()
    empty = np.full((8, 8), 10.0, dtype='float32')
    angles = np.array([0., 10., 20.], dtype='float32')

    t4.show_frames_with_border(data, empty, angles, 0, 1, 7, 1, 7)
    plt.close('all')

    assert np.array_equal(data, before)


def test_read_series_length_raises_without_metadata(tmp_path):
    import h5py

    path = tmp_path / 'noinfo.h5'
    with h5py.File(path, 'w') as f:
        f.create_group('empty')
        f.create_group('data')
    with pytest.raises(ValueError, match='series_length'):
        t4._read_series_length_from_hdf5(str(path), 10)


def test_read_series_length_from_exp_info(tmp_path):
    import h5py
    import json as _json

    path = tmp_path / 'ok.h5'
    with h5py.File(path, 'w') as f:
        f.attrs['exp_info'] = _json.dumps(
            {'experiment parameters': {'series_length': 7}})
    assert t4._read_series_length_from_hdf5(str(path), 10) == 7
