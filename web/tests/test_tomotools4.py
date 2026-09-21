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


def test_apply_repositioning_no_shifts_is_noop():
    obj = make_object()
    adv, _ = build_adv([obj], [])
    norm = np.stack([normalize(f) for f in adv.data_images]).astype('float32')
    before = norm.copy()
    t4.apply_repositioning_correction(norm, adv.data_numbers, adv,
                                      np.array([]), np.array([]))
    assert np.array_equal(norm, before)
