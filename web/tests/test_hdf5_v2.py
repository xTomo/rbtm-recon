"""Тесты читателя HDF5 v2 (hdf5_v2.py) на синтетических файлах."""
import numpy as np
import pytest

import hdf5_v2
from helpers import FRAME_MODES, build_advanced_timeline, make_v2_file


@pytest.fixture
def frames():
    return build_advanced_timeline(n_dark=4, series_length=3, n_periodic=2,
                                   data_per_segment=4)


def test_is_hdf5_v2(tmp_path, frames):
    path, _ = make_v2_file(tmp_path / 'exp.h5', frames)
    assert hdf5_v2.is_hdf5_v2(path) is True
    assert hdf5_v2.is_hdf5_v2(str(tmp_path / 'nope.h5')) is False


def test_experiment_info(tmp_path, frames):
    path, _ = make_v2_file(tmp_path / 'exp.h5', frames)
    info = hdf5_v2.get_experiment_info_v2(path)
    assert info['format_version'] == 'v2'
    assert info['is_advanced'] is True
    assert info['total_frames'] == len(frames)
    assert info['dark_count'] == 4
    assert info['data_check_count'] == 2


def test_frame_group_matches_timeline(tmp_path, frames):
    path, images = make_v2_file(tmp_path / 'exp.h5', frames)
    imgs, angles, fnums = hdf5_v2.get_frame_group_v2(
        path, 'data', str(tmp_path), return_frame_numbers=True)
    expected = [i for i, f in enumerate(frames) if f['mode'] == 'data']
    assert list(fnums) == expected
    assert np.allclose(imgs, images[expected])


def test_frame_group_without_mapping_equals_with_mapping(tmp_path, frames):
    """Прерванный эксперимент (без finalize → без mapping) должен читаться."""
    with_map, _ = make_v2_file(tmp_path / 'with.h5', frames, with_mapping=True)
    no_map, _ = make_v2_file(tmp_path / 'without.h5', frames, with_mapping=False)

    for group in ('dark', 'empty', 'data', 'data_check'):
        a = hdf5_v2.get_frame_group_v2(with_map, group, str(tmp_path),
                                       return_frame_numbers=True)
        b = hdf5_v2.get_frame_group_v2(no_map, group, str(tmp_path),
                                       return_frame_numbers=True)
        for x, y in zip(a, b):
            assert np.array_equal(x, y), group


def test_load_tomo_data_v2_identical_with_and_without_mapping(tmp_path, frames):
    with_map, _ = make_v2_file(tmp_path / 'with.h5', frames, with_mapping=True)
    no_map, _ = make_v2_file(tmp_path / 'without.h5', frames, with_mapping=False)

    e1, d1, a1 = hdf5_v2.load_tomo_data_v2(with_map, str(tmp_path))
    e2, d2, a2 = hdf5_v2.load_tomo_data_v2(no_map, str(tmp_path))

    assert np.array_equal(e1, e2)
    assert np.array_equal(d1, d2)
    assert np.array_equal(a1, a2)


def test_load_tomo_data_advanced_v2_identical_with_and_without_mapping(tmp_path, frames):
    with_map, _ = make_v2_file(tmp_path / 'with.h5', frames, with_mapping=True)
    no_map, _ = make_v2_file(tmp_path / 'without.h5', frames, with_mapping=False)

    a = hdf5_v2.load_tomo_data_advanced_v2(with_map, str(tmp_path))
    b = hdf5_v2.load_tomo_data_advanced_v2(no_map, str(tmp_path))

    assert np.array_equal(a.dark_image, b.dark_image)
    assert np.array_equal(a.initial_empty, b.initial_empty)
    assert a.initial_empty_fnumber == b.initial_empty_fnumber
    assert a.periodic_empty_fnumbers == b.periodic_empty_fnumbers
    assert np.array_equal(a.data_images, b.data_images)
    assert np.array_equal(a.data_angles, b.data_angles)
    assert np.array_equal(a.data_numbers, b.data_numbers)
    assert np.array_equal(a.data_check_images, b.data_check_images)
    assert a.series_length == b.series_length


def test_advanced_fields_are_consistent(tmp_path, frames):
    path, _ = make_v2_file(tmp_path / 'exp.h5', frames)
    adv = hdf5_v2.load_tomo_data_advanced_v2(path, str(tmp_path))

    assert adv.series_length == 3
    assert len(adv.periodic_empties) == 2
    assert adv.data_images.shape[0] == 12
    assert adv.data_check_images.shape[0] == 2
    # initial_empty_fnumber — первый empty-кадр начальной серии (после 4 dark)
    assert adv.initial_empty_fnumber == 4
    # periodic_empty_fnumbers — первые кадры periodic-серий
    empties = [i for i, f in enumerate(frames) if f['mode'] == 'empty']
    assert adv.periodic_empty_fnumbers == [empties[3], empties[6]]


def test_empty_group_returns_3d_zero_length_array(tmp_path):
    """Отсутствующая группа (например, data_check) → массив формы (0, H, W)."""
    frames = build_advanced_timeline(n_dark=2, series_length=2, n_periodic=0,
                                     data_per_segment=3)
    path, _ = make_v2_file(tmp_path / 'exp.h5', frames, series_length=2)

    imgs, angles, fnums = hdf5_v2.get_frame_group_v2(
        path, 'data_check', str(tmp_path), return_frame_numbers=True)
    assert imgs.shape == (0, 8, 10)
    assert angles.shape == (0,)
    assert fnums.shape == (0,)


def test_advanced_without_data_check(tmp_path):
    frames = build_advanced_timeline(n_dark=2, series_length=2, n_periodic=0,
                                     data_per_segment=3)
    path, _ = make_v2_file(tmp_path / 'exp.h5', frames, series_length=2,
                           with_mapping=False)
    adv = hdf5_v2.load_tomo_data_advanced_v2(path, str(tmp_path))
    assert adv.data_check_images.shape == (0, 8, 10)
    assert adv.data_check_angles.shape == (0,)
    assert adv.periodic_empties == []


def test_advanced_without_dark_uses_zero_dark(tmp_path):
    """Прерванный эксперимент без dark-серии — dark_image нулевой."""
    frames = build_advanced_timeline(n_dark=0, series_length=2, n_periodic=1,
                                     data_per_segment=3)
    path, _ = make_v2_file(tmp_path / 'exp.h5', frames, series_length=2,
                           with_mapping=False)
    adv = hdf5_v2.load_tomo_data_advanced_v2(path, str(tmp_path))
    assert adv.dark_image.shape == (8, 10)
    assert np.all(adv.dark_image == 0)


def test_unknown_group_raises(tmp_path, frames):
    path, _ = make_v2_file(tmp_path / 'exp.h5', frames, with_mapping=False)
    with pytest.raises(ValueError):
        hdf5_v2.get_frame_group_v2(path, 'nonsense', str(tmp_path))


def test_checkpoint_mapping(tmp_path, frames):
    path, _ = make_v2_file(tmp_path / 'exp.h5', frames)
    data_idx, dc_idx = hdf5_v2.get_checkpoint_mapping_v2(path)
    assert len(data_idx) == len(dc_idx) == 2

    no_map, _ = make_v2_file(tmp_path / 'without.h5', frames, with_mapping=False)
    with pytest.raises(ValueError, match='checkpoint'):
        hdf5_v2.get_checkpoint_mapping_v2(no_map)
