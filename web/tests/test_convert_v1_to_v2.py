"""Тест конвертера HDF5 v1 → v2 на синтетическом файле."""
import numpy as np
import pytest

import hdf5_v2
from convert_v1_to_v2 import convert_v1_to_v2
from helpers import FRAME_MODES, build_advanced_timeline, make_v1_file


@pytest.fixture
def converted(tmp_path):
    frames = build_advanced_timeline(n_dark=4, series_length=3, n_periodic=2,
                                     data_per_segment=4)
    v1_path, images = make_v1_file(tmp_path / 'exp.h5', frames, series_length=3)
    v2_path = str(tmp_path / 'exp_v2.h5')
    convert_v1_to_v2(v1_path, v2_path)
    return frames, images, v2_path


def test_converted_file_is_v2(converted):
    _, _, v2_path = converted
    assert hdf5_v2.is_hdf5_v2(v2_path)


def test_converted_metadata(converted):
    frames, _, v2_path = converted
    info = hdf5_v2.get_experiment_info_v2(v2_path)
    assert info['is_advanced'] is True
    assert info['series_length'] == 3
    assert info['total_frames'] == len(frames)
    assert info['dark_count'] == 4
    assert info['empty_count'] == 9
    assert info['data_count'] == 12
    assert info['data_check_count'] == 2


def test_converted_images_match_source(converted, tmp_path):
    frames, images, v2_path = converted
    imgs, angles, fnums = hdf5_v2.get_frame_group_v2(
        v2_path, 'data', str(tmp_path), return_frame_numbers=True)
    expected = [i for i, f in enumerate(frames) if f['mode'] == 'data']
    assert list(fnums) == expected
    assert np.array_equal(imgs.astype('uint16'), images[expected])


def test_converted_segment_ids_follow_storage_semantics(converted):
    import h5py

    frames, _, v2_path = converted
    with h5py.File(v2_path, 'r') as f:
        modes = f['timeline/modes'][:]
        segments = f['timeline/segment_ids'][:]
        fnums = f['timeline/frame_numbers'][:]

    # timeline упорядочен по frame_number (порядок съёмки)
    assert list(fnums) == sorted(fnums)

    expected = [fr['segment'] for fr in frames]
    assert list(segments) == expected

    assert set(segments[modes == FRAME_MODES['dark']]) == {-1}
    # data_check принадлежит своей periodic-вставке
    assert list(segments[modes == FRAME_MODES['data_check']]) == [1, 2]
    # начальная empty-серия и первый сегмент data — 0
    assert list(segments[modes == FRAME_MODES['empty']]) == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert list(segments[modes == FRAME_MODES['data']]) == [0] * 4 + [1] * 4 + [2] * 4


def test_converted_checkpoint_pairs(converted):
    frames, _, v2_path = converted
    data_idx, dc_idx = hdf5_v2.get_checkpoint_mapping_v2(v2_path)

    # ровно одна пара на periodic-серию
    assert len(data_idx) == len(dc_idx) == 2

    angles = [f['angle'] for f in frames]
    for d, c in zip(data_idx, dc_idx):
        assert d < c
        assert frames[d]['mode'] == 'data'
        assert frames[c]['mode'] == 'data_check'
        assert angles[d] == pytest.approx(angles[c], abs=0.5)
        # data-кадр — последний перед вставкой, то есть из предыдущего сегмента
        assert frames[d]['segment'] == frames[c]['segment'] - 1


def test_reader_loads_converted_advanced_data(converted, tmp_path):
    _, _, v2_path = converted
    adv = hdf5_v2.load_tomo_data_advanced_v2(v2_path, str(tmp_path))

    assert adv.series_length == 3
    assert len(adv.periodic_empties) == 2
    assert adv.data_images.shape[0] == 12
    assert adv.data_check_images.shape[0] == 2
    assert adv.initial_empty_fnumber == 4
    assert adv.dark_image.shape == (8, 10)


def test_converted_matches_direct_v2_file(tmp_path):
    """Конвертация v1 даёт то же, что прямая запись v2 storage-ом."""
    from helpers import make_v2_file

    frames = build_advanced_timeline(n_dark=2, series_length=2, n_periodic=1,
                                     data_per_segment=3)
    rng = np.random.default_rng(11)
    images = rng.integers(100, 4000,
                          size=(len(frames), 8, 10)).astype('uint16')

    v1_path, _ = make_v1_file(tmp_path / 'a.h5', frames, images=images,
                              series_length=2)
    v2_direct, _ = make_v2_file(tmp_path / 'b.h5', frames, images=images,
                                series_length=2)
    v2_converted = str(tmp_path / 'a_v2.h5')
    convert_v1_to_v2(v1_path, v2_converted)

    a = hdf5_v2.load_tomo_data_advanced_v2(v2_converted, str(tmp_path))
    b = hdf5_v2.load_tomo_data_advanced_v2(v2_direct, str(tmp_path))

    assert np.array_equal(a.dark_image, b.dark_image)
    assert np.array_equal(a.initial_empty, b.initial_empty)
    assert a.periodic_empty_fnumbers == b.periodic_empty_fnumbers
    assert np.array_equal(a.data_images, b.data_images)
    assert np.array_equal(a.data_angles, b.data_angles)
    assert np.array_equal(a.data_check_images, b.data_check_images)

    assert np.array_equal(*[np.asarray(x) for x in (
        hdf5_v2.get_checkpoint_mapping_v2(v2_converted)[1],
        hdf5_v2.get_checkpoint_mapping_v2(v2_direct)[1])])
