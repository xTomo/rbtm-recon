"""Тесты reconengine.pixelsize: выбор размера пикселя и его источника."""
import math

from reconengine import pixelsize as ps


def test_user_value_wins_over_everything():
    r = ps.resolve({'pixel_size': 0.02, 'detector_model': 'MH110XC-KK-FA'},
                   {'pixel_size': 0.03}, user_value=0.0123)
    assert r.value_mm == 0.0123
    assert r.source == 'user'
    assert r.warnings == []


def test_mongo_value_wins_over_detector_and_hdf5():
    r = ps.resolve({'pixel_size': 0.009, 'detector_model': 'MH110XC-KK-FA'},
                   {'pixel_size': 0.05}, None)
    assert math.isclose(r.value_mm, 0.009)
    assert r.source == 'mongo'


def test_mongo_zero_or_negative_is_ignored():
    r = ps.resolve({'pixel_size': 0.0, 'detector_model': 'MH110XC-KK-FA'}, None, None)
    assert r.source == 'detector'
    assert math.isclose(r.value_mm, ps.DETECTOR_PIXEL_MM['MH110XC-KK-FA'])


def test_detector_model_from_mongo_wins_over_hdf5_value():
    r = ps.resolve({'detector_model': 'MH110XC-KK-FA'}, {'pixel_size': 0.05}, None)
    assert r.source == 'detector'
    assert math.isclose(r.value_mm, 0.009)


def test_detector_model_from_hdf5_used_when_mongo_absent():
    r = ps.resolve(None, {'detector_model': 'MH110XC-KK-FA'}, None)
    assert r.source == 'detector'
    assert math.isclose(r.value_mm, 0.009)


def test_hdf5_value_used_when_not_writer_default():
    r = ps.resolve(None, {'pixel_size': 0.011, 'detector_model': 'unknown-model'}, None)
    assert r.source == 'hdf5'
    assert math.isclose(r.value_mm, 0.011)
    assert r.warnings == []


def test_hdf5_writer_default_is_treated_as_unknown():
    r = ps.resolve(None, {'pixel_size': ps.WRITER_DEFAULT_MM}, None)
    assert r.source == 'default'
    assert math.isclose(r.value_mm, ps.DEFAULT_MM)
    assert any('по умолчанию' in w for w in r.warnings)


def test_nothing_known_falls_back_to_default_with_warning():
    r = ps.resolve(None, None, None)
    assert r.source == 'default'
    assert math.isclose(r.value_mm, 0.00425)
    assert len(r.warnings) == 1
    assert '4,25 мкм' in r.warnings[0]


def test_mismatch_between_mongo_and_detector_model_warns():
    r = ps.resolve({'pixel_size': 0.02, 'detector_model': 'MH110XC-KK-FA'}, None, None)
    assert r.source == 'mongo'
    assert math.isclose(r.value_mm, 0.02)
    assert len(r.warnings) == 1
    assert 'расходится' in r.warnings[0]


def test_small_mismatch_below_tolerance_does_not_warn():
    close_value = ps.DETECTOR_PIXEL_MM['MH110XC-KK-FA'] * 1.001
    r = ps.resolve({'pixel_size': close_value, 'detector_model': 'MH110XC-KK-FA'}, None, None)
    assert r.warnings == []


def test_unknown_detector_model_falls_through_to_hdf5():
    r = ps.resolve({'detector_model': 'no-such-model'}, {'pixel_size': 0.012}, None)
    assert r.source == 'hdf5'
    assert math.isclose(r.value_mm, 0.012)


def test_mj150xr_writer_default_value_is_known_via_detector_model():
    # у MJ150XR пиксель действительно 4,25 мкм: при известной модели это не «значение по умолчанию»
    r = ps.resolve(None, {'pixel_size': ps.WRITER_DEFAULT_MM, 'detector_model': 'MJ150XR-GP-FA-GO'}, None)
    assert r.source == 'detector'
    assert math.isclose(r.value_mm, 0.00425)
    assert r.warnings == []
