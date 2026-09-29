"""Проверка контрольных кадров advanced-скана: поворот образца во время вставки (сбой угла, которого не видит
счётчик мотора) и его последствия — предупреждения и отказ от ложных «сдвигов» (скан 524efd6e от 15.09.2026)."""
import numpy as np
import pytest

import engine_phantom as ph
from reconengine import gpu, pipeline, preprocess
from reconengine import recipe as recipe_mod
from reconengine.model import ROI, Axis

ANGLES = np.arange(0, 180, 1.0)


@pytest.fixture(autouse=True)
def _cpu_backend(monkeypatch):
    monkeypatch.setenv('RECON_ENGINE_CPU', '1')
    gpu.reset_backend()
    yield
    gpu.reset_backend()


def _scan(angle_offsets=None, noise=0.01):
    ss = ph.make_synthetic_scan(ANGLES, height=64, width=128, center_x=63.0, y_ref=31.5,
                                advanced=True, n_segments=4, series_length=3,
                                segment_angle_offsets=angle_offsets, noise=noise, seed=3)
    roi = ROI(0, 128, 0, 64, 32)
    crop = ph.make_crop(ss.frames, roi)
    return ss, crop, preprocess.dark_empty_from_crop(ss.scan, crop)


def test_clean_scan_all_ok():
    ss, crop, de = _scan()
    checks = preprocess.check_checkpoints(ss.scan, crop, de)
    assert [c['status'] for c in checks] == ['ok'] * 3
    assert all(c['ratio'] < preprocess.CHECK_SUSPECT_RATIO for c in checks)
    assert preprocess.checks_summary(checks) is None


def test_rotation_during_insertion_detected_and_measured():
    # во время вставки перед сегментом 2 стол провернулся назад на 3°; сбой сохраняется в сегменте 3
    ss, crop, de = _scan(angle_offsets=[0.0, 0.0, -3.0, -3.0])
    checks = preprocess.check_checkpoints(ss.scan, crop, de)
    assert [c['status'] for c in checks] == ['ok', 'rotated', 'ok']
    rot = checks[1]
    assert rot['offset_deg'] == pytest.approx(-3.0, abs=0.5)
    assert rot['rms_best'] < 0.5 * rot['rms_same']
    assert '3.0°' in rot['message'] or '2.9°' in rot['message'] or '3.1°' in rot['message']
    summary = preprocess.checks_summary(checks)
    assert summary and '1 из 3' in summary


def test_forward_rotation_flagged_as_changed():
    # вперёд: кадров «после» для сравнения нет (они сняты уже со сбоем) — угол не найти, но вставка помечена
    ss, crop, de = _scan(angle_offsets=[0.0, 4.0, 4.0, 4.0])
    checks = preprocess.check_checkpoints(ss.scan, crop, de)
    assert checks[0]['status'] == 'changed' and checks[1]['status'] == 'ok' and checks[2]['status'] == 'ok'


def test_prepare_drops_false_shifts_and_warns(monkeypatch):
    ss, crop, de = _scan(angle_offsets=[0.0, 0.0, -3.0, -3.0])
    # без проверки корреляция пары дала бы «сдвиг» — подменяем измерение заведомо ложным значением
    monkeypatch.setattr(preprocess, 'repositioning_shifts',
                        lambda scan, crop, de: (np.zeros(3, 'float32'), np.array([0.0, 2.5, 0.0]),
                                                np.array([0.0, 7.0, 0.0])))
    r = recipe_mod.default_recipe('synthetic', 'synthetic', crop.roi, 0.01, 'default', True)
    r.axis = Axis(63.0, 31.5, 0.0, 'manual')              # пары 0°/180° в синтетике нет
    prep = pipeline.prepare(ss.scan, crop, r)
    assert prep.shifts == {'sy': [0.0, 0.0, 0.0], 'sx': [0.0, 0.0, 0.0]}
    assert any('контрольные кадры' in w for w in prep.warnings)
    assert any('повернулся назад' in w for w in prep.warnings)


def test_non_advanced_scan_has_no_checks():
    ss = ph.make_synthetic_scan(ANGLES[::4], height=32, width=64)
    crop = ph.make_crop(ss.frames, ROI(0, 64, 0, 32, 16))
    de = preprocess.dark_empty_from_crop(ss.scan, crop)
    assert preprocess.check_checkpoints(ss.scan, crop, de) == []
