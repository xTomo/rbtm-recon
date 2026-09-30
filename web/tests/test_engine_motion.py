"""Тесты reconengine.motion: оценка смещения образца по центру масс полос, решение о компенсации, применение в
pipeline.prepare; пропуск первых кадров серий empty (preprocess.dark_empty_from_crop)."""
import numpy as np
import pytest

import engine_phantom as ph
from reconengine import gpu, motion, pipeline, preprocess
from reconengine import recipe as recipe_mod
from reconengine.model import ROI

ANG = np.arange(0.0, 184.0, 1.0)


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv('RECON_ENGINE_CPU', '1')
    gpu.reset_backend()
    yield
    gpu.reset_backend()


def drift_pattern(n, amp):
    t = np.arange(n) / max(n - 1, 1)
    return amp * np.interp(t, [0, 0.3, 0.6, 1.0], [1.0, -0.45, 0.2, -0.45])


def sinusoid_part_removed(values, angles):
    A = np.stack([np.ones(len(angles)), np.cos(np.radians(angles)), np.sin(np.radians(angles))], 1)
    c, *_ = np.linalg.lstsq(A, values, rcond=None)
    return values - A @ c


def synthetic(amp=0.0, advanced=True, **kw):
    tl = ph.advanced_timeline(ANG, 4) if advanced else ph.simple_timeline(ANG)
    fdx = drift_pattern(len(tl.modes), amp)
    ss = ph.make_synthetic_scan(ANG, height=96, width=160, center_x=81.3, y_ref=47.5, tilt_deg=0.6,
                                advanced=advanced, n_segments=4, frame_dx=fdx, **kw)
    return ss, fdx


def estimate_for(ss, tilt=0.6):
    sc = ss.scan
    crop = ph.make_crop(ss.frames, ROI(0, sc.width, 0, sc.height))
    de = preprocess.dark_empty_from_crop(sc, crop)
    idx, angles, fnums = pipeline.data_frames(sc)
    z = np.zeros(len(idx))
    prof, _ = motion.band_profiles(crop, idx, fnums, de, z, z, tilt)
    b = motion.effective_bin(sc.height, sc.width, motion.DEFAULTS['bin'])
    return motion.estimate(prof, angles, fnums, de.periodic_empty_fnumbers, bin_used=b), idx, angles


# --- оценка на синтетическом скане -------------------------------------------------------------------------------

@pytest.mark.parametrize('advanced', [True, False])
def test_estimate_recovers_drift(advanced):
    ss, fdx = synthetic(6.0, advanced=advanced, noise=0.01)
    est, idx, angles = estimate_for(ss)
    truth = sinusoid_part_removed(fdx[idx], angles)          # синусоидальную часть сдвига не видно (и не нужно)
    assert est.status == 'detected' and est.common > 0.9
    assert np.sqrt(np.mean((est.dx - truth) ** 2)) < 0.15
    assert abs(est.rms - truth.std()) < 0.15
    ok, why = motion.decide(est, 'auto')
    assert ok and 'компенсируется' in why and not motion.decide(est, 'off')[0]
    assert not motion.decide(est, 'auto', rotated=True)[0]  # сбой угла на вставках — авто не включается


def test_no_drift_is_none():
    ss, _ = synthetic(0.0)
    est, _, _ = estimate_for(ss)
    assert est.status == 'none' and est.rms < 0.1
    ok, why = motion.decide(est, 'auto')
    assert not ok and 'нет' in why
    assert motion.decide(est, 'on')[0]                       # вручную — компенсируется


def test_to_dict_and_summary():
    ss, _ = synthetic(6.0)
    est, _, _ = estimate_for(ss)
    d = est.to_dict()
    assert set(est.summary()) <= set(d)
    assert len(d['dx']) == len(d['fnums']) == len(d['angles']) == len(d['dx_raw'])
    assert any(v is None for v in d['dx_raw'])               # кадры после вставок не учитывались


# --- оценка на искусственных профилях: сдвиг против сбоя угла ------------------------------------------------------

def profiles_from_centers(centers, width=240, sigma=6.0):
    xs = np.arange(width)
    return np.exp(-0.5 * ((xs[None, None, :] - centers[:, :, None]) / sigma) ** 2).astype('float32')


def band_centers(angles, err_angle=None, shift=None):
    rng = np.random.default_rng(3)
    amps = np.linspace(20, 80, 12)
    phases = rng.uniform(0, 2 * np.pi, 12)
    th = np.radians(angles)[:, None] + (0 if err_angle is None else np.radians(err_angle)[:, None])
    c = 120 + amps[None, :] * np.cos(th - phases[None, :])
    if shift is not None:
        c = c + shift[:, None]
    return c


def test_angle_error_is_inconsistent():
    n = len(ANG)
    err = np.where(np.arange(n) > n // 2, -6.0, 0.0)          # проворот на полпути
    est = motion.estimate(profiles_from_centers(band_centers(ANG, err_angle=err)), ANG, np.arange(n), [],
                          bin_used=1)
    assert est.status == 'inconsistent' and est.common < 0.5
    ok, why = motion.decide(est, 'auto')
    assert not ok and 'не сдвиг' in why


def test_common_shift_is_detected_on_profiles():
    n = len(ANG)
    shift = drift_pattern(n, 5.0)
    est = motion.estimate(profiles_from_centers(band_centers(ANG, shift=shift)), ANG, np.arange(n), [],
                          bin_used=1)
    assert est.status == 'detected' and est.common > 0.9
    assert np.sqrt(np.mean((est.dx - sinusoid_part_removed(shift, ANG)) ** 2)) < 0.1


def test_no_object():
    n = 40
    est = motion.estimate(np.zeros((n, 0, 50), 'float32'), ANG[:n], np.arange(n), [])
    assert est.status == 'no_object'
    assert not motion.decide(est, 'on')[0] and not motion.decide(est, 'auto')[0]
    assert not motion.decide(None, 'auto')[0]
    with pytest.raises(ValueError):
        motion.decide(est, 'bad')


def test_excluded_and_smooth():
    fn = np.arange(10, 30)
    ex = motion._excluded(fn, [14, 25], 2)
    assert list(fn[ex]) == [15, 16, 26, 27]
    v = np.arange(20, dtype=float)
    w = (~ex).astype(float)
    sm = motion._smooth(np.where(ex, 1e6, v), w, 1.0)
    assert np.all(np.abs(sm - v) < 1.0)                       # исключённые значения не тянут, восстановлены соседями


def test_band_lines_inside_frame():
    coords, ys = motion.band_lines(100, 300, 1.25, 16, 7)
    assert coords.shape == (2, len(ys), 7, 300) and 1 <= len(ys) <= 16
    assert coords[0].min() >= 0 and coords[0].max() <= 99
    coords, ys = motion.band_lines(8, 17, 0.6, 16, 7)         # кадр слишком низкий — полос нет, без исключения
    assert len(ys) == 0
    assert motion.effective_bin(1397, 3216, 4) == 4 and motion.effective_bin(96, 160, 4) == 1


def test_from_block():
    blk = recipe_mod.motion_block('auto', True, [0.5, -0.5], [10, 11])
    assert np.allclose(motion.from_block(blk, [10, 11]), [0.5, -0.5])
    assert motion.from_block(recipe_mod.motion_block('auto'), [10, 11]) is None
    with pytest.raises(ValueError, match='других кадров'):
        motion.from_block(blk, [10, 12])


# --- pipeline.prepare ---------------------------------------------------------------------------------------------

def recipe_for(ss, **motion_kw):
    sc = ss.scan
    roi = ROI(0, sc.width, 0, sc.height)
    r = recipe_mod.default_recipe('synthetic', sc.fingerprint, roi, 0.001, 'user', sc.is_advanced)
    r.repositioning['enabled'] = False
    if motion_kw:
        r.motion = recipe_mod.motion_block(**motion_kw)
    return r, ph.make_crop(ss.frames, roi)


def test_prepare_auto_compensates_and_records():
    ss, fdx = synthetic(6.0)
    r, crop = recipe_for(ss)
    assert r.motion['mode'] == 'auto' and r.empty_skip_first == recipe_mod.EMPTY_SKIP_DEFAULT
    prep = pipeline.prepare(ss.scan, crop, r)
    truth = sinusoid_part_removed(fdx[prep.idx], prep.angles)
    assert prep.motion['applied'] is True and prep.motion['summary']['status'] == 'detected'
    assert np.sqrt(np.mean((-prep.frame_sx - truth) ** 2)) < 0.15
    rr = pipeline.resolved_recipe(r, prep)
    assert rr.motion['applied'] is True and len(rr.motion['dx']) == len(prep.idx)
    # рецепт с решением воспроизводит те же сдвиги без оценки
    prep2 = pipeline.prepare(ss.scan, crop, rr)
    assert np.allclose(prep2.frame_sx, prep.frame_sx, atol=1e-3)


def test_prepare_respects_decision_and_off():
    ss, _ = synthetic(6.0)
    r, crop = recipe_for(ss, mode='auto', applied=False)
    prep = pipeline.prepare(ss.scan, crop, r)
    assert not np.any(prep.frame_sx) and prep.motion['applied'] is False
    r, crop = recipe_for(ss, mode='off')
    prep = pipeline.prepare(ss.scan, crop, r)
    assert not np.any(prep.frame_sx) and prep.motion == recipe_mod.motion_block('off', applied=False)


def test_prepare_old_recipe_without_blocks_is_unchanged():
    ss, _ = synthetic(6.0)
    r, crop = recipe_for(ss)
    d = recipe_mod.to_dict(r)
    d.pop('motion'); d.pop('empty_skip_first')
    old = recipe_mod.from_dict(d)
    assert old.motion['mode'] == 'off' and old.empty_skip_first == 0
    prep = pipeline.prepare(ss.scan, crop, old)
    assert not np.any(prep.frame_sx)


# --- empty без первых кадров серии -------------------------------------------------------------------------------

def test_series_tail():
    assert list(preprocess.series_tail(np.arange(5), 2)) == [2, 3, 4]
    assert list(preprocess.series_tail(np.arange(3), 2)) == [1, 2]     # остаётся не меньше двух
    assert list(preprocess.series_tail(np.arange(1), 2)) == [0]
    assert list(preprocess.series_tail(np.arange(5), 0)) == [0, 1, 2, 3, 4]


def test_dark_empty_skip_first_uses_tail_of_each_series():
    ss = ph.make_synthetic_scan(ANG, height=32, width=48, advanced=True, n_segments=3, series_length=5, noise=0.02,
                                seed=5)
    sc = ss.scan
    crop = ph.make_crop(ss.frames, ROI(0, sc.width, 0, sc.height))
    de0 = preprocess.dark_empty_from_crop(sc, crop)
    de2 = preprocess.dark_empty_from_crop(sc, crop, skip_first=2)
    fr = ss.frames.astype('float32')
    emp = np.asarray(sc.empty_idx)
    dark = np.median(fr[sc.dark_idx], axis=0)
    series = [emp[i:i + 5] for i in range(0, len(emp), 5)]
    np.testing.assert_allclose(de2.initial_empty, np.median(fr[series[0][2:]] - dark, axis=0), atol=1e-3)
    for k, s in enumerate(series[1:]):
        np.testing.assert_allclose(de2.periodic_empties[k], np.median(fr[s[2:]] - dark, axis=0), atol=1e-3)
        np.testing.assert_allclose(de0.periodic_empties[k], np.median(fr[s] - dark, axis=0), atol=1e-3)
    assert de2.periodic_empty_fnumbers == de0.periodic_empty_fnumbers  # номер серии — её первый кадр, как раньше
