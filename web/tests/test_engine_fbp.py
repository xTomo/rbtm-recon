"""reconengine.fbp: выбор углов, FBP на CPU (соглашения skimage/astra), масштаб, выбор бэкенда."""
import sys

import numpy as np
import pytest
from skimage.transform import iradon

import engine_phantom as ph
from reconengine import fbp, gpu

ANGLES_180 = np.arange(0, 180, 1.0)


@pytest.fixture(autouse=True)
def _cpu_backend(monkeypatch):
    monkeypatch.setenv('RECON_ENGINE_CPU', '1')
    gpu.reset_backend()
    yield
    gpu.reset_backend()


def _corr(a, b):
    return float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])


# --- выбор углов -------------------------------------------------------------------------------------------

def test_select_angles_partial_second_half():
    a = np.arange(0, 200, 0.5)                        # 0..199.5
    first = fbp.select_angles(a, 'first_180')
    assert first.dtype == bool and first.sum() == 360 and a[first].max() == 179.5
    np.testing.assert_array_equal(fbp.select_angles(a, 'full_halves'), first)
    assert fbp.halves_count(a, 'first_180') == 1
    assert fbp.halves_count(a, 'full_halves') == 1


def test_select_angles_full_turn():
    a = np.arange(0, 360, 0.5)                        # 0..359.5
    assert fbp.select_angles(a, 'full_halves').all()
    assert fbp.halves_count(a, 'full_halves') == 2
    assert fbp.select_angles(a, 'first_180').sum() == 360
    assert fbp.halves_count(a, 'first_180') == 1


@pytest.mark.parametrize('angles, k, n_sel', [
    (np.arange(0, 360, 0.1, dtype='float32'), 2, 3600),   # шаг, непредставимый в float32
    (np.arange(10, 370, 1.0), 2, 360),                    # скан начинается не с нуля
    (np.arange(0, 120, 1.0), 1, 120),                     # меньше полуоборота — все углы
    (np.arange(0, 540, 2.0), 3, 270),
    (np.arange(0, 400, 1.0), 2, 360),
])
def test_halves_count_cases(angles, k, n_sel):
    assert fbp.halves_count(angles, 'full_halves') == k
    assert fbp.select_angles(angles, 'full_halves').sum() == n_sel


def test_select_angles_keeps_old_behaviour_for_first_180():
    a = np.array([5.0, 90.0, 184.9, 185.0, 300.0], dtype='float32')
    np.testing.assert_array_equal(fbp.select_angles(a), (a - a.min()) < 180)


def test_unknown_angle_mode():
    with pytest.raises(ValueError):
        fbp.select_angles(ANGLES_180, 'all')
    with pytest.raises(ValueError):
        fbp.halves_count(ANGLES_180, 'all')


# --- CPU FBP -----------------------------------------------------------------------------------------------

def test_cpu_fbp_equals_iradon_for_odd_width():
    """При нечётной ширине центр iradon (w//2) совпадает с (w−1)/2 — результаты равны."""
    w = 65
    sino = ph.sinogram(ph.make_blobs(seed=1, n=12, r_max=30), ANGLES_180, w)
    ref = iradon(sino.T, theta=ANGLES_180, filter_name='ramp', circle=False, output_size=w)
    rec = fbp.recon_slice(sino, ANGLES_180, 1.0, backend='cpu')
    assert rec.shape == (w, w) and rec.dtype == np.float32
    np.testing.assert_allclose(rec, ref, atol=1e-6 * np.abs(ref).max())


@pytest.mark.parametrize('w', [64, 97])
def test_cpu_fbp_reconstructs_phantom(w):
    blobs = ph.make_blobs(seed=2, n=12, r_max=w / 2 - 4)
    truth = ph.slice_truth(blobs, w, zeta=1.5)
    sino = ph.sinogram(blobs, ANGLES_180, w, zeta=1.5)
    rec = fbp.recon_slice(sino, ANGLES_180, 1.0, backend='cpu')
    assert _corr(rec, truth) > 0.99
    assert rec.max() == pytest.approx(truth.max(), rel=0.1)
    # ось в (w−1)/2: сдвиг синограммы на пиксель заметно портит срез
    shifted = fbp.recon_slice(np.roll(sino, 1, axis=1), ANGLES_180, 1.0, backend='cpu')
    assert _corr(shifted, truth) < _corr(rec, truth) - 0.01


def test_pixel_size_scaling():
    sino = ph.sinogram(ph.asymmetric_blobs(), ANGLES_180, 64)
    r1 = fbp.recon_slice(sino, ANGLES_180, 1.0, backend='cpu')
    r2 = fbp.recon_slice(sino, ANGLES_180, 0.5, backend='cpu')
    np.testing.assert_allclose(r2, 2 * r1, rtol=1e-6, atol=1e-9)
    with pytest.raises(ValueError):
        fbp.recon_slice(sino, ANGLES_180, 0.0, backend='cpu')


def test_full_halves_scale_matches_first_180():
    blobs = ph.asymmetric_blobs()
    w = 80
    a360 = np.arange(0, 360, 1.0)
    sino = ph.sinogram(blobs, a360, w, zeta=3.0)
    half = fbp.recon_slice(sino, a360, 1.0, backend='cpu', angle_mode='first_180')
    full = fbp.recon_slice(sino, a360, 1.0, backend='cpu', angle_mode='full_halves')
    assert full.mean() / half.mean() == pytest.approx(1.0, abs=0.01)
    assert _corr(full, half) > 0.999
    truth = ph.slice_truth(blobs, w, zeta=3.0)
    assert _corr(full, truth) > 0.99


def test_recon_rows_matches_slices():
    blobs = ph.asymmetric_blobs()
    rows = np.stack([ph.sinogram(blobs, ANGLES_180, 48, zeta=z) for z in (-5.0, 0.0, 7.0)])
    out = fbp.recon_rows(rows, ANGLES_180, 0.01, backend='cpu')
    assert out.shape == (3, 48, 48) and out.dtype == np.float32
    for i in range(3):
        np.testing.assert_allclose(out[i], fbp.recon_slice(rows[i], ANGLES_180, 0.01, backend='cpu'),
                                   rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError):
        fbp.recon_rows(rows[:, :-1], ANGLES_180, 1.0, backend='cpu')
    with pytest.raises(ValueError):
        fbp.recon_slice(rows, ANGLES_180, 1.0, backend='cpu')


# --- бэкенды -----------------------------------------------------------------------------------------------

def test_auto_backend_is_cpu_without_gpu():
    assert fbp.resolve_backend('auto') == 'cpu'
    assert fbp.resolve_backend('cpu') == 'cpu'
    assert fbp.resolve_backend('astra') == 'astra'
    with pytest.raises(ValueError):
        fbp.resolve_backend('gpu')


def test_auto_backend_needs_gpu_and_astra(monkeypatch):
    monkeypatch.setattr(gpu, 'is_gpu', lambda xp=None: True)
    assert fbp.resolve_backend('auto') == 'astra'
    monkeypatch.setitem(sys.modules, 'tomo.recon.astra_utils', None)   # импорт упадёт
    monkeypatch.delattr(sys.modules['tomo.recon'], 'astra_utils', raising=False)
    assert not fbp.astra_available()
    assert fbp.resolve_backend('auto') == 'cpu'


def test_astra_backend_calls_astra_like_old_code(monkeypatch):
    """backend='astra': astra_recon_2d_parallel(sino, углы в градусах, [['FBP_CUDA']]) на каждую строку и
    полуоборот, среднее по полуоборотам, деление на pixel_size."""
    calls = []

    def fake_recon(sino, angles, method):
        calls.append((np.array(sino), np.array(angles), method))
        return np.full((sino.shape[1], sino.shape[1]), float(len(calls)), dtype='float32')

    from tomo.recon import astra_utils
    monkeypatch.setattr(astra_utils, 'astra_recon_2d_parallel', fake_recon)
    a = np.arange(0, 360, 2.0)
    sino_rows = np.random.default_rng(0).random((2, len(a), 16)).astype('float32')
    out = fbp.recon_rows(sino_rows, a, 0.5, backend='astra', angle_mode='full_halves')
    assert len(calls) == 4                                     # 2 строки × 2 полуоборота
    assert all(c[2] == [['FBP_CUDA']] for c in calls)
    # порядок вызовов: полуоборот 0 (строки 0, 1), затем полуоборот 1 (строки 0, 1)
    np.testing.assert_array_equal(calls[0][1], a[a < 180])
    np.testing.assert_array_equal(calls[2][1], a[a >= 180])
    np.testing.assert_array_equal(calls[0][0], sino_rows[0][a < 180])
    np.testing.assert_array_equal(calls[1][0], sino_rows[1][a < 180])
    np.testing.assert_array_equal(calls[3][0], sino_rows[1][a >= 180])
    assert out.shape == (2, 16, 16)
    np.testing.assert_allclose(out[0], (1 + 3) / 2 / 0.5)
    np.testing.assert_allclose(out[1], (2 + 4) / 2 / 0.5)

    calls.clear()
    out1 = fbp.recon_slice(sino_rows[0], a, 0.5, backend='astra')     # first_180
    assert len(calls) == 1 and out1.shape == (16, 16)
    np.testing.assert_allclose(out1, 1 / 0.5)


def test_astra_stub_error_propagates():
    with pytest.raises(RuntimeError, match='astra'):
        fbp.recon_slice(np.ones((10, 8), 'float32'), np.arange(10.0), 1.0, backend='astra')
