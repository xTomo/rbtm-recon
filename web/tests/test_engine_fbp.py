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


# --- фрагмент среза ----------------------------------------------------------------------------------------

@pytest.mark.parametrize('w', [48, 51])
def test_cpu_region_equals_crop_of_full(w):
    """Фрагмент (x0, y0, x1, y1): обратная проекция только в его пикселях — те же значения, что обрезка полного
    среза (в т.ч. полуоборотами)."""
    blobs = ph.asymmetric_blobs()
    a360 = np.arange(0, 360, 2.0)
    rows = np.stack([ph.sinogram(blobs, a360, w, zeta=z) for z in (-3.0, 4.0)])
    full = fbp.recon_rows(rows, a360, 0.01, backend='cpu', angle_mode='full_halves')
    for region in ((5, 7, 30, 19), (0, 0, w, 1), (w - 3, w - 5, w, w)):
        x0, y0, x1, y1 = region
        part = fbp.recon_rows(rows, a360, 0.01, backend='cpu', angle_mode='full_halves', region=region)
        assert part.shape == (2, y1 - y0, x1 - x0) and part.dtype == np.float32
        np.testing.assert_allclose(part, full[:, y0:y1, x0:x1], rtol=0, atol=1e-6 * np.abs(full).max())
    assert fbp.recon_slice(rows[0], a360, 0.01, backend='cpu', region=(0, 0, w, w)).shape == (w, w)
    for bad in ((0, 0, w + 1, 5), (5, 5, 5, 9), (-1, 0, 4, 4), (0, 3, 4, 2)):
        with pytest.raises(ValueError):
            fbp.recon_rows(rows, a360, 0.01, backend='cpu', region=bad)


def test_astra_window_default_and_fragment():
    assert fbp.astra_window(64, (0, 0, 64, 64)) == (-32.0, 32.0, -32.0, 32.0)      # окно astra по умолчанию
    # пиксель (строка i, столбец j) окна: центр (min_x + j + 0,5, max_y − i − 0,5) — как у пикселя (y0 + i, x0 + j)
    min_x, max_x, min_y, max_y = fbp.astra_window(64, (10, 20, 30, 24))
    assert (max_x - min_x, max_y - min_y) == (20, 4)                               # шаг пикселя 1
    assert (min_x + 0.5, max_y - 0.5) == (-32 + 10 + 0.5, 32 - 20 - 0.5)


class _FakeAstra:
    """Минимальная astra для проверки окна фрагмента: FBP_CUDA по соглашениям astra (центр пикселя объёма
    (min_x + j + 0,5, max_y − i − 0,5), точка (X, Y) проецируется в t = X·cos θ + Y·sin θ от центра детектора) с тем
    же ramp-фильтром, что у CPU-бэкенда. Совпадение с CPU-бэкендом на полном срезе проверяет соглашения, а фрагмент —
    арифметику окна (реальная astra на GPU — вручную)."""

    def __init__(self):
        self.store = {}
        self.vol_geoms = []
        self.next_id = 0

    def create_vol_geom(self, rows, cols, min_x=None, max_x=None, min_y=None, max_y=None):
        if min_x is None:
            min_x, max_x, min_y, max_y = -cols / 2.0, cols / 2.0, -rows / 2.0, rows / 2.0
        g = {'rows': rows, 'cols': cols, 'window': (min_x, max_x, min_y, max_y)}
        self.vol_geoms.append(g)
        return g

    def astra_dict(self, name):
        assert name == 'FBP_CUDA'
        return {'type': name}

    @property
    def data2d(self):
        fake = self

        class D:
            @staticmethod
            def create(kind, geom, data=None):
                fake.next_id += 1
                key = fake.next_id
                fake.store[key] = (kind, geom, None if data is None else np.array(data, dtype='float32'))
                return key

            @staticmethod
            def get(key):
                return fake.store[key][2]

            @staticmethod
            def delete(key):
                fake.store.pop(key)
        return D

    @property
    def algorithm(self):
        fake = self

        class A:
            @staticmethod
            def create(cfg):
                return cfg

            @staticmethod
            def run(cfg, iterations):
                _, proj, sino = fake.store[cfg['ProjectionDataId']]
                kind, vol, _ = fake.store[cfg['ReconstructionDataId']]
                n, w = sino.shape
                size = max(64, int(2 ** np.ceil(np.log2(2 * w))))
                padded = np.zeros((n, size))
                padded[:, :w] = sino
                filt = np.real(np.fft.ifft(np.fft.fft(padded, axis=-1) * fbp.ramp_filter(size), axis=-1))[:, :w]
                min_x, max_x, min_y, max_y = vol['window']
                px, py = (max_x - min_x) / vol['cols'], (max_y - min_y) / vol['rows']
                xx = (min_x + (np.arange(vol['cols']) + 0.5) * px)[None, :]
                yy = (max_y - (np.arange(vol['rows']) + 0.5) * py)[:, None]
                rec = np.zeros((vol['rows'], vol['cols']))
                for j, th in enumerate(proj['angles']):
                    t = xx * np.cos(th) + yy * np.sin(th) + (w - 1) / 2.0
                    rec += np.interp(t, np.arange(w), filt[j], left=0.0, right=0.0)
                fake.store[cfg['ReconstructionDataId']] = (kind, vol, (rec * np.pi / (2 * n)).astype('float32'))

            @staticmethod
            def delete(cfg):
                pass
        return A


def test_astra_region_window_maps_onto_full_slice(monkeypatch):
    """backend='astra' с region: объём astra — окно фрагмента; на поддельной astra (соглашения astra, см.
    _FakeAstra) фрагмент совпадает с обрезкой полного среза, а полный срез — с CPU-бэкендом."""
    from tomo.recon import astra_utils
    fake = _FakeAstra()
    monkeypatch.setattr(astra_utils, 'astra', fake, raising=False)
    monkeypatch.setattr(astra_utils, 'build_proj_geometry_parallel_2d',
                        lambda det, angles, spacing: {'det': det, 'angles': np.deg2rad(angles)}, raising=False)

    def full_recon(sino, angles, method):
        assert method == [['FBP_CUDA']]
        g = fake.create_vol_geom(sino.shape[1], sino.shape[1])
        sid = fake.data2d.create('-sino', {'angles': np.deg2rad(angles)}, data=sino)
        rid = fake.data2d.create('-vol', g)
        fake.algorithm.run({'ProjectionDataId': sid, 'ReconstructionDataId': rid}, 1)
        rec = fake.data2d.get(rid)
        fake.data2d.delete(rid)
        fake.data2d.delete(sid)
        return rec

    monkeypatch.setattr(astra_utils, 'astra_recon_2d_parallel', full_recon)
    w = 40
    rows = np.stack([ph.sinogram(ph.asymmetric_blobs(), ANGLES_180, w, zeta=z) for z in (0.0, 2.0)])
    full = fbp.recon_rows(rows, ANGLES_180, 0.02, backend='astra')
    cpu = fbp.recon_rows(rows, ANGLES_180, 0.02, backend='cpu')
    np.testing.assert_allclose(full, cpu, rtol=0, atol=1e-4 * np.abs(cpu).max())
    fake.vol_geoms.clear()
    region = (3, 11, 29, 17)
    part = fbp.recon_rows(rows, ANGLES_180, 0.02, backend='astra', region=region)
    assert part.shape == (2, 6, 26)
    assert fake.vol_geoms[0] == {'rows': 6, 'cols': 26, 'window': fbp.astra_window(w, region)}
    np.testing.assert_allclose(part, full[:, 11:17, 3:29], rtol=0, atol=1e-5 * np.abs(full).max())
    assert not fake.store                                          # объекты astra удалены


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
