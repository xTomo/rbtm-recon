"""reconengine.smoothing: сглаживание проекций + деблюринг тем же ядром одним фильтром проекций."""
import math

import numpy as np
import pytest
from scipy import ndimage
from skimage.restoration import wiener
from skimage.transform import radon

from reconengine import fbp, gpu, smoothing as sm

ANGLES = np.arange(0, 180, 1.0)


@pytest.fixture(autouse=True)
def _cpu_backend(monkeypatch):
    monkeypatch.setenv('RECON_ENGINE_CPU', '1')
    gpu.reset_backend()
    yield
    gpu.reset_backend()


def _gauss_psf(sigma):
    r = int(math.ceil(4 * sigma))
    x = np.arange(-r, r + 1)
    g = np.exp(-x * x / (2 * sigma * sigma))
    g /= g.sum()
    return np.outer(g, g)


# --- параметры -----------------------------------------------------------------------------------------------

def test_resolve_off_defaults_and_errors():
    assert sm.resolve(None) is None and sm.resolve({}) is None and sm.resolve(sm.default_block()) is None
    assert sm.resolve({'sigma': 0}) is None
    assert sm.resolve({'sigma': 1.5}) == {'sigma': 1.5, 'deblur': 'wiener', 'balance': 0.02, 'amount': 1.5}
    assert sm.resolve({'sigma': 1, 'deblur': 'unsharp', 'amount': 1.0})['amount'] == 1.0
    for bad in ({'sigma': 0.1}, {'sigma': 5}, {'sigma': 1.5, 'deblur': 'rl'}, {'sigma': 1.5, 'balance': 0},
                {'sigma': 1.5, 'amount': -1}, {'sigma': float('nan')}):
        with pytest.raises(ValueError):
            sm.resolve(bad)


@pytest.mark.parametrize('method', sm.DEBLUR_METHODS)
def test_kernel_normalized_symmetric_and_halo_grows_with_sigma(method):
    hs = []
    for s in (0.7, 1.0, 1.5, 2.0):
        p = sm.resolve({'sigma': s, 'deblur': method})
        k = sm.kernel(p)
        assert k.shape[0] == k.shape[1] == 2 * sm.halo_rows(p) + 1
        assert abs(k.sum() - 1) < 1e-12
        assert np.allclose(k, k[::-1, :]) and np.allclose(k, k[:, ::-1]) and np.allclose(k, k.T)
        hs.append(sm.halo_rows(p))
    assert hs == sorted(hs) and sm.halo_rows(None) == 0


def test_wiener_transfer_is_lowpass_unit_at_dc():
    p = sm.resolve({'sigma': 1.5})
    t = sm.transfer(64, p)
    assert abs(t[0, 0] - 1) < 1e-12 and t.max() <= 1 + 1e-12 and t.min() >= 0


# --- применение ----------------------------------------------------------------------------------------------

def test_slab_with_halo_equals_full_block():
    rng = np.random.default_rng(0)
    x = rng.random((70, 4, 83)).astype('float32')
    for method in sm.DEBLUR_METHODS:
        p = sm.resolve({'sigma': 1.5, 'deblur': method})
        h = sm.halo_rows(p)
        full = sm.apply(x, p, xp=np)
        a, b = 30, 37
        part = sm.apply(x[a - h:b + h], p, keep=(h, h + b - a), xp=np)
        assert np.abs(part - full[a:b]).max() < 1e-5
        top = sm.apply(x[:9 + h], p, keep=(0, 9), xp=np)          # у края кропа — то же отражение
        assert np.abs(top - full[:9]).max() < 1e-5
    assert sm.apply(x, None, keep=(3, 5), xp=np).shape == (2, 4, 83)


def test_none_equals_scipy_gaussian_filter():
    rng = np.random.default_rng(1)
    img = rng.random((60, 70))
    p = sm.resolve({'sigma': 1.5, 'deblur': 'none'})
    got = sm.apply(img[:, None, :].astype('float32'), p, xp=np)[:, 0, :]
    ref = ndimage.gaussian_filter(img, 1.5, mode='mirror', truncate=4.0)      # numpy 'reflect' = scipy 'mirror'
    assert np.abs(got - ref).max() < 3e-3                 # хвост ядра за ±h отброшен (TAIL)


def test_wiener_equals_gauss_then_skimage_wiener_like_mars():
    """«Марс»: gaussian_filter(σ) проекций …, затем skimage wiener(psf = гаусс σ, balance) — одна свёртка."""
    yy, xx = np.mgrid[:96, :96]
    img = ((np.hypot(yy - 48, xx - 40) < 20) * 1.0 + 0.5 * (np.abs(xx - 60) < 3) * (np.abs(yy - 50) < 30))
    p = sm.resolve({'sigma': 1.5, 'deblur': 'wiener', 'balance': 0.02})
    got = sm.apply(img[:, None, :].astype('float32'), p, xp=np)[:, 0, :]
    blurred = ndimage.gaussian_filter(img, 1.5, mode='wrap', truncate=4.0)
    ref = wiener(blurred, _gauss_psf(1.5), 0.02, clip=False)
    c = slice(24, 72)                                    # вдали от краёв (у skimage — циклическое продолжение)
    assert np.abs(got[c, c] - ref[c, c]).max() < 0.02 * np.ptp(img)


def test_projection_filter_equals_volume_filter():
    """Параллельный пучок: фильтр проекций перед FBP ≡ 3D-фильтр с тем же T по восстановленному объёму."""
    rng = np.random.default_rng(2)
    nz, w = 40, 64
    vol = ndimage.gaussian_filter(rng.standard_normal((nz, w, w)), 1.2)
    yy, xx = np.mgrid[:w, :w]
    vol *= (np.hypot(yy - (w - 1) / 2, xx - (w - 1) / 2) < w / 2 - 6)[None]
    proj = np.stack([radon(sl, ANGLES, circle=True).T for sl in vol]).astype('float32')    # (z, n, w)

    p = sm.resolve({'sigma': 1.5, 'deblur': 'wiener'})
    rec_a = fbp.recon_rows(sm.apply(proj, p, xp=np), ANGLES, 1.0, backend='cpu')
    rec0 = fbp.recon_rows(proj, ANGLES, 1.0, backend='cpu')

    # 3D-фильтр того же вида по объёму: H = h(kz)h(ky)h(kx), L — 3D-лапласиан (отражение по краям)
    pad = 24
    v = np.pad(rec0, pad, mode='reflect')
    n3 = v.shape

    def g1(n, real):
        return sm._gauss_spectrum_1d(n, 1.5, real)

    hz, hy, hx = g1(n3[0], False)[:, None, None], g1(n3[1], False)[None, :, None], g1(n3[2], True)[None, None, :]
    kz = np.fft.fftfreq(n3[0])[:, None, None]
    ky = np.fft.fftfreq(n3[1])[None, :, None]
    kx = np.fft.rfftfreq(n3[2])[None, None, :]
    lap = 6 - 2 * np.cos(2 * np.pi * kz) - 2 * np.cos(2 * np.pi * ky) - 2 * np.cos(2 * np.pi * kx)
    h = hz * hy * hx
    t3 = h * h / (h * h + 0.02 * lap * lap)
    rec_b = np.fft.irfftn(np.fft.rfftn(v) * t3, s=n3, axes=(0, 1, 2))[pad:-pad, pad:-pad, pad:-pad]

    inner = (np.hypot(yy - (w - 1) / 2, xx - (w - 1) / 2) < w / 2 - 12)[None]
    zc = slice(12, nz - 12)
    diff = (rec_a - rec_b)[zc] * inner
    ref = rec_b[zc] * inner
    rel = np.sqrt((diff ** 2).sum() / (ref ** 2).sum())
    moved = np.sqrt((((rec0 - rec_b)[zc] * inner) ** 2).sum() / (ref ** 2).sum())
    assert rel < 0.05 and rel < moved / 5, (rel, moved)
