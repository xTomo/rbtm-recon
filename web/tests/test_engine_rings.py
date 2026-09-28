"""reconengine.rings: пресеты и вызов remove_all_stripe (реальная функция — на cupy, здесь подменяется)."""
import sys

import numpy as np
import pytest

from reconengine import gpu, rings


@pytest.fixture(autouse=True)
def _cpu_backend(monkeypatch):
    monkeypatch.setenv('RECON_ENGINE_CPU', '1')
    gpu.reset_backend()
    yield
    gpu.reset_backend()


def test_resolve_presets():
    assert rings.resolve('off') is None
    assert rings.resolve() == rings.resolve('medium') == {'snr': 3.0, 'la_size': 61, 'sm_size': 21}
    for name in ('weak', 'medium', 'strong'):
        p = rings.resolve(name)
        assert set(p) == {'snr', 'la_size', 'sm_size'}
        assert isinstance(p['snr'], float) and isinstance(p['la_size'], int) and isinstance(p['sm_size'], int)
    assert rings.resolve('weak')['snr'] > rings.resolve('strong')['snr']


def test_resolve_explicit_params_override_preset():
    p = rings.resolve('off', params={'snr': '2.5', 'la_size': 51.0, 'sm_size': 11})
    assert p == {'snr': 2.5, 'la_size': 51, 'sm_size': 11}
    assert isinstance(p['la_size'], int)
    # пустые params — используется пресет
    assert rings.resolve('strong', params={}) == rings.resolve('strong')


def test_resolve_unknown_preset():
    with pytest.raises(ValueError, match='пресет'):
        rings.resolve('extreme')


def test_apply_without_params_returns_input():
    x = np.ones((2, 3, 4), dtype='float32')
    assert rings.apply(x, None) is x
    assert rings.apply(x, {}) is x


def test_apply_calls_remove_all_stripe_with_projection_major_layout(monkeypatch):
    calls = []

    def fake_remove_all_stripe(tomo, snr=None, la_size=None, sm_size=None, dim=None):
        calls.append({'shape': tomo.shape, 'dtype': tomo.dtype, 'contiguous': tomo.flags['C_CONTIGUOUS'],
                      'snr': snr, 'la_size': la_size, 'sm_size': sm_size, 'dim': dim, 'data': np.array(tomo)})
        # метка номера проекции — чтобы проверить обратную перестановку осей
        return tomo + np.arange(tomo.shape[0], dtype='float32')[:, None, None]

    monkeypatch.setattr(sys.modules['tomo.remove_stripe'], 'remove_all_stripe', fake_remove_all_stripe)
    s, n, w = 3, 5, 7
    sino_rows = np.random.default_rng(0).random((s, n, w))          # float64 на входе
    params = rings.resolve('medium')
    out = rings.apply(sino_rows, params)

    assert len(calls) == 1
    c = calls[0]
    assert c['shape'] == (n, s, w)                                   # [проекции, строки, столбцы]
    assert c['dtype'] == np.float32 and c['contiguous']
    assert (c['snr'], c['la_size'], c['sm_size'], c['dim']) == (3.0, 61, 21, 1)
    np.testing.assert_allclose(c['data'], np.swapaxes(sino_rows, 0, 1), rtol=1e-6)
    assert out.shape == (s, n, w)
    np.testing.assert_allclose(out, sino_rows + np.arange(n)[None, :, None], rtol=1e-6)
