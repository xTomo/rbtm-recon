"""reconengine.rings: пресеты версий 1 и 2, вызов remove_all_stripe (реальная функция — на cupy, здесь подменяется),
БПФ-фильтр полос, подавление с учётом сдвигов кадров."""
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


# --- пресеты -----------------------------------------------------------------------------------------------------

def test_resolve_presets_v2():
    assert rings.resolve('off') is None
    assert rings.resolve() == rings.resolve('medium')
    med = rings.resolve('medium')
    assert med == {'vo': {'snr': 3.0, 'la_size': 61, 'sm_size': 21}, 'fft': {'period': 50.0, 'n': 8, 'v': 2},
                   'unshift': True}
    assert rings.resolve('weak') == {'vo': None, 'fft': {'period': 50.0, 'n': 8, 'v': 1}, 'unshift': True}
    strong = rings.resolve('strong')
    assert strong['vo']['snr'] == 2.0 and strong['fft']['v'] == 3
    assert set(rings.DESCRIPTIONS) == set(rings.PRESETS)


def test_resolve_v1_is_old_vo_only():
    assert rings.resolve('off', version=1) is None
    assert rings.resolve('medium', version=1) == {'vo': {'snr': 3.0, 'la_size': 61, 'sm_size': 21}, 'fft': None,
                                                  'unshift': False}
    assert rings.resolve('weak', version=1)['vo']['snr'] > rings.resolve('strong', version=1)['vo']['snr']
    p = rings.resolve('off', params={'snr': '2.5', 'la_size': 51.0, 'sm_size': 11}, version=1)
    assert p['vo'] == {'snr': 2.5, 'la_size': 51, 'sm_size': 11} and isinstance(p['vo']['la_size'], int)


def test_resolve_explicit_params_v2():
    p = rings.resolve('off', params={'vo': None, 'fft': {'period': 30, 'n': 4, 'v': 2}})
    assert p == {'vo': None, 'fft': {'period': 30.0, 'n': 4, 'v': 2}, 'unshift': True}
    with pytest.raises(ValueError, match='period'):
        rings.resolve('off', params={'fft': {'period': 0.5, 'n': 4, 'v': 2}})
    assert rings.resolve('strong', params={}) == rings.resolve('strong')     # пустые params — пресет
    assert rings.resolve('medium', params={'vo': None, 'fft': None}) is None


def test_resolve_errors():
    with pytest.raises(ValueError, match='пресет'):
        rings.resolve('extreme')
    with pytest.raises(ValueError, match='пресет'):
        rings.resolve('extreme', version=1)
    with pytest.raises(ValueError, match='версия'):
        rings.resolve('medium', version=3)


# --- применение --------------------------------------------------------------------------------------------------

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
    out = rings.apply(sino_rows, rings.resolve('medium', version=1))
    assert len(calls) == 1
    c = calls[0]
    assert c['shape'] == (n, s, w)                                   # [проекции, строки, столбцы]
    assert c['dtype'] == np.float32 and c['contiguous']
    assert (c['snr'], c['la_size'], c['sm_size'], c['dim']) == (3.0, 61, 21, 1)
    np.testing.assert_allclose(c['data'], np.swapaxes(sino_rows, 0, 1), rtol=1e-6)
    assert out.shape == (s, n, w)
    np.testing.assert_allclose(out, sino_rows + np.arange(n)[None, :, None], rtol=1e-6)


def sinogram_with_stripes(n=180, w=256, seed=0, partial=True, r_min=5.0):
    """Синограмма гладкого объекта (сумма «пятен» на синусоидах) + полосы детектора: постоянные и (partial) на части
    углов. Возвращает (синограмма, полосы)."""
    rng = np.random.default_rng(seed)
    th = np.radians(np.arange(n) * 180.0 / n)
    x = np.arange(w)
    sino = np.zeros((n, w))
    for _ in range(12):
        r, ph, a, s = rng.uniform(r_min, 80), rng.uniform(0, 2 * np.pi), rng.uniform(0.2, 1.0), rng.uniform(3, 9)
        pos = w / 2 + r * np.cos(th - ph)
        sino += a * np.exp(-0.5 * ((x[None, :] - pos[:, None]) / s) ** 2)
    stripes = np.zeros((n, w))
    for col in rng.choice(np.arange(20, w - 20), 10, replace=False):
        stripes[:, col] += rng.uniform(0.03, 0.08)
    if partial:
        stripes[: n // 2, w // 2 + 7] += 0.08                        # полоса на половине углов (дуга)
    return sino.astype('float32'), stripes.astype('float32')


def thin(a):
    """Тонкая по x составляющая (минус скользящее среднее по 9 столбцам) — то, что даёт видимые тонкие кольца;
    широкую размытую часть полосы БПФ-фильтр оставляет (её берёт Vo в пресетах «средне» и «сильно»)."""
    import scipy.ndimage as ndi
    return a - ndi.uniform_filter1d(a, 9, axis=-1)


def test_stripe_fft_removes_stripes_keeps_object():
    # фильтр линейный: действие на полосы и на объект проверяется по отдельности
    _, st = sinogram_with_stripes(w=1024, partial=False)
    left = rings.stripe_fft(st[None], 50, 8, 2)[0]
    assert left.shape == st.shape and left.dtype == np.float32
    assert np.std(thin(left)) < 0.1 * np.std(thin(st))               # тонкие полные полосы убраны
    # неполная полоса (дуга): чем больше v, тем меньше остаётся
    _, stp = sinogram_with_stripes(w=1024, partial=True)
    arc = stp - st
    rest = [np.std(thin(rings.stripe_fft(arc[None], 50, 8, v)[0])) for v in (1, 2, 4)]
    assert rest[0] > rest[1] > rest[2] and rest[2] < 0.6 * np.std(thin(arc))
    # объект (детали дальше 40 px от оси) меняется мало: в среднем < 2 %, у 99 % отсчётов < 15 % максимума. Больше —
    # в точках поворота траекторий: фильтр колец трогает и круговое среднее объекта на мелких масштабах (как Vo);
    # детали у самой оси почти не движутся с углом и для фильтра похожи на полосы
    sino, _ = sinogram_with_stripes(w=1024, r_min=40.0)
    d = np.abs(rings.stripe_fft(sino[None], 50, 8, 2)[0] - sino)
    assert d.mean() < 0.02 * sino.max() and np.percentile(d, 99) < 0.15 * sino.max()


def test_shift_angles_roundtrip_and_direction():
    rng = np.random.default_rng(1)
    # гладкий сигнал: сдвиг на полпикселя в частотной области не восстанавливает частоту Найквиста точно
    x = np.convolve(rng.normal(size=300), np.hanning(25) / 12, mode='same')[None, None, :].repeat(4, axis=1).astype(
        'float32')
    d = np.array([0.0, 2.0, -3.0, 1.5])
    y = rings.shift_angles(x, d)
    np.testing.assert_allclose(y[0, 1, 20:-20], np.roll(x[0, 1], 2)[20:-20], atol=1e-4)   # вправо на 2
    np.testing.assert_allclose(y[0, 2, 20:-20], np.roll(x[0, 2], -3)[20:-20], atol=1e-4)
    back = rings.shift_angles(y, -d)
    np.testing.assert_allclose(back[0, :, 20:-20], x[0, :, 20:-20], atol=1e-4)


def test_apply_with_frame_shifts_equals_cleaning_before_shift():
    """Полосы — в координатах детектора; кадры затем сдвинуты на d. С frame_dx поправка считается на проекциях без
    сдвига — как если бы кольца подавлялись до сдвига; без frame_dx полосы «гуляют» и остаются."""
    sino, st = sinogram_with_stripes(partial=False)
    n = sino.shape[0]
    d = 6.0 * np.sin(np.linspace(0, 3 * np.pi, n))                   # смещение образца по кадрам
    raw = (sino + st)[None]
    shifted = rings.shift_angles(raw, d)
    params = rings.resolve('weak')                                    # только БПФ — работает на CPU
    before = rings.shift_angles(rings.apply(raw, params), d)          # эталон: кольца до сдвига
    fixed = rings.apply(shifted, params, frame_dx=d)
    naive = rings.apply(shifted, dict(params, unshift=False), frame_dx=d)
    np.testing.assert_allclose(fixed[0, :, 20:-20], before[0, :, 20:-20], atol=2e-3)
    # фильтр линейный — на одних полосах видно, что без возврата сдвига «гуляющие» полосы остаются
    st_shifted = rings.shift_angles(st[None], d)
    left_fixed = rings.apply(st_shifted, params, frame_dx=d)[0, :, 20:-20]
    left_naive = rings.apply(st_shifted, dict(params, unshift=False), frame_dx=d)[0, :, 20:-20]
    assert np.std(thin(left_fixed)) < 0.3 * np.std(thin(left_naive))
    with pytest.raises(ValueError, match='frame_dx'):
        rings.apply(shifted, params, frame_dx=d[:-1])
    # нулевые сдвиги — как без них
    np.testing.assert_allclose(rings.apply(raw, params, frame_dx=np.zeros(n)), rings.apply(raw, params), atol=1e-6)
