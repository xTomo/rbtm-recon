"""reconengine.noise: шум среза и ошибка фильтра по двум половинам углов (Noise2Noise)."""
import numpy as np
import pytest
from scipy import ndimage

from reconengine import noise


def test_split_halves_interleaved_equal_disjoint():
    a = np.arange(0, 360, 3.0)                       # first_180 — 60 углов
    ev, od = noise.split_halves(a, 'first_180')
    assert len(ev) == len(od) == 30 and not set(ev) & set(od)
    assert list(ev[:3]) == [0, 2, 4] and list(od[:3]) == [1, 3, 5] and a[ev].max() < 180 and a[od].max() < 180
    ev, od = noise.split_halves(a, 'full_halves')
    assert len(ev) == len(od) == 60
    ev, od = noise.split_halves(np.arange(0, 181.0), 'first_180')          # нечётное число — поровну
    assert len(ev) == len(od) == 90
    with pytest.raises(ValueError):
        noise.split_halves(np.array([0.0, 90.0, 180.0]), 'first_180')


def test_pick_smallest_within_tolerance():
    assert noise.pick([0.1, 0.05, 0.03, 0.0305, 0.03]) == (2, 2)
    assert noise.pick([0.1, 0.05, 0.0314, 0.0301, 0.03]) == (2, 4)
    assert noise.pick([0.1, 0.05, 0.0316, 0.0301, 0.03]) == (3, 4)
    assert noise.pick([0.2]) == (0, 0)
    with pytest.raises(ValueError):
        noise.pick([])


def _phantom(n=256):
    rng = np.random.default_rng(1)
    x = np.zeros((n, n))
    for _ in range(40):
        cy, cx = rng.integers(20, n - 20, 2)
        r = rng.integers(4, 18)
        yy, xx = np.ogrid[:n, :n]
        x[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] += rng.uniform(0.5, 2.0)
    return x


def _halves(x, sigma_full, corr=0.0, seed=0):
    """Срезы половин: шум у каждой σ_полн·√2 (белый или сглаженный — коррелированный, как у FBP)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(2):
        n = rng.standard_normal(x.shape)
        if corr:
            n = ndimage.gaussian_filter(n, corr)
            n /= n.std()
        out.append(x + sigma_full * np.sqrt(2) * n)
    return out


def test_noise_sigma_ignores_object_and_edges():
    x = _phantom()
    re, ro = _halves(x, 0.3)
    assert noise.noise_sigma(re, ro) == pytest.approx(0.3, rel=0.03)
    assert noise.robust_std(np.zeros(0)) == 0.0


@pytest.mark.parametrize('corr', [0.0, 1.5])
def test_filter_mse_matches_true_error_of_full_angle_slice(corr):
    """Оценка по половинам = настоящая ошибка фильтра на срезе по всем углам (шум вдвое меньше по дисперсии),
    и белого, и коррелированного шума; без фильтра — дисперсия шума."""
    x = _phantom()
    re, ro = _halves(x, 0.5, corr)
    full = (re + ro) / 2
    mse0, var0 = noise.filter_mse(re, ro, re, ro)
    assert mse0 == pytest.approx(0.25, rel=0.05) and var0 == pytest.approx(0.25, rel=0.05)
    ests, trues = {}, {}
    for s in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        f = lambda a: ndimage.gaussian_filter(a, s)          # noqa: E731
        ests[s], var = noise.filter_mse(f(re), f(ro), re, ro)
        trues[s] = float(np.mean((f(full) - x) ** 2))
        assert ests[s] == pytest.approx(trues[s], rel=0.15, abs=2e-3), s
        assert var == pytest.approx(float(np.mean((f(full) - f(x)) ** 2)), rel=0.1, abs=1e-3)
    # выбранная по оценке σ — почти оптимальна (минимум пологий: соседние σ различаются на проценты)
    assert trues[min(ests, key=ests.get)] <= 1.1 * min(trues.values())
