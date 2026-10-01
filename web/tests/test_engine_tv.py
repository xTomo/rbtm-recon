"""reconengine.tv: TV-шумоподавление (FGP) — то же решение ROF, что у skimage; плитки и порции по z."""
import numpy as np
import pytest
from skimage.restoration import denoise_tv_chambolle

from reconengine import gpu, tv


@pytest.fixture(autouse=True)
def _cpu_backend(monkeypatch):
    monkeypatch.setenv('RECON_ENGINE_CPU', '1')
    gpu.reset_backend()
    yield
    gpu.reset_backend()


def _vol(shape=(10, 40, 48), seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(shape, np.float32)
    x[:, 10:30, 12:36] = 1.0
    x[3:7, 15:25, 20:28] += 0.5
    return x + 0.3 * rng.standard_normal(shape).astype(np.float32)


def _rof_energy(u, f, w, nd):
    """½‖u − f‖² + w·Σ|∇u| (разности вперёд, 0 на последнем отсчёте; nd = 2 — без оси 0)."""
    u = np.asarray(u, np.float64)
    axes = range(u.ndim - nd, u.ndim)
    g2 = sum(np.diff(u, axis=a, append=np.take(u, [-1], axis=a)) ** 2 for a in axes)
    return 0.5 * ((u - f) ** 2).sum() + w * np.sqrt(g2).sum()


@pytest.mark.parametrize('nd', [3, 2])
def test_solves_rof_like_skimage(nd):
    """То же решение задачи ROF, что у skimage: энергия не выше (skimage останавливается чуть раньше оптимума),
    отличие ≤ 1 % перепада; 50 итераций — уже близко в среднем."""
    v = _vol()
    if nd == 3:
        ref = denoise_tv_chambolle(v, weight=0.2, eps=1e-9, max_num_iter=5000)
    else:
        ref = np.stack([denoise_tv_chambolle(s, weight=0.2, eps=1e-9, max_num_iter=5000) for s in v])
    got = tv.denoise(v, 0.2, iterations=1500, ndim=nd, xp=np)
    assert got.dtype == np.float32 and got.shape == v.shape
    assert _rof_energy(got, v, 0.2, nd) <= _rof_energy(ref, v, 0.2, nd) + 1e-3
    assert np.abs(got - ref).max() < 0.01
    assert np.abs(tv.denoise(v, 0.2, iterations=50, ndim=nd, xp=np) - ref).mean() < 0.01


def test_2d_input_and_errors():
    v = _vol()[0]
    assert np.abs(tv.denoise(v, 0.2, iterations=300, xp=np) - tv.denoise(v[None], 0.2, 300, ndim=2, xp=np)[0]).max() == 0
    for bad in (0, -1, float('nan')):
        with pytest.raises(ValueError):
            tv.denoise(v, bad, xp=np)
    with pytest.raises(ValueError):
        tv.denoise(v, 0.1, ndim=3, xp=np)


def test_reduces_noise_keeps_edges():
    v = _vol()
    clean = np.zeros_like(v)
    clean[:, 10:30, 12:36] = 1.0
    clean[3:7, 15:25, 20:28] += 0.5
    out = tv.denoise(v, 0.3, xp=np)
    assert np.abs(out - clean).mean() < 0.5 * np.abs(v - clean).mean()
    # перепад на границе остаётся резким: средний скачок через край близок к 1
    assert (out[:, 20, 12] - out[:, 20, 11]).mean() > 0.6


def test_denoise_volume_tiles_match_whole():
    v = _vol((8, 90, 100), seed=1)
    whole = tv.denoise(v, 0.2, 50, xp=np)
    tiled = tv.denoise_volume(v, 0.2, 50, xp=np, budget_voxels=8 * 64 * 64)      # плитки 64², ядро 40
    assert tv._tile_side(8, 8 * 64 * 64) == 64 + 2 * tv.HALO or tv._tile_side(8, 8 * 64 * 64) == 64
    assert np.abs(tiled - whole).max() < 0.03 * 0.3                             # ≪ σ шума 0,3


class _Sink:
    def __init__(self, shape):
        self.vol = np.full(shape, np.nan, np.float32)
        self.next = 0
        self.closed = self.aborted = False

    def write(self, z0, slab):
        assert z0 == self.next
        self.vol[z0:z0 + len(slab)] = slab
        self.next += len(slab)

    def close(self):
        self.closed = True
        return {'ok': 1}

    def abort(self):
        self.aborted = True


@pytest.mark.parametrize('pieces', [[7, 7, 7, 7, 2], [30], [1] * 30, [13, 17]])
def test_denoise_writer_chunks_match_whole(pieces):
    v = _vol((30, 36, 40), seed=2)
    whole = tv.denoise(v, 0.2, 50, xp=np)
    sink = _Sink(v.shape)
    seen = []
    w = tv.DenoiseWriter(sink, {'weight': 0.2, 'iterations': 50}, nz=30, chunk=5, xp=np,
                         on_chunk=lambda z0, r: seen.append((z0, len(r))))
    z = 0
    for k in pieces:
        w.write(z, v[z:z + k])
        z += k
    assert w.close() == {'ok': 1} and sink.closed and sink.next == 30
    assert [s for s, _ in seen] == list(range(0, 30, 5))
    assert np.abs(sink.vol - whole).max() < 0.03 * 0.3
    with pytest.raises(ValueError):
        tv.DenoiseWriter(_Sink(v.shape), {'weight': 0.2, 'iterations': 5}, nz=30, xp=np).write(3, v[:2])


def test_resolve_block():
    assert tv.resolve(None) is None and tv.resolve(tv.default_block()) is None
    p = tv.resolve({'method': 'tv', 'weight': 0.05})
    assert p == {'method': 'tv', 'strength': 2.0, 'weight': 0.05, 'iterations': 50}
    assert tv.resolve({'method': 'tv', 'strength': 3}, need_weight=False)['weight'] is None
    for bad in ({'method': 'nlm', 'weight': 1}, {'method': 'tv'}, {'method': 'tv', 'weight': 0},
                {'method': 'tv', 'weight': 1, 'strength': 20}, {'method': 'tv', 'weight': 1, 'iterations': 2.5},
                {'method': 'tv', 'weight': 1, 'iterations': 1000}):
        with pytest.raises(ValueError):
            tv.resolve(bad)
    assert tv.chunk_slices(3216, 3216) >= 16 and tv.chunk_slices(100, 100) == 256
