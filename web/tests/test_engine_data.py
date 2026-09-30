"""Тесты reconengine.data: open_scan, ChunkSampler, выборка обзора, синограмма строки, CropLoader."""
import json
import os
import threading
import time
import zlib

import h5py
import numpy as np
import pytest

import engine_synth as es
from reconengine import data
from reconengine.model import ROI, Cancelled, MODE_DATA

REAL_H5 = os.environ.get('RECON_REAL_H5', '')


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv('RECON_ENGINE_CPU', '1')


@pytest.fixture
def adv(tmp_path):
    """advanced, 38 кадров, чанк 7 (последний чанк — 3 кадра)."""
    return es.write_scan(tmp_path / 'adv.h5')


@pytest.fixture
def adv_scan(adv):
    return data.open_scan(adv.path)


def ref_bin(a, b):
    """Эталонный биннинг через float64."""
    if b == 1:
        return a.astype(np.float32)
    h, w = a.shape[-2] // b * b, a.shape[-1] // b * b
    v = a[..., :h, :w].astype(np.float64)
    v = v.reshape(a.shape[:-2] + (h // b, b, w // b, b)).mean(axis=(-3, -1))
    return v.astype(np.float32)


def h5_frames(path, sel):
    with h5py.File(path, 'r') as f:
        return f['images/all'][sel]


# --------------------------------------------------------------------------- open_scan

def test_synth_layout(adv):
    assert adv.n_frames == 38 and adv.n_frames % adv.chunk_frames != 0
    with h5py.File(adv.path, 'r') as f:
        ds = f['images/all']
        assert ds.chunks == (7, 48, 64) and ds.compression == 'gzip' and ds.compression_opts == 4
        assert not ds.shuffle
        np.testing.assert_array_equal(ds[()], adv.images)


def test_open_scan_fields(adv, adv_scan):
    s = adv_scan
    assert s.path == os.path.abspath(adv.path)
    assert (s.n_frames, s.height, s.width) == (38, 48, 64)
    assert s.dtype == 'uint16'
    assert s.chunk_frames == 7
    assert s.compression == 'gzip' and s.shuffle is False and s.fast_path is True
    assert s.angles.dtype == np.float64 and s.angles.shape == (38,)
    np.testing.assert_array_equal(s.angles, adv.angles.astype(np.float64))
    np.testing.assert_array_equal(s.modes, adv.modes)
    np.testing.assert_array_equal(s.frame_numbers, adv.frame_numbers)
    np.testing.assert_array_equal(s.dark_idx, [0, 1, 2])
    np.testing.assert_array_equal(s.empty_idx, adv.idx('empty'))
    np.testing.assert_array_equal(s.data_idx, adv.idx('data'))
    np.testing.assert_array_equal(s.check_idx, adv.idx('data_check'))
    assert len(s.check_idx) == 2 and len(s.data_idx) == 24
    assert s.is_advanced is True and s.series_length == 3 and s.empty_period == 8
    assert s.exp_id == 'synthetic-scan'
    md = s.metadata
    assert md['format_version'] == 'v2' and isinstance(md['format_version'], str)
    assert md['specimen'] == 'образец 50% Ni'
    assert md['experiment_id'] == 'synthetic-scan'
    assert md['is_advanced'] is True and md['pixel_size'] == pytest.approx(4.25e-3)
    assert all(not isinstance(v, bytes) for v in md.values())
    assert len(s.fingerprint) == 64
    assert data.open_scan(adv.path).fingerprint == s.fingerprint
    assert s.data_angle_range == pytest.approx(23 * 7.5)


def test_open_scan_without_mapping_same_indices(tmp_path, adv_scan):
    nm = es.write_scan(tmp_path / 'nomap.h5', with_mapping=False)
    s = data.open_scan(nm.path, exp_id='override')
    assert s.exp_id == 'override'
    for name in ('dark_idx', 'empty_idx', 'data_idx', 'check_idx', 'angles', 'modes', 'frame_numbers'):
        np.testing.assert_array_equal(getattr(s, name), getattr(adv_scan, name))
    assert s.fingerprint != adv_scan.fingerprint  # mapping входит в fingerprint


def test_open_scan_simple(tmp_path):
    sm = es.write_scan(tmp_path / 'simple.h5', advanced=False, n_dark=4, n_empty_series=2, series_length=3,
                       n_data=20, chunk_frames=8)
    s = data.open_scan(sm.path)
    assert s.is_advanced is False and s.series_length == 0 and s.empty_period == 0
    assert len(s.check_idx) == 0
    np.testing.assert_array_equal(s.empty_idx, np.arange(4, 10))
    np.testing.assert_array_equal(data.initial_empty_indices(s), np.arange(4, 10))


def test_open_scan_is_advanced_fallback(tmp_path, adv):
    """Без metadata/is_advanced и series_length: advanced — по наличию data_check, серия — по первому блоку."""
    with h5py.File(adv.path, 'r+') as f:
        del f['metadata/is_advanced']
        del f['metadata/series_length']
    s = data.open_scan(adv.path)
    assert s.is_advanced is True and s.series_length == 3


def test_fingerprint_depends_on_structure(tmp_path, adv_scan):
    other = es.write_scan(tmp_path / 'c10.h5', chunk_frames=10)
    assert data.open_scan(other.path).fingerprint != adv_scan.fingerprint


def test_fast_path_false_with_shuffle(tmp_path):
    sh = es.write_scan(tmp_path / 'shuffle.h5', shuffle=True)
    s = data.open_scan(sh.path)
    assert s.shuffle is True and s.fast_path is False
    assert data.ChunkSampler(s).frame_cost(7) == 7


def test_fast_path_false_without_compression(tmp_path, adv):
    path = str(tmp_path / 'raw.h5')
    with h5py.File(path, 'w') as f, h5py.File(adv.path, 'r') as src:
        for grp in ('timeline', 'metadata'):
            src.copy(grp, f)
        f.create_dataset('images/all', data=adv.images, chunks=(7, 48, 64))
    s = data.open_scan(path)
    assert s.compression is None and s.fast_path is False
    with data.ChunkSampler(s) as smp:
        np.testing.assert_array_equal(smp.read_frame(9), adv.images[9])


# --------------------------------------------------------------------------- ChunkSampler

IDX = [0, 3, 6, 7, 10, 13, 14, 20, 27, 28, 34, 35, 36, 37]   # начало/середина/конец чанков, последний неполный
ROWS = [None, (0, 1), (5, 17), (47, 48), (0, 48)]


@pytest.mark.parametrize('rows', ROWS)
def test_read_frame_matches_h5py(adv, adv_scan, rows):
    y0, y1 = rows if rows else (0, 48)
    ref = h5_frames(adv.path, (slice(None), slice(y0, y1)))
    with data.ChunkSampler(adv_scan) as smp:
        for i in IDX:
            got = smp.read_frame(i, rows=rows)
            assert got.dtype == np.uint16 and got.shape == (y1 - y0, 64)
            np.testing.assert_array_equal(got, ref[i], err_msg='кадр {}'.format(i))


@pytest.mark.parametrize('bin_', [2, 3, 5])
def test_read_frame_bin(adv, adv_scan, bin_):
    with data.ChunkSampler(adv_scan) as smp:
        for i in (0, 13, 37):
            got = smp.read_frame(i, bin=bin_)
            assert got.dtype == np.float32 and got.shape == (48 // bin_, 64 // bin_)
            np.testing.assert_allclose(got, ref_bin(adv.images[i], bin_), rtol=1e-6)
            got = smp.read_frame(i, rows=(3, 20), bin=bin_)
            np.testing.assert_allclose(got, ref_bin(adv.images[i, 3:20], bin_), rtol=1e-6)


def test_read_frame_stops_early(adv, adv_scan, monkeypatch):
    """Для начала чанка распаковывается не больше одного кадра (плюс максимум один вызов decompress)."""
    produced = []
    real = zlib.decompressobj

    class Spy:
        def __init__(self):
            self._d = real()

        def decompress(self, buf, max_length=0):
            out = self._d.decompress(buf, max_length)
            produced.append(len(out))
            return out

        def __getattr__(self, item):
            return getattr(self._d, item)

    monkeypatch.setattr(data.zlib, 'decompressobj', Spy)
    with data.ChunkSampler(adv_scan) as smp:
        smp.read_frame(7, rows=(0, 2))
    assert sum(produced) == 2 * 64 * 2   # две строки первого кадра чанка


def test_read_frame_errors(adv_scan):
    with data.ChunkSampler(adv_scan) as smp:
        with pytest.raises(IndexError):
            smp.read_frame(38)
        with pytest.raises(ValueError):
            smp.read_frame(0, rows=(10, 10))
        with pytest.raises(ValueError):
            smp.read_frame(0, bin=0)


def test_read_frames_order_duplicates_workers(adv, adv_scan):
    order = [37, 0, 10, 10, 3, 36, 21, 7]
    with data.ChunkSampler(adv_scan) as smp:
        a1 = smp.read_frames(order, workers=1)
        a4 = smp.read_frames(order, rows=(2, 30), workers=4)
        ab = smp.read_frames(order, bin=4, workers=3)
        empty = smp.read_frames([], bin=2)
    np.testing.assert_array_equal(a1, adv.images[order])
    np.testing.assert_array_equal(a4, adv.images[order, 2:30])
    np.testing.assert_allclose(ab, ref_bin(adv.images[order], 4), rtol=1e-6)
    assert empty.shape == (0, 24, 32)


def test_read_frames_threads_share_sampler(adv, adv_scan):
    errors = []
    with data.ChunkSampler(adv_scan) as smp:
        def worker(k):
            try:
                for i in range(k, 38, 5):
                    if not np.array_equal(smp.read_frame(i), adv.images[i]):
                        errors.append(i)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        threads = [threading.Thread(target=worker, args=(k,)) for k in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert errors == []


def test_frame_cost(adv_scan):
    with data.ChunkSampler(adv_scan) as smp:
        assert [smp.frame_cost(i) for i in (0, 6, 7, 8, 13, 35, 37)] == [1, 7, 1, 2, 7, 1, 3]


def test_fallback_shuffle_matches(tmp_path, monkeypatch):
    sh = es.write_scan(tmp_path / 'shuffle.h5', shuffle=True)
    s = data.open_scan(sh.path)

    def boom(*a, **k):
        raise AssertionError('быстрый путь не должен использоваться')

    monkeypatch.setattr(data.ChunkSampler, '_stream_rows', boom)
    with data.ChunkSampler(s) as smp:
        for i in IDX:
            np.testing.assert_array_equal(smp.read_frame(i, rows=(4, 9)), sh.images[i, 4:9])
        np.testing.assert_array_equal(smp.read_frames([36, 1, 2, 1]), sh.images[[36, 1, 2, 1]])
        np.testing.assert_allclose(smp.read_frame(20, bin=2), ref_bin(sh.images[20], 2), rtol=1e-6)


def test_chunk_with_filter_mask_goes_through_h5py(adv, monkeypatch):
    """Чанк, записанный без фильтра (filter_mask=1), читается через h5py; остальные — быстро."""
    raw = np.ascontiguousarray(adv.images[7:14]).tobytes()
    with h5py.File(adv.path, 'r+') as f:
        f['images/all'].id.write_direct_chunk((7, 0, 0), raw, filter_mask=1)
    s = data.open_scan(adv.path)
    assert s.fast_path is True
    calls = []
    real = data.ChunkSampler._h5_read

    def spy(self, sel):
        calls.append(sel)
        return real(self, sel)

    monkeypatch.setattr(data.ChunkSampler, '_h5_read', spy)
    with data.ChunkSampler(s) as smp:
        assert smp.frame_cost(8) == 7 and smp.frame_cost(15) == 2
        for i in (6, 7, 9, 13, 14):
            np.testing.assert_array_equal(smp.read_frame(i), adv.images[i])
    assert len(calls) == 3   # кадры 7, 9, 13


# --------------------------------------------------------------------------- выборка

@pytest.fixture
def big(tmp_path):
    """advanced, 90 позиций, чанк 10, 4 empty-серии."""
    return es.write_scan(tmp_path / 'big.h5', n_dark=5, series_length=5, n_empty_series=4, n_data=90,
                         chunk_frames=10, angle_step=2.0, H=16, W=24)


@pytest.mark.parametrize('n', [2, 5, 8, 16])
def test_pick_sample_indices(big, n):
    s = data.open_scan(big.path)
    got = data.pick_sample_indices(s, n)
    assert len(got) == n and len(set(got.tolist())) == n
    assert np.all(s.modes[got] == MODE_DATA)
    ang = s.angles[got]
    assert np.all(np.diff(ang) >= 0)
    da = s.angles[s.data_idx]
    targets = np.linspace(da.min(), da.max(), n)
    half = (da.max() - da.min()) / (n - 1) / 2
    cost = s.data_idx % s.chunk_frames + 1
    for t, i in zip(targets, got):
        assert abs(s.angles[i] - t) <= half + 1e-9
        window = np.abs(da - t) <= half + 1e-9
        assert i % s.chunk_frames + 1 == cost[window].min()


def test_pick_sample_indices_edge_cases(big, adv_scan):
    s = data.open_scan(big.path)
    assert len(data.pick_sample_indices(s, 0)) == 0
    one = data.pick_sample_indices(s, 1)
    assert len(one) == 1 and s.modes[one[0]] == MODE_DATA
    allidx = data.pick_sample_indices(adv_scan, 100)
    np.testing.assert_array_equal(np.sort(allidx), adv_scan.data_idx)
    assert np.all(np.diff(adv_scan.angles[allidx]) >= 0)


def test_pick_prefers_cheap_without_mask(tmp_path):
    """При fast_path=False стоимость одинакова — выбирается ближайший к цели кадр."""
    sh = es.write_scan(tmp_path / 'sh.h5', shuffle=True, n_data=40, angle_step=1.0)
    s = data.open_scan(sh.path)
    got = data.pick_sample_indices(s, 5)
    da = s.angles[s.data_idx]
    for t, i in zip(np.linspace(da.min(), da.max(), 5), got):
        assert abs(s.angles[i] - t) == np.abs(da - t).min()


def test_sample_overview(big):
    s = data.open_scan(big.path)
    steps = []
    ov = data.sample_overview(s, n=6, bin=4, n_dark=3, n_empty=3, workers=4,
                              progress=lambda frac, stage: steps.append((frac, stage)))
    assert ov.bin == 4 and (ov.full_height, ov.full_width) == (16, 24)
    assert ov.dark.shape == (4, 6) and ov.empty.shape == (4, 6) and ov.samples.shape == (6, 4, 6)
    assert ov.dark.dtype == ov.empty.dtype == ov.samples.dtype == np.float32
    np.testing.assert_array_equal(ov.sample_idx, data.pick_sample_indices(s, 6))
    np.testing.assert_array_equal(ov.sample_angles, s.angles[ov.sample_idx])
    imgs = big.images
    # dark 0..4 (стоимость 1..5) → 0, 1, 2; начальная empty 5..9 (6..10) → 5, 6, 7
    np.testing.assert_allclose(ov.dark, np.median(ref_bin(imgs[[0, 1, 2]], 4), axis=0), rtol=1e-6)
    np.testing.assert_allclose(ov.empty, np.median(ref_bin(imgs[[5, 6, 7]], 4), axis=0), rtol=1e-6)
    np.testing.assert_allclose(ov.samples, ref_bin(imgs[ov.sample_idx], 4), rtol=1e-6)
    fr = [f for f, _ in steps]
    assert fr == sorted(fr) and fr[-1] == 1.0 and all(st == 'overview' for _, st in steps)


def test_sample_overview_cheapest_empty_across_chunk(tmp_path):
    """Начальная empty-серия 5..9 при чанке 7: самые дешёвые — 7, 8, 9 (начало второго чанка)."""
    sc = es.write_scan(tmp_path / 'x.h5', n_dark=5, series_length=5, n_empty_series=2, n_data=20,
                       chunk_frames=7, H=16, W=20)
    s = data.open_scan(sc.path)
    ov = data.sample_overview(s, n=4, bin=2, n_dark=2, n_empty=3)
    np.testing.assert_allclose(ov.empty, np.median(ref_bin(sc.images[[7, 8, 9]], 2), axis=0), rtol=1e-6)
    np.testing.assert_allclose(ov.dark, np.median(ref_bin(sc.images[[0, 1]], 2), axis=0), rtol=1e-6)


def test_sample_overview_simple_uses_first_block(tmp_path):
    sm = es.write_scan(tmp_path / 's.h5', advanced=False, n_dark=2, n_empty_series=2, series_length=2,
                       n_data=12, chunk_frames=5, H=12, W=16)
    s = data.open_scan(sm.path)
    ov = data.sample_overview(s, n=3, bin=1, n_empty=10)
    np.testing.assert_allclose(ov.empty, np.median(sm.images[2:6].astype(np.float32), axis=0))
    assert ov.samples.dtype == np.float32 and ov.samples.shape == (3, 12, 16)


def test_read_row_sinogram(adv, adv_scan):
    idx = adv_scan.data_idx
    for row in (0, 23, 47):
        sino = data.read_row_sinogram(adv_scan, idx, row, workers=3)
        assert sino.dtype == np.float32 and sino.shape == (len(idx), 64)
        np.testing.assert_array_equal(sino, h5_frames(adv.path, (slice(None), row))[idx].astype(np.float32))


# --------------------------------------------------------------------------- CropLoader

ROI_A = ROI(5, 50, 7, 40)


def test_crop_matches_h5py(tmp_path, adv, adv_scan, monkeypatch):
    def boom(self, sel):
        raise AssertionError('быстрый путь не должен уходить в h5py')

    monkeypatch.setattr(data.ChunkSampler, '_h5_read', boom)
    steps = []
    loader = data.CropLoader(adv_scan, str(tmp_path / 'cache'))
    crop = loader.load(ROI_A, progress=lambda f, st: steps.append(f), workers=3)
    ref = h5_frames(adv.path, (slice(None), slice(7, 40), slice(5, 50)))
    assert crop.frames.shape == (38, 33, 45) and crop.frames.dtype == np.uint16
    np.testing.assert_array_equal(np.asarray(crop.frames), ref)
    assert crop.fingerprint == adv_scan.fingerprint and crop.roi == ROI_A
    assert crop.path == loader.cache_path(ROI_A) and os.path.basename(crop.path).startswith('crop-')
    meta = json.load(open(crop.path[:-4] + '.json', encoding='utf8'))
    assert meta['complete'] is True and meta['fingerprint'] == adv_scan.fingerprint
    assert meta['roi'] == {'x0': 5, 'x1': 50, 'y0': 7, 'y1': 40} and meta['shape'] == [38, 33, 45]
    assert steps == sorted(steps) and steps[-1] == 1.0 and len(steps) == 6 + 1   # 0.0 + 6 чанков
    del crop


def test_crop_reads_sequentially_in_one_thread(tmp_path, adv, adv_scan, monkeypatch):
    """HDD: сжатые чанки читает один поток подряд по файлу, распаковка — в пуле; в памяти не больше workers + 2
    прочитанных и не записанных чанков."""
    workers = 2
    reads, done = [], []
    orig_read, orig_crop = data.ChunkSampler._read_raw, data.ChunkSampler._crop_from_raw

    def read_raw(self, offset, size):
        reads.append((threading.get_ident(), offset, len(reads) - len(done)))
        return orig_read(self, offset, size)

    def crop_from_raw(self, c, raw, *box):
        time.sleep(0.02)             # распаковка медленнее чтения — чтение упирается в предел буфера
        return orig_crop(self, c, raw, *box)

    monkeypatch.setattr(data.ChunkSampler, '_read_raw', read_raw)
    monkeypatch.setattr(data.ChunkSampler, '_crop_from_raw', crop_from_raw)
    crop = data.CropLoader(adv_scan, str(tmp_path / 'cache')).load(
        ROI_A, progress=lambda f, st: f > 0 and done.append(f), workers=workers)
    np.testing.assert_array_equal(np.asarray(crop.frames), h5_frames(adv.path, (slice(None), slice(7, 40), slice(5, 50))))
    assert len(reads) == 6
    assert len({t for t, _, _ in reads}) == 1
    offsets = [o for _, o, _ in reads]
    assert offsets == sorted(offsets)
    assert max(backlog for _, _, backlog in reads) <= workers + 2
    del crop


@pytest.mark.parametrize('where', ['_read_raw', '_crop_from_raw'])
def test_crop_error_propagates_and_stops_reader(tmp_path, adv_scan, monkeypatch, where):
    """Ошибка чтения (поток чтения) или распаковки (пул) поднимается из load, недописанный файл удаляется,
    поток чтения не остаётся висеть."""
    orig = getattr(data.ChunkSampler, where)
    calls = []

    def failing(self, *args):
        calls.append(1)
        if len(calls) == 3:
            raise OSError('сбой диска')
        return orig(self, *args)

    monkeypatch.setattr(data.ChunkSampler, where, failing)
    loader = data.CropLoader(adv_scan, str(tmp_path / 'cache'))
    with pytest.raises(OSError, match='сбой диска'):
        loader.load(ROI_A, workers=2)
    assert os.listdir(str(tmp_path / 'cache')) == []
    assert not [t for t in threading.enumerate() if t.name == 'crop-reader']


def test_crop_bad_chunk_falls_back_to_h5py(tmp_path, adv, adv_scan, monkeypatch):
    orig_read = data.ChunkSampler._read_raw
    bad = []

    def read_raw(self, offset, size):
        raw = orig_read(self, offset, size)
        if not bad:
            bad.append(offset)
            return b'\0' * len(raw)       # битый zlib-поток первого чанка → h5py
        return raw

    monkeypatch.setattr(data.ChunkSampler, '_read_raw', read_raw)
    crop = data.CropLoader(adv_scan, str(tmp_path / 'cache')).load(ROI_A, workers=3)
    np.testing.assert_array_equal(np.asarray(crop.frames), h5_frames(adv.path, (slice(None), slice(7, 40), slice(5, 50))))
    assert bad
    del crop


@pytest.mark.parametrize('workers', [1, 4])
def test_crop_fallback_shuffle(tmp_path, workers):
    sh = es.write_scan(tmp_path / 'sh.h5', shuffle=True)
    s = data.open_scan(sh.path)
    crop = data.CropLoader(s, str(tmp_path / 'cache')).load(ROI(0, 64, 0, 48), workers=workers)
    np.testing.assert_array_equal(np.asarray(crop.frames), sh.images)
    del crop


def test_crop_cache_reused_without_hdf5(tmp_path, adv_scan, monkeypatch):
    loader = data.CropLoader(adv_scan, str(tmp_path / 'cache'))
    first = np.array(loader.load(ROI_A).frames)
    path = loader.cache_path(ROI_A)
    mtime = os.stat(path).st_mtime_ns

    def boom(*a, **k):
        raise AssertionError('HDF5 не должен читаться')

    monkeypatch.setattr(data, 'ChunkSampler', boom)
    monkeypatch.setattr(data.h5py, 'File', boom)
    steps = []
    again = loader.load(ROI(5, 50, 7, 40, preview_row=8), progress=lambda f, st: steps.append(f))
    np.testing.assert_array_equal(np.asarray(again.frames), first)
    assert again.path == path and os.stat(path).st_mtime_ns == mtime and steps == [1.0]
    del again


def test_crop_new_roi_new_file(tmp_path, adv, adv_scan):
    loader = data.CropLoader(adv_scan, str(tmp_path / 'cache'))
    a = loader.load(ROI_A)
    b = loader.load(ROI(0, 64, 40, 48))
    assert a.path != b.path and os.path.isfile(a.path) and os.path.isfile(b.path)
    np.testing.assert_array_equal(np.asarray(b.frames), adv.images[:, 40:48, :])
    del a, b


def test_crop_incomplete_cache_rebuilt(tmp_path, adv, adv_scan):
    loader = data.CropLoader(adv_scan, str(tmp_path / 'cache'))
    crop = loader.load(ROI_A)
    path = crop.path
    del crop
    meta_path = path[:-4] + '.json'
    meta = json.load(open(meta_path, encoding='utf8'))
    meta['complete'] = False
    json.dump(meta, open(meta_path, 'w', encoding='utf8'))
    with open(path, 'r+b') as fh:
        fh.write(b'\xff' * 100)
    crop = loader.load(ROI_A)
    np.testing.assert_array_equal(np.asarray(crop.frames), adv.images[:, 7:40, 5:50])
    assert json.load(open(meta_path, encoding='utf8'))['complete'] is True
    del crop


def test_crop_fingerprint_mismatch_rebuilt(tmp_path, adv, adv_scan):
    loader = data.CropLoader(adv_scan, str(tmp_path / 'cache'))
    crop = loader.load(ROI_A)
    del crop
    path = loader.cache_path(ROI_A)
    meta_path = path[:-4] + '.json'
    meta = json.load(open(meta_path, encoding='utf8'))
    meta['fingerprint'] = 'other'
    json.dump(meta, open(meta_path, 'w', encoding='utf8'))
    assert loader._open_cached(ROI_A, path) is None


@pytest.mark.parametrize('workers', [1, 3])
def test_crop_cancel(tmp_path, adv_scan, workers):
    loader = data.CropLoader(adv_scan, str(tmp_path / 'cache'))
    cancel = threading.Event()
    steps = []

    def progress(frac, stage):
        steps.append(frac)
        if frac > 0:
            cancel.set()

    with pytest.raises(Cancelled):
        loader.load(ROI_A, progress=progress, cancel=cancel, workers=workers)
    path = loader.cache_path(ROI_A)
    assert not os.path.exists(path) and not os.path.exists(path[:-4] + '.json')
    assert steps[-1] < 1.0
    assert os.listdir(str(tmp_path / 'cache')) == []


def test_crop_cancel_before_start(tmp_path, adv_scan):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        data.CropLoader(adv_scan, str(tmp_path / 'cache')).load(ROI_A, cancel=cancel)


def test_crop_bad_roi(tmp_path, adv_scan):
    with pytest.raises(ValueError):
        data.CropLoader(adv_scan, str(tmp_path / 'cache')).load(ROI(0, 65, 0, 10))


def test_last_partial_chunk_is_stored_full(adv):
    """HDF5 хранит последний неполный чанк целиком: распакованный размер = C·H·W·2."""
    with h5py.File(adv.path, 'r') as f:
        ds = f['images/all']
        mask, raw = ds.id.read_direct_chunk((35, 0, 0))
    assert mask == 0
    dec = zlib.decompress(raw)
    assert len(dec) == 7 * 48 * 64 * 2
    np.testing.assert_array_equal(np.frombuffer(dec, np.uint16)[:3 * 48 * 64].reshape(3, 48, 64), adv.images[35:])


# --------------------------------------------------------------------------- реальный файл

@pytest.mark.slow
@pytest.mark.skipif(not os.path.isfile(REAL_H5), reason='нет реального файла (RECON_REAL_H5)')
def test_real_file():
    t0 = time.perf_counter()
    s = data.open_scan(REAL_H5)
    t_open = time.perf_counter() - t0
    assert (s.n_frames, s.height, s.width, s.chunk_frames) == (452, 2968, 5056, 10)
    assert s.fast_path is True

    with data.ChunkSampler(s) as smp, h5py.File(REAL_H5, 'r') as f:
        ds = f['images/all']
        times = {}
        for i in (20, 125):   # начало чанка и середина
            t0 = time.perf_counter()
            got = smp.read_frame(i)
            times[i] = time.perf_counter() - t0
            np.testing.assert_array_equal(got, ds[i])

    t0 = time.perf_counter()
    ov = data.sample_overview(s, n=16, bin=4)
    t_ov = time.perf_counter() - t0
    print('\nreal: open_scan {:.3f} s, read_frame(20) {:.2f} s, read_frame(125) {:.2f} s, overview {:.2f} s'
          .format(t_open, times[20], times[125], t_ov))
    assert ov.samples.shape == (16, 742, 1264)
    assert t_ov < 5.0
