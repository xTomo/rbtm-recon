"""Тесты reconengine.outputs: запись Amira raw + .hx, биннинг «по ходу», result.json."""
import json
import re

import numpy as np
import pytest

import tomotools4 as t4
from reconengine import outputs as out


# --- sanitize / имена файлов, как в tomotools4 --------------------------------

def test_amira_raw_name_matches_tomotools4():
    assert out.amira_raw_name('обр 1 2', (3, 4, 5), 1) == t4.amira_raw_name('обр 1 2', (3, 4, 5), 1)
    assert out.amira_raw_name('обр 1 2', (3, 4, 5), 1) == 'обр_1_2.3_4_5.1.raw'


def test_sanitize_amira_name_matches_tomotools4():
    assert out.sanitize_amira_name('a b  c') == t4.sanitize_amira_name('a b  c')


# --- hx_text ровно как save_amira ----------------------------------------------

def test_hx_text_matches_save_amira_full(tmp_path):
    vol = np.arange(2 * 3 * 4, dtype='float32').reshape(2, 3, 4)
    t4.save_amira(vol, str(tmp_path), 'образец 1', reshape=1, pixel_size=0.01)

    raw_name = t4.amira_raw_name('образец 1', vol.shape, 1)
    hx_path = tmp_path / 'tomo.образец_1.1.hx'
    expected = hx_path.read_text(encoding='utf-8')

    assert out.hx_text(raw_name, vol.shape, voxel_mm=0.01) == expected


def test_hx_text_matches_save_amira_binned(tmp_path):
    vol = np.ones((4, 4, 4), dtype='float32')
    t4.save_amira(vol, str(tmp_path), 'sample2', reshape=2, pixel_size=0.009)

    binned_shape = (2, 2, 2)
    raw_name = t4.amira_raw_name('sample2', binned_shape, 2)
    hx_path = tmp_path / 'tomo.sample2.2.hx'
    expected = hx_path.read_text(encoding='utf-8')

    assert out.hx_text(raw_name, binned_shape, voxel_mm=0.009 * 2) == expected


# --- VolumeWriter ----------------------------------------------------------------

def _make_volume(shape, seed=1):
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype='float64').astype('float32')


def test_volume_writer_full_and_binned_match_reshape_volume(tmp_path):
    shape = (8, 6, 10)  # nz, ny, nx
    vol = _make_volume(shape)
    writer = out.VolumeWriter(tmp_path, 'sample', shape, voxel_mm=0.0123, binning=(4,))

    # пишем неровными кусками, не кратными фактору биннинга
    writer.write(0, vol[0:3])
    writer.write(3, vol[3:5])
    writer.write(5, vol[5:8])
    result = writer.close()

    assert result['full']['shape'] == list(shape)
    full_path = tmp_path / result['full']['raw']
    full_mm = np.memmap(full_path, dtype='<f4', mode='r', shape=shape)
    assert np.allclose(full_mm, vol)

    size_vals = np.loadtxt(str(full_path) + '.size').astype(int)
    assert tuple(size_vals) == shape

    hx_text = (tmp_path / result['full']['hx']).read_text(encoding='utf-8')
    assert result['full']['raw'] in hx_text

    assert len(result['binned']) == 1
    binned = result['binned'][0]
    assert binned['factor'] == 4
    expected_binned = t4.reshape_volume(vol, 4)
    assert binned['shape'] == list(expected_binned.shape)

    binned_path = tmp_path / binned['raw']
    binned_mm = np.memmap(binned_path, dtype='<f4', mode='r', shape=tuple(binned['shape']))
    assert np.allclose(binned_mm, expected_binned, atol=1e-6)

    binned_hx_text = (tmp_path / binned['hx']).read_text(encoding='utf-8')
    assert binned['raw'] in binned_hx_text


def test_volume_writer_multiple_binning_factors(tmp_path):
    shape = (8, 8, 8)
    vol = _make_volume(shape, seed=2)
    writer = out.VolumeWriter(tmp_path, 'multi', shape, voxel_mm=0.005, binning=(2, 4))
    writer.write(0, vol)
    result = writer.close()

    factors = {b['factor']: b for b in result['binned']}
    assert set(factors) == {2, 4}
    for b, entry in factors.items():
        expected = t4.reshape_volume(vol, b)
        mm = np.memmap(tmp_path / entry['raw'], dtype='<f4', mode='r', shape=tuple(entry['shape']))
        assert np.allclose(mm, expected, atol=1e-6)


def test_volume_writer_single_write_call(tmp_path):
    shape = (4, 4, 4)
    vol = _make_volume(shape, seed=3)
    writer = out.VolumeWriter(tmp_path, 'onego', shape, voxel_mm=0.01, binning=(4,))
    writer.write(0, vol)
    result = writer.close()

    expected = t4.reshape_volume(vol, 4)
    entry = result['binned'][0]
    mm = np.memmap(tmp_path / entry['raw'], dtype='<f4', mode='r', shape=tuple(entry['shape']))
    assert np.allclose(mm, expected, atol=1e-6)


def test_volume_writer_out_of_order_raises(tmp_path):
    shape = (4, 4, 4)
    writer = out.VolumeWriter(tmp_path, 'oop', shape, voxel_mm=0.01, binning=(4,))
    slab = np.zeros((2, 4, 4), dtype='float32')
    writer.write(0, slab)
    with pytest.raises(ValueError):
        writer.write(3, slab)  # пропуск строки 2
    writer.abort()


def test_volume_writer_shape_mismatch_raises(tmp_path):
    shape = (4, 4, 4)
    writer = out.VolumeWriter(tmp_path, 'mismatch', shape, voxel_mm=0.01, binning=(4,))
    bad_slab = np.zeros((2, 5, 5), dtype='float32')
    with pytest.raises(ValueError):
        writer.write(0, bad_slab)
    writer.abort()


def test_volume_writer_overflow_raises(tmp_path):
    shape = (4, 4, 4)
    writer = out.VolumeWriter(tmp_path, 'overflow', shape, voxel_mm=0.01, binning=())
    with pytest.raises(ValueError):
        writer.write(0, np.zeros((5, 4, 4), dtype='float32'))
    writer.abort()


def test_volume_writer_binning_larger_than_volume_yields_no_binned_output(tmp_path):
    shape = (2, 4, 4)
    vol = _make_volume(shape, seed=4)
    writer = out.VolumeWriter(tmp_path, 'short', shape, voxel_mm=0.01, binning=(4,))
    writer.write(0, vol)
    result = writer.close()

    assert result['binned'] == []
    files = sorted(p.name for p in tmp_path.iterdir())
    # только файлы полного объёма — для биннинга 4 при nz=2 файлы не создаются
    assert all('.4.' not in name for name in files)


def test_volume_writer_abort_removes_all_created_files(tmp_path):
    shape = (8, 4, 4)
    writer = out.VolumeWriter(tmp_path, 'aborted', shape, voxel_mm=0.01, binning=(4,))
    writer.write(0, np.zeros((3, 4, 4), dtype='float32'))
    assert any(tmp_path.iterdir())  # файлы уже созданы

    writer.abort()
    assert list(tmp_path.iterdir()) == []


# --- volume_stats -----------------------------------------------------------------

def test_volume_stats_basic():
    sample = np.linspace(0.0, 100.0, 1001, dtype='float32')
    stats = out.volume_stats(sample, bins=100)

    assert stats['min'] == pytest.approx(0.0, abs=1e-4)
    assert stats['max'] == pytest.approx(100.0, abs=1e-4)
    assert stats['p0_1'] < stats['p99_9']
    assert len(stats['hist']['edges']) == 101
    assert len(stats['hist']['counts']) == 100
    assert sum(stats['hist']['counts']) == sample.size


def test_volume_stats_constant_array_does_not_crash():
    sample = np.full((5, 5), 3.0, dtype='float32')
    stats = out.volume_stats(sample, bins=16)
    assert stats['min'] == pytest.approx(3.0)
    assert stats['max'] == pytest.approx(3.0)
    assert sum(stats['hist']['counts']) == sample.size


# --- write_json -----------------------------------------------------------------

def test_write_json_atomic_roundtrip_and_utf8(tmp_path):
    path = tmp_path / 'out.json'
    obj = {'a': 1, 'text': 'кириллица'}
    out.write_json(path, obj)

    assert json.loads(path.read_text(encoding='utf-8')) == obj
    assert 'кириллица' in path.read_text(encoding='utf-8')  # ensure_ascii=False
    leftovers = [p for p in tmp_path.iterdir() if p.name != 'out.json']
    assert leftovers == []


# --- result_document --------------------------------------------------------------

def test_result_document_schema_and_fields():
    files = {
        'full': {'raw': 'vol.4_5_6.1.raw', 'hx': 'tomo.vol.1.hx', 'shape': [4, 5, 6]},
        'binned': [{'factor': 4, 'raw': 'vol.1_1_1.4.raw', 'hx': 'tomo.vol.4.hx', 'shape': [1, 1, 1]}],
    }
    recipe_dict = {'schema': 'rbtm-recon-recipe/1'}
    doc = out.result_document(
        recipe_dict=recipe_dict, recipe_sha='deadbeef', files=files, shape=(4, 5, 6),
        voxel_mm=0.009, stats={'min': 0.0, 'max': 1.0}, timings={'total_s': 12.5},
        warnings=['w1', 'w2'], engine_version='0.1.0', gpu_name='RTX 4090',
    )

    assert doc['schema'] == 'rbtm-recon-result/1'
    assert re.fullmatch(r'[0-9a-f]{32}', doc['run_id'])
    assert doc['recipe_sha256'] == 'deadbeef'
    assert doc['recipe'] == recipe_dict
    assert doc['volume'] == {
        'file': 'vol.4_5_6.1.raw', 'shape': [4, 5, 6], 'dtype': 'float32',
        'byteorder': 'little', 'voxel_mm': 0.009, 'units': '1/mm',
    }
    assert doc['binned'] == files['binned']
    assert doc['stats'] == {'min': 0.0, 'max': 1.0}
    assert doc['timings'] == {'total_s': 12.5}
    assert doc['warnings'] == ['w1', 'w2']
    assert doc['engine'] == {'version': '0.1.0'}
    assert doc['gpu'] == 'RTX 4090'

    # сериализуемо в JSON
    json.dumps(doc, ensure_ascii=False)


def test_result_document_handles_missing_files():
    doc = out.result_document(
        recipe_dict={}, recipe_sha='x', files=None, shape=(1, 2, 3), voxel_mm=0.01,
        stats={}, timings={}, warnings=[], engine_version='0.1.0', gpu_name=None,
    )
    assert doc['volume']['file'] is None
    assert doc['binned'] == []
    assert doc['gpu'] is None
