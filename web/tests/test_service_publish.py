"""Тесты публикации результата в хранилище (reconservice.publish): перенос, история, чужие файлы, повтор после
прерывания, перенос между дисками, архивная копия HDF5."""
import errno
import os

import numpy as np
import pytest

from jobs_helpers import EXP, engine_run, scan_and_recipe
from reconengine import gpu
from reconservice import publish
from service_helpers import make_config


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv('RECON_ENGINE_CPU', '1')
    gpu.reset_backend()
    yield
    gpu.reset_backend()


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path)


def volume(dest, doc):
    return np.fromfile(os.path.join(dest, doc['volume']['file']), '<f4').reshape(doc['volume']['shape'])


def test_publish_moves_result_and_keeps_foreign_files(cfg):
    _, recipe = scan_and_recipe(cfg)
    run_dir, doc = engine_run(cfg, EXP, recipe, 'run1', name='обр 1')
    expected = volume(run_dir, doc)
    dest = cfg.reconstruction_dir(EXP)
    os.makedirs(dest)
    foreign = {'rec_config.ini': b'[roi]', 'обр_1.10_10_10.1.raw': b'notebook volume', 'report.html': b'<html/>'}
    for name, body in foreign.items():
        with open(os.path.join(dest, name), 'wb') as fh:
            fh.write(body)

    published = publish.publish(cfg, EXP, run_dir)

    assert published['run_id'] == 'run1'
    assert not os.path.exists(run_dir)                           # пустой run_dir удалён
    names = set(os.listdir(dest))
    assert set(publish.engine_files(doc)) | {'recipe.json', 'result.json'} | set(foreign) == names
    assert 'tomo.обр_1.1.hx' in names                            # .hx полного объёма — по имени raw
    assert np.array_equal(volume(dest, published), expected)
    for name, body in foreign.items():
        with open(os.path.join(dest, name), 'rb') as fh:
            assert fh.read() == body
    assert not os.path.exists(os.path.join(dest, 'history'))
    assert publish.read_published(cfg, EXP) == published


def test_republish_moves_previous_run_to_history(cfg):
    _, recipe = scan_and_recipe(cfg)
    dest = cfg.reconstruction_dir(EXP)
    run1, doc1 = engine_run(cfg, EXP, recipe, 'run1')
    publish.publish(cfg, EXP, run1)
    with open(os.path.join(dest, 'notes.txt'), 'w') as fh:
        fh.write('чужой файл')
    # тот же объём другого размера: у файлов другие имена
    run2, doc2 = engine_run(cfg, EXP, dict(recipe, recon=dict(recipe['recon'], slices=[8, 32])), 'run2')
    publish.publish(cfg, EXP, run2)

    names = set(os.listdir(dest))
    assert names == set(publish.engine_files(doc2)) | {'recipe.json', 'result.json', 'notes.txt', 'history'}
    stale = set(publish.engine_files(doc1)) - set(publish.engine_files(doc2))    # .hx — те же имена
    assert stale and not stale & names
    hist = os.path.join(dest, 'history', 'run1')
    assert sorted(os.listdir(hist)) == ['recipe.json', 'result.json']
    assert publish.read_json(os.path.join(hist, 'result.json'))['run_id'] == 'run1'
    assert publish.read_json(os.path.join(hist, 'recipe.json'))['recon']['slices'] == [4, 36]
    assert publish.read_json(os.path.join(dest, 'recipe.json'))['recon']['slices'] == [8, 32]
    assert publish.read_published(cfg, EXP)['run_id'] == 'run2'


def test_republish_same_names_overwrites(cfg):
    _, recipe = scan_and_recipe(cfg)
    dest = cfg.reconstruction_dir(EXP)
    run1, doc1 = engine_run(cfg, EXP, recipe, 'run1')
    publish.publish(cfg, EXP, run1)
    recipe2 = dict(recipe, axis=dict(recipe['axis'], center_x=recipe['axis']['center_x'] + 3))   # другой объём
    run2, doc2 = engine_run(cfg, EXP, recipe2, 'run2')
    new = volume(run2, doc2)
    publish.publish(cfg, EXP, run2)
    assert publish.engine_files(doc1) == publish.engine_files(doc2)
    for name in publish.engine_files(doc2):
        assert os.path.isfile(os.path.join(dest, name)), name
    assert np.array_equal(volume(dest, doc2), new)
    assert os.listdir(os.path.join(dest, 'history')) == ['run1']


def test_publish_resumes_after_interruption(cfg):
    """Прерванная публикация (часть файлов перенесена, осталась временная копия) повторяется тем же вызовом."""
    _, recipe = scan_and_recipe(cfg)
    dest = cfg.reconstruction_dir(EXP)
    run1, _ = engine_run(cfg, EXP, recipe, 'run1')
    publish.publish(cfg, EXP, run1)
    run2, doc2 = engine_run(cfg, EXP, dict(recipe, recon=dict(recipe['recon'], slices=[8, 32])), 'run2')
    moved = publish.engine_files(doc2)[:2]
    for name in moved:
        os.replace(os.path.join(run2, name), os.path.join(dest, name))
    stale = os.path.join(dest, '.tmp-publish-0000-' + moved[0])
    with open(stale, 'wb') as fh:
        fh.write(b'partial')

    publish.publish(cfg, EXP, run2)

    assert publish.read_published(cfg, EXP)['run_id'] == 'run2'
    assert not os.path.exists(stale)
    for name in publish.engine_files(doc2):
        assert os.path.isfile(os.path.join(dest, name)), name
    assert os.listdir(os.path.join(dest, 'history')) == ['run1']


def test_publish_across_devices(cfg, monkeypatch):
    """os.replace между дисками невозможен (EXDEV) → копия во временное имя рядом с назначением и замена."""
    _, recipe = scan_and_recipe(cfg)
    run_dir, doc = engine_run(cfg, EXP, recipe, 'run1')
    expected = volume(run_dir, doc)
    real_replace = os.replace
    run_abs = os.path.abspath(run_dir)
    crossed = []

    def replace(src, dst):
        if os.path.dirname(os.path.abspath(src)) == run_abs:
            crossed.append(os.path.basename(src))
            raise OSError(errno.EXDEV, 'Invalid cross-device link')
        return real_replace(src, dst)

    monkeypatch.setattr(publish.os, 'replace', replace)
    publish.publish(cfg, EXP, run_dir)
    dest = cfg.reconstruction_dir(EXP)
    assert set(crossed) == set(publish.engine_files(doc)) | {'recipe.json', 'result.json'}
    assert not os.path.exists(run_dir)
    assert not [n for n in os.listdir(dest) if n.startswith('.tmp-publish-')]
    assert np.array_equal(volume(dest, doc), expected)


def test_publish_incomplete_run_fails_without_changes(cfg):
    _, recipe = scan_and_recipe(cfg)
    run_dir, doc = engine_run(cfg, EXP, recipe, 'run1')
    os.remove(os.path.join(run_dir, doc['binned'][0]['raw']))
    with pytest.raises(FileNotFoundError):
        publish.publish(cfg, EXP, run_dir)
    assert not os.path.exists(os.path.join(cfg.reconstruction_dir(EXP), 'result.json'))
    os.remove(os.path.join(run_dir, 'result.json'))
    with pytest.raises(FileNotFoundError):
        publish.publish(cfg, EXP, run_dir)


def test_engine_files_names():
    doc = {'volume': {'file': 'обр.4_5_6.1.raw', 'shape': [4, 5, 6]},
           'binned': [{'factor': 4, 'raw': 'обр.1_1_1.4.raw', 'hx': 'tomo.обр.4.hx', 'shape': [1, 1, 1]},
                      {'factor': 2, 'raw': '../evil.raw', 'hx': 'a/b.hx'}, 'мусор']}
    assert publish.engine_files(doc) == ['обр.4_5_6.1.raw', 'обр.4_5_6.1.raw.size', 'tomo.обр.1.hx',
                                         'обр.1_1_1.4.raw', 'обр.1_1_1.4.raw.size', 'tomo.обр.4.hx']
    assert publish.binned_files(doc) == ['обр.1_1_1.4.raw', 'обр.1_1_1.4.raw.size', 'tomo.обр.4.hx']
    # имя raw не по схеме — .hx полного объёма не угадываем
    assert publish.engine_files({'volume': {'file': 'x.raw', 'shape': [1, 2, 3]}}) == ['x.raw', 'x.raw.size']
    for bad in ('', '.', '..', '../x', 'a\\b', None, 5):
        assert not publish.safe_name(bad)


def test_archive_h5(cfg):
    src = cfg.scan_path(EXP)
    with pytest.raises(FileNotFoundError):
        publish.archive_h5(cfg, EXP)
    os.makedirs(os.path.dirname(src))
    with open(src, 'wb') as fh:
        fh.write(b'hdf5' * 1000)
    os.utime(src, (1_600_000_000, 1_600_000_000))
    assert publish.archive_h5(cfg, EXP) is True
    dst = os.path.join(cfg.storage_dir, EXP + '.h5')
    with open(dst, 'rb') as fh:
        assert fh.read() == b'hdf5' * 1000
    assert int(os.path.getmtime(dst)) == 1_600_000_000
    with open(src, 'wb') as fh:
        fh.write(b'changed')
    assert publish.archive_h5(cfg, EXP) is False                 # копия уже есть — не трогаем
    with open(dst, 'rb') as fh:
        assert fh.read() == b'hdf5' * 1000
    assert sorted(os.listdir(cfg.storage_dir)) == [EXP + '.h5']  # временных файлов нет
