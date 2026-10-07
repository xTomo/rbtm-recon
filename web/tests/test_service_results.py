"""Тесты /results/*: сведения о результате и история, срезы копии ×4 по трём осям, файлы по белому списку."""
import json
import os

import numpy as np
import pytest

from jobs_helpers import EXP, engine_run, jobs_service, scan_and_recipe
from reconengine import gpu
from reconservice import publish
from service_helpers import HEADERS, decode


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv('RECON_ENGINE_CPU', '1')
    gpu.reset_backend()
    yield
    gpu.reset_backend()


@pytest.fixture
def service(tmp_path, monkeypatch):
    """Сервис с опубликованным результатом движка: (client, cfg, recipe, result.json)."""
    _, client, cfg, _ = jobs_service(tmp_path, monkeypatch)
    _, recipe = scan_and_recipe(cfg)
    run_dir, _ = engine_run(cfg, EXP, recipe, 'run1', name='обр 1')
    doc = publish.publish(cfg, EXP, run_dir)
    return client, cfg, recipe, doc


def binned_volume(cfg, doc):
    b = max(doc['binned'], key=lambda x: x['factor'])
    path = os.path.join(cfg.reconstruction_dir(EXP), b['raw'])
    return np.fromfile(path, '<f4').reshape(b['shape']), b


def get(client, url):
    r = client.get(url, headers=HEADERS)
    data = r.get_data()
    r.close()                    # send_file держит файл открытым до закрытия ответа (Windows)
    return r, data


def test_result_info_and_history(service):
    client, cfg, recipe, doc = service
    r = client.get('/results/' + EXP, headers=HEADERS)
    assert r.status_code == 200
    body = r.get_json()
    assert body['result']['run_id'] == 'run1' and body['history'] == []
    assert body['dir'] == cfg.reconstruction_dir(EXP)
    names = [f['name'] for f in body['files']]
    assert names == ['recipe.json', 'result.json', 'обр_1.8_17_17.4.raw', 'обр_1.8_17_17.4.raw.size',
                     'tomo.обр_1.4.hx']
    assert dict((f['name'], f['size']) for f in body['files'])['обр_1.8_17_17.4.raw'] == 8 * 17 * 17 * 4
    # полный объём через сервис не отдаётся, но о нём сказано: путь относительно хранилища и размер
    vol = body['result']['volume']
    full = {f['name']: f for f in body['full']}
    assert set(full) == {vol['file'], vol['hx']}
    assert full[vol['file']]['rel'] == EXP + '/reconstruction/' + vol['file']
    assert full[vol['file']]['size'] == int(np.prod(vol['shape'])) * 4
    assert vol['file'] not in names

    run2, _ = engine_run(cfg, EXP, dict(recipe, recon=dict(recipe['recon'], slices=[8, 32])), 'run2', name='обр 1')
    publish.publish(cfg, EXP, run2)
    body = client.get('/results/' + EXP, headers=HEADERS).get_json()
    assert body['result']['run_id'] == 'run2'
    assert [(h['run_id'], h['recipe_sha256'], h['has_recipe']) for h in body['history']] == \
        [('run1', doc['recipe_sha256'], True)]

    assert client.get('/results/none', headers=HEADERS).status_code == 404
    assert client.get('/results/..', headers=HEADERS).status_code in (400, 404)
    assert client.get('/results/' + EXP).status_code == 403


@pytest.mark.parametrize('axis', ['z', 'y', 'x'])
def test_slice_of_binned_copy(service, axis):
    client, cfg, _, doc = service
    vol, b = binned_volume(cfg, doc)
    k = 'zyx'.index(axis)
    n = vol.shape[k]
    lo, hi = doc['stats']['p0_1'], doc['stats']['p99_9']
    for i in (None, 0, n - 1):
        url = '/results/{}/slice?axis={}'.format(EXP, axis) + ('' if i is None else '&i={}'.format(i))
        r = client.get(url, headers=HEADERS)
        assert r.status_code == 200, r.get_data(as_text=True)
        arr, meta = decode(r)
        idx = n // 2 if i is None else i
        expected = np.take(vol, idx, axis=k)
        assert arr.shape == expected.shape
        assert np.abs(arr - np.clip(expected, lo, hi)).max() <= (hi - lo) / 65535 * 0.51 + 1e-6
        assert meta['axis'] == axis and meta['i'] == idx and meta['n'] == n
        assert meta['binning'] == 4 and meta['shape'] == b['shape']
        assert meta['voxel_mm'] == pytest.approx(0.009 * 4)
        assert meta['window'] == [lo, hi]


def test_slice_errors(service):
    client, cfg, _, doc = service
    for q in ('axis=w', 'i=8', 'i=-1', 'i=abc'):
        r = client.get('/results/{}/slice?{}'.format(EXP, q), headers=HEADERS)
        assert r.status_code == 400, q
    assert client.get('/results/none/slice', headers=HEADERS).status_code == 404
    b = doc['binned'][0]
    os.remove(os.path.join(cfg.reconstruction_dir(EXP), b['raw']))
    assert client.get('/results/{}/slice'.format(EXP), headers=HEADERS).status_code == 404


def test_slice_does_not_block_republish(service):
    """Срез читается через отображение файла, которое закрывается сразу: публикация может заменить копию
    (на Windows замена отображённого файла невозможна)."""
    client, cfg, recipe, _ = service
    assert client.get('/results/{}/slice?axis=x'.format(EXP), headers=HEADERS).status_code == 200
    recipe2 = dict(recipe, axis=dict(recipe['axis'], center_x=recipe['axis']['center_x'] + 2))
    run2, doc2 = engine_run(cfg, EXP, recipe2, 'run2', name='обр 1')
    new = np.fromfile(os.path.join(run2, doc2['binned'][0]['raw']), '<f4')
    publish.publish(cfg, EXP, run2)
    vol, _ = binned_volume(cfg, doc2)
    assert np.array_equal(vol.ravel(), new)


def test_files_whitelist(service):
    client, cfg, _, doc = service
    dest = cfg.reconstruction_dir(EXP)
    r, data = get(client, '/results/{}/file/result.json'.format(EXP))
    assert r.status_code == 200 and json.loads(data)['run_id'] == 'run1'
    assert 'attachment' in r.headers['Content-Disposition']
    raw = doc['binned'][0]['raw']
    r, data = get(client, '/results/{}/file/{}'.format(EXP, raw))
    assert r.status_code == 200
    with open(os.path.join(dest, raw), 'rb') as fh:
        assert data == fh.read()
    for name in ('recipe.json', raw + '.size', doc['binned'][0]['hx']):
        assert get(client, '/results/{}/file/{}'.format(EXP, name))[0].status_code == 200, name

    with open(os.path.join(dest, 'report.html'), 'w') as fh:
        fh.write('<html/>')
    for name in (doc['volume']['file'], 'tomo.обр_1.1.hx', 'report.html', 'history', '..', 'x.raw'):
        assert get(client, '/results/{}/file/{}'.format(EXP, name))[0].status_code == 404, name
    for url in ('/results/{}/file/../result.json', '/results/{}/file/..%2Fresult.json',
                '/results/{}/file/%2E%2E%2F%2E%2E%2Fexp_src', '/results/{}/file/history/run1/result.json'):
        r, _ = get(client, url.format(EXP))
        assert r.status_code in (400, 404), url
    assert get(client, '/results/none/file/result.json')[0].status_code == 404


def test_recipes_current_and_history(service):
    """Рецепт опубликованного запуска и прежних (history/<run_id>) — JSON и файлом; ошибки — 404/400."""
    client, cfg, recipe, doc = service
    r = client.get('/results/{}/recipes/current'.format(EXP), headers=HEADERS)
    assert r.status_code == 200
    body = r.get_json()
    assert body['run_id'] == 'run1' and body['current'] is True and body['recipe_sha256'] == doc['recipe_sha256']
    assert body['recipe']['input']['exp_id'] == EXP and body['recipe']['recon'] == recipe['recon']
    # опубликованный запуск находится и по своему id
    assert client.get('/results/{}/recipes/run1'.format(EXP), headers=HEADERS).get_json()['current'] is True

    run2, _ = engine_run(cfg, EXP, dict(recipe, recon=dict(recipe['recon'], slices=[8, 32])), 'run2', name='обр 1')
    publish.publish(cfg, EXP, run2)
    cur = client.get('/results/{}/recipes/current'.format(EXP), headers=HEADERS).get_json()
    old = client.get('/results/{}/recipes/run1'.format(EXP), headers=HEADERS).get_json()
    assert cur['run_id'] == 'run2' and cur['recipe']['recon']['slices'] == [8, 32]
    assert old['run_id'] == 'run1' and old['current'] is False and old['recipe']['recon'] == recipe['recon']
    assert old['recipe_sha256'] == doc['recipe_sha256']

    r, data = get(client, '/results/{}/recipes/run1?download=1'.format(EXP))
    assert r.status_code == 200 and 'attachment' in r.headers['Content-Disposition']
    assert '{}.run1.recipe.json'.format(EXP) in r.headers['Content-Disposition']
    assert json.loads(data) == old['recipe']

    for url, code in (('/results/{}/recipes/nope', 404), ('/results/{}/recipes/..', (400, 404)),
                      ('/results/{}/recipes/a%2F..%2Fb', (400, 404)), ('/results/none/recipes/current', 404)):
        st = client.get(url.format(EXP), headers=HEADERS).status_code
        assert st == code or (isinstance(code, tuple) and st in code), url
    assert client.get('/results/{}/recipes/current'.format(EXP)).status_code == 403
    # запуск в истории без рецепта — 404
    os.remove(os.path.join(cfg.reconstruction_dir(EXP), publish.HISTORY, 'run1', publish.RECIPE))
    assert client.get('/results/{}/recipes/run1'.format(EXP), headers=HEADERS).status_code == 404
