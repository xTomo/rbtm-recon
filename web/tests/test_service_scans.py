"""Тесты /scans/*: сведения о скане, размер пикселя из storage, обзор (кэш в памяти и в npz), огибающая, кадры
выборки, синограмма строки."""
import os
import threading
import time

import numpy as np
import pytest
import requests

from reconengine import data, gpu
from reconservice import scans as scans_mod

from engine_scans import advanced_scan, simple_scan
from service_helpers import HEADERS, decode, make_service, write_scan


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv('RECON_ENGINE_CPU', '1')
    gpu.reset_backend()
    yield
    gpu.reset_backend()


@pytest.fixture
def storage(monkeypatch):
    """Подмена requests.post к storage: по умолчанию «недоступен» сразу (на Windows отказ порта 9 ждёт ~2 с).
    storage.doc = {...} — ответить документом; storage.calls — вызовы."""
    class Storage:
        doc = None
        calls = []

    st = Storage()

    class Resp:
        def __init__(self, body):
            self._body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self._body

    def post(url, **kw):
        st.calls.append((url, kw))
        if st.doc is None:
            raise requests.ConnectionError('storage недоступен')
        return Resp([st.doc])

    monkeypatch.setattr(scans_mod.requests, 'post', post)
    return st


def wide_scan():
    """Объект занимает середину кадра (не весь), с шумом — у авто-ROI есть что отрезать."""
    return simple_scan(height=64, width=160, center_x=70.3, y_ref=31.5, noise=0.3)


@pytest.fixture
def svc(tmp_path, storage):
    app, client, cfg = make_service(tmp_path)
    ss = wide_scan()
    write_scan(cfg, 'exp1', ss)
    return app, client, cfg, ss


def registry(app):
    return app.extensions['recon'].scans


# --- info ----------------------------------------------------------------------------------------------------

def test_info(svc, storage):
    _, client, cfg, ss = svc
    r = client.get('/scans/exp1/info', headers=HEADERS)
    assert r.status_code == 200
    body = r.get_json()
    assert body['shape'] == [ss.scan.n_frames, 64, 160]
    assert body['frames'] == {'dark': 3, 'empty': 4, 'data': 120, 'data_check': 0}
    assert body['advanced'] is False and body['pair_0_180'] is True
    assert body['angles']['min'] == 0.0 and body['angles']['max'] == 357.0 and body['angles']['step'] == 3.0
    assert body['fingerprint'] == data.open_scan(cfg.scan_path('exp1')).fingerprint
    # storage недоступен: размер пикселя из HDF5 (модель детектора неизвестна) и предупреждение
    ps = body['pixel_size']
    assert ps['value_mm'] == 0.009 and ps['source'] == 'hdf5'
    assert any('storage' in w for w in ps['warnings'])
    # запрос к storage — как в ноутбуке, но JSON и с таймаутом; неудача кэшируется (второй /info без запроса)
    url, kw = storage.calls[0]
    assert url == cfg.storage_server + 'storage/experiments/get'
    assert kw['json'] == {'_id': 'exp1'} and kw['timeout'] == 5
    client.get('/scans/exp1/info', headers=HEADERS)
    assert len(storage.calls) == 1


def test_pixel_size_from_storage_document(svc, storage):
    app, client, _, _ = svc
    storage.doc = {'_id': 'exp1', 'pixel_size': 0.005}
    ps = client.get('/scans/exp1/info', headers=HEADERS).get_json()['pixel_size']
    assert ps == {'value_mm': 0.005, 'source': 'mongo', 'warnings': []}
    assert registry(app).pixel_size('exp1', user_value=0.002).source == 'user'
    assert len(storage.calls) == 1


def test_info_advanced_and_missing(tmp_path, storage):
    _, client, cfg = make_service(tmp_path)
    write_scan(cfg, 'adv1', advanced_scan())
    body = client.get('/scans/adv1/info', headers=HEADERS).get_json()
    assert body['advanced'] is True and body['frames']['data_check'] == 2 and body['series_length'] == 3
    assert client.get('/scans/nope/info', headers=HEADERS).status_code == 404
    assert client.get('/scans/nope/overview', headers=HEADERS).status_code == 404


def test_info_cache_follows_file(svc):
    """ScanInfo кэшируется по (путь, размер, mtime): перезапись файла — новые сведения."""
    app, _, cfg, _ = svc
    reg = registry(app)
    a = reg.info('exp1')
    assert reg.info('exp1') is a
    write_scan(cfg, 'exp1', simple_scan())
    st = os.stat(cfg.scan_path('exp1'))
    os.utime(cfg.scan_path('exp1'), ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
    b = reg.info('exp1')
    assert b is not a and b.width == 72


# --- обзор ---------------------------------------------------------------------------------------------------

def test_overview_roi_covers_object(svc):
    _, client, _, ss = svc
    r = client.get('/scans/exp1/overview', headers=HEADERS)
    assert r.status_code == 200
    body = r.get_json()
    assert body['bin'] == 4 and body['n'] == 16 and body['shape'] == [16, 16, 40]
    assert body['full_shape'] == [64, 160]
    assert len(body['sample_angles']) == 16 and body['sample_angles'] == sorted(body['sample_angles'])
    roi = body['roi']
    proj = ss.projections[ss.scan.data_idx]
    cols = np.where(proj.max(axis=(0, 1)) > 0.01)[0]
    rows = np.where(proj.max(axis=(0, 2)) > 0.01)[0]
    assert roi['x0'] <= cols.min() and cols.max() < roi['x1']
    assert roi['y0'] <= rows.min() and rows.max() < roi['y1']
    assert roi['x1'] - roi['x0'] < 0.6 * 160                 # ROI уже кадра: объект в середине
    assert body['angles_outside'] == []
    assert body['pixel_size']['source'] == 'hdf5'


def test_envelope_thumbs_sample(svc):
    _, client, _, _ = svc
    re = client.get('/scans/exp1/envelope', headers=HEADERS)
    env, _ = decode(re)
    r = client.get('/scans/exp1/thumbs', headers=HEADERS)
    thumbs, meta = decode(r)
    assert env.shape == (16, 40) and thumbs.shape == (16, 16, 40)
    assert len(meta['angles']) == 16 and len(meta['indices']) == 16
    # общее окно: кадр выборки отдельно — те же коды, что в thumbs
    s5, m5 = decode(client.get('/scans/exp1/sample/5', headers=HEADERS))
    assert np.array_equal(s5, thumbs[5]) and m5['index'] == meta['indices'][5]
    # огибающая — максимум −ln T по кадрам выборки (с точностью до квантования; вне окна thumbs значения обрезаны)
    step = float(r.headers['X-Scale'])
    lo = max(float(h.headers['X-Offset']) for h in (r, re))
    hi = min(float(h.headers['X-Offset']) + 65535 * float(h.headers['X-Scale']) for h in (r, re))
    inside = (env > lo + 0.01) & (env < hi - 0.01)
    assert inside.mean() > 0.2                               # объект — середина кадра
    assert np.abs(env - thumbs.max(axis=0))[inside].max() < 1e-4 * env.max() + 2 * step
    assert client.get('/scans/exp1/sample/16', headers=HEADERS).status_code == 400
    # другое число углов и биннинг
    small, _ = decode(client.get('/scans/exp1/thumbs?n=4&bin=8', headers=HEADERS))
    assert small.shape == (4, 8, 20)


@pytest.mark.parametrize('query', ['n=0', 'n=65', 'bin=0', 'bin=abc'])
def test_overview_bad_params(svc, query):
    _, client, _, _ = svc
    assert client.get('/scans/exp1/overview?' + query, headers=HEADERS).status_code == 400


def test_overview_cached_in_memory_and_on_disk(svc, monkeypatch, tmp_path):
    app, client, cfg, _ = svc
    calls = []
    orig = data.sample_overview

    def counting(*a, **kw):
        calls.append(1)
        return orig(*a, **kw)

    monkeypatch.setattr(data, 'sample_overview', counting)
    first = client.get('/scans/exp1/overview', headers=HEADERS).get_json()
    client.get('/scans/exp1/envelope', headers=HEADERS)
    client.get('/scans/exp1/thumbs', headers=HEADERS)
    assert len(calls) == 1
    files = [f for f in os.listdir(cfg.fast_exp_dir('exp1')) if f.startswith('overview-')]
    assert len(files) == 1 and files[0].endswith('-n16-b4.npz')

    # «рестарт сервиса»: новый реестр читает npz, HDF5 не распаковывается
    app2, client2, _ = make_service(tmp_path)
    again = client2.get('/scans/exp1/overview', headers=HEADERS).get_json()
    assert len(calls) == 1
    assert again['roi'] == first['roi'] and again['sample_angles'] == first['sample_angles']
    env1, _ = decode(client.get('/scans/exp1/envelope', headers=HEADERS))
    env2, _ = decode(client2.get('/scans/exp1/envelope', headers=HEADERS))
    assert np.array_equal(env1, env2)

    # другие параметры — другой обзор
    client2.get('/scans/exp1/overview?n=8', headers=HEADERS)
    assert len(calls) == 2


def test_overview_concurrent_requests_compute_once(svc, monkeypatch):
    app, _, _, _ = svc
    calls = []
    orig = data.sample_overview

    def slow(*a, **kw):
        calls.append(1)
        time.sleep(0.3)
        return orig(*a, **kw)

    monkeypatch.setattr(data, 'sample_overview', slow)
    reg = registry(app)
    out = []
    threads = [threading.Thread(target=lambda: out.append(reg.overview('exp1'))) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1 and len(out) == 3 and out[0] is out[1] is out[2]


def test_broken_npz_recomputed(svc, monkeypatch):
    app, client, cfg, _ = svc
    reg = registry(app)
    scan = reg.info('exp1')
    path = reg.overview_path('exp1', scan, 16, 4)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(b'not a npz')
    assert client.get('/scans/exp1/overview', headers=HEADERS).status_code == 200
    with np.load(path) as z:                   # перезаписан нормальным
        assert str(z['fingerprint']) == scan.fingerprint


# --- синограмма ----------------------------------------------------------------------------------------------

def test_sinogram(svc):
    _, client, cfg, ss = svc
    r = client.get('/scans/exp1/sinogram?row=30&n=30', headers=HEADERS)
    assert r.status_code == 200
    sino, meta = decode(r)
    assert sino.shape == (30, 160) and meta['row'] == 30 and len(meta['angles']) == 30
    assert meta['angles'] == sorted(meta['angles'])
    scan = data.open_scan(cfg.scan_path('exp1'))
    idx = data.pick_sample_indices(scan, 30)
    assert np.allclose(meta['angles'], scan.angles[idx], atol=1e-3)
    # −ln T строки ≈ линейные интегралы фантома (шум 0,3 отсчёта при 3000 в пучке)
    truth = ss.projections[idx, 30, :]
    assert np.sqrt(np.mean((sino - truth) ** 2)) < 0.01 and np.abs(sino - truth).max() < 0.05

    default, meta = decode(client.get('/scans/exp1/sinogram', headers=HEADERS))
    assert default.shape == (90, 160) and meta['row'] == 32          # по умолчанию середина кадра


@pytest.mark.parametrize('query', ['row=64', 'row=-1', 'n=0', 'n=361', 'row=x'])
def test_sinogram_bad_params(svc, query):
    _, client, _, _ = svc
    assert client.get('/scans/exp1/sinogram?' + query, headers=HEADERS).status_code == 400


def test_sinogram_cached(svc, monkeypatch):
    app, client, _, _ = svc
    client.get('/scans/exp1/sinogram?row=10&n=20', headers=HEADERS)
    monkeypatch.setattr(data, 'ChunkSampler', None)             # повторный запрос не открывает HDF5
    r = client.get('/scans/exp1/sinogram?row=10&n=20', headers=HEADERS)
    assert r.status_code == 200
