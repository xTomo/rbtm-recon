"""Тесты /sessions/*: одна сессия на сервис, владелец, перехват (force), TTL, загрузка кропа в фоне с прогрессом
и отменой. Вычисления превью — test_service_preview.py (помощники оттуда берутся отсюда)."""
import os
import threading
import time

import pytest
import requests

from reconengine import data, gpu
from reconengine.model import Cancelled
from reconservice import preview
from reconservice import scans as scans_mod

from engine_scans import simple_scan
from service_helpers import HEADERS, headers, make_service, write_scan

ROI = {'x0': 2, 'x1': 70, 'y0': 4, 'y1': 36}


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv('RECON_ENGINE_CPU', '1')
    gpu.reset_backend()
    yield
    gpu.reset_backend()


@pytest.fixture(autouse=True)
def _no_storage(monkeypatch):
    """storage «недоступен» мгновенно (размер пикселя — из HDF5)."""
    def post(url, **kw):
        raise requests.ConnectionError('storage недоступен')
    monkeypatch.setattr(scans_mod.requests, 'post', post)


@pytest.fixture
def svc(tmp_path):
    app, client, cfg = make_service(tmp_path)
    ss = simple_scan()
    write_scan(cfg, 'exp1', ss)
    write_scan(cfg, 'exp2', ss)
    yield app, client, cfg, ss
    manager(app).stop_reaper()            # отменить незавершённую загрузку
    manager(app).wait_loaded(10)


# --- помощники (их же использует test_service_preview) --------------------------------------------------------

def manager(app):
    return app.extensions['recon'].sessions


def open_session(client, exp_id='exp1', user='alice', **kw):
    return client.post('/sessions', json=dict(exp_id=exp_id, **kw), headers=headers(user))


def wait_state(client, sid, user='alice', timeout=30.0, states=('ready', 'error', 'open')):
    """Опрашивать состояние, пока оно не станет одним из states: (последнее состояние, увиденные (state, progress))."""
    seen = []
    deadline = time.time() + timeout
    while True:
        st = client.get('/sessions/' + sid, headers=headers(user)).get_json()
        seen.append((st['state'], st['progress'], st['stage']))
        if st['state'] in states or time.time() > deadline:
            return st, seen
        time.sleep(0.01)


def load_ready(client, sid, roi=None, user='alice'):
    r = client.post('/sessions/{}/load'.format(sid), json={'roi': roi or ROI}, headers=headers(user))
    assert r.status_code == 202, r.get_json()
    st, _ = wait_state(client, sid, user)
    assert st['state'] == 'ready', st
    return st


# --- открытие, владелец, перехват ------------------------------------------------------------------------------

def test_open_same_user_same_exp_returns_existing(svc):
    _, client, _, _ = svc
    r = open_session(client)
    assert r.status_code == 200
    s = r.get_json()
    assert s['state'] == 'open' and s['owner'] == 'alice' and s['exp_id'] == 'exp1'
    assert s['scan']['width'] == 72 and s['pixel_size']['source'] == 'hdf5' and s['crop_ready'] is False
    assert open_session(client).get_json()['id'] == s['id']


def test_busy_force_taken_over(svc):
    _, client, _, _ = svc
    sid = open_session(client).get_json()['id']
    r = open_session(client, user='bob')
    assert r.status_code == 409
    body = r.get_json()
    assert body['error'] == 'busy' and body['owner'] == 'alice' and body['exp_id'] == 'exp1'
    assert body['idle_s'] >= 0

    r = open_session(client, 'exp2', user='bob', force=True)
    assert r.status_code == 200
    sid2 = r.get_json()['id']
    assert sid2 != sid
    r = client.get('/sessions/' + sid, headers=headers('alice'))
    assert r.status_code == 410 and r.get_json() == {'error': 'taken_over', 'by': 'bob'}
    assert client.post('/sessions/{}/ping'.format(sid), headers=headers('alice')).status_code == 410
    # чужая — 403, неизвестная — 404
    assert client.get('/sessions/' + sid2, headers=headers('alice')).status_code == 403
    assert client.delete('/sessions/' + sid2, headers=headers('alice')).status_code == 403
    assert client.get('/sessions/nope', headers=headers('bob')).status_code == 404
    # закрытие владельцем — дальше 404
    assert client.delete('/sessions/' + sid2, headers=headers('bob')).status_code == 200
    assert client.get('/sessions/' + sid2, headers=headers('bob')).status_code == 404
    # сервис свободен
    assert open_session(client, user='carol').status_code == 200


def test_same_user_other_exp_replaces(svc):
    _, client, _, _ = svc
    sid1 = open_session(client, 'exp1').get_json()['id']
    sid2 = open_session(client, 'exp2').get_json()['id']
    assert sid2 != sid1
    assert client.get('/sessions/' + sid1, headers=HEADERS).status_code == 404
    assert client.get('/sessions/' + sid2, headers=HEADERS).get_json()['exp_id'] == 'exp2'


def test_open_errors(svc):
    _, client, _, _ = svc
    assert client.post('/sessions', json={'exp_id': 'exp1'},
                       headers={'X-Recon-Token': HEADERS['X-Recon-Token']}).status_code == 400   # нет пользователя
    assert open_session(client, 'no-such').status_code == 404
    assert open_session(client, '../x').status_code == 400
    assert client.post('/sessions', json={}, headers=HEADERS).status_code == 400


def test_health_and_ttl(svc):
    app, client, cfg, _ = svc
    mgr = manager(app)
    assert client.get('/health').get_json()['session'] == {'active': False}
    sid = open_session(client).get_json()['id']
    h = client.get('/health').get_json()['session']
    assert h['active'] and h['owner'] == 'alice' and h['exp_id'] == 'exp1' and h['state'] == 'open'

    s = mgr._session
    assert mgr.reap(now=s.last_seen + cfg.session_ttl_s - 1) is False
    client.post('/sessions/{}/ping'.format(sid), headers=HEADERS)           # ping продлевает
    t = mgr._session.last_seen
    assert mgr.reap(now=t + cfg.session_ttl_s - 1) is False
    assert mgr.reap(now=t + cfg.session_ttl_s + 1) is True
    assert client.get('/sessions/' + sid, headers=HEADERS).status_code == 404
    assert client.get('/health').get_json()['session'] == {'active': False}


def test_reaper_thread_starts_and_stops(svc):
    app, _, _, _ = svc
    mgr = manager(app)
    mgr.start_reaper()
    assert mgr._reaper.is_alive()
    t0 = time.time()
    mgr.stop_reaper()
    assert time.time() - t0 < 2 and mgr._reaper is None


def test_preview_before_load_not_ready(svc):
    _, client, _, _ = svc
    sid = open_session(client).get_json()['id']
    r = client.get('/sessions/{}/slice'.format(sid), headers=HEADERS)
    assert r.status_code == 409 and r.get_json() == {'error': 'not_ready', 'state': 'open'}
    assert client.post('/sessions/{}/axis/auto'.format(sid), headers=HEADERS).status_code == 409
    assert client.get('/sessions/{}/repositioning'.format(sid), headers=HEADERS).status_code == 409


# --- загрузка ------------------------------------------------------------------------------------------------

def test_load_progress_to_ready(svc, monkeypatch):
    app, client, cfg, _ = svc
    mgr = manager(app)
    seen = []
    orig = data.CropLoader.load

    def recording(self, roi, progress=None, cancel=None, workers=1):
        def prog(frac, stage):
            progress(frac, stage)
            seen.append((mgr._session.progress, mgr._session.stage))
        return orig(self, roi, progress=prog, cancel=cancel, workers=workers)

    monkeypatch.setattr(data.CropLoader, 'load', recording)
    sid = open_session(client).get_json()['id']
    r = client.post('/sessions/{}/load'.format(sid), json={'roi': ROI}, headers=HEADERS)
    assert r.status_code == 202 and r.get_json()['state'] == 'loading'
    st, _ = wait_state(client, sid)
    assert st['state'] == 'ready' and st['progress'] == 1.0 and st['crop_ready'] is True
    assert st['roi'] == dict(ROI, preview_row=20)
    fr = [p for p, _ in seen]
    assert fr == sorted(fr) and 0 < fr[-1] <= 0.85 and {stage for _, stage in seen} == {'crop'}
    assert any(f.startswith('crop-') for f in os.listdir(cfg.cache_dir('exp1')))
    # повторная загрузка того же ROI — из кэша кропа
    seen.clear()
    load_ready(client, sid)


@pytest.mark.parametrize('roi', [{'x0': 0, 'x1': 80, 'y0': 0, 'y1': 10}, {'x0': 5, 'x1': 5, 'y0': 0, 'y1': 10},
                                 {'x0': 1, 'x1': 10}, 'bad'])
def test_load_bad_roi(svc, roi):
    _, client, _, _ = svc
    sid = open_session(client).get_json()['id']
    r = client.post('/sessions/{}/load'.format(sid), json={'roi': roi}, headers=HEADERS)
    assert r.status_code == 400
    assert client.get('/sessions/' + sid, headers=HEADERS).get_json()['state'] == 'open'


def _blocking_first_load(monkeypatch):
    """Первая загрузка висит до отмены (затем Cancelled), следующие — настоящие."""
    orig = data.CropLoader.load
    started = threading.Event()
    cancels = []

    def load(self, roi, progress=None, cancel=None, workers=1):
        cancels.append(cancel)
        if len(cancels) == 1:
            progress(0.5, 'crop')
            started.set()
            assert cancel.wait(10)
            raise Cancelled()
        return orig(self, roi, progress=progress, cancel=cancel, workers=workers)

    monkeypatch.setattr(data.CropLoader, 'load', load)
    return started, cancels


def test_load_cancel(svc, monkeypatch):
    _, client, _, _ = svc
    started, cancels = _blocking_first_load(monkeypatch)
    sid = open_session(client).get_json()['id']
    client.post('/sessions/{}/load'.format(sid), json={'roi': ROI}, headers=HEADERS)
    assert started.wait(5)
    st = client.get('/sessions/' + sid, headers=HEADERS).get_json()
    assert st['state'] == 'loading' and abs(st['progress'] - 0.425) < 1e-6
    r = client.post('/sessions/{}/load/cancel'.format(sid), headers=HEADERS)
    assert r.status_code == 200 and r.get_json()['stage'] == 'canceling'
    st, _ = wait_state(client, sid)
    assert st['state'] == 'open' and st['stage'] == 'canceled' and st['crop_ready'] is False
    assert cancels[0].is_set()


def test_new_load_cancels_current(svc, monkeypatch):
    _, client, _, _ = svc
    started, cancels = _blocking_first_load(monkeypatch)
    sid = open_session(client).get_json()['id']
    client.post('/sessions/{}/load'.format(sid), json={'roi': ROI}, headers=HEADERS)
    assert started.wait(5)
    roi2 = {'x0': 3, 'x1': 69, 'y0': 6, 'y1': 30, 'preview_row': 12}
    st = load_ready(client, sid, roi2)
    assert cancels[0].is_set() and len(cancels) == 2
    assert st['roi'] == roi2


def test_close_cancels_load(svc, monkeypatch):
    app, client, _, _ = svc
    started, cancels = _blocking_first_load(monkeypatch)
    sid = open_session(client).get_json()['id']
    client.post('/sessions/{}/load'.format(sid), json={'roi': ROI}, headers=HEADERS)
    assert started.wait(5)
    assert client.delete('/sessions/' + sid, headers=HEADERS).status_code == 200
    assert manager(app).wait_loaded(5)
    assert cancels[0].is_set() and manager(app)._session is None


def test_load_error_state(svc, monkeypatch):
    _, client, _, _ = svc

    def broken(*a, **kw):
        raise RuntimeError('сломалось')

    monkeypatch.setattr(preview, 'build_context', broken)
    sid = open_session(client).get_json()['id']
    client.post('/sessions/{}/load'.format(sid), json={'roi': ROI}, headers=HEADERS)
    st, _ = wait_state(client, sid)
    assert st['state'] == 'error' and 'сломалось' in st['error']
    assert client.get('/sessions/{}/slice'.format(sid), headers=HEADERS).status_code == 409
