"""Тесты reconservice.prefetch: предзагрузка исходного HDF5 в кэш ОС и её остановка загрузкой области и задачей."""
import io
import os
import threading
import time

import pytest
import requests

from engine_scans import simple_scan
from jobs_helpers import EXP, jobs_service, scan_and_recipe
from reconengine import gpu
from reconservice import prefetch as prefetch_mod
from reconservice import scans as scans_mod
from reconservice.config import Config
from service_helpers import HEADERS, make_service, write_scan

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
    write_scan(cfg, 'exp1', simple_scan(height=40, width=72))
    write_scan(cfg, 'exp2', simple_scan(height=40, width=72))
    st = app.extensions['recon']
    st.prefetch.meminfo = lambda: None
    yield app, client, cfg, st.prefetch
    st.prefetch.stop('test')
    st.sessions.stop_reaper()             # отменить незавершённую загрузку
    st.sessions.wait_loaded(10)


@pytest.fixture
def slow_read(monkeypatch):
    """Чтение по 1 КБ с паузой 10 мс: предзагрузка идёт секунды, пока тест её не остановит."""
    real_open = open

    class Slow(io.RawIOBase):
        def __init__(self, path):
            self.fh = real_open(path, 'rb', buffering=0)

        def fileno(self):
            return self.fh.fileno()

        def readinto(self, b):
            time.sleep(0.01)
            return self.fh.readinto(memoryview(b)[:1024])

        def close(self):
            self.fh.close()
            super().close()

    monkeypatch.setattr(prefetch_mod, 'open', lambda path, *a, **kw: Slow(path), raising=False)


def wait_state(pf, states, timeout=10.0):
    deadline = time.time() + timeout
    while True:
        st = pf.status()
        if st['state'] in states or time.time() > deadline:
            return st
        time.sleep(0.01)


def test_prefetch_reads_whole_file(svc):
    app, client, cfg, pf = svc
    r = client.post('/scans/exp1/prefetch', headers=HEADERS)
    assert r.status_code == 202
    assert r.get_json()['state'] in ('running', 'done') and r.get_json()['exp_id'] == 'exp1'
    st = wait_state(pf, ('done',))
    size = os.path.getsize(cfg.scan_path('exp1'))
    assert st['state'] == 'done' and st['done_bytes'] == st['total_bytes'] == size
    assert client.get('/health').get_json()['prefetch']['state'] == 'done'

    t = pf._thread
    again = client.post('/scans/exp1/prefetch', headers=HEADERS).get_json()
    assert again['state'] == 'done' and pf._thread is t         # законченная не перезапускается


def test_prefetch_needs_token_and_existing_scan(svc):
    _, client, _, _ = svc
    assert client.post('/scans/exp1/prefetch').status_code == 403
    assert client.post('/scans/nope/prefetch', headers=HEADERS).status_code == 404


def test_prefetch_stopped_by_crop_load(svc, slow_read):
    _, client, _, pf = svc
    assert client.post('/scans/exp1/prefetch', headers=HEADERS).get_json()['state'] == 'running'
    sid = client.post('/sessions', json={'exp_id': 'exp1'}, headers=HEADERS).get_json()['id']
    assert pf.status()['state'] == 'running'                     # открытие сессии не мешает
    deadline = time.time() + 5
    while pf.status()['done_bytes'] == 0 and time.time() < deadline:
        time.sleep(0.005)
    assert client.post('/sessions/{}/load'.format(sid), json={'roi': ROI}, headers=HEADERS).status_code == 202
    st = wait_state(pf, ('stopped',))
    assert st['state'] == 'stopped' and st['reason'] == 'load'
    assert 0 < st['done_bytes'] < st['total_bytes']
    assert not pf._thread.is_alive()


def test_prefetch_same_scan_twice_keeps_running(svc, slow_read):
    _, client, _, pf = svc
    client.post('/scans/exp1/prefetch', headers=HEADERS)
    t = pf._thread
    assert client.post('/scans/exp1/prefetch', headers=HEADERS).get_json()['state'] == 'running'
    assert pf._thread is t


def test_prefetch_other_scan_replaces(svc, slow_read):
    _, client, _, pf = svc
    client.post('/scans/exp1/prefetch', headers=HEADERS)
    t1 = pf._thread
    st = client.post('/scans/exp2/prefetch', headers=HEADERS).get_json()
    assert st['exp_id'] == 'exp2' and st['state'] == 'running'
    assert not t1.is_alive()
    pf.stop('test')
    assert wait_state(pf, ('stopped',))['reason'] == 'test'


def test_prefetch_skipped_when_file_does_not_fit_memory(svc):
    _, client, cfg, pf = svc
    size = os.path.getsize(cfg.scan_path('exp1'))
    pf.meminfo = lambda: size          # файл больше половины доступной памяти
    st = client.post('/scans/exp1/prefetch', headers=HEADERS).get_json()
    assert st['state'] == 'skipped' and 'доступной памяти' in st['reason']
    assert pf._thread is None
    pf.meminfo = lambda: 3 * size
    assert client.post('/scans/exp1/prefetch', headers=HEADERS).get_json()['state'] in ('running', 'done')
    assert wait_state(pf, ('done',))['state'] == 'done'


def test_prefetch_disabled(tmp_path):
    app, client, cfg = make_service(tmp_path, prefetch=False)
    write_scan(cfg, 'exp1', simple_scan(height=40, width=72))
    st = client.post('/scans/exp1/prefetch', headers=HEADERS).get_json()
    assert st['state'] == 'skipped' and 'RECON_PREFETCH' in st['reason']
    assert Config.from_env({'RECON_PREFETCH': '0'}).prefetch is False
    assert Config.from_env({}).prefetch is True


def test_prefetch_read_error_is_state_not_crash(svc, monkeypatch):
    _, client, _, pf = svc

    def broken(path, *a, **kw):
        raise OSError('диск отвалился')

    monkeypatch.setattr(prefetch_mod, 'open', broken, raising=False)
    client.post('/scans/exp1/prefetch', headers=HEADERS)
    st = wait_state(pf, ('error',))
    assert st['state'] == 'error' and 'диск отвалился' in st['reason']


def test_service_wires_prefetch_stop(svc):
    app, _, _, pf = svc
    st = app.extensions['recon']
    assert st.sessions.on_source_read == pf.stop and st.jobs.on_source_read == pf.stop


def test_job_calls_source_read_hook(tmp_path, monkeypatch):
    _, client, cfg, jsvc = jobs_service(tmp_path, monkeypatch)
    _, recipe = scan_and_recipe(cfg)
    calls = []
    jsvc.on_source_read = calls.append
    assert client.post('/jobs', json={'recipe': recipe}, headers=HEADERS).status_code == 201
    assert jsvc.runner.run_once() is True
    assert calls == ['job']
