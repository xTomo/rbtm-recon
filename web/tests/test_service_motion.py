"""Тесты смещения образца в сервисе: оценка при загрузке области, GET/POST sessions/<sid>/motion, сводка в состоянии
сессии, блок motion в рецепте и совпадение сдвигов кадров превью и задачи."""
import time

import numpy as np
import pytest
import requests

import engine_phantom as ph
from engine_scans import simple_scan
from reconengine import data, gpu, pipeline
from reconengine import recipe as recipe_mod
from reconservice import scans as scans_mod
from service_helpers import HEADERS, make_service, write_scan

ANG = np.arange(0.0, 184.0, 1.0)
ROI = {'x0': 0, 'x1': 160, 'y0': 0, 'y1': 96}


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv('RECON_ENGINE_CPU', '1')
    gpu.reset_backend()
    yield
    gpu.reset_backend()


@pytest.fixture(autouse=True)
def _no_storage(monkeypatch):
    def post(url, **kw):
        raise requests.ConnectionError('storage недоступен')
    monkeypatch.setattr(scans_mod.requests, 'post', post)


def drift_scan(amp=6.0):
    tl = ph.advanced_timeline(ANG, 4)
    t = np.arange(len(tl.modes)) / (len(tl.modes) - 1)
    fdx = amp * np.interp(t, [0, 0.3, 0.6, 1.0], [1.0, -0.45, 0.2, -0.45])
    return ph.make_synthetic_scan(ANG, height=96, width=160, center_x=81.3, y_ref=47.5, tilt_deg=0.6,
                                  advanced=True, n_segments=4, frame_dx=fdx)


@pytest.fixture
def svc(tmp_path):
    app, client, cfg = make_service(tmp_path)
    write_scan(cfg, 'drift', drift_scan())
    write_scan(cfg, 'still', drift_scan(0.0))
    yield app, client, cfg
    mgr = app.extensions['recon'].sessions
    mgr.stop_reaper()
    mgr.wait_loaded(10)


def load(client, exp_id, roi=None):
    sid = client.post('/sessions', json={'exp_id': exp_id}, headers=HEADERS).get_json()['id']
    assert client.post('/sessions/{}/load'.format(sid), json={'roi': roi or ROI}, headers=HEADERS).status_code == 202
    deadline = time.time() + 60
    while True:
        st = client.get('/sessions/' + sid, headers=HEADERS).get_json()
        if st['state'] in ('ready', 'error') or time.time() > deadline:
            break
        time.sleep(0.02)
    assert st['state'] == 'ready', st
    return sid, st


def ctx_of(app):
    return app.extensions['recon'].sessions._session.ctx()


def test_drift_detected_and_compensated_on_load(svc):
    app, client, _ = svc
    sid, st = load(client, 'drift')
    assert st['motion']['applied'] is True and st['motion']['status'] == 'detected'
    assert 'dx' not in st['motion']                         # в состоянии сессии — только сводка
    assert st['axis'] is None                               # ось сессии по-прежнему задаёт axis/auto
    m = client.get('/sessions/{}/motion'.format(sid), headers=HEADERS).get_json()
    assert m['mode'] == 'auto' and m['applied'] is True and 'компенсируется' in m['message']
    ctx = ctx_of(app)
    assert len(m['dx']) == len(m['fnums']) == len(ctx.prep.idx)
    # сдвиги кадров = сдвиги после вставок (repositioning) − смещение образца
    np.testing.assert_allclose(ctx.prep.frame_sx, ctx._base_frame_sx - np.asarray(m['dx']), atol=1e-3)
    assert ctx.empty_skip == recipe_mod.EMPTY_SKIP_DEFAULT


def test_switch_modes(svc):
    app, client, _ = svc
    sid, _ = load(client, 'drift')
    ctx = ctx_of(app)
    ax = client.post('/sessions/{}/axis/auto'.format(sid), headers=HEADERS).get_json()['axis']
    r = client.post('/sessions/{}/motion'.format(sid), json={'mode': 'off'}, headers=HEADERS)
    assert r.status_code == 200 and r.get_json()['applied'] is False
    np.testing.assert_array_equal(ctx.prep.frame_sx, ctx._base_frame_sx)
    assert ctx.axis is None                                            # авто-ось забыта — найдётся по новой паре
    ax_off = client.post('/sessions/{}/axis/auto'.format(sid), headers=HEADERS).get_json()['axis']
    assert abs(ax_off['center_x'] - ax['center_x']) > 0.1              # без компенсации пара 0/180 сдвинута дрейфом
    on = client.post('/sessions/{}/motion'.format(sid), json={'mode': 'on'}, headers=HEADERS).get_json()
    assert on['applied'] is True and 'вручную' in on['message']
    assert client.post('/sessions/{}/motion'.format(sid), json={'mode': 'maybe'}, headers=HEADERS).status_code == 400
    # режим запоминается в сессии и действует на новую загрузку
    assert app.extensions['recon'].sessions._session.extra['motion_mode'] == 'on'


def test_manual_axis_survives_mode_change(svc):
    app, client, _ = svc
    sid, _ = load(client, 'drift')
    client.post('/sessions/{}/axis/set'.format(sid), json={'center': 81.0, 'tilt': 0.5}, headers=HEADERS)
    client.post('/sessions/{}/motion'.format(sid), json={'mode': 'off'}, headers=HEADERS)
    assert ctx_of(app).axis.method == 'manual'


def test_recipe_carries_decision_and_job_reproduces_shifts(svc):
    app, client, cfg = svc
    sid, _ = load(client, 'drift')
    rd = client.post('/sessions/{}/recipe'.format(sid), json={}, headers=HEADERS).get_json()
    assert rd['motion']['mode'] == 'auto' and rd['motion']['applied'] is True and rd['empty_skip_first'] == 2
    ctx = ctx_of(app)
    scan = data.open_scan(cfg.scan_path('drift'))
    prep = pipeline.prepare(scan, ctx.crop, recipe_mod.from_dict(rd))
    np.testing.assert_allclose(prep.frame_sx, ctx.prep.frame_sx, atol=1e-3)
    client.post('/sessions/{}/motion'.format(sid), json={'mode': 'off'}, headers=HEADERS)
    rd = client.post('/sessions/{}/recipe'.format(sid), json={}, headers=HEADERS).get_json()
    assert rd['motion'] == recipe_mod.motion_block('off', applied=False)


def test_still_scan_not_compensated(svc):
    app, client, _ = svc
    sid, st = load(client, 'still')
    assert st['motion']['status'] == 'none' and st['motion']['applied'] is False
    rd = client.post('/sessions/{}/recipe'.format(sid), json={}, headers=HEADERS).get_json()
    assert rd['motion']['applied'] is False and rd['motion']['dx'] is None
    np.testing.assert_array_equal(ctx_of(app).prep.frame_sx, ctx_of(app)._base_frame_sx)


def test_small_scan_no_object(tmp_path):
    app, client, cfg = make_service(tmp_path)
    write_scan(cfg, 'tiny', simple_scan())
    try:
        sid, st = load(client, 'tiny', roi={'x0': 2, 'x1': 70, 'y0': 4, 'y1': 36})
        assert st['motion']['applied'] is False and st['motion']['measured'] is True
    finally:
        app.extensions['recon'].sessions.stop_reaper()
        app.extensions['recon'].sessions.wait_loaded(10)
