"""Эндпоинты для студии в rbtm-web: углы выхода объекта за рамку и рецепт по состоянию сессии."""
import pytest

from engine_scans import simple_scan
from service_helpers import HEADERS, make_service, write_scan
from test_service_preview import ROI_SIMPLE, loaded, row_center, url  # noqa: F401 — фикстуры
from test_service_sessions import _cpu, _no_storage  # noqa: F401


def test_outside_for_user_roi(tmp_path):
    _, client, cfg = make_service(tmp_path)
    ss = simple_scan()
    write_scan(cfg, 'exp1', ss)
    full = client.get('/scans/exp1/outside?x0=0&x1=72&y0=0&y1=40', headers=HEADERS).get_json()
    assert full['angles_outside'] == [] and full['indices'] == []
    narrow = client.get('/scans/exp1/outside?x0=30&x1=42&y0=0&y1=40', headers=HEADERS).get_json()
    assert narrow['angles_outside'] and len(narrow['indices']) == len(narrow['angles_outside'])
    assert narrow['roi'] == {'x0': 30, 'x1': 42, 'y0': 0, 'y1': 40, 'preview_row': 20}
    assert client.get('/scans/exp1/outside?x0=50&x1=10', headers=HEADERS).status_code == 400


def test_recipe_from_session_defaults_and_overrides(loaded):
    app, client, cfg, ss, sid, ctx = loaded
    r = client.post(url(sid, 'recipe'), json={}, headers=HEADERS)
    assert r.status_code == 200
    d = r.get_json()
    assert d['schema'] == 'rbtm-recon-recipe/1' and d['input']['exp_id'] == 'exp1'
    assert {k: d['fov'][k] for k in ('x0', 'x1', 'y0', 'y1')} == {k: ROI_SIMPLE[k] for k in ('x0', 'x1', 'y0', 'y1')}
    assert d['axis']['method'] == 'auto' and d['rings']['preset'] == 'medium'
    assert d['recon']['slices'] == [ROI_SIMPLE['y0'], ROI_SIMPLE['y1']]
    assert d['provenance']['steps']['axis'] == 'auto' and d['author'] == HEADERS['X-Recon-User']

    c = row_center(ss, 20)
    d = client.post(url(sid, 'recipe'), json={'row': 20, 'center': c, 'tilt': 0.6, 'rings': 'off',
                                              'angles': 'full_halves', 'slices': [10, 20], 'pixel_size_mm': 0.011},
                    headers=HEADERS).get_json()
    assert d['axis']['center_x'] == pytest.approx(c) and d['axis']['tilt_deg'] == pytest.approx(0.6)
    assert d['rings']['preset'] == 'off' and d['recon']['angles'] == 'full_halves' and d['recon']['slices'] == [10, 20]
    assert d['pixel_size'] == {'value_mm': 0.011, 'source': 'user', 'user_edited': True}
    assert d['provenance']['steps']['axis'] == 'checked' and d['provenance']['steps']['rings'] == 'checked'

    # рецепт принимает очередь задач
    job = client.post('/jobs', json={'recipe': d, 'name': 'образец'}, headers=HEADERS)
    assert job.status_code == 201, job.get_json()

    for bad in ({'slices': [0, 100]}, {'rings': 'ultra'}, {'angles': 'all'}, {'slices': [5]}):
        assert client.post(url(sid, 'recipe'), json=bad, headers=HEADERS).status_code == 400


def test_recipe_binning(loaded):
    _, client, _, _, sid, _ = loaded
    d = client.post(url(sid, 'recipe'), json={}, headers=HEADERS).get_json()
    assert d['outputs']['binning'] == [4]                                   # по умолчанию — как у ноутбука
    d = client.post(url(sid, 'recipe'), json={'binning': [8, 2, 2]}, headers=HEADERS).get_json()
    assert d['outputs']['binning'] == [2, 8]
    est = client.post(url(sid, 'estimate'), json={'recipe': d}, headers=HEADERS).get_json()
    assert sorted(int(k) for k in est['binned_bytes']) == [2, 8]
    assert client.post(url(sid, 'recipe'), json={'binning': []}, headers=HEADERS).get_json()['outputs']['binning'] == []
    for bad in ([1], [64], 'x', [2.5], [True], 4):
        assert client.post(url(sid, 'recipe'), json={'binning': bad}, headers=HEADERS).status_code == 400, bad


def test_recipe_smoothing(loaded):
    """smoothing в теле recipe → блок рецепта (недостающие поля — по умолчанию), provenance.steps.smoothing; без
    ключа — выключено и 'auto'; ошибки значений — 400."""
    _, client, _, _, sid, _ = loaded
    d = client.post(url(sid, 'recipe'), json={}, headers=HEADERS).get_json()
    assert d['smoothing'] == {'sigma': None, 'deblur': 'none', 'balance': 0.02, 'amount': 1.5}
    assert d['provenance']['steps']['smoothing'] == 'auto'
    d = client.post(url(sid, 'recipe'), json={'smoothing': {'sigma': 1.5, 'deblur': 'unsharp', 'amount': 2}},
                    headers=HEADERS).get_json()
    assert d['smoothing'] == {'sigma': 1.5, 'deblur': 'unsharp', 'balance': 0.02, 'amount': 2}
    assert d['provenance']['steps']['smoothing'] == 'checked'
    assert client.post('/jobs', json={'recipe': d, 'name': 'образец'}, headers=HEADERS).status_code == 201
    est = client.post(url(sid, 'estimate'), json={'recipe': d}, headers=HEADERS).get_json()
    assert est['halo_rows'] > 0 and est['ring_rows_factor'] > 1
    d = client.post(url(sid, 'recipe'), json={'smoothing': None}, headers=HEADERS).get_json()
    assert d['smoothing']['sigma'] is None and d['provenance']['steps']['smoothing'] == 'checked'
    for bad in ({'sigma': 0.1}, {'sigma': 1.5, 'deblur': 'rl'}, {'sigma': 1.5, 'balance': 2}, {'sigma': 'x'},
                {'sigma': 1.5, 'radius': 2}, 1.5, 'wiener', [1.5]):
        assert client.post(url(sid, 'recipe'), json={'smoothing': bad}, headers=HEADERS).status_code == 400, bad
