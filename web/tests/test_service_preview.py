"""Тесты превью сессии (reconservice.preview через /sessions/<sid>/...): срез совпадает со срезом run_recipe (и со
сглаживанием), кэш полосы и блока строк, авто-ось, перебор центра, вид 0°−180°, кольца, сравнение вариантов, сдвиги
образца, оценка, «последний выигрывает»."""
import os
import sys
import threading
import time
import types

import numpy as np
import pytest

from reconengine import axis as axis_mod
from reconengine import data, fbp, gpu, pipeline, preprocess, rings, smoothing
from reconengine import recipe as recipe_mod
from reconengine.model import Axis, ROI
from reconservice import preview

from engine_scans import advanced_scan, simple_scan
from service_helpers import HEADERS, decode, make_service, write_scan
from test_service_sessions import _cpu, _no_storage, load_ready, manager, open_session  # noqa: F401 — фикстуры

ROI_SIMPLE = {'x0': 2, 'x1': 70, 'y0': 4, 'y1': 36}     # кроп 32×68, строка превью 20


def true_axis(ss):
    return Axis(center_x=ss.center_x, y_ref=ss.y_ref, tilt_deg=ss.tilt_deg)


def row_center(ss, row):
    return true_axis(ss).center_at(row)


def session_for(tmp_path, ss, roi, exp_id='exp1'):
    app, client, cfg = make_service(tmp_path)
    write_scan(cfg, exp_id, ss)
    sid = open_session(client, exp_id).get_json()['id']
    load_ready(client, sid, roi)
    return app, client, cfg, sid, manager(app)._session.ctx()


@pytest.fixture
def loaded(tmp_path):
    ss = simple_scan()
    app, client, cfg, sid, ctx = session_for(tmp_path, ss, ROI_SIMPLE)
    yield app, client, cfg, ss, sid, ctx
    manager(app).stop_reaper()
    manager(app).wait_loaded(10)


def url(sid, path, **q):
    s = '/sessions/{}/{}'.format(sid, path)
    if q:
        s += '?' + '&'.join('{}={}'.format(k, v) for k, v in q.items())
    return s


def make_recipe(scan, roi, ax, pixel_size, **recon):
    r = recipe_mod.default_recipe(scan.exp_id, scan.fingerprint, roi, pixel_size, 'hdf5', scan.is_advanced)
    r.axis = ax
    r.recon.update(recon)
    return r


def run_volume(tmp_path, cfg, exp_id, r, slab_rows=5):
    res = pipeline.run_recipe(r, cfg.scan_path(exp_id), str(tmp_path / 'run'), cfg.cache_dir(exp_id),
                              backend='cpu', slab_rows=slab_rows)
    doc = res.result
    return np.fromfile(os.path.join(res.out_dir, doc['volume']['file']), '<f4').reshape(doc['volume']['shape'])


# --- срез = срез run_recipe ------------------------------------------------------------------------------------

@pytest.mark.parametrize('make, dx, dy, n_slices', [(simple_scan, 2, 4, None), (advanced_scan, 6, 10, 24)],
                         ids=['simple', 'advanced'])
def test_slice_matches_run_recipe(tmp_path, make, dx, dy, n_slices):
    """Превью строки при тех же ROI, оси, кольцах и углах — тот же срез, что в объёме run_recipe (в т.ч. крайние
    строки кропа и сдвиги образца advanced-скана)."""
    ss = make()
    roi = ROI(dx, ss.scan.width - dx, dy, ss.scan.height - dy)
    app, client, cfg, sid, ctx = session_for(tmp_path, ss, roi.to_dict())
    scan = data.open_scan(cfg.scan_path('exp1'))
    ax = true_axis(ss)
    z1 = roi.y1 if n_slices is None else roi.y0 + n_slices          # advanced: часть срезов — быстрее
    r = make_recipe(scan, roi, ax, ctx.pixel_size, angles='full_halves', slices=[roi.y0, z1])
    r.rings = {'preset': 'medium', 'params': None}
    vol = run_volume(tmp_path, cfg, 'exp1', r)

    for row in (roi.y0, roi.y0 + 3, (roi.y0 + z1) // 2, z1 - 1):
        img, meta = ctx.slice(row, Axis(ax.center_at(row), row, ax.tilt_deg), 'medium', 'full_halves')
        ref = vol[row - roi.y0]
        assert img.shape == ref.shape == (roi.width, roi.width)
        assert np.abs(img - ref).max() < 1e-4 * (ref.max() - ref.min()), row
        assert meta['n_angles'] == 120

    # то же через HTTP: uint16 в окне персентилей — совпадает до шага квантования (вне окна значения обрезаны)
    row = (roi.y0 + z1) // 2
    resp = client.get(url(sid, 'slice', row=row, center=ax.center_at(row), tilt=ax.tilt_deg, rings='medium',
                          angles='full_halves'), headers=HEADERS)
    assert resp.status_code == 200
    got, meta = decode(resp)
    ref = vol[row - roi.y0]
    step = float(resp.headers['X-Scale'])
    lo = float(resp.headers['X-Offset'])
    inside = (ref > lo + step) & (ref < lo + 65534 * step)
    assert inside.mean() > 0.95
    assert np.abs(got - ref)[inside].max() <= step
    assert meta['row'] == row and meta['rings'] == 'medium' and meta['angles'] == 'full_halves'
    assert set(meta['timings']) >= {'band_s', 'align_s', 'rings_s', 'fbp_s', 'total_s'}


@pytest.fixture
def per_row_rings(monkeypatch):
    """remove_all_stripe (заглушка — тождество) → вычитание половины среднего по углам: заметно меняет синограмму,
    по строкам независимо, как настоящая."""
    monkeypatch.setattr(sys.modules['tomo.remove_stripe'], 'remove_all_stripe',
                        lambda tomo, **kw: tomo - 0.5 * tomo.mean(axis=0, keepdims=True))


SMOOTH = {'sigma': 1.5, 'deblur': 'wiener', 'balance': 0.02, 'amount': 1.5}


@pytest.mark.parametrize('make, dx, dy, n_slices', [(simple_scan, 2, 4, None), (advanced_scan, 6, 10, 24)],
                         ids=['simple', 'advanced'])
def test_slice_with_smoothing_matches_run_recipe(tmp_path, per_row_rings, make, dx, dy, n_slices):
    """Сглаживание: превью строки (блок строк ± ореол после колец → фильтр → FBP строки) — тот же срез, что в объёме
    run_recipe со слоями по 5 строк (ореол слоёв, края кропа, сдвиги образца); фрагмент — FBP только его."""
    ss = make()
    roi = ROI(dx, ss.scan.width - dx, dy, ss.scan.height - dy)
    app, client, cfg, sid, ctx = session_for(tmp_path, ss, roi.to_dict())
    scan = data.open_scan(cfg.scan_path('exp1'))
    ax = true_axis(ss)
    z1 = roi.y1 if n_slices is None else roi.y0 + n_slices
    r = make_recipe(scan, roi, ax, ctx.pixel_size, angles='full_halves', slices=[roi.y0, z1])
    r.rings = {'preset': 'medium', 'params': None}
    r.smoothing = dict(SMOOTH)
    vol = run_volume(tmp_path, cfg, 'exp1', r)

    for row in (roi.y0, roi.y0 + 3, (roi.y0 + z1) // 2, z1 - 1):
        rax = Axis(ax.center_at(row), row, ax.tilt_deg)
        img, meta = ctx.slice(row, rax, 'medium', 'full_halves', smooth=SMOOTH)
        ref = vol[row - roi.y0]
        assert img.shape == ref.shape == (roi.width, roi.width)
        assert np.abs(img - ref).max() < 1e-4 * (ref.max() - ref.min()), row
        assert meta['smoothing'] == SMOOTH and meta['timings']['block_rows'] >= 17
        # фрагмент — FBP только его пикселей, те же значения
        part, meta = ctx.slice(row, rax, 'medium', 'full_halves', region=(5, 9, 30, 21), smooth=SMOOTH)
        assert part.shape == (12, 25) and meta['timings']['fbp_px'] == 12 * 25
        assert np.abs(part - ref[9:21, 5:30]).max() < 1e-4 * (ref.max() - ref.min())
    manager(app).stop_reaper()


def test_slice_smoothing_changes_reuse_block(loaded, per_row_rings, monkeypatch):
    """Смена σ и метода (в пределах запаса) — только фильтр и FBP: без выравнивания, колец и нормировки; смена центра
    — сдвиг блока; смена колец — кольца заново на том же блоке строк."""
    _, client, _, ss, sid, ctx = loaded
    calls = {'align': 0, 'rings': 0, 'norm': 0}

    def counting(name, fn):
        def inner(*a, **kw):
            calls[name] += 1
            return fn(*a, **kw)
        return inner

    monkeypatch.setattr(axis_mod, 'align_rows', counting('align', axis_mod.align_rows))
    monkeypatch.setattr(rings, 'apply', counting('rings', rings.apply))
    monkeypatch.setattr(preprocess, 'normalize_slab', counting('norm', preprocess.normalize_slab))
    c = row_center(ss, 20)
    q = dict(row=20, center=c, tilt=ss.tilt_deg, rings='medium', max_px=4000)
    first, meta = decode(client.get(url(sid, 'slice', smooth=1.5, **q), headers=HEADERS))
    t = meta['timings']
    assert meta['smoothing'] == SMOOTH and meta['exact'] is True
    assert t['block_rows'] == 32 and t['align_s'] > 0 and t['smooth_s'] > 0        # запас до σ = 2 — весь кроп
    assert calls['align'] == 4 and calls['rings'] == 2                             # порции по 8 и по 16 строк
    before = dict(calls)
    for extra in (dict(smooth=2.0), dict(smooth=1.0, deblur='unsharp', amount=2), dict(smooth=0.8, deblur='none'),
                  dict(smooth=1.5, balance=0.05)):
        img, meta = decode(client.get(url(sid, 'slice', **dict(q, **extra)), headers=HEADERS))
        assert meta['timings']['align_s'] == 0 and meta['timings']['rings_s'] == 0, extra
        assert meta['smoothing']['sigma'] == extra['smooth'] and meta['exact'] is True
        assert np.abs(img - first).max() > 0                                         # фильтр другой
    assert calls == before
    # центр: быстрый путь — сдвиг блока, близко к полному
    fast, meta = decode(client.get(url(sid, 'slice', smooth=1.5, **dict(q, center=c + 0.6)), headers=HEADERS))
    assert meta['exact'] is False and 'fast_shift_px' in meta['timings'] and calls == before
    exact, meta = decode(client.get(url(sid, 'slice', smooth=1.5, exact=1, **dict(q, center=c + 0.6)),
                                    headers=HEADERS))
    assert meta['exact'] is True and _corr(fast, exact) > 0.999
    assert calls['norm'] == before['norm'] and calls['rings'] > before['rings']
    # другие кольца — кольца заново, без нормировки полосы
    before = dict(calls)
    decode(client.get(url(sid, 'slice', smooth=1.5, **dict(q, rings='strong')), headers=HEADERS))
    assert calls['rings'] == before['rings'] + 2 and calls['norm'] == before['norm']
    # выключенное сглаживание — прежний путь по одной строке
    _, meta = decode(client.get(url(sid, 'slice', smooth=0, **q), headers=HEADERS))
    assert meta['smoothing'] is None and meta['timings']['block_rows'] == 1


def test_slice_http_smoothing_params(loaded):
    _, client, _, ss, sid, _ = loaded
    base = dict(row=20, center=row_center(ss, 20), tilt=ss.tilt_deg, rings='off')
    _, meta = decode(client.get(url(sid, 'slice', smooth=1.2, deblur='unsharp', amount=2, **base), headers=HEADERS))
    assert meta['smoothing'] == {'sigma': 1.2, 'deblur': 'unsharp', 'balance': 0.02, 'amount': 2.0}
    assert {'smooth_s', 'block_rows', 'fbp_s', 'rings_s'} <= set(meta['timings'])
    for off in ('', '0'):
        _, meta = decode(client.get(url(sid, 'slice', smooth=off, **base), headers=HEADERS))
        assert meta['smoothing'] is None
    for bad in (dict(smooth=0.1), dict(smooth=5), dict(smooth='x'), dict(smooth='nan'), dict(smooth=1.5, deblur='rl'),
                dict(smooth=1.5, balance=0), dict(smooth=1.5, amount=-1), dict(smooth=1.5, balance='x')):
        r = client.get(url(sid, 'slice', **dict(base, **bad)), headers=HEADERS)
        assert r.status_code == 400, bad


def test_slice_http_params(loaded):
    _, client, _, ss, sid, _ = loaded
    c = row_center(ss, 20)
    base = dict(row=20, center=c, tilt=ss.tilt_deg, rings='off')
    img, meta = decode(client.get(url(sid, 'slice', **base), headers=HEADERS))
    assert img.shape == (68, 68) and meta['region'] == [0, 0, 68, 68] and meta['n_angles'] == 60
    part, meta = decode(client.get(url(sid, 'slice', region='10,20,40,30', **base), headers=HEADERS))
    assert part.shape == (10, 30) and meta['region'] == [10, 20, 40, 30]
    small, meta = decode(client.get(url(sid, 'slice', max_px=32, **base), headers=HEADERS))
    assert small.shape == (22, 22) and meta['downsample'] == 3
    for bad in (dict(row=3), dict(row=36), dict(region='0,0,69,10'), dict(region='1,2,3'), dict(rings='x'),
                dict(angles='x'), dict(center='nan'), dict(max_px=0)):
        q = dict(base, **bad)
        assert client.get(url(sid, 'slice', **q), headers=HEADERS).status_code == 400, bad


# --- кэш полосы ------------------------------------------------------------------------------------------------

def test_band_cache_reused_for_center_and_tilt(loaded, monkeypatch):
    _, client, _, ss, sid, _ = loaded
    calls = []
    orig = preprocess.normalize_slab

    def counting(frames, *a, **kw):
        calls.append(np.shape(frames))
        return orig(frames, *a, **kw)

    monkeypatch.setattr(preprocess, 'normalize_slab', counting)
    c = row_center(ss, 20)
    metas = []
    for dc, tilt in [(0, 0.6), (0.7, 0.6), (-1.2, 0.4), (0, 0.9), (0.3, -0.5)]:
        r = client.get(url(sid, 'slice', row=20, center=c + dc, tilt=tilt, rings='off'), headers=HEADERS)
        assert r.status_code == 200
        metas.append(decode(r)[1])
    assert len(calls) == 1                                   # нормировка — один раз, на всю полосу
    assert calls[0][0] == 120                                # все data-кадры
    assert [m['timings']['band_cached'] for m in metas] == [False, True, True, True, True]
    # соседняя строка в пределах полосы — тоже без нормировки
    client.get(url(sid, 'slice', row=22, center=c, tilt=0.6, rings='off'), headers=HEADERS)
    assert len(calls) == 1
    # строка у края кропа — вне полосы [3, 30) строк кропа, пересчёт
    client.get(url(sid, 'slice', row=5, center=c, tilt=0.6, rings='off'), headers=HEADERS)
    assert len(calls) == 2
    # наклон больше резерва — полоса шире, пересчёт; теперь полоса — весь кроп (32 строки)
    client.get(url(sid, 'slice', row=20, center=c, tilt=10.0, rings='off'), headers=HEADERS)
    assert len(calls) == 3 and calls[-1][1] == 32
    client.get(url(sid, 'slice', row=5, center=c, tilt=0.6, rings='off'), headers=HEADERS)
    assert len(calls) == 3


def test_band_on_host_when_gpu_memory_is_short(loaded, monkeypatch):
    """Полоса не влезает в долю свободной памяти GPU — хранится в RAM, срез тот же."""
    _, _, _, ss, _, ctx = loaded
    ax = Axis(row_center(ss, 20), 20, ss.tilt_deg)
    ref, _ = ctx.slice(20, ax, 'off')
    ctx.release()
    # preview «видит» GPU с 1 МБ свободной памяти (полоса ~1 МБ), движок остаётся на numpy
    fake = types.SimpleNamespace(is_gpu=lambda xp=None: True, mem_info=lambda: (1 << 20, 1 << 30),
                                 free_memory=lambda: None, to_numpy=gpu.to_numpy, ndimage=gpu.ndimage,
                                 get_xp=gpu.get_xp)
    monkeypatch.setattr(preview, 'gpu', fake)
    img, meta = ctx.slice(20, ax, 'off')
    assert ctx._band.on_gpu is False and isinstance(ctx._band.data, np.ndarray)
    assert np.array_equal(img, ref)


# --- ось -------------------------------------------------------------------------------------------------------

def test_auto_axis_close_to_truth_and_same_as_pipeline(loaded):
    _, client, cfg, ss, sid, ctx = loaded
    r = client.post(url(sid, 'axis/auto'), headers=HEADERS)
    assert r.status_code == 200
    body = r.get_json()
    ax = Axis.from_dict(body['axis'])
    assert abs(ax.center_at(ss.y_ref) - ss.center_x) < 0.3
    assert abs(ax.tilt_deg - ss.tilt_deg) < 0.1
    assert body['pair']['angles'] == [0.0, 180.0]
    assert (body['shift_x'], body['alfa']) == pytest.approx(axis_mod.to_crop_params(ax, ctx.roi))
    # та же ось, что у run_recipe без оси в рецепте (pipeline.prepare)
    scan = data.open_scan(cfg.scan_path('exp1'))
    prep = pipeline.prepare(scan, ctx.crop, make_recipe(scan, ctx.roi, None, ctx.pixel_size))
    assert prep.axis.center_x == pytest.approx(ax.center_x, abs=1e-9)
    assert prep.axis.tilt_deg == pytest.approx(ax.tilt_deg, abs=1e-9)
    # стала осью сессии: срез без center/tilt — по ней
    _, meta = decode(client.get(url(sid, 'slice', row=24), headers=HEADERS))
    assert meta['axis']['center_x'] == pytest.approx(ax.center_at(24))
    assert meta['axis']['tilt_deg'] == pytest.approx(ax.tilt_deg)
    assert client.get('/sessions/' + sid, headers=HEADERS).get_json()['axis'] == body['axis']


def test_slice_without_axis_computes_auto_axis(loaded):
    _, client, _, ss, sid, ctx = loaded
    assert ctx.axis is None
    _, meta = decode(client.get(url(sid, 'slice', tilt=0.0), headers=HEADERS))   # центр — авто, наклон — задан
    assert ctx.axis is not None and ctx.axis.method == 'auto'
    assert meta['axis']['tilt_deg'] == 0.0
    assert meta['axis']['center_x'] == pytest.approx(ctx.axis.center_at(20))


def test_axis_tilt_sets_session_axis(loaded):
    _, client, _, _, sid, ctx = loaded
    r = client.post(url(sid, 'axis/tilt'), json={'y_top': 10, 'c_top': 35.0, 'y_bottom': 30, 'c_bottom': 35.4},
                    headers=HEADERS)
    assert r.status_code == 200
    ax = Axis.from_dict(r.get_json()['axis'])
    assert ax.y_ref == 20.0 and ax.center_x == pytest.approx(35.2) and ax.tilt_deg == pytest.approx(
        np.degrees(np.arctan(0.4 / 20)))
    assert ctx.axis == ax
    assert client.post(url(sid, 'axis/tilt'), json={'y_top': 10}, headers=HEADERS).status_code == 400


def test_axis_set_manual_axis_used_by_recipe_until_auto(loaded):
    _, client, _, _, sid, ctx = loaded
    r = client.post(url(sid, 'axis/set'), json={'center': 35.25, 'tilt': 0.5, 'row': 20}, headers=HEADERS)
    assert r.status_code == 200
    ax = Axis.from_dict(r.get_json()['axis'])
    assert (ax.center_x, ax.y_ref, ax.tilt_deg, ax.method) == (35.25, 20.0, 0.5, 'manual')
    assert ctx.axis == ax
    # ось сессии: восстановление страницы и рецепт без center/tilt
    assert client.get('/sessions/' + sid, headers=HEADERS).get_json()['axis']['method'] == 'manual'
    rec = client.post(url(sid, 'recipe'), json={}, headers=HEADERS).get_json()
    assert rec['axis']['center_x'] == 35.25 and rec['axis']['tilt_deg'] == 0.5
    assert rec['provenance']['steps']['axis'] == 'checked'
    # ошибки ввода
    assert client.post(url(sid, 'axis/set'), json={'tilt': 0.5}, headers=HEADERS).status_code == 400
    assert client.post(url(sid, 'axis/set'), json={'center': 35, 'tilt': 60}, headers=HEADERS).status_code == 400
    assert client.post(url(sid, 'axis/set'), json={'center': 35, 'tilt': 0, 'row': 10 ** 6},
                       headers=HEADERS).status_code == 400
    # сброс — авто-ось
    assert client.post(url(sid, 'axis/auto'), headers=HEADERS).get_json()['axis']['method'] == 'auto'
    assert ctx.axis.method == 'auto'


def test_center_scan_finds_true_center(loaded):
    _, client, _, ss, sid, _ = loaded
    row = 20
    c = row_center(ss, row)
    r = client.post(url(sid, 'axis/scan'), json={'row': row, 'center': c + 1.5, 'tilt': ss.tilt_deg, 'step': 0.5,
                                                 'n': 11, 'rings': 'off', 'seq': 1}, headers=HEADERS)
    assert r.status_code == 200
    frags, meta = decode(r)
    assert frags.shape == (11, 68, 68)                           # центральный квадрат 256 → весь срез 68×68
    assert meta['centers'] == pytest.approx(list(c + 1.5 + (np.arange(11) - 5) * 0.5))
    assert len(meta['metrics']) == 11 and meta['best'] == meta['centers'][int(np.argmin(meta['metrics']))]
    assert abs(meta['best'] - c) <= 0.5
    # фрагмент и tv
    r = client.post(url(sid, 'axis/scan'), json={'row': row, 'center': c - 1, 'tilt': ss.tilt_deg, 'step': 0.5,
                                                 'n': 9, 'metric': 'tv', 'region': [10, 10, 58, 58],
                                                 'rings': 'off'}, headers=HEADERS)
    frags, meta = decode(r)
    assert frags.shape == (9, 48, 48) and abs(meta['best'] - c) <= 1.0
    for bad in ({'n': 0}, {'n': 26}, {'step': 0}, {'metric': 'x'}, {'region': [0, 0, 100, 10]}):
        body = dict({'row': row, 'center': c, 'tilt': 0.6}, **bad)
        assert client.post(url(sid, 'axis/scan'), json=body, headers=HEADERS).status_code == 400, bad


def test_center_scan_same_as_engine_center_scan(loaded):
    """Сдвиг строки на xp и метрика по фрагментам во весь срез — ровно axis.center_scan."""
    _, _, _, ss, _, ctx = loaded
    row = 20
    ax = Axis(row_center(ss, row) + 0.8, row, ss.tilt_deg)
    frags, meta = ctx.center_scan(row, ax, step=0.7, n=5, metric='entropy', region=(0, 0, 68, 68), preset='off')
    sino, _ = ctx.aligned_row(row, ax)
    cc = (68 - 1) / 2.0
    centers = [cc + o for o in (np.arange(5) - 2) * 0.7]

    def recon_fn(s, angles, pixel_size):
        return fbp.recon_slice(s, angles, pixel_size, angle_mode='first_180')

    slices, metrics = axis_mod.center_scan(sino, ctx.prep.angles, centers, cc, ctx.pixel_size, recon_fn, 'entropy')
    assert np.allclose(frags, np.stack(slices), atol=1e-6 * np.abs(slices[0]).max())
    assert meta['metrics'] == pytest.approx(list(metrics), rel=1e-9)


def test_diff_view(loaded):
    _, client, _, ss, sid, _ = loaded
    c = row_center(ss, 20)
    good, meta = decode(client.get(url(sid, 'axis/diff', center=c, tilt=ss.tilt_deg), headers=HEADERS))
    bad, _ = decode(client.get(url(sid, 'axis/diff', center=c + 2, tilt=ss.tilt_deg), headers=HEADERS))
    assert good.shape == bad.shape == (32, 68)
    assert meta['axis']['center_x'] == pytest.approx(c)
    inner = (slice(4, -4), slice(8, -8))
    assert np.abs(good[inner]).mean() < 0.3 * np.abs(bad[inner]).mean()


# --- кольца, сдвиги образца, оценка ------------------------------------------------------------------------

def test_noise_level_of_white_noise():
    rng = np.random.default_rng(0)
    img = 5.0 + 0.3 * rng.standard_normal((200, 200))
    img[:, 100:] += 10.0                                                # край объекта не мешает (MAD)
    assert preview.noise_level(img) == pytest.approx(0.3, rel=0.05)
    m = preview.compare_metrics([img, img * 2])
    assert m[0]['sharpness'] == 1.0 and m[1]['sharpness'] == pytest.approx(4.0)
    assert m[1]['noise'] == pytest.approx(2 * m[0]['noise'])


def test_rings_preview(loaded, monkeypatch):
    _, client, _, ss, sid, ctx = loaded
    # заглушка remove_all_stripe — тождество; подменяем на вычитание среднего по углам (заметно меняет срез)
    monkeypatch.setattr(sys.modules['tomo.remove_stripe'], 'remove_all_stripe',
                        lambda tomo, **kw: tomo - tomo.mean(axis=0, keepdims=True))
    c = row_center(ss, 20)
    r = client.get(url(sid, 'rings/preview', row=20, center=c, tilt=ss.tilt_deg, preset='strong', seq=1),
                   headers=HEADERS)
    assert r.status_code == 200
    pair, meta = decode(r)
    assert pair.shape == (2, 68, 68) and meta['rings'] == 'strong' and meta['params']['snr'] == 2.0
    off, _ = ctx.slice(20, Axis(c, 20, ss.tilt_deg), 'off')
    on, _ = ctx.slice(20, Axis(c, 20, ss.tilt_deg), 'strong')
    step = float(r.headers['X-Scale'])
    lo, hi = float(r.headers['X-Offset']), float(r.headers['X-Offset']) + 65535 * step
    for got, ref in ((pair[0], off), (pair[1], on)):                # общее окно: сравнимы оба кадра
        inside = (ref > lo + step) & (ref < hi - step)
        assert np.abs(got - ref)[inside].max() <= step
    assert np.abs(on - off).max() > 0.1 * np.abs(off).max()


def test_repositioning_advanced(tmp_path):
    ss = advanced_scan()
    roi = ROI(6, ss.scan.width - 6, 10, ss.scan.height - 10)
    app, client, cfg, sid, ctx = session_for(tmp_path, ss, roi.to_dict(), 'adv1')
    r = client.get(url(sid, 'repositioning'), headers=HEADERS)
    assert r.status_code == 200
    body = r.get_json()
    assert body['advanced'] is True and body['applicable'] is True and len(body['checkpoints']) == 2
    # ndi.shift(data_check, s) ≈ data: накопленный сдвиг сегмента k = −смещение образца в нём
    assert np.allclose(body['cumulative']['sy'], [0.0, -1.3, -2.1], atol=0.11)
    assert np.allclose(body['cumulative']['sx'], [0.0, 0.8, -0.6], atol=0.11)
    angles = ss.scan.angles
    assert body['checkpoints'][0]['angle'] == pytest.approx(float(angles[ss.scan.check_idx[0]]))
    # те же сдвиги кадров, что у pipeline.prepare
    scan = data.open_scan(cfg.scan_path('adv1'))
    prep = pipeline.prepare(scan, ctx.crop, make_recipe(scan, roi, true_axis(ss), ctx.pixel_size))
    assert np.array_equal(prep.frame_sy, ctx.prep.frame_sy) and np.array_equal(prep.frame_sx, ctx.prep.frame_sx)
    assert body['max_shift']['sy'] == pytest.approx(np.abs(prep.frame_sy).max())


def test_repositioning_simple_not_applicable(loaded):
    _, client, _, _, sid, _ = loaded
    body = client.get(url(sid, 'repositioning'), headers=HEADERS).get_json()
    assert body['advanced'] is False and body['applicable'] is False and body['checkpoints'] == []


def test_estimate(loaded):
    _, client, _, ss, sid, ctx = loaded
    scan = ctx.scan
    r = make_recipe(scan, ctx.roi, true_axis(ss), ctx.pixel_size)
    body = client.post(url(sid, 'estimate'), json={'recipe': recipe_mod.to_dict(r)}, headers=HEADERS).get_json()
    assert body['volume_shape'] == [32, 68, 68] and body['n_data_frames'] == 120 and body['n_angles_used'] == 60
    assert body['time'] is None                                  # превью ещё не считалось
    client.get(url(sid, 'slice', row=20, center=row_center(ss, 20), tilt=0.6), headers=HEADERS)
    t = client.post(url(sid, 'estimate'), json={'recipe': recipe_mod.to_dict(r)}, headers=HEADERS).get_json()['time']
    assert t['s_per_slice'] > 0 and t['n_slices'] == 32 and t['recon_s'] == pytest.approx(32 * t['s_per_slice'])
    assert t['prepare_s'] is not None and t['prepare_s'] >= 0    # подготовка сессии — задача повторит её
    # быстрый путь (сдвиг готовой строки) не занижает оценку: она по последнему срезу полным путём
    _, meta = decode(client.get(url(sid, 'slice', row=20, center=row_center(ss, 20) + 0.5, tilt=0.6), headers=HEADERS))
    assert 'fast_shift_px' in meta['timings']
    t2 = client.post(url(sid, 'estimate'), json={'recipe': recipe_mod.to_dict(r)}, headers=HEADERS).get_json()['time']
    assert t2['s_per_slice'] == t['s_per_slice']
    d = recipe_mod.to_dict(r)
    d['recon']['slices'] = [0, 100]                              # вне fov
    assert client.post(url(sid, 'estimate'), json={'recipe': d}, headers=HEADERS).status_code == 400
    assert client.post(url(sid, 'estimate'), json={'recipe': {'schema': 'x'}}, headers=HEADERS).status_code == 400


# --- «последний выигрывает» ----------------------------------------------------------------------------------

def test_stale_seq_superseded(loaded):
    app, client, _, ss, sid, _ = loaded
    q = dict(row=20, center=row_center(ss, 20), tilt=0.6, rings='off')
    assert client.get(url(sid, 'slice', seq=5, **q), headers=HEADERS).status_code == 200
    r = client.get(url(sid, 'slice', seq=3, **q), headers=HEADERS)
    assert r.status_code == 409 and r.get_json()['error'] == 'superseded'
    assert client.get(url(sid, 'slice', seq=6, **q), headers=HEADERS).status_code == 200
    # другой канал — свой счётчик
    assert client.get(url(sid, 'rings/preview', seq=1, **q), headers=HEADERS).status_code == 200


def test_request_waiting_for_gpu_is_superseded(loaded):
    """Запрос ждёт compute_lock, тем временем приходит более новый — после захвата он выходит с 409."""
    app, _, _, ss, sid, _ = loaded
    mgr = manager(app)
    s = mgr._session
    key = (sid, 'slice')
    out = {}

    def run():
        out['r'] = app.test_client().get(url(sid, 'slice', seq=10, row=20, center=row_center(ss, 20), tilt=0.6),
                                         headers=HEADERS)

    with s.compute_lock:
        t = threading.Thread(target=run)
        t.start()
        deadline = time.time() + 5
        while mgr.arbiter.latest(key) != 10 and time.time() < deadline:
            time.sleep(0.005)
        mgr.arbiter.begin(key, 11)
    t.join(10)
    assert out['r'].status_code == 409 and out['r'].get_json()['error'] == 'superseded'


def test_close_forgets_channels_and_releases(loaded):
    app, client, _, ss, sid, ctx = loaded
    client.get(url(sid, 'slice', seq=7, row=20, center=row_center(ss, 20), tilt=0.6), headers=HEADERS)
    mgr = manager(app)
    assert mgr.arbiter.latest((sid, 'slice')) == 7 and ctx._band is not None
    client.delete('/sessions/' + sid, headers=HEADERS)
    assert mgr.arbiter.latest((sid, 'slice')) is None and ctx._band is None
    assert client.get(url(sid, 'slice'), headers=HEADERS).status_code == 404


# --- быстрый путь смены центра, фрагмент с краями -------------------------------------------------------------

def _corr(a, b):
    a = np.asarray(a, 'float64').ravel() - np.mean(a)
    b = np.asarray(b, 'float64').ravel() - np.mean(b)
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def test_center_change_uses_fast_shift_close_to_exact(loaded):
    """Смена только центра: готовая строка после колец сдвигается в частотной области (без выравнивания и колец
    заново) — срез почти тот же, что полным путём (exact=1)."""
    app, client, cfg, ss, sid, ctx = loaded
    row = 20
    c = row_center(ss, row)
    tilt = ss.tilt_deg
    first = client.get(url(sid, 'slice', row=row, center=c, tilt=tilt, max_px=4000), headers=HEADERS)
    assert decode(first)[1]['exact'] is True
    for dc in (0.4, -1.3):
        fast = client.get(url(sid, 'slice', row=row, center=c + dc, tilt=tilt, max_px=4000), headers=HEADERS)
        img_fast, meta = decode(fast)
        assert meta['exact'] is False and 'fast_shift_px' in meta['timings']
        exact = client.get(url(sid, 'slice', row=row, center=c + dc, tilt=tilt, max_px=4000, exact=1),
                           headers=HEADERS)
        img_exact, meta_e = decode(exact)
        assert meta_e['exact'] is True
        assert _corr(img_fast, img_exact) > 0.999
    # другой наклон — полный путь
    other = client.get(url(sid, 'slice', row=row, center=c, tilt=tilt + 0.5, max_px=4000), headers=HEADERS)
    assert decode(other)[1]['exact'] is True


# --- сравнение вариантов на фрагменте --------------------------------------------------------------------------

def test_compare_variants_fragment_and_metrics(tmp_path, per_row_rings):
    """compare: стопка (k, th, tw) с общим окном; каждый вариант — тот же фрагмент, что slice с его кольцами и
    сглаживанием; варианты нормализованы; на шумном скане сглаживание снижает шум; кольца — раз на пресет."""
    ss = simple_scan(noise=1.0)
    app, client, cfg, sid, ctx = session_for(tmp_path, ss, ROI_SIMPLE)
    c = row_center(ss, 20)
    variants = [{'rings': 'medium', 'smoothing': None}, {'rings': 'medium', 'smoothing': {'sigma': 1.5}},
                {'rings': 'medium', 'smoothing': {'sigma': 2.0, 'deblur': 'none'}},
                {'rings': 'strong', 'smoothing': {'sigma': 1.0, 'deblur': 'unsharp'}}, {'rings': 'off'}]
    region = [8, 12, 60, 50]
    calls = []
    orig = rings.apply

    def counting(sino, params, xp=None):
        calls.append((np.shape(sino)[0], params and params['snr']))
        return orig(sino, params, xp=xp)

    rings.apply = counting
    try:
        r = client.post(url(sid, 'compare'), json={'row': 20, 'center': c, 'tilt': ss.tilt_deg, 'region': region,
                                                   'variants': variants, 'angles': 'first_180', 'seq': 1},
                        headers=HEADERS)
    finally:
        rings.apply = orig
    assert r.status_code == 200, r.get_json()
    stack, meta = decode(r)
    assert stack.shape == (5, 38, 52) and meta['region'] == region and meta['row'] == 20
    assert meta['downsample'] == 1
    small, meta_s = decode(client.post(url(sid, 'compare'), json={'row': 20, 'center': c, 'tilt': ss.tilt_deg,
                                                                  'region': region, 'variants': variants[:2],
                                                                  'max_px': 20}, headers=HEADERS))
    assert small.shape == (2, 12, 17) and meta_s['downsample'] == 3
    # кольца: medium — на строках под наибольший ореол его вариантов (σ = 2 Винер нет, 2.0 'none' → h = 6,
    # 1.5 Винер → h = 16: весь кроп 32 строки, порции по 16), strong — ореол 1.0 'unsharp' (h = 4, 9 строк)
    assert sorted(calls) == sorted([(16, 3.0), (16, 3.0), (9, 2.0)])
    assert meta['variants'] == [
        {'rings': 'medium', 'smoothing': None}, {'rings': 'medium', 'smoothing': SMOOTH},
        {'rings': 'medium', 'smoothing': {'sigma': 2.0, 'deblur': 'none', 'balance': 0.02, 'amount': 1.5}},
        {'rings': 'strong', 'smoothing': {'sigma': 1.0, 'deblur': 'unsharp', 'balance': 0.02, 'amount': 1.5}},
        {'rings': 'off', 'smoothing': None}]
    step = float(r.headers['X-Scale'])
    lo, hi = float(r.headers['X-Offset']), float(r.headers['X-Offset']) + 65535 * step
    for got, v in zip(stack, meta['variants']):                       # общее окно, значения — как у slice
        ref, _ = ctx.slice(20, Axis(c, 20, ss.tilt_deg), v['rings'], region=tuple(region), smooth=v['smoothing'])
        inside = (ref > lo + step) & (ref < hi - step)
        assert inside.mean() > 0.9 and np.abs(got - ref)[inside].max() <= step
    m = meta['metrics']
    assert len(m) == 5 and m[0]['sharpness'] == 1.0 and all(x['noise'] > 0 for x in m)
    assert m[2]['noise'] < 0.6 * m[0]['noise'] and m[2]['sharpness'] < 1      # гаусс без деблюра: тише и мягче
    assert m[1]['noise'] < m[0]['noise']                                        # Винер σ 1,5: шум ниже
    assert set(meta['timings']) >= {'band_s', 'align_s', 'rings_s', 'smooth_s', 'fbp_s', 'total_s', 'block_rows'}
    manager(app).stop_reaper()


def test_compare_default_region_limits_and_seq(loaded):
    _, client, _, ss, sid, ctx = loaded
    c = row_center(ss, 20)
    base = {'row': 20, 'center': c, 'tilt': ss.tilt_deg}
    # без region — квадрат size с краями по срезу первого варианта (срез 68 — меньше 384: весь срез)
    stack, meta = decode(client.post(url(sid, 'compare'), json=dict(base, variants=[{'rings': 'off'}]),
                                     headers=HEADERS))
    assert stack.shape == (1, 68, 68) and meta['region'] == [0, 0, 68, 68] and meta['metrics'][0]['sharpness'] == 1
    stack, meta = decode(client.post(url(sid, 'compare'), json=dict(base, size=24, variants=[
        {'rings': 'off'}, {'rings': 'off', 'smoothing': {'sigma': 1.0}}]), headers=HEADERS))
    x0, y0, x1, y1 = meta['region']
    assert stack.shape == (2, 24, 24) and (x1 - x0, y1 - y0) == (24, 24)
    full, _ = ctx.slice(20, Axis(c, 20, ss.tilt_deg), 'off')
    assert meta['region'] == list(preview.structured_region(full, 24))
    # ось сессии, если center/tilt не заданы
    assert client.post(url(sid, 'compare'), json={'variants': [{}]}, headers=HEADERS).status_code == 200
    for bad in ({'variants': []}, {'variants': [{}] * 9}, {'variants': 'off'}, {}, {'variants': [{'rings': 'x'}]},
                {'variants': [{'smoothing': {'sigma': 9}}]}, {'variants': [{'smoothing': 1.5}]}, {'variants': [3]},
                {'variants': [{}], 'region': [0, 0, 100, 10]}, {'variants': [{}], 'size': 4},
                {'variants': [{}], 'angles': 'x'}):
        assert client.post(url(sid, 'compare'), json=dict(base, **bad), headers=HEADERS).status_code == 400, bad
    # «последний выигрывает» — свой канал
    body = dict(base, variants=[{'rings': 'off'}], region=[0, 0, 20, 20])
    assert client.post(url(sid, 'compare'), json=dict(body, seq=5), headers=HEADERS).status_code == 200
    r = client.post(url(sid, 'compare'), json=dict(body, seq=3), headers=HEADERS)
    assert r.status_code == 409 and r.get_json()['error'] == 'superseded'
    assert client.get(url(sid, 'slice', seq=1, row=20), headers=HEADERS).status_code == 200


def test_compare_reuses_slice_block(loaded, per_row_rings, monkeypatch):
    """Блок строк после колец, посчитанный срезом со сглаживанием, сравнение σ при той же оси берёт из кэша: без
    выравнивания и колец."""
    _, client, _, ss, sid, _ = loaded
    c = row_center(ss, 20)
    q = dict(row=20, center=c, tilt=ss.tilt_deg, rings='medium', smooth=1.5)
    assert client.get(url(sid, 'slice', **q), headers=HEADERS).status_code == 200
    calls = []
    monkeypatch.setattr(rings, 'apply', lambda *a, **kw: calls.append(1))
    monkeypatch.setattr(axis_mod, 'align_rows', lambda *a, **kw: calls.append(2))
    body = {'row': 20, 'center': c, 'tilt': ss.tilt_deg, 'region': [10, 10, 50, 50],
            'variants': [{'rings': 'medium', 'smoothing': s} for s in (None, {'sigma': 0.7}, {'sigma': 2.0})]}
    r = client.post(url(sid, 'compare'), json=body, headers=HEADERS)
    assert r.status_code == 200 and calls == []
    assert decode(r)[1]['timings']['rings_s'] == 0


def test_structured_region_finds_edges():
    img = np.zeros((600, 600), 'float32')
    img[400:520, 380:470] = 1.0                   # единственный объект с краями — справа внизу
    x0, y0, x1, y1 = preview.structured_region(img, 128)
    assert x1 - x0 == 128 and y1 - y0 == 128
    assert x0 < 470 and x1 > 380 and y0 < 520 and y1 > 400       # фрагмент захватывает края объекта


def test_estimate_uses_recent_jobs_rate(loaded):
    """С выполненными задачами оценка берётся по их фактической скорости (на срез · ширина² · угол)."""
    app, client, _, ss, sid, ctx = loaded
    r = make_recipe(ctx.scan, ctx.roi, true_axis(ss), ctx.pixel_size)
    jobs = app.extensions['recon'].jobs
    # задача: 64 среза ширины 100 по 200 углам за 32 с → 32 / (64·100²·200) с на срез·px²·угол
    jobs.coll.insert_one({'_id': 'j1', 'status': 'done', 'finished': 1,
                          'recipe': {'fov': {'x0': 0, 'x1': 100, 'y0': 0, 'y1': 64}},
                          'result': {'volume': {'shape': [64, 100, 100]},
                                     'timings': {'recon_s': 32.0, 'prepare_s': 5.0, 'n_angles': 200}}})
    t = client.post(url(sid, 'estimate'), json={'recipe': recipe_mod.to_dict(r)}, headers=HEADERS).get_json()['time']
    assert t['source'] == 'jobs' and t['prepare_s'] == 5.0
    unit = 32.0 / (64 * 100 * 100 * 200)
    assert t['s_per_slice'] == pytest.approx(unit * 68 * 68 * 60)            # ширина 68, 60 углов (first_180)
    assert t['recon_s'] == pytest.approx(t['s_per_slice'] * 32)
