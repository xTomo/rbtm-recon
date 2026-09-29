"""Тесты reconengine.pipeline и cli: конвейер по слоям, совпадение со старым путём, запуск по рецепту."""
import json
import os
import sys
import threading

import numpy as np
import pytest

import engine_phantom as ph
from reconengine import axis as axis_mod
from reconengine import cli, data, fbp, gpu, pipeline, preprocess, rings, smoothing
from reconengine import recipe as recipe_mod
from reconengine.model import Axis, Cancelled, ROI

from engine_scans import ANGLES, advanced_scan, simple_scan, write_h5  # noqa: F401


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv('RECON_ENGINE_CPU', '1')
    gpu.reset_backend()
    yield
    gpu.reset_backend()


def inner_roi(ss, dx, dy):
    return ROI(dx, ss.scan.width - dx, dy, ss.scan.height - dy)


def make_recipe(scan, roi, ax=None, **changes):
    r = recipe_mod.default_recipe(scan.exp_id, scan.fingerprint, roi, 0.009, 'user', scan.is_advanced)
    r.axis = ax
    r.rings = {'preset': 'off', 'params': None}
    for key, value in changes.items():
        setattr(r, key, value)
    return r


def full_crop_reference(crop, prep, ring_params=None, smooth_params=None):
    """Та же обработка, что в process_slab, но сразу на всём кропе (как в ноутбуке): (h, n, w); с ring_params и
    smooth_params — затем кольца и сглаживание по всему кропу."""
    raw = np.asarray(crop.frames[prep.idx])
    norm = np.asarray(preprocess.normalize_slab(raw, prep.fnums, prep.de, xp=np))
    norm = pipeline._apply_shifts(norm, prep.frame_sy, prep.frame_sx, np)
    out = np.stack([np.asarray(axis_mod.transform_image(fr, prep.shift_x, prep.alfa, xp=np)) for fr in norm])
    sino = np.ascontiguousarray(np.swapaxes(out, 0, 1))
    sino = np.asarray(rings.apply(sino, ring_params, xp=np))
    return np.asarray(smoothing.apply(sino, smooth_params, xp=np))


@pytest.fixture
def per_row_rings(monkeypatch):
    """remove_all_stripe (заглушка — тождество) → вычитание половины среднего по углам: заметно меняет синограмму,
    по строкам независимо, как настоящая."""
    monkeypatch.setattr(sys.modules['tomo.remove_stripe'], 'remove_all_stripe',
                        lambda tomo, **kw: tomo - 0.5 * tomo.mean(axis=0, keepdims=True))


# --- мелкие функции ------------------------------------------------------------------------------------------

def test_pair_0_180_first_pair_like_tomotools4():
    import tomotools4
    a = np.array([10.0, 0.0, 90.0, 180.0, 190.0, 270.0])
    p0, p180 = pipeline.pair_0_180(a)
    t0, t180 = tomotools4.get_angles_at_180_deg(a)
    assert (p0, p180) == (t0[0], t180[0])
    with pytest.raises(ValueError):
        pipeline.pair_0_180(np.array([0.0, 10.0, 20.0]))


def test_output_window_rect_and_circle():
    ss = simple_scan()
    roi = ROI(4, 68, 2, 38)
    r = make_recipe(ss.scan, roi)
    assert pipeline.output_shape(r) == (36, 64, 64)
    r.recon['xy_roi'] = {'kind': 'rect', 'x0': 10, 'x1': 30, 'y0': 5, 'y1': 25}
    assert pipeline.output_shape(r) == (36, 20, 20)
    r.recon['xy_roi'] = {'kind': 'circle', 'cx': 31.5, 'cy': 31.5, 'r': 10}
    (y0, y1, x0, x1), mask = pipeline.output_window(r)
    assert mask.shape == (y1 - y0, x1 - x0)
    assert mask.sum() > 0.9 * np.pi * 100


# --- конвейер по слоям -----------------------------------------------------------------------------------------

@pytest.mark.parametrize('make', [simple_scan, advanced_scan], ids=['simple', 'advanced'])
def test_slabs_equal_full_crop(make):
    """Слои с запасом дают то же, что обработка всего кропа сразу (наклон оси и сдвиги образца)."""
    ss = make()
    roi = inner_roi(ss, 3, 1)
    crop = ph.make_crop(ss.frames, roi)
    ax = Axis(center_x=ss.center_x, y_ref=ss.y_ref, tilt_deg=ss.tilt_deg)
    prep = pipeline.prepare(ss.scan, crop, make_recipe(ss.scan, roi, ax), xp=np)
    if ss.scan.is_advanced:
        assert np.abs(prep.frame_sy).max() > 1.0          # сдвиги измерены и участвуют в запасе
    ref = full_crop_reference(crop, prep)
    m = pipeline.margin(prep, roi.width)
    for rows in (5, 16):
        got = np.concatenate([np.asarray(pipeline.process_slab(crop, prep, (a, min(a + rows, roi.height)), m, None,
                                                               xp=np))
                              for a in range(0, roi.height, rows)])
        scale = np.abs(ref).max()
        assert np.abs(got - ref).max() < 1e-4 * scale


@pytest.mark.parametrize('make', [simple_scan, advanced_scan], ids=['simple', 'advanced'])
def test_smoothed_slabs_equal_full_crop(make, per_row_rings):
    """Сглаживание: слой с ореолом ±h (кольца на всех его строках, затем фильтр) даёт то же, что кольца и фильтр по
    всему кропу сразу — и у краёв кропа, и при слое меньше ореола."""
    ss = make()
    roi = inner_roi(ss, 3, 1)
    crop = ph.make_crop(ss.frames, roi)
    ax = Axis(center_x=ss.center_x, y_ref=ss.y_ref, tilt_deg=ss.tilt_deg)
    prep = pipeline.prepare(ss.scan, crop, make_recipe(ss.scan, roi, ax), xp=np)
    rp = rings.resolve('strong')
    m = pipeline.margin(prep, roi.width)
    base = full_crop_reference(crop, prep, rp)
    cases = [({'sigma': 1.5}, 5)] if ss.scan.is_advanced else [({'sigma': 1.5}, 5), ({'sigma': 1.5}, 16),
                                                               ({'sigma': 0.8, 'deblur': 'unsharp'}, 5)]
    for block, rows in cases:
        sp = smoothing.resolve(block)
        assert smoothing.halo_rows(sp) >= 3
        ref = np.asarray(smoothing.apply(base, sp, xp=np))
        scale = ref.max() - ref.min()
        assert np.abs(ref - base).max() > 0.01 * scale                                   # фильтр заметен
        got = np.concatenate([np.asarray(pipeline.smoothed_slab(crop, prep, (a, min(a + rows, roi.height)), m,
                                                                rp, sp, xp=np))
                              for a in range(0, roi.height, rows)])
        assert np.abs(got - ref).max() < 1e-4 * scale, (block, rows)


def test_standard_scan_matches_notebook_path():
    """Простой скан: нормировка + выравнивание оси как в ноутбуке (normalize_projections + apply_axis_correction)."""
    import tomotools4
    ss = simple_scan()
    roi = ROI(3, 69, 1, 39)
    crop = ph.make_crop(ss.frames, roi)
    sc = ss.scan
    ax = Axis(center_x=ss.center_x, y_ref=ss.y_ref, tilt_deg=ss.tilt_deg)
    prep = pipeline.prepare(sc, crop, make_recipe(sc, roi, ax), xp=np)

    frames = crop.frames.astype('float32')
    dark = np.median(frames[sc.dark_idx], axis=0).astype('float32')
    empty = np.median(frames[sc.empty_idx], axis=0).astype('float32') - dark
    empty[empty < 1] = 1
    data_crop = frames[prep.idx] - dark
    tomotools4.normalize_projections(data_crop, empty)
    ref = tomotools4.apply_axis_correction(data_crop, prep.shift_x, prep.alfa)

    m = pipeline.margin(prep, roi.width)
    got = np.concatenate([np.asarray(pipeline.process_slab(crop, prep, (a, min(a + 8, roi.height)), m, None, xp=np))
                          for a in range(0, roi.height, 8)])
    assert np.abs(got - ref).max() < 1e-4 * np.abs(ref).max()


# --- запуск по рецепту -----------------------------------------------------------------------------------------

def _corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


@pytest.mark.parametrize('make', [simple_scan, advanced_scan], ids=['simple', 'advanced'])
def test_run_recipe_end_to_end(tmp_path, make):
    ss = make()
    path = write_h5(ss, tmp_path / 'scan.h5')
    scan = data.open_scan(path)
    roi = inner_roi(ss, 2, 4)
    r = make_recipe(scan, roi)                                  # ось — авто
    r.recon['angles'] = 'full_halves'
    out = tmp_path / 'out'
    res = pipeline.run_recipe(r, path, str(out), str(tmp_path / 'cache'), name='образец 1', backend='cpu',
                              slab_rows=7)

    # авто-ось близка к истинной
    ax = res.recipe.axis
    assert abs(ax.center_at(ss.y_ref) - ss.center_x) < 0.25
    assert abs(ax.tilt_deg - ss.tilt_deg) < 0.05
    if scan.is_advanced:
        sy = res.recipe.repositioning['shifts']['sy']
        # ndi.shift(data_check, s) ≈ data: накопленный сдвиг сегмента k = −смещение образца в нём
        assert np.allclose(np.cumsum(sy), [-1.3, -2.1], atol=0.11)
        assert np.allclose(np.cumsum(res.recipe.repositioning['shifts']['sx']), [0.8, -0.6], atol=0.11)

    # файлы
    doc = json.loads((out / 'result.json').read_text(encoding='utf-8'))
    shape = tuple(doc['volume']['shape'])
    assert shape == (roi.height, roi.width, roi.width)
    raw = out / doc['volume']['file']
    assert raw.name == 'образец_1.{}_{}_{}.1.raw'.format(*shape)
    vol = np.fromfile(str(raw), dtype='<f4').reshape(shape)
    assert (out / 'tomo.образец_1.1.hx').is_file()
    assert doc['binned'] and doc['binned'][0]['shape'] == [s // 4 for s in shape]
    assert recipe_mod.load(out / 'recipe.json').axis is not None
    assert doc['recipe_sha256'] == recipe_mod.sha256(res.recipe)

    # срез совпадает с истинным объектом (ось в центре кропа, единицы 1/мм)
    z = ss.scan.height // 2 - roi.y0
    zeta = (roi.y0 + z) - ss.y_ref
    truth = ph.slice_truth(ss.blobs, roi.width, zeta=zeta)
    assert _corr(vol[z], truth) > 0.95
    assert np.isclose(vol[z].sum() * 0.009, truth.sum(), rtol=0.1)


def test_run_recipe_with_smoothing_equals_full_crop_filter(tmp_path, per_row_rings):
    """Задача со сглаживанием (слои по 5 строк, часть срезов): срезы — FBP синограмм после колец и фильтра по всему
    кропу; ореол слоёв берёт соседние строки кропа и за пределами recon.slices."""
    ss = simple_scan()
    path = write_h5(ss, tmp_path / 'scan.h5')
    scan = data.open_scan(path)
    roi = ROI(3, 69, 2, 38)
    ax = Axis(ss.center_x, ss.y_ref, ss.tilt_deg)
    r = make_recipe(scan, roi, ax)
    r.rings = {'preset': 'medium', 'params': None}
    r.smoothing = dict(smoothing.default_block(), sigma=1.5)
    r.recon['slices'] = [roi.y0 + 4, roi.y0 + 17]
    res = pipeline.run_recipe(r, path, str(tmp_path / 'out'), str(tmp_path / 'cache'), backend='cpu', slab_rows=5)
    doc = res.result
    vol = np.fromfile(os.path.join(res.out_dir, doc['volume']['file']), '<f4').reshape(doc['volume']['shape'])
    assert vol.shape == (13, 66, 66)
    assert doc['timings']['halo_rows'] == smoothing.halo_rows(smoothing.resolve(r.smoothing)) == 16
    assert res.recipe.smoothing['sigma'] == 1.5 and doc['recipe']['smoothing']['sigma'] == 1.5
    assert recipe_mod.load(tmp_path / 'out' / 'recipe.json').smoothing == res.recipe.smoothing

    crop = ph.make_crop(ss.frames, roi)
    prep = pipeline.prepare(scan, crop, r, xp=np)
    sp = smoothing.resolve(r.smoothing)
    ref_sino = full_crop_reference(crop, prep, rings.resolve('medium'), sp)[4:17]
    ref = fbp.recon_rows(ref_sino, prep.angles, 0.009, backend='cpu')
    assert np.abs(vol - ref).max() < 1e-4 * (ref.max() - ref.min())
    # без сглаживания — другой объём (фильтр действительно применён)
    r.smoothing = smoothing.default_block()
    off = pipeline.run_recipe(r, path, str(tmp_path / 'off'), str(tmp_path / 'cache'), backend='cpu', slab_rows=5)
    vol_off = np.fromfile(os.path.join(off.out_dir, off.result['volume']['file']), '<f4').reshape(vol.shape)
    assert np.abs(vol_off - vol).max() > 0.01 * (ref.max() - ref.min())
    assert 'halo_rows' not in off.result['timings']


def test_axis_is_invariant_to_roi_shift(tmp_path):
    """Одна и та же ось в координатах детектора при сдвинутом ROI даёт тот же срез."""
    ss = simple_scan(tilt_deg=0.0)
    path = write_h5(ss, tmp_path / 'scan.h5')
    scan = data.open_scan(path)
    ax = Axis(center_x=ss.center_x, y_ref=ss.y_ref, tilt_deg=0.0)
    vols = []
    for x0 in (2, 5):
        roi = ROI(x0, x0 + 64, 10, 20)
        res = pipeline.run_recipe(make_recipe(scan, roi, ax), path, str(tmp_path / 'o{}'.format(x0)),
                                  str(tmp_path / 'cache'), backend='cpu')
        doc = res.result
        vols.append(np.fromfile(os.path.join(res.out_dir, doc['volume']['file']), '<f4').reshape(doc['volume']['shape']))
    inner = (slice(None), slice(12, 52), slice(12, 52))
    assert _corr(vols[0][inner], vols[1][inner]) > 0.999


def test_run_recipe_cancel_removes_volume(tmp_path):
    ss = simple_scan()
    path = write_h5(ss, tmp_path / 'scan.h5')
    scan = data.open_scan(path)
    r = make_recipe(scan, ROI(2, 70, 4, 36), Axis(ss.center_x, ss.y_ref, ss.tilt_deg))
    cancel = threading.Event()

    def progress(frac, stage):
        if stage == 'recon' and frac > 0.5:
            cancel.set()

    out = tmp_path / 'out'
    with pytest.raises(Cancelled):
        pipeline.run_recipe(r, path, str(out), str(tmp_path / 'cache'), progress=progress, cancel=cancel,
                            backend='cpu', slab_rows=4)
    assert not any(p.suffix in ('.raw', '.hx', '.json') for p in out.iterdir())


def test_estimate(tmp_path):
    ss = simple_scan()
    scan = data.open_scan(write_h5(ss, tmp_path / 'scan.h5'))
    r = make_recipe(scan, ROI(2, 70, 4, 36))
    est = pipeline.estimate(r, scan)
    assert est['volume_shape'] == [32, 68, 68]
    assert est['volume_bytes'] == 32 * 68 * 68 * 4
    assert est['n_angles_used'] == 60                          # first_180 из 0..357
    assert est['halo_rows'] == 0 and est['ring_rows_factor'] == 1.0
    # сглаживание: слои (на CPU — по 16 строк) с ореолом ±16 — строк через кольца больше, чем срезов
    r.smoothing = dict(smoothing.default_block(), sigma=1.5)
    est = pipeline.estimate(r, scan)
    assert est['halo_rows'] == 16
    assert est['ring_rows_factor'] == pytest.approx((32 + 32) / 32.0)       # 2 слоя, у каждого ореол до краёв
    assert pipeline.ring_rows_total((0, 32), 16, 16, 32) == 64
    assert pipeline.ring_rows_total((10, 20), 4, 2, 32) == 10 + 3 * 4


# --- cli -------------------------------------------------------------------------------------------------------

def test_cli_suggest_and_run(tmp_path, capsys):
    ss = simple_scan()
    path = write_h5(ss, tmp_path / 'scan.h5', exp_id='exp-42')
    rpath = tmp_path / 'recipe.json'
    assert cli.main(['suggest', path, '--out', str(rpath), '--bin', '2']) == 0
    r = recipe_mod.load(rpath)
    assert r.input['exp_id'] == 'exp-42'
    assert r.pixel_size['source'] == 'hdf5' and r.pixel_size['value_mm'] == 0.009
    # предложенная область охватывает объект
    assert r.fov.x0 <= 20 and r.fov.x1 >= 52
    capsys.readouterr()

    r.recon['slices'] = [r.fov.y0 + 5, r.fov.y0 + 9]
    recipe_mod.save(r, rpath)
    out = tmp_path / 'out'
    assert cli.main(['run', path, '--recipe', str(rpath), '--out', str(out), '--backend', 'cpu']) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary['volume']['shape'][0] == 4
    assert (out / 'result.json').is_file()


def test_cli_resolves_exp_id_in_src_dir(tmp_path):
    """id эксперимента: раскладка rbtm-storage <src>/<id>/before_processing/<id>.h5, затем плоская <src>/<id>.h5."""
    ss = simple_scan()
    write_h5(ss, tmp_path / 'abc.h5')
    assert cli.resolve_scan_path('abc', str(tmp_path)) == os.path.join(str(tmp_path), 'abc.h5')
    storage_layout = tmp_path / 'abc' / 'before_processing'
    storage_layout.mkdir(parents=True)
    write_h5(ss, storage_layout / 'abc.h5')
    assert cli.resolve_scan_path('abc', str(tmp_path)) == str(storage_layout / 'abc.h5')     # приоритет
    # путь к файлу — как есть
    assert cli.resolve_scan_path(str(tmp_path / 'abc.h5'), None) == str(tmp_path / 'abc.h5')
    with pytest.raises(FileNotFoundError) as exc:
        cli.resolve_scan_path('nope', str(tmp_path))
    assert os.path.join('nope', 'before_processing', 'nope.h5') in str(exc.value)
    assert os.path.join(str(tmp_path), 'nope.h5') in str(exc.value)


def test_cli_migrate_rec_config_ini(tmp_path, capsys):
    """Рецепт из rec_config.ini ноутбука: ROI и ось ноутбука (относительно кропа) → координаты детектора."""
    ss = simple_scan()
    path = write_h5(ss, tmp_path / 'scan.h5')
    roi = ROI(3, 69, 2, 38)
    shift_x, alfa = axis_mod.to_crop_params(Axis(ss.center_x, ss.y_ref, ss.tilt_deg), roi)
    ini = tmp_path / 'rec_config.ini'
    ini.write_text('[roi]\nx_min = 3\nx_max = 69\ny_min = 2\ny_max = 38\n\n'
                   '[axis_corr]\nshift_x = {}\nalfa = {}\n'.format(shift_x, alfa), encoding='utf-8')
    out = tmp_path / 'recipe.json'
    assert cli.main(['migrate', path, '--ini', str(ini), '--out', str(out)]) == 0
    capsys.readouterr()
    r = recipe_mod.load(out)
    assert r.fov == roi
    assert r.axis.center_at(ss.y_ref) == pytest.approx(ss.center_x)
    assert r.axis.tilt_deg == pytest.approx(ss.tilt_deg)
    assert r.pixel_size == {'value_mm': 0.009, 'source': 'hdf5', 'user_edited': False}
    assert r.repositioning['enabled'] is False


def test_cli_run_slices_and_compare(tmp_path, capsys):
    """Пробный запуск части срезов (--slices) сравнивается с полным объёмом по тем же строкам детектора."""
    ss = simple_scan()
    path = write_h5(ss, tmp_path / 'scan.h5')
    scan = data.open_scan(path)
    rpath = tmp_path / 'recipe.json'
    recipe_mod.save(make_recipe(scan, ROI(3, 69, 2, 38), Axis(ss.center_x, ss.y_ref, ss.tilt_deg)), rpath)
    cache = str(tmp_path / 'cache')
    assert cli.main(['run', path, '--recipe', str(rpath), '--out', str(tmp_path / 'full'), '--cache', cache,
                     '--backend', 'cpu']) == 0
    assert cli.main(['run', path, '--recipe', str(rpath), '--out', str(tmp_path / 'part'), '--cache', cache,
                     '--backend', 'cpu', '--slices', '10', '20', '--slab-rows', '3']) == 0
    capsys.readouterr()
    full = json.loads((tmp_path / 'full' / 'result.json').read_text(encoding='utf-8'))
    old_raw = str(tmp_path / 'full' / full['volume']['file'])

    from reconengine import compare
    rows = compare.compare(old_raw, str(tmp_path / 'part'), step=1)
    assert [z for z, _, _ in rows] == list(range(10, 20))
    assert all(c > 0.99999 and abs(ratio - 1) < 1e-4 for _, c, ratio in rows)
    assert cli.main(['compare', old_raw, str(tmp_path / 'part'), '--step', '4']) == 0
    assert 'минимум корреляции' in capsys.readouterr().out


def test_slab_rows_for_memory():
    n, w = 400, 3216
    six_gb = pipeline.slab_rows_for_memory(int(5.8e9), n, w, 43)
    eight_gb = pipeline.slab_rows_for_memory(int(7.5e9), n, w, 43)
    assert 40 <= six_gb < eight_gb <= 256                     # на 6 ГБ — десятки строк, с ростом памяти — больше
    assert pipeline.slab_rows_for_memory(int(0.5e9), n, w, 43) == 4          # не меньше минимума
    assert pipeline.slab_rows_for_memory(int(80e9), n, w, 43) == 256         # не больше максимума
    # сглаживание: ореол 2h строк на входе и выходе, копия результата фильтра и его рабочие порции — слой меньше
    smooth = pipeline.slab_rows_for_memory(int(5.8e9), n, w, 43, halo=16)
    assert 20 <= smooth < six_gb
    assert pipeline.slab_rows_for_memory(int(5.8e9), n, w, 43, halo=0) == six_gb


@pytest.mark.parametrize('sigma', [None, 1.0], ids=['plain', 'smoothing'])
def test_run_recipe_retries_slab_on_gpu_oom(tmp_path, monkeypatch, sigma):
    """Нехватка памяти GPU на слое — слой вдвое меньше и заново; объём тот же, что без сбоя (и со сглаживанием:
    слой берётся с ореолом)."""
    ss = simple_scan()
    path = write_h5(ss, tmp_path / 'scan.h5')
    scan = data.open_scan(path)
    r = make_recipe(scan, ROI(3, 69, 2, 38), Axis(ss.center_x, ss.y_ref, ss.tilt_deg))
    r.smoothing = dict(smoothing.default_block(), sigma=sigma)
    hs = smoothing.halo_rows(smoothing.resolve(r.smoothing))
    ref = pipeline.run_recipe(r, path, str(tmp_path / 'ref'), str(tmp_path / 'cache'), backend='cpu', slab_rows=16)

    class OutOfMemoryError(MemoryError):
        pass

    real = pipeline.process_slab
    calls = []

    def flaky(crop, prep, out_rows, *args, **kwargs):
        calls.append(out_rows)
        if len(calls) == 1:
            raise OutOfMemoryError('out of memory allocating')
        return real(crop, prep, out_rows, *args, **kwargs)

    monkeypatch.setattr(pipeline, 'process_slab', flaky)
    res = pipeline.run_recipe(r, path, str(tmp_path / 'oom'), str(tmp_path / 'cache'), backend='cpu', slab_rows=16)
    assert calls[0] == (0, 16 + hs) and calls[1] == (0, 8 + hs)
    vol = [np.fromfile(os.path.join(x.out_dir, x.result['volume']['file']), '<f4') for x in (ref, res)]
    assert np.allclose(vol[0], vol[1], atol=1e-6 * np.abs(vol[0]).max())
