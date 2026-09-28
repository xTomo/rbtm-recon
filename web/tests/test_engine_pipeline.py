"""Тесты reconengine.pipeline и cli: конвейер по слоям, совпадение со старым путём, запуск по рецепту."""
import json
import os
import threading

import h5py
import numpy as np
import pytest

import engine_phantom as ph
from reconengine import axis as axis_mod
from reconengine import cli, data, gpu, pipeline, preprocess
from reconengine import recipe as recipe_mod
from reconengine.model import Axis, Cancelled, ROI

ANGLES = np.arange(0.0, 360.0, 3.0)          # 120 углов, полный оборот


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv('RECON_ENGINE_CPU', '1')
    gpu.reset_backend()
    yield
    gpu.reset_backend()


def write_h5(ss: ph.SyntheticScan, path, chunk_frames: int = 5, exp_id: str = 'synthetic') -> str:
    """Синтетический скан → HDF5 v2 в раскладке rbtm-storage (gzip-чанки из целых кадров)."""
    sc = ss.scan
    n, h, w = ss.frames.shape
    with h5py.File(str(path), 'w') as f:
        md = f.create_group('metadata')
        md.create_dataset('experiment_id', data=exp_id.encode('utf8'))
        md.create_dataset('is_advanced', data=bool(sc.is_advanced))
        md.create_dataset('series_length', data=int(sc.series_length))
        md.create_dataset('pixel_size', data=0.009)
        md.create_dataset('detector_model', data=b'synthetic-detector')
        tl = f.create_group('timeline')
        tl.create_dataset('modes', data=sc.modes.astype('uint8'))
        tl.create_dataset('angles', data=sc.angles.astype('float32'))
        tl.create_dataset('frame_numbers', data=sc.frame_numbers.astype('int64'))
        ds = f.create_group('images').create_dataset(
            'all', shape=(n, h, w), dtype='uint16', chunks=(chunk_frames, h, w), compression='gzip',
            compression_opts=4, shuffle=False)
        for i in range(n):
            ds[i] = ss.frames[i]
    return str(path)


def simple_scan(**kw):
    kw.setdefault('height', 40)
    kw.setdefault('width', 72)
    kw.setdefault('center_x', 35.3)
    kw.setdefault('y_ref', 19.5)
    kw.setdefault('tilt_deg', 0.6)
    return ph.make_synthetic_scan(ANGLES, **kw)


def advanced_scan(**kw):
    # кадр крупнее: на 40×72 фантом обрезается краями и фазовая корреляция пары ошибается
    kw.setdefault('height', 96)
    kw.setdefault('width', 128)
    kw.setdefault('center_x', 63.3)
    kw.setdefault('y_ref', 47.5)
    kw.setdefault('tilt_deg', 0.6)
    return ph.make_synthetic_scan(ANGLES, advanced=True, n_segments=3,
                                  segment_offsets=[(0.0, 0.0), (1.3, -0.8), (2.1, 0.6)], **kw)


def inner_roi(ss, dx, dy):
    return ROI(dx, ss.scan.width - dx, dy, ss.scan.height - dy)


def make_recipe(scan, roi, ax=None, **changes):
    r = recipe_mod.default_recipe(scan.exp_id, scan.fingerprint, roi, 0.009, 'user', scan.is_advanced)
    r.axis = ax
    r.rings = {'preset': 'off', 'params': None}
    for key, value in changes.items():
        setattr(r, key, value)
    return r


def full_crop_reference(crop, prep):
    """Та же обработка, что в process_slab, но сразу на всём кропе (как в ноутбуке): (h, n, w)."""
    raw = np.asarray(crop.frames[prep.idx])
    norm = np.asarray(preprocess.normalize_slab(raw, prep.fnums, prep.de, xp=np))
    norm = pipeline._apply_shifts(norm, prep.frame_sy, prep.frame_sx, np)
    out = np.stack([np.asarray(axis_mod.transform_image(fr, prep.shift_x, prep.alfa, xp=np)) for fr in norm])
    return np.swapaxes(out, 0, 1)


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
    ss = simple_scan()
    write_h5(ss, tmp_path / 'abc.h5')
    assert cli.resolve_scan_path('abc', str(tmp_path)) == os.path.join(str(tmp_path), 'abc.h5')
    with pytest.raises(FileNotFoundError):
        cli.resolve_scan_path('nope', str(tmp_path))


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
