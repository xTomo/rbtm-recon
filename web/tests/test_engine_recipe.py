"""Тесты reconengine.recipe: рецепт реконструкции, схема rbtm-recon-recipe/1."""
import copy
import json
import logging

import pytest

import reconengine
from reconengine import fbp, recipe as recipe_mod, rings, smoothing
from reconengine.model import ROI, Axis


def make_roi(x0=10, x1=90, y0=5, y1=45):
    return ROI(x0, x1, y0, y1)


def make_default(exp_id='exp-1', fingerprint='abc123', roi=None, is_advanced=True):
    roi = roi or make_roi()
    return recipe_mod.default_recipe(exp_id, fingerprint, roi, pixel_size_value=0.009,
                                     pixel_size_source='detector', is_advanced=is_advanced)


# --- default_recipe ----------------------------------------------------------

def test_default_recipe_structure():
    roi = make_roi()
    r = make_default(roi=roi, is_advanced=True)

    assert r.schema == 'rbtm-recon-recipe/1'
    assert r.engine['version'] == reconengine.__version__
    assert isinstance(r.engine['git'], str)
    assert r.author == ''
    assert r.input == {'exp_id': 'exp-1', 'format': 'hdf5-v2', 'fingerprint': 'abc123'}
    assert r.pixel_size == {'value_mm': 0.009, 'source': 'detector', 'user_edited': False}
    assert r.fov == roi
    assert r.axis is None
    assert r.repositioning == {'enabled': True, 'shifts': None}
    assert r.recon['slices'] == [roi.y0, roi.y1]
    assert r.recon['xy_roi'] == {'kind': None}
    assert r.recon['algorithm'] == 'FBP'
    assert r.recon['angles'] == 'first_180'
    assert r.normalization == 'auto'
    assert r.rings == {'preset': 'medium', 'params': None}
    assert r.outputs == {'full': True, 'binning': [4], 'dtype': 'float32'}
    assert r.smoothing == {'sigma': None, 'deblur': 'wiener', 'balance': 0.02, 'amount': 1.5}   # выключено
    assert smoothing.resolve(r.smoothing) is None
    assert r.provenance == {'steps': {'fov': 'auto', 'axis': 'auto', 'rings': 'auto', 'smoothing': 'auto',
                                      'run': 'auto'}}


def test_default_recipe_repositioning_disabled_when_not_advanced():
    r = make_default(is_advanced=False)
    assert r.repositioning['enabled'] is False


# --- to_dict / from_dict ------------------------------------------------------

def test_to_dict_from_dict_roundtrip():
    r = make_default()
    d = recipe_mod.to_dict(r)
    # JSON-сериализуемость
    json.dumps(d, ensure_ascii=False)

    r2 = recipe_mod.from_dict(d)
    assert recipe_mod.to_dict(r2) == d


def test_from_dict_rejects_unknown_major_schema_version():
    d = recipe_mod.to_dict(make_default())
    d['schema'] = 'rbtm-recon-recipe/2'
    with pytest.raises(ValueError):
        recipe_mod.from_dict(d)


def test_from_dict_rejects_foreign_schema_name():
    d = recipe_mod.to_dict(make_default())
    d['schema'] = 'something-else/1'
    with pytest.raises(ValueError):
        recipe_mod.from_dict(d)


def test_from_dict_ignores_unknown_top_level_keys_with_warning(caplog):
    d = recipe_mod.to_dict(make_default())
    d['totally_unknown_field'] = 42
    with caplog.at_level(logging.WARNING, logger='reconengine.recipe'):
        r = recipe_mod.from_dict(d)
    assert isinstance(r, recipe_mod.Recipe)
    assert any('totally_unknown_field' in rec.message for rec in caplog.records)


@pytest.mark.parametrize('mutate,message_part', [
    (lambda d: d['pixel_size'].update(value_mm=-1.0), 'value_mm'),
    (lambda d: d['pixel_size'].update(source='guess'), 'source'),
    (lambda d: d['recon'].update(angles='full_360'), 'angles'),
    (lambda d: d['recon'].update(algorithm='SIRT'), 'algorithm'),
    (lambda d: d['rings'].update(preset='ultra'), 'preset'),
    (lambda d: d.update(normalization='weird'), 'normalization'),
    (lambda d: d['outputs'].update(binning=[4, 0]), 'binning'),
    (lambda d: d['outputs'].update(binning=[4, -2]), 'binning'),
    (lambda d: d['outputs'].update(binning=[4, 2.5]), 'binning'),
    (lambda d: d['smoothing'].update(sigma=0.1), 'smoothing.sigma'),
    (lambda d: d['smoothing'].update(sigma=5), 'smoothing.sigma'),
    (lambda d: d['smoothing'].update(sigma='1.5'), 'smoothing.sigma'),
    (lambda d: d['smoothing'].update(sigma=True), 'smoothing.sigma'),
    (lambda d: d['smoothing'].update(sigma=1.5, deblur='rl'), 'smoothing.deblur'),
    (lambda d: d['smoothing'].update(sigma=1.5, balance=0), 'smoothing.balance'),
    (lambda d: d['smoothing'].update(sigma=1.5, amount=-1), 'smoothing.amount'),
    (lambda d: d['smoothing'].update(radius=3), 'smoothing'),
    (lambda d: d.update(smoothing=[1.5]), 'smoothing'),
    (lambda d: d['provenance']['steps'].update(smoothing='maybe'), 'smoothing'),
])
def test_from_dict_rejects_invalid_values(mutate, message_part):
    d = recipe_mod.to_dict(make_default())
    mutate(d)
    with pytest.raises(ValueError, match=message_part):
        recipe_mod.from_dict(d)


def test_smoothing_block_roundtrip_and_partial():
    d = recipe_mod.to_dict(make_default())
    d['smoothing'] = {'sigma': 1.5, 'deblur': 'unsharp'}                    # недостающие поля — по умолчанию
    r = recipe_mod.from_dict(d)
    assert r.smoothing == {'sigma': 1.5, 'deblur': 'unsharp', 'balance': 0.02, 'amount': 1.5}
    assert smoothing.resolve(r.smoothing)['deblur'] == 'unsharp'
    d2 = recipe_mod.to_dict(r)
    assert recipe_mod.to_dict(recipe_mod.from_dict(json.loads(json.dumps(d2)))) == d2
    # выключено: sigma null или 0; null вместо блока — блок по умолчанию
    for off in ({'sigma': 0}, {'sigma': None, 'deblur': 'none'}, None):
        d['smoothing'] = off
        assert smoothing.resolve(recipe_mod.from_dict(d).smoothing) is None


def test_recipe_without_smoothing_block_reads_as_off_with_same_sha():
    """Рецепт, записанный до появления блока, читается как выключенный; хэш — как у рецепта с выключенным блоком
    (выключенный блок в хэш не входит — хэши старых result.json не меняются)."""
    r = make_default()
    old = recipe_mod.to_dict(r)
    del old['smoothing']
    old['provenance']['steps'].pop('smoothing')
    r_old = recipe_mod.from_dict(old)
    assert r_old.smoothing == smoothing.default_block()
    r.provenance['steps'].pop('smoothing')
    assert recipe_mod.sha256(r_old) == recipe_mod.sha256(r)
    off = copy.deepcopy(r)
    off.smoothing = {'sigma': 0, 'deblur': 'none', 'balance': 0.5, 'amount': 0.0}     # выключено иначе
    assert recipe_mod.sha256(off) == recipe_mod.sha256(r)
    on = copy.deepcopy(r)
    on.smoothing = dict(smoothing.default_block(), sigma=1.5)
    assert recipe_mod.sha256(on) != recipe_mod.sha256(r)
    on2 = copy.deepcopy(on)
    on2.smoothing['balance'] = 0.05
    assert recipe_mod.sha256(on2) != recipe_mod.sha256(on)


def test_angle_modes_match_fbp_module():
    # angles валидны ровно те, что объявлены в fbp.ANGLE_MODES
    d = recipe_mod.to_dict(make_default())
    for mode in fbp.ANGLE_MODES:
        d['recon']['angles'] = mode
        recipe_mod.from_dict(d)  # не должно бросать


def test_rings_presets_match_rings_module():
    d = recipe_mod.to_dict(make_default())
    for preset in rings.PRESETS:
        d['rings']['preset'] = preset
        recipe_mod.from_dict(d)  # не должно бросать


# --- validate (границы кадра) -------------------------------------------------

def test_validate_ok_for_matching_frame():
    r = make_default(roi=make_roi(0, 100, 0, 50))
    recipe_mod.validate(r, height=200, width=200)


def test_validate_roi_outside_frame_raises():
    r = make_default(roi=make_roi(0, 100, 0, 50))
    with pytest.raises(ValueError):
        recipe_mod.validate(r, height=40, width=200)


def test_validate_slices_outside_fov_raises():
    roi = make_roi(0, 100, 10, 50)
    r = make_default(roi=roi)
    r.recon['slices'] = [5, 20]  # 5 < roi.y0=10
    with pytest.raises(ValueError):
        recipe_mod.validate(r, height=200, width=200)


def test_validate_xy_roi_rect_outside_slice_raises():
    roi = make_roi(0, 64, 0, 40)
    r = make_default(roi=roi)
    r.recon['xy_roi'] = {'kind': 'rect', 'x0': 0, 'x1': 1000, 'y0': 0, 'y1': 10}
    with pytest.raises(ValueError):
        recipe_mod.validate(r, height=200, width=200)


def test_validate_xy_roi_circle_within_slice_ok():
    roi = make_roi(0, 64, 0, 40)
    r = make_default(roi=roi)
    r.recon['xy_roi'] = {'kind': 'circle', 'cx': 32, 'cy': 32, 'r': 20}
    recipe_mod.validate(r, height=200, width=200)


# --- sha256 --------------------------------------------------------------------

def test_sha256_ignores_created_and_author():
    r1 = make_default()
    r2 = copy.deepcopy(r1)
    r2.created = '2000-01-01T00:00:00+00:00'
    r2.author = 'someone else'
    assert recipe_mod.sha256(r1) == recipe_mod.sha256(r2)


def test_sha256_changes_with_content():
    r1 = make_default()
    r2 = copy.deepcopy(r1)
    r2.normalization = 'standard'
    assert recipe_mod.sha256(r1) != recipe_mod.sha256(r2)


# --- save / load -----------------------------------------------------------------

def test_save_load_roundtrip_and_atomic(tmp_path):
    r = make_default()
    path = tmp_path / 'recipe.json'
    recipe_mod.save(r, path)

    assert path.exists()
    text = path.read_text(encoding='utf-8')
    assert '\n  ' in text  # indent=2
    # никаких временных файлов не осталось
    leftovers = [p for p in tmp_path.iterdir() if p.name != 'recipe.json']
    assert leftovers == []

    r2 = recipe_mod.load(path)
    assert recipe_mod.to_dict(r2) == recipe_mod.to_dict(r)


# --- transferable_part / apply_template -----------------------------------------

def test_transferable_part_contains_only_portable_fields():
    r = make_default()
    r.rings['preset'] = 'strong'
    r.outputs['binning'] = [2, 4]
    r.recon['angles'] = 'full_halves'
    r.smoothing = dict(smoothing.default_block(), sigma=1.2)
    part = recipe_mod.transferable_part(r)

    assert set(part.keys()) == {'normalization', 'rings', 'smoothing', 'outputs', 'recon'}
    assert part['rings']['preset'] == 'strong'
    assert part['smoothing']['sigma'] == 1.2
    assert part['outputs']['binning'] == [2, 4]
    assert set(part['recon'].keys()) == {'algorithm', 'angles'}
    assert part['recon']['angles'] == 'full_halves'


def test_apply_template_keeps_scan_bound_fields_and_overrides_portable():
    template_source = make_default(exp_id='exp-A', fingerprint='fp-A', roi=make_roi(0, 64, 0, 64))
    template_source.rings['preset'] = 'weak'
    template_source.outputs['binning'] = [8]
    template_source.recon['angles'] = 'full_halves'
    template_source.smoothing['sigma'] = 1.5
    template = recipe_mod.transferable_part(template_source)

    target = make_default(exp_id='exp-B', fingerprint='fp-B', roi=make_roi(0, 32, 0, 32))
    target.recon['slices'] = [0, 20]
    result = recipe_mod.apply_template(target, template)

    # перенесено
    assert result.rings['preset'] == 'weak'
    assert result.outputs['binning'] == [8]
    assert result.recon['angles'] == 'full_halves'
    assert result.smoothing['sigma'] == 1.5
    # шаблон без блока сглаживания (сохранён до его появления) — сглаживание цели не меняется
    old_template = {k: v for k, v in template.items() if k != 'smoothing'}
    target.smoothing['sigma'] = 0.8
    assert recipe_mod.apply_template(target, old_template).smoothing['sigma'] == 0.8
    # не перенесено — осталось от target
    assert result.input['exp_id'] == 'exp-B'
    assert result.fov == target.fov
    assert result.recon['slices'] == [0, 20]
    assert result.recon['algorithm'] == 'FBP'


# --- from_rec_config_ini -----------------------------------------------------------

def _write_ini(path, roi_section, axis_section=None):
    lines = ['[roi]']
    lines += ['{} = {}'.format(k, v) for k, v in roi_section.items()]
    if axis_section is not None:
        lines.append('')
        lines.append('[axis_corr]')
        lines += ['{} = {}'.format(k, v) for k, v in axis_section.items()]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def test_from_rec_config_ini_migrates_half_open_roi(tmp_path, caplog):
    ini_path = tmp_path / 'rec_config.ini'
    _write_ini(ini_path, {'x_min': 2, 'x_max': 8, 'y_min': 1, 'y_max': 5},
              {'shift_x': 1.2, 'alfa': 0.3})

    with caplog.at_level(logging.INFO, logger='reconengine.recipe'):
        r = recipe_mod.from_rec_config_ini(ini_path, exp_id='exp-old', fingerprint='fp-old',
                                           frame_height=10, frame_width=10)

    assert r.fov == ROI(2, 8, 1, 5)
    assert r.input['exp_id'] == 'exp-old'
    assert r.input['fingerprint'] == 'fp-old'
    assert r.pixel_size['source'] == 'default'
    assert any('shift_x=1.2' in rec.message for rec in caplog.records)
    # ось ноутбука (относительно кропа) переведена в координаты детектора и обратно даёт те же параметры
    from reconengine import axis as axis_mod
    assert r.axis.method == 'notebook'
    shift_x, alfa = axis_mod.to_crop_params(r.axis, r.fov)
    assert shift_x == pytest.approx(1.2) and alfa == pytest.approx(0.3)


def test_from_rec_config_ini_without_axis_corr_leaves_axis_auto(tmp_path):
    ini_path = tmp_path / 'rec_config.ini'
    _write_ini(ini_path, {'x_min': 0, 'x_max': 10, 'y_min': 0, 'y_max': 10})
    r = recipe_mod.from_rec_config_ini(ini_path, exp_id='e', fingerprint='f', frame_height=20, frame_width=20)
    assert r.axis is None


def test_from_rec_config_ini_without_axis_corr_section(tmp_path):
    ini_path = tmp_path / 'rec_config.ini'
    _write_ini(ini_path, {'x_min': 0, 'x_max': 10, 'y_min': 0, 'y_max': 10})

    r = recipe_mod.from_rec_config_ini(ini_path, exp_id='e', fingerprint='f',
                                       frame_height=20, frame_width=20)
    assert r.fov == ROI(0, 10, 0, 10)


def test_from_rec_config_ini_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        recipe_mod.from_rec_config_ini(tmp_path / 'nope.ini', 'e', 'f', 10, 10)


def test_from_rec_config_ini_roi_outside_frame_raises(tmp_path):
    ini_path = tmp_path / 'rec_config.ini'
    _write_ini(ini_path, {'x_min': 0, 'x_max': 100, 'y_min': 0, 'y_max': 10})
    with pytest.raises(ValueError):
        recipe_mod.from_rec_config_ini(ini_path, 'e', 'f', frame_height=10, frame_width=10)
