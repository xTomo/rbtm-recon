"""Рецепт реконструкции — JSON-схема ``rbtm-recon-recipe/1``.

Часть полей привязана к конкретному файлу скана (``pixel_size``, ``fov``, ``axis``, ``repositioning``,
``recon.slices``, ``recon.xy_roi``) и не должна переноситься на другой скан. Остальные поля
(``normalization``, ``rings``, ``smoothing``, ``recon.algorithm``, ``recon.angles``, ``outputs``) — «переносимые»:
их можно сохранить как шаблон и применить к рецепту другого скана через :func:`apply_template`.

``smoothing`` — сглаживание проекций гауссом и деблюринг тем же ядром (:mod:`smoothing`):
``{sigma: null | число, deblur: 'wiener' | 'unsharp' | 'none', balance, amount}``; ``sigma`` null или 0 — выключено
(по умолчанию). Рецепт без блока (записанный до его появления) читается как выключенный; выключенный блок в
:func:`sha256` не входит — хэш старых рецептов не меняется.

Геометрия (``fov``, ``axis``) хранится через :class:`model.ROI` / :class:`model.Axis`, чтобы переиспользовать
их валидацию и (де)сериализацию; остальные секции — обычные словари с проверкой типов/диапазонов
в :func:`from_dict` (что рецепт синтаксически корректен) и :func:`validate` (что он согласован с конкретным
кадром — размеры, единственная точка, где нужны ``height``/``width``).
"""
from __future__ import annotations

import configparser
import copy
import dataclasses
import datetime
import functools
import hashlib
import json
import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

from . import __version__ as _ENGINE_VERSION
from . import fbp, pixelsize, rings, smoothing
from .model import ROI, Axis

logger = logging.getLogger(__name__)

SCHEMA_NAME = 'rbtm-recon-recipe'
SCHEMA = SCHEMA_NAME + '/1'
_SUPPORTED_MAJOR = '1'

_PIXEL_SIZE_SOURCES = {'mongo', 'detector', 'hdf5', 'default', 'user'}
_NORMALIZATION_MODES = {'auto', 'standard', 'timeline'}
_XY_ROI_KINDS = {None, 'rect', 'circle'}
_PROVENANCE_STATES = {'auto', 'checked'}
_ALGORITHMS = {'FBP'}

_TOP_LEVEL_KEYS = {
    'schema', 'engine', 'created', 'author', 'input', 'pixel_size', 'fov', 'axis',
    'repositioning', 'recon', 'normalization', 'rings', 'smoothing', 'outputs', 'provenance',
}
#: Верхнеуровневые поля, переносимые в другой рецепт целиком.
_TRANSFERABLE_TOP = ('normalization', 'rings', 'smoothing', 'outputs')
#: Шаги студии, чьё происхождение (auto | checked) записывается в ``provenance.steps``.
_PROVENANCE_STEPS = ('fov', 'axis', 'rings', 'smoothing', 'run')
#: Поля секции ``recon``, переносимые в другой рецепт (``slices``/``xy_roi`` привязаны к скану).
_TRANSFERABLE_RECON = ('algorithm', 'angles')


@dataclasses.dataclass
class Recipe:
    """Рецепт реконструкции. См. модульный docstring про привязанные/переносимые поля."""
    schema: str
    engine: Dict[str, Any]
    created: str
    author: str
    input: Dict[str, Any]
    pixel_size: Dict[str, Any]
    fov: ROI
    axis: Optional[Axis]
    repositioning: Dict[str, Any]
    recon: Dict[str, Any]
    normalization: str
    rings: Dict[str, Any]
    outputs: Dict[str, Any]
    provenance: Dict[str, Any]
    smoothing: Dict[str, Any] = dataclasses.field(default_factory=smoothing.default_block)


@functools.lru_cache(maxsize=1)
def _git_revision() -> str:
    """Короткий git-хэш HEAD этого чекаута, либо '' (git недоступен / это не репозиторий)."""
    try:
        proc = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=2, check=False,
        )
    except Exception:  # noqa: BLE001 — git может отсутствовать в окружении
        return ''
    return proc.stdout.strip() if proc.returncode == 0 else ''


def default_recipe(exp_id: str, fingerprint: str, roi: ROI, pixel_size_value: float,
                   pixel_size_source: str, is_advanced: bool) -> Recipe:
    """Рецепт по умолчанию для только что открытого скана.

    ``fov`` берётся из ``roi``; ``recon.slices`` по умолчанию = ``[roi.y0, roi.y1)``; ``axis`` не задана
    (определяется автоматически при запуске); ``repositioning.enabled`` — True для advanced-скана.
    """
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    fov = ROI(roi.x0, roi.x1, roi.y0, roi.y1, roi.preview_row)
    return Recipe(
        schema=SCHEMA,
        engine={'version': _ENGINE_VERSION, 'git': _git_revision()},
        created=now,
        author='',
        input={'exp_id': exp_id, 'format': 'hdf5-v2', 'fingerprint': fingerprint},
        pixel_size={
            'value_mm': float(pixel_size_value),
            'source': pixel_size_source,
            'user_edited': False,
        },
        fov=fov,
        axis=None,
        repositioning={'enabled': bool(is_advanced), 'shifts': None},
        recon={
            'slices': [fov.y0, fov.y1],
            'xy_roi': {'kind': None},
            'algorithm': 'FBP',
            'angles': 'first_180',
        },
        normalization='auto',
        rings={'preset': 'medium', 'params': None},
        outputs={'full': True, 'binning': [4], 'dtype': 'float32'},
        provenance={'steps': {'fov': 'auto', 'axis': 'auto', 'rings': 'auto', 'smoothing': 'auto', 'run': 'auto'}},
        smoothing=smoothing.default_block(),
    )


def _check_schema(schema: Any) -> None:
    if not isinstance(schema, str) or '/' not in schema:
        raise ValueError('recipe: некорректная схема: {!r}'.format(schema))
    name, _, version = schema.rpartition('/')
    if name != SCHEMA_NAME:
        raise ValueError('recipe: неизвестная схема: {!r}'.format(schema))
    major = version.split('.')[0]
    if major != _SUPPORTED_MAJOR:
        raise ValueError('recipe: неизвестная мажорная версия схемы: {!r}'.format(schema))


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError('recipe: ' + message)


def _validate_types(recipe: Recipe) -> None:
    """Проверка типов/диапазонов полей рецепта, не зависящая от конкретного кадра (кадр — см. :func:`validate`)."""
    ps = recipe.pixel_size
    _require(isinstance(ps.get('value_mm'), (int, float)) and ps['value_mm'] > 0,
             'pixel_size.value_mm должен быть положительным числом: {!r}'.format(ps.get('value_mm')))
    _require(ps.get('source') in _PIXEL_SIZE_SOURCES,
             'pixel_size.source неизвестен: {!r}'.format(ps.get('source')))
    _require(isinstance(ps.get('user_edited'), bool),
             'pixel_size.user_edited должен быть bool: {!r}'.format(ps.get('user_edited')))

    _require(isinstance(recipe.repositioning.get('enabled'), bool),
             'repositioning.enabled должен быть bool: {!r}'.format(recipe.repositioning.get('enabled')))
    shifts = recipe.repositioning.get('shifts')
    if shifts is not None:
        _require(isinstance(shifts, dict) and 'sy' in shifts and 'sx' in shifts,
                 'repositioning.shifts должен быть {{sy: [...], sx: [...]}} или None')
        _require(len(shifts['sy']) == len(shifts['sx']),
                 'repositioning.shifts: sy и sx разной длины')

    recon = recipe.recon
    slices = recon.get('slices')
    _require(isinstance(slices, (list, tuple)) and len(slices) == 2,
             'recon.slices должен быть [z0, z1]: {!r}'.format(slices))
    _require(int(slices[0]) < int(slices[1]), 'recon.slices: z0 должен быть < z1: {!r}'.format(slices))

    xy_roi = recon.get('xy_roi') or {'kind': None}
    kind = xy_roi.get('kind')
    _require(kind in _XY_ROI_KINDS, 'recon.xy_roi.kind неизвестен: {!r}'.format(kind))
    if kind == 'rect':
        for key in ('x0', 'x1', 'y0', 'y1'):
            _require(key in xy_roi, 'recon.xy_roi (rect): нет поля {!r}'.format(key))
        _require(xy_roi['x0'] < xy_roi['x1'], 'recon.xy_roi (rect): x0 должен быть < x1')
        _require(xy_roi['y0'] < xy_roi['y1'], 'recon.xy_roi (rect): y0 должен быть < y1')
    elif kind == 'circle':
        for key in ('cx', 'cy', 'r'):
            _require(key in xy_roi, 'recon.xy_roi (circle): нет поля {!r}'.format(key))
        _require(xy_roi['r'] > 0, 'recon.xy_roi (circle): r должен быть положительным')

    _require(recon.get('algorithm') in _ALGORITHMS,
             'recon.algorithm неизвестен: {!r}'.format(recon.get('algorithm')))
    _require(recon.get('angles') in fbp.ANGLE_MODES,
             'recon.angles неизвестен: {!r}'.format(recon.get('angles')))

    _require(recipe.normalization in _NORMALIZATION_MODES,
             'normalization неизвестен: {!r}'.format(recipe.normalization))

    preset = recipe.rings.get('preset')
    _require(preset in rings.PRESETS, 'rings.preset неизвестен: {!r}'.format(preset))
    params = recipe.rings.get('params')
    _require(params is None or isinstance(params, dict), 'rings.params должен быть словарём или None')

    outputs = recipe.outputs
    _require(isinstance(outputs.get('full'), bool), 'outputs.full должен быть bool')
    binning = outputs.get('binning')
    _require(isinstance(binning, (list, tuple)),
             'outputs.binning должен быть списком: {!r}'.format(binning))
    for b in binning:
        _require(isinstance(b, int) and not isinstance(b, bool) and b > 0,
                 'outputs.binning: значения должны быть положительными целыми: {!r}'.format(b))
    _require(isinstance(outputs.get('dtype'), str) and outputs['dtype'],
             'outputs.dtype должен быть непустой строкой')

    _validate_smoothing(recipe.smoothing)

    steps = recipe.provenance.get('steps', {})
    for key in _PROVENANCE_STEPS:
        if key in steps:
            _require(steps[key] in _PROVENANCE_STATES,
                     'provenance.steps.{} неизвестен: {!r}'.format(key, steps[key]))


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _validate_smoothing(block: Any) -> None:
    """Блок ``smoothing``: словарь с полями ``smoothing.DEFAULTS`` (типы), значения — :func:`smoothing.resolve`."""
    _require(isinstance(block, dict), 'smoothing должен быть словарём: {!r}'.format(block))
    unknown = set(block) - set(smoothing.DEFAULTS)
    _require(not unknown, 'smoothing: неизвестные поля {}'.format(sorted(unknown)))
    sigma = block.get('sigma')
    _require(sigma is None or _is_number(sigma), 'smoothing.sigma должен быть числом или null: {!r}'.format(sigma))
    _require(isinstance(block.get('deblur'), str), 'smoothing.deblur должен быть строкой: {!r}'.format(
        block.get('deblur')))
    for key in ('balance', 'amount'):
        _require(_is_number(block.get(key)), 'smoothing.{} должен быть числом: {!r}'.format(key, block.get(key)))
    try:
        smoothing.resolve(block)
    except ValueError as exc:
        raise ValueError('recipe: {}'.format(exc)) from None


def smoothing_block(block: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Блок ``smoothing`` рецепта из частичного (недостающие поля — по умолчанию; None — выключено), без проверки."""
    out = smoothing.default_block()
    if block is not None:
        _require(isinstance(block, dict), 'smoothing должен быть словарём или null: {!r}'.format(block))
        out.update(copy.deepcopy(block))
    return out


def validate(recipe: Recipe, height: int, width: int) -> None:
    """Проверить, что рецепт согласован с конкретным кадром ``height`` × ``width``.

    ``fov`` — через :meth:`model.ROI.validate`; ``recon.slices`` — внутри ``fov`` по y;
    ``recon.xy_roi`` — в пикселях среза (сторона среза = ширина ``fov``, т.к. FBP восстанавливает
    квадратные срезы шириной кропа).
    """
    recipe.fov.validate(height, width)

    z0, z1 = int(recipe.recon['slices'][0]), int(recipe.recon['slices'][1])
    _require(recipe.fov.y0 <= z0 < z1 <= recipe.fov.y1,
             'recon.slices [{}, {}) вне fov.y [{}, {})'.format(z0, z1, recipe.fov.y0, recipe.fov.y1))

    xy_roi = recipe.recon.get('xy_roi') or {'kind': None}
    kind = xy_roi.get('kind')
    side = recipe.fov.width
    if kind == 'rect':
        _require(0 <= xy_roi['x0'] < xy_roi['x1'] <= side,
                 'recon.xy_roi (rect) x вне среза шириной {}'.format(side))
        _require(0 <= xy_roi['y0'] < xy_roi['y1'] <= side,
                 'recon.xy_roi (rect) y вне среза шириной {}'.format(side))
    elif kind == 'circle':
        _require(0 <= xy_roi['cx'] <= side and 0 <= xy_roi['cy'] <= side,
                 'recon.xy_roi (circle) центр вне среза шириной {}'.format(side))


def to_dict(recipe: Recipe) -> Dict[str, Any]:
    """Рецепт → JSON-совместимый словарь (глубокая копия, безопасно изменять результат)."""
    return {
        'schema': recipe.schema,
        'engine': copy.deepcopy(recipe.engine),
        'created': recipe.created,
        'author': recipe.author,
        'input': copy.deepcopy(recipe.input),
        'pixel_size': copy.deepcopy(recipe.pixel_size),
        'fov': recipe.fov.to_dict(),
        'axis': recipe.axis.to_dict() if recipe.axis is not None else None,
        'repositioning': copy.deepcopy(recipe.repositioning),
        'recon': copy.deepcopy(recipe.recon),
        'normalization': recipe.normalization,
        'rings': copy.deepcopy(recipe.rings),
        'smoothing': copy.deepcopy(recipe.smoothing),
        'outputs': copy.deepcopy(recipe.outputs),
        'provenance': copy.deepcopy(recipe.provenance),
    }


def from_dict(d: Dict[str, Any]) -> Recipe:
    """Словарь (например, из JSON) → :class:`Recipe`, с валидацией типов/диапазонов.

    Неизвестные ключи верхнего уровня — игнорируются с предупреждением в лог. Неизвестная мажорная
    версия схемы — :class:`ValueError`. Нет блока ``smoothing`` (или null) — выключено; недостающие поля блока —
    по умолчанию.
    """
    _check_schema(d.get('schema'))

    unknown = set(d) - _TOP_LEVEL_KEYS
    if unknown:
        logger.warning('recipe: неизвестные поля верхнего уровня проигнорированы: %s', sorted(unknown))

    fov_d = d.get('fov')
    _require(isinstance(fov_d, dict), 'нет обязательного поля fov')
    fov = ROI.from_dict(fov_d)
    axis_d = d.get('axis')
    axis = Axis.from_dict(axis_d) if axis_d is not None else None

    recipe = Recipe(
        schema=d['schema'],
        engine=dict(d.get('engine') or {}),
        created=d.get('created', ''),
        author=d.get('author', ''),
        input=dict(d.get('input') or {}),
        pixel_size=dict(d.get('pixel_size') or {}),
        fov=fov,
        axis=axis,
        repositioning=copy.deepcopy(d.get('repositioning') or {'enabled': True, 'shifts': None}),
        recon=copy.deepcopy(d.get('recon') or {}),
        normalization=d.get('normalization', 'auto'),
        rings=copy.deepcopy(d.get('rings') or {'preset': 'medium', 'params': None}),
        outputs=copy.deepcopy(d.get('outputs') or {}),
        provenance=copy.deepcopy(d.get('provenance') or {'steps': {}}),
        smoothing=smoothing_block(d.get('smoothing')),
    )
    _validate_types(recipe)
    return recipe


def _atomic_write_json(path: Any, obj: Any) -> None:
    path = str(path)
    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix='.tmp-recipe-', suffix='.json', dir=directory)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def save(recipe: Recipe, path: Any) -> None:
    """Сохранить рецепт в JSON (UTF-8, indent=2) атомарно (временный файл + ``os.replace``)."""
    _atomic_write_json(path, to_dict(recipe))


def load(path: Any) -> Recipe:
    """Загрузить и провалидировать рецепт из JSON-файла."""
    with open(str(path), 'r', encoding='utf-8') as fh:
        d = json.load(fh)
    return from_dict(d)


def sha256(recipe: Recipe) -> str:
    """Sha256 канонического JSON рецепта (sort_keys, без ``created``/``author``). Выключенный ``smoothing`` не
    входит: рецепт без блока (записанный до его появления) и с выключенным блоком — один и тот же хэш."""
    d = to_dict(recipe)
    d.pop('created', None)
    d.pop('author', None)
    if smoothing.resolve(d.get('smoothing')) is None:
        d.pop('smoothing', None)
    canon = json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(canon.encode('utf-8')).hexdigest()


def transferable_part(recipe: Recipe) -> Dict[str, Any]:
    """Только переносимые поля рецепта (для сохранения как шаблон)."""
    d = to_dict(recipe)
    part: Dict[str, Any] = {key: d[key] for key in _TRANSFERABLE_TOP}
    part['recon'] = {key: d['recon'][key] for key in _TRANSFERABLE_RECON if key in d['recon']}
    return part


def apply_template(recipe: Recipe, template: Dict[str, Any]) -> Recipe:
    """Перенести переносимые поля ``template`` (см. :func:`transferable_part`) поверх ``recipe``.

    Поля, привязанные к скану (``pixel_size``, ``fov``, ``axis``, ``repositioning``, ``recon.slices``,
    ``recon.xy_roi``), из ``template`` не берутся — они остаются от ``recipe``.
    """
    d = to_dict(recipe)
    for key in _TRANSFERABLE_TOP:
        if key in template:
            d[key] = copy.deepcopy(template[key])
    if 'recon' in template:
        recon = dict(d['recon'])
        for key in _TRANSFERABLE_RECON:
            if key in template['recon']:
                recon[key] = template['recon'][key]
        d['recon'] = recon
    return from_dict(d)


def from_rec_config_ini(path: Any, exp_id: str, fingerprint: str,
                        frame_height: int, frame_width: int) -> Recipe:
    """Миграция старого ``rec_config.ini`` (см. ``tomotools4.load_recon_config``/``save_recon_config``) в рецепт.

    ``[roi] x_min x_max y_min y_max`` — полуоткрытые границы (как использует ``reconstructor4.py``:
    ``data_images[:, y_min:y_max, x_min:x_max]``), переносятся напрямую в ``fov``. ``[axis_corr]``
    (``shift_x``, ``alfa``) задан относительно кропа ``[roi]`` из того же файла и переводится в координаты
    детектора (:func:`axis.from_crop_params`, ``method='notebook'``) — по такому рецепту движок повторяет
    ноутбук. Размер пикселя в старом конфиге не хранился — берётся значение по умолчанию с source='default'.
    """
    from .axis import from_crop_params  # noqa: WPS433 — axis тянет scipy только при вызове
    cfg = configparser.ConfigParser()
    if not cfg.read(str(path), encoding='utf-8'):
        raise FileNotFoundError('rec_config.ini не найден: {}'.format(path))
    if 'roi' not in cfg:
        raise ValueError('в {} нет секции [roi]'.format(path))

    sec = cfg['roi']
    roi = ROI(x0=int(sec['x_min']), x1=int(sec['x_max']), y0=int(sec['y_min']), y1=int(sec['y_max']))
    roi.validate(frame_height, frame_width)

    ps = pixelsize.resolve(None, None, None)
    r = default_recipe(exp_id, fingerprint, roi, ps.value_mm, ps.source, is_advanced=True)
    if cfg.has_section('axis_corr'):
        sec = cfg['axis_corr']
        r.axis = from_crop_params(float(sec['shift_x']), float(sec['alfa']), roi, method='notebook')
        logger.info('from_rec_config_ini(%s): ось ноутбука shift_x=%s alfa=%s → center_x=%.3f на y=%.1f, '
                    'наклон %.4f°', path, sec['shift_x'], sec['alfa'], r.axis.center_x, r.axis.y_ref,
                    r.axis.tilt_deg)
    return r
