"""Результат реконструкции для шага «Результат» (``/results/<exp_id>/...``).

| Метод и путь                          | Ответ |
|---------------------------------------|-------|
| GET /results/<id>                     | JSON: опубликованный ``result.json``, история (``history/*/result.json``: run_id, created, recipe_sha256), список доступных файлов; 404 — результата движка нет |
| GET /results/<id>/slice?axis=z|y|x&i=&max_px= | binary uint16: срез копии с наибольшим биннингом из ``result.json['binned']`` (memmap, без чтения объёма целиком); окно квантования — ``stats.p0_1``/``p99_9`` результата; X-Meta: axis, i, n, binning, voxel_mm |
| GET /results/<id>/volume3d?max_voxels=&max_side= | binary uint8 (nz, ny, nx): копия с наибольшим биннингом, уменьшенная ещё в f раз (среднее по кубам f×f×f, неполные кубы у краёв — по тому, что есть) так, чтобы вокселей было ≤ max_voxels и сторона ≤ max_side; окно квантования — как у среза; кэш в ``cfg.view3d_dir(id)``; X-Meta: binning (полный: копия × f), source_binning, downsample (f), source_shape, voxel_mm, window, run_id |
| POST /results/<id>/view3d-html?max_voxels=&max_side= | тело — HTML-оболочка 3D-вида из студии (text/html, UTF-8, ≤ 2 МБ) с одной меткой ``__RBTM_VOLUME__``; сервис ставит на её место JSON объёма — тот же, что отдаёт ``volume3d`` с теми же пределами (``{w, h, k, scale, offset, meta, b64}``, b64 — uint8 (nz, ny, nx) в base64) — и пишет ``view3d-<run_id>.html`` в каталог результата (повторное сохранение того же запуска заменяет файл); 201 ``{name, size}`` |
| GET /results/<id>/file/<name>         | файл потоком (as_attachment): только ``recipe.json``, ``result.json``, файлы копий с биннингом (raw, .size, .hx) из ``result.json`` и сохранённые 3D-виды ``view3d-<run_id>.html``; полный объём через сервис не отдаётся (он доступен как раньше, через ``/reconstruct/static``) |
| GET /results/<id>/recipes/<run_id>    | JSON: рецепт запуска — ``current`` (опубликованный) или из истории (``history/<run_id>/recipe.json``): ``{exp_id, run_id, created, recipe_sha256, current, recipe}``; ``?download=1`` — тот же ``recipe.json`` файлом (``<id>.<run_id>.recipe.json``); 404 — запуска или его рецепта нет |

``GET /results/<id>``: ``{exp_id, dir, result, history: [{run_id, created, recipe_sha256, has_recipe}] (новые первыми),
files: [{name, size}], full: [{name, rel, size}]}``; ``dir`` — каталог результата в хранилище (путь к полному объёму —
``dir``/``volume.file``); ``full`` — полный объём (raw) и его .hx, если файлы есть: ``rel`` — путь относительно
хранилища (``<id>/reconstruction/<имя>``, через «/») — по нему студия строит ссылку на старую раздачу статики
(``/reconstruct/static/tomo_data/`` смотрит в тот же каталог).

Срез: ``axis`` по умолчанию ``z``, ``i`` — по умолчанию середина, ``max_px`` — по умолчанию ``cfg.preview_max_px``.
Срез по z — ``(ny, nx)``, по y — ``(nz, nx)``, по x — ``(nz, ny)`` (в осях копии с биннингом). Файл отображается
только на время запроса и закрывается сразу (на Windows открытое отображение не дало бы публикации заменить файл).

Объём для 3D-вида (``volume3d``) рисуется в браузере (WebGL2, текстура целиком в видеопамяти), отсюда предел
``max_voxels`` (по умолчанию 320³ ≈ 33 МБ uint8, как в сегментаторе Tomat) и ``max_side`` (браузер знает свой
``MAX_3D_TEXTURE_SIZE``; по умолчанию 1024). Коэффициент f — наименьший целый, при котором оба предела выполнены,
одинаковый по трём осям (воксель остаётся кубическим). Копия читается слоями по f плоскостей z через отображение в
память; первое чтение копии ×4 большого скана (~2 ГБ) — десятки секунд на HDD, поэтому результат кэшируется
файлом ``vol-<run_id>-f<f>.u8`` + ``.json`` (ключ — запуск, размер и mtime копии, окно); при записи новый файл
заменяет прежние объёмы этого эксперимента.
"""
from __future__ import annotations

import base64
import json
import logging
import math
import mmap
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flask import Blueprint, current_app, jsonify, request, send_file

from . import auth, binary, publish
from .config import Config

logger = logging.getLogger(__name__)

bp = Blueprint('results', __name__, url_prefix='/results')

AXES = ('z', 'y', 'x')


def _cfg() -> Config:
    return current_app.extensions['recon'].cfg


def load_result(cfg: Config, exp_id: str) -> Dict[str, Any]:
    """Опубликованный result.json; FileNotFoundError — результата движка нет."""
    path = os.path.join(cfg.reconstruction_dir(exp_id), publish.RESULT)
    if not os.path.isfile(path):
        raise FileNotFoundError('результата реконструкции движком для {} нет'.format(exp_id))
    return publish.read_json(path)


def history(cfg: Config, exp_id: str) -> List[Dict[str, Any]]:
    """Прошлые запуски из ``history/<run_id>/result.json`` — новые первыми; нечитаемые пропускаются."""
    hdir = os.path.join(cfg.reconstruction_dir(exp_id), publish.HISTORY)
    items = []
    if not os.path.isdir(hdir):
        return items
    for name in os.listdir(hdir):
        run_dir = os.path.join(hdir, name)
        try:
            doc = publish.read_json(os.path.join(run_dir, publish.RESULT))
        except (OSError, ValueError):
            continue
        items.append({'run_id': doc.get('run_id') or name, 'created': doc.get('created'),
                      'recipe_sha256': doc.get('recipe_sha256'),
                      'has_recipe': os.path.isfile(os.path.join(run_dir, publish.RECIPE))})
    items.sort(key=lambda it: str(it.get('created') or ''), reverse=True)
    return items


#: сохранённый 3D-вид запуска (POST view3d-html): имя по run_id — файл не путается с видом другого запуска
VIEW3D_HTML_RE = re.compile(r'view3d-[A-Za-z0-9_-]{1,64}\.html\Z')


def view3d_html_files(dest: str) -> List[str]:
    """Сохранённые 3D-виды в каталоге результата (публикация нового запуска их не трогает: имён нет в result.json)."""
    if not os.path.isdir(dest):
        return []
    return sorted(n for n in os.listdir(dest) if VIEW3D_HTML_RE.match(n) and os.path.isfile(os.path.join(dest, n)))


def allowed_files(doc: Dict[str, Any], dest: Optional[str] = None) -> List[str]:
    """Белый список файлов для скачивания: рецепт, result.json, файлы копий с биннингом и (если указан каталог
    результата dest) сохранённые 3D-виды."""
    return [publish.RECIPE, publish.RESULT] + publish.binned_files(doc) + (view3d_html_files(dest) if dest else [])


def largest_binned(cfg: Config, exp_id: str, doc: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Копия с наибольшим биннингом, файл которой есть: (запись из result.json, путь к raw)."""
    dest = cfg.reconstruction_dir(exp_id)
    best: Optional[Tuple[Dict[str, Any], str]] = None
    for b in doc.get('binned') or []:
        if not isinstance(b, dict) or not publish.safe_name(b.get('raw')):
            continue
        path = os.path.join(dest, b['raw'])
        if os.path.isfile(path) and (best is None or int(b.get('factor', 1)) > int(best[0].get('factor', 1))):
            best = (b, path)
    if best is None:
        raise FileNotFoundError('в результате {} нет копии с биннингом'.format(exp_id))
    return best


def read_slice(path: str, shape: Tuple[int, int, int], axis: str, i: int) -> np.ndarray:
    """Срез float32 объёма (nz, ny, nx) из raw-файла через отображение в память; читаются только нужные страницы."""
    count = int(np.prod(shape))
    size = os.path.getsize(path)
    if size != count * 4:
        raise RuntimeError('{}: размер {} байт, по result.json ожидалось {}'.format(path, size, count * 4))
    with open(path, 'rb') as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        vol = np.frombuffer(mm, dtype='<f4', count=count).reshape(shape)
        if axis == 'z':
            out = np.array(vol[i])
        elif axis == 'y':
            out = np.array(vol[:, i, :])
        else:
            out = np.array(vol[:, :, i])
        del vol                     # иначе mmap не закрыть (BufferError): на него ссылается массив
    return out


def _window(doc: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    stats = doc.get('stats') or {}
    lo, hi = stats.get('p0_1'), stats.get('p99_9')
    try:
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        return None, None
    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
        return None, None
    return lo, hi


@bp.get('/<exp_id>')
def result_info(exp_id):
    cfg = _cfg()
    exp_id = auth.valid_exp_id(exp_id)
    doc = load_result(cfg, exp_id)
    dest = cfg.reconstruction_dir(exp_id)
    files = []
    for name in allowed_files(doc, dest):
        path = os.path.join(dest, name)
        if os.path.isfile(path):
            files.append({'name': name, 'size': os.path.getsize(path)})
    return jsonify({'exp_id': exp_id, 'dir': dest, 'result': doc, 'history': history(cfg, exp_id), 'files': files,
                    'full': full_volume_files(cfg, exp_id, doc)})


def full_volume_files(cfg: Config, exp_id: str, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Полный объём и его .hx (``result.volume.file``/``hx``), которые есть в каталоге результата: [{name, rel, size}];
    rel — относительно хранилища, через «/»."""
    dest = cfg.reconstruction_dir(exp_id)
    vol = doc.get('volume') or {}
    rel_dir = os.path.relpath(dest, cfg.storage_dir).replace(os.sep, '/')
    out = []
    for name in (vol.get('file'), vol.get('hx')):
        if not name or os.path.basename(name) != name:
            continue
        path = os.path.join(dest, name)
        if os.path.isfile(path):
            out.append({'name': name, 'rel': rel_dir + '/' + name, 'size': os.path.getsize(path)})
    return out


@bp.get('/<exp_id>/slice')
def result_slice(exp_id):
    cfg = _cfg()
    exp_id = auth.valid_exp_id(exp_id)
    axis = request.args.get('axis', 'z')
    if axis not in AXES:
        raise ValueError('axis: одно из z, y, x')
    doc = load_result(cfg, exp_id)
    b, path = largest_binned(cfg, exp_id, doc)
    shape = tuple(int(s) for s in b['shape'])
    if len(shape) != 3:
        raise RuntimeError('в result.json некорректная форма копии: {}'.format(b.get('shape')))
    n = shape[AXES.index(axis)]
    i_arg = request.args.get('i')
    i = n // 2 if i_arg in (None, '') else int(i_arg)
    if not 0 <= i < n:
        raise ValueError('i={} вне [0, {})'.format(i, n))
    max_px = request.args.get('max_px', type=int) or cfg.preview_max_px
    lo, hi = _window(doc)
    factor = int(b.get('factor', 1))
    voxel = (doc.get('volume') or {}).get('voxel_mm')
    meta = {'axis': axis, 'i': i, 'n': n, 'binning': factor, 'shape': list(shape),
            'voxel_mm': float(voxel) * factor if voxel else None, 'window': [lo, hi]}
    return binary.array_response(read_slice(path, shape, axis, i), lo=lo, hi=hi, meta=meta, max_px=max_px)


#: пределы 3D-вида по умолчанию и допустимые
VIEW3D_MAX_VOXELS = 320 ** 3
VIEW3D_MAX_VOXELS_LIMIT = 512 ** 3
VIEW3D_MAX_SIDE = 1024


def view3d_factor(shape: Tuple[int, int, int], max_voxels: int, max_side: int) -> int:
    """Наименьший целый f ≥ 1: у объёма ceil(shape / f) вокселей ≤ max_voxels и каждая сторона ≤ max_side."""
    f = max(1, max(int(math.ceil(s / float(max_side))) for s in shape))
    while int(np.prod([-(-s // f) for s in shape])) > max_voxels:
        f += 1
    return f


def block_mean(path: str, shape: Tuple[int, int, int], f: int) -> np.ndarray:
    """Среднее по кубам f×f×f float32-объёма из raw-файла: (ceil(nz/f), ceil(ny/f), ceil(nx/f)), float32.
    Неполные кубы у краёв усредняются по имеющимся вокселям; NaN — по остальным (весь куб NaN — NaN).
    Читается слоями по f плоскостей z через отображение в память."""
    count = int(np.prod(shape))
    size = os.path.getsize(path)
    if size != count * 4:
        raise RuntimeError('{}: размер {} байт, по result.json ожидалось {}'.format(path, size, count * 4))
    nz, ny, nx = shape
    oz, oy, ox = -(-nz // f), -(-ny // f), -(-nx // f)
    iy, ix = np.arange(0, ny, f), np.arange(0, nx, f)
    out = np.empty((oz, oy, ox), dtype='float32')
    with open(path, 'rb') as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        vol = np.frombuffer(mm, dtype='<f4', count=count).reshape(shape)
        for k in range(oz):
            slab = np.array(vol[k * f:(k + 1) * f], dtype='float64')
            good = np.isfinite(slab)
            slab[~good] = 0.0
            s = np.add.reduceat(np.add.reduceat(slab.sum(axis=0), iy, axis=0), ix, axis=1)
            n = np.add.reduceat(np.add.reduceat(good.sum(axis=0, dtype='int64'), iy, axis=0), ix, axis=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                out[k] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        del vol                     # иначе mmap не закрыть (BufferError)
    return out


def quantize_u8(arr: np.ndarray, lo: Optional[float], hi: Optional[float]) -> Tuple[np.ndarray, float, float]:
    """float → uint8 по окну [lo, hi] (без окна — персентили 0,1 и 99,9): (коды, scale, offset); NaN → 0."""
    a = np.asarray(arr, dtype='float32')
    if lo is None or hi is None:
        finite = a[np.isfinite(a)]
        lo, hi = (float(v) for v in np.percentile(finite, [0.1, 99.9])) if finite.size else (0.0, 1.0)
    if not hi > lo:
        hi = lo + 1.0
    scale = (hi - lo) / 255.0
    codes = np.clip(np.round((np.nan_to_num(a, nan=lo) - lo) / scale), 0, 255).astype('u1')
    return codes, float(scale), float(lo)


def _view3d_volume(cfg: Config, exp_id: str, doc: Dict[str, Any], b: Dict[str, Any], path: str, f: int
                   ) -> Tuple[np.ndarray, float, float]:
    """(коды uint8, scale, offset) 3D-вида — из кэша или посчитанные и записанные в кэш."""
    shape = tuple(int(s) for s in b['shape'])
    st = os.stat(path)
    lo, hi = _window(doc)
    run_id = str(doc.get('run_id') or 'current')
    key = {'run_id': run_id, 'raw': b['raw'], 'size': st.st_size, 'mtime': st.st_mtime, 'shape': list(shape),
           'factor': f, 'window': [lo, hi]}
    cdir = cfg.view3d_dir(exp_id)
    stem = 'vol-{}-f{}'.format(run_id if publish.safe_run_id(run_id) else 'current', f)
    data_path, meta_path = os.path.join(cdir, stem + '.u8'), os.path.join(cdir, stem + '.json')
    out_shape = tuple(-(-s // f) for s in shape)
    try:
        meta = publish.read_json(meta_path)
        if meta.get('key') == key and os.path.getsize(data_path) == int(np.prod(out_shape)):
            codes = np.fromfile(data_path, dtype='u1').reshape(out_shape)
            return codes, float(meta['scale']), float(meta['offset'])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    codes, scale, offset = quantize_u8(block_mean(path, shape, f), lo, hi)
    try:
        os.makedirs(cdir, exist_ok=True)
        tmp = '.tmp{}-{}'.format(os.getpid(), threading.get_ident())    # параллельный запрос пишет свой файл
        codes.tofile(data_path + tmp)
        with open(meta_path + tmp, 'w', encoding='utf-8') as fh:
            json.dump({'key': key, 'scale': scale, 'offset': offset}, fh)
        os.replace(data_path + tmp, data_path)
        os.replace(meta_path + tmp, meta_path)
        for name in os.listdir(cdir):                    # прежние объёмы этого эксперимента не нужны
            if name.startswith('vol-') and '.tmp' not in name and name not in (stem + '.u8', stem + '.json'):
                try:
                    os.remove(os.path.join(cdir, name))
                except OSError:
                    pass
    except OSError as exc:
        logger.warning('3D-вид %s: кэш не записан: %s', exp_id, exc)
    return codes, scale, offset


def _view3d_request(cfg: Config, exp_id: str):
    """Объём 3D-вида по пределам из query (max_voxels, max_side): (result.json, коды uint8, scale, offset, meta)."""
    max_voxels = request.args.get('max_voxels', type=int) or VIEW3D_MAX_VOXELS
    max_side = request.args.get('max_side', type=int) or VIEW3D_MAX_SIDE
    if not 64 ** 3 <= max_voxels <= VIEW3D_MAX_VOXELS_LIMIT:
        raise ValueError('max_voxels: от {} до {}'.format(64 ** 3, VIEW3D_MAX_VOXELS_LIMIT))
    if not 64 <= max_side <= 4096:
        raise ValueError('max_side: от 64 до 4096')
    doc = load_result(cfg, exp_id)
    b, path = largest_binned(cfg, exp_id, doc)
    shape = tuple(int(s) for s in b['shape'])
    if len(shape) != 3:
        raise RuntimeError('в result.json некорректная форма копии: {}'.format(b.get('shape')))
    f = view3d_factor(shape, max_voxels, max_side)
    codes, scale, offset = _view3d_volume(cfg, exp_id, doc, b, path, f)
    factor = int(b.get('factor', 1))
    voxel = (doc.get('volume') or {}).get('voxel_mm')
    meta = {'binning': factor * f, 'source_binning': factor, 'downsample': f, 'source_shape': list(shape),
            'voxel_mm': float(voxel) * factor * f if voxel else None, 'window': list(_window(doc)),
            'run_id': doc.get('run_id')}
    return doc, codes, scale, offset, meta


@bp.get('/<exp_id>/volume3d')
def result_volume3d(exp_id):
    cfg = _cfg()
    exp_id = auth.valid_exp_id(exp_id)
    _, codes, scale, offset, meta = _view3d_request(cfg, exp_id)
    resp = binary.array_response(codes, quantized=False, meta=meta)
    resp.headers['X-Scale'] = repr(scale)
    resp.headers['X-Offset'] = repr(offset)
    return resp


def run_files(cfg: Config, exp_id: str, run_id: str) -> Tuple[str, Dict[str, Any], bool]:
    """Рецепт запуска: (путь к recipe.json, его result.json, опубликован ли он сейчас). run_id — 'current' или id из
    истории (каталог history/<run_id>); опубликованный запуск находится и по своему id. FileNotFoundError — нет."""
    dest = cfg.reconstruction_dir(exp_id)
    current = publish.read_published(cfg, exp_id)
    if run_id == 'current' or (current is not None and current.get('run_id') == run_id):
        if current is None:
            raise FileNotFoundError('результата реконструкции движком для {} нет'.format(exp_id))
        return os.path.join(dest, publish.RECIPE), current, True
    if not publish.safe_run_id(run_id):
        raise ValueError('некорректный run_id')
    run_dir = os.path.join(dest, publish.HISTORY, run_id)
    try:
        doc = publish.read_json(os.path.join(run_dir, publish.RESULT))
    except (OSError, ValueError):
        raise FileNotFoundError('запуска {} в истории {} нет'.format(run_id, exp_id)) from None
    return os.path.join(run_dir, publish.RECIPE), doc, False


@bp.get('/<exp_id>/recipes/<run_id>')
def result_recipe(exp_id, run_id):
    cfg = _cfg()
    exp_id = auth.valid_exp_id(exp_id)
    path, doc, is_current = run_files(cfg, exp_id, run_id)
    if not os.path.isfile(path):
        raise FileNotFoundError('у запуска {} нет рецепта'.format(run_id))
    rid = doc.get('run_id') or run_id
    if request.args.get('download') in ('1', 'true', 'yes'):
        name = '{}.{}.recipe.json'.format(exp_id, rid if publish.safe_run_id(rid) else 'current')
        return send_file(path, as_attachment=True, download_name=name, conditional=True, max_age=0)
    return jsonify({'exp_id': exp_id, 'run_id': rid, 'created': doc.get('created'),
                    'recipe_sha256': doc.get('recipe_sha256'), 'current': is_current,
                    'recipe': publish.read_json(path)})


#: метка в HTML-оболочке 3D-вида, на место которой встаёт JSON объёма; предел оболочки (код вида и настройки)
VIEW3D_HTML_MARK = '__RBTM_VOLUME__'
VIEW3D_HTML_MAX_SHELL = 2 * 1024 * 1024


def view3d_html_name(run_id: Any) -> str:
    return 'view3d-{}.html'.format(run_id if publish.safe_run_id(run_id) else 'current')


def view3d_volume_json(codes: np.ndarray, scale: float, offset: float, meta: Dict[str, Any]) -> str:
    """JSON объёма для вставки в <script>: «</» экранируется, чтобы строка не закрыла тег."""
    nz, ny, nx = codes.shape
    data = np.ascontiguousarray(codes, dtype=np.uint8).tobytes()
    payload = {'w': int(nx), 'h': int(ny), 'k': int(nz), 'scale': float(scale), 'offset': float(offset),
               'meta': meta, 'b64': base64.b64encode(data).decode('ascii')}
    return json.dumps(payload, ensure_ascii=False).replace('</', '<\\/')


@bp.post('/<exp_id>/view3d-html')
def result_view3d_html(exp_id):
    """Сохранить 3D-вид в HTML: оболочка из студии + объём (тот же, что у volume3d) → view3d-<run_id>.html."""
    cfg = _cfg()
    exp_id = auth.valid_exp_id(exp_id)
    raw = request.get_data(cache=False)
    if not raw or len(raw) > VIEW3D_HTML_MAX_SHELL:
        raise ValueError('нужна HTML-оболочка 3D-вида до {} байт'.format(VIEW3D_HTML_MAX_SHELL))
    try:
        shell = raw.decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError('оболочка 3D-вида — не UTF-8') from None
    if shell.count(VIEW3D_HTML_MARK) != 1:
        raise ValueError('в оболочке 3D-вида должна быть ровно одна метка {}'.format(VIEW3D_HTML_MARK))
    doc, codes, scale, offset, meta = _view3d_request(cfg, exp_id)
    html = shell.replace(VIEW3D_HTML_MARK, view3d_volume_json(codes, scale, offset, meta), 1)
    dest = cfg.reconstruction_dir(exp_id)
    name = view3d_html_name(doc.get('run_id'))
    path = os.path.join(dest, name)
    tmp = path + '.tmp-{}'.format(os.getpid())
    try:
        with open(tmp, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(html)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    size = os.path.getsize(path)
    logger.info('%s: 3D-вид сохранён в %s (%d байт, пользователь %s)', exp_id, name, size, auth.current_user())
    return jsonify({'name': name, 'size': size}), 201


@bp.get('/<exp_id>/file/<name>')
def result_file(exp_id, name):
    cfg = _cfg()
    exp_id = auth.valid_exp_id(exp_id)
    doc = load_result(cfg, exp_id)
    if name not in allowed_files(doc, cfg.reconstruction_dir(exp_id)):
        raise FileNotFoundError('файл {} недоступен'.format(name))
    path = os.path.join(cfg.reconstruction_dir(exp_id), name)
    if not os.path.isfile(path):
        raise FileNotFoundError('файла {} нет'.format(name))
    return send_file(path, as_attachment=True, download_name=name, conditional=True, max_age=0)
