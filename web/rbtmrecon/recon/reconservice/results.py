"""Результат реконструкции для шага «Результат» (``/results/<exp_id>/...``).

| Метод и путь                          | Ответ |
|---------------------------------------|-------|
| GET /results/<id>                     | JSON: опубликованный ``result.json``, история (``history/*/result.json``: run_id, created, recipe_sha256), список доступных файлов; 404 — результата движка нет |
| GET /results/<id>/slice?axis=z|y|x&i=&max_px= | binary uint16: срез копии с наибольшим биннингом из ``result.json['binned']`` (memmap, без чтения объёма целиком); окно квантования — ``stats.p0_1``/``p99_9`` результата; X-Meta: axis, i, n, binning, voxel_mm |
| GET /results/<id>/file/<name>         | файл потоком (as_attachment): только ``recipe.json``, ``result.json`` и файлы копий с биннингом (raw, .size, .hx) из ``result.json``; полный объём через сервис не отдаётся (он доступен как раньше, через ``/reconstruct/static``) |

``GET /results/<id>``: ``{exp_id, dir, result, history: [{run_id, created, recipe_sha256, has_recipe}] (новые первыми),
files: [{name, size}]}``; ``dir`` — каталог результата в хранилище (путь к полному объёму — ``dir``/``volume.file``).

Срез: ``axis`` по умолчанию ``z``, ``i`` — по умолчанию середина, ``max_px`` — по умолчанию ``cfg.preview_max_px``.
Срез по z — ``(ny, nx)``, по y — ``(nz, nx)``, по x — ``(nz, ny)`` (в осях копии с биннингом). Файл отображается
только на время запроса и закрывается сразу (на Windows открытое отображение не дало бы публикации заменить файл).
"""
from __future__ import annotations

import math
import mmap
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flask import Blueprint, current_app, jsonify, request, send_file

from . import auth, binary, publish
from .config import Config

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


def allowed_files(doc: Dict[str, Any]) -> List[str]:
    """Белый список файлов для скачивания: рецепт, result.json и файлы копий с биннингом."""
    return [publish.RECIPE, publish.RESULT] + publish.binned_files(doc)


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
    for name in allowed_files(doc):
        path = os.path.join(dest, name)
        if os.path.isfile(path):
            files.append({'name': name, 'size': os.path.getsize(path)})
    return jsonify({'exp_id': exp_id, 'dir': dest, 'result': doc, 'history': history(cfg, exp_id), 'files': files})


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


@bp.get('/<exp_id>/file/<name>')
def result_file(exp_id, name):
    cfg = _cfg()
    exp_id = auth.valid_exp_id(exp_id)
    doc = load_result(cfg, exp_id)
    if name not in allowed_files(doc):
        raise FileNotFoundError('файл {} недоступен'.format(name))
    path = os.path.join(cfg.reconstruction_dir(exp_id), name)
    if not os.path.isfile(path):
        raise FileNotFoundError('файла {} нет'.format(name))
    return send_file(path, as_attachment=True, download_name=name, conditional=True, max_age=0)
