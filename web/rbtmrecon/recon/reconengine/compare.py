"""Сравнение объёма движка с объёмом ноутбука по тем же срезам (``python -m reconengine compare``).

Форма обоих объёмов берётся из имени raw (``<имя>.<nz>_<ny>_<nx>.<b>.raw``). Объём ноутбука покрывает строки
детектора ``[fov.y0, fov.y1)`` рецепта (ROI ноутбука), объём движка — ``recon.slices``; сравнение идёт по общим
срезам, срезы должны быть одного размера (рецепт из ``python -m reconengine migrate`` — тот же ROI и ось).
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

_SHAPE = re.compile(r'\.(\d+)_(\d+)_(\d+)\.\d+\.raw$')


def open_raw(path: str) -> np.memmap:
    m = _SHAPE.search(os.path.basename(path))
    if not m:
        raise ValueError('в имени {} нет формы <nz>_<ny>_<nx>'.format(path))
    return np.memmap(path, dtype='<f4', mode='r', shape=tuple(int(g) for g in m.groups()))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype('float64')
    b = b.ravel().astype('float64')
    a -= a.mean()
    b -= b.mean()
    den = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / den) if den > 0 else float('nan')


def compare(old_raw: str, result: str, step: int = 16):
    """[(строка детектора, корреляция, отношение средних new/old)] по общим срезам."""
    if os.path.isdir(result):
        result = os.path.join(result, 'result.json')
    with open(result, encoding='utf-8') as fh:
        doc = json.load(fh)
    new = open_raw(os.path.join(os.path.dirname(result), doc['volume']['file']))
    old = open_raw(old_raw)
    if new.shape[1:] != old.shape[1:]:
        raise ValueError('срезы разного размера: движок {} и ноутбук {} — нужен тот же ROI'.format(
            new.shape[1:], old.shape[1:]))
    y0 = int(doc['recipe']['fov']['y0'])
    z0 = int(doc['recipe']['recon']['slices'][0])
    rows = []
    for i in range(0, new.shape[0], max(1, int(step))):
        j = z0 + i - y0
        if not 0 <= j < old.shape[0]:
            continue
        a, b = np.asarray(old[j]), np.asarray(new[i])
        ma = float(a.mean())
        rows.append((z0 + i, corr(a, b), float(b.mean()) / ma if ma else float('nan')))
    return rows
