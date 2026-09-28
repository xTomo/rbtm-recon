"""Перенос результата запуска в хранилище.

``publish(cfg, exp_id, run_dir)`` — ``run_dir`` на SSD (``cfg.runs_dir(exp)/<run_id>``) содержит то, что написал
``pipeline.run_recipe``: raw/.size/.hx полного объёма и копий с биннингом, ``recipe.json``, ``result.json``.
Назначение — ``cfg.reconstruction_dir(exp)`` (``/storage/<id>/reconstruction``, туда же копирует ноутбук):

1. если там уже есть ``result.json`` прошлого запуска движка — его ``recipe.json``/``result.json`` переносятся в
   ``history/<run_id прошлого>/``, а файлы объёма, перечисленные в прошлом ``result.json``, удаляются
   (пересчёт заменяет объём; история хранит только рецепты и метаданные);
2. файлы, которые движок не создавал (результаты ноутбука, отчёты), не удаляются никогда; совпадающие по имени
   перезаписываются новым объёмом;
3. файлы объёма переносятся (``shutil.move``: между дисками — копия и удаление), ``recipe.json`` — затем,
   ``result.json`` — последним и атомарно: читатель видит либо прошлый результат целиком, либо новый;
4. пустой ``run_dir`` удаляется.

Порядок внутри: прошлые ``recipe.json``/``result.json`` копируются в историю (на месте они остаются до замены
новыми — между шагами каталог не бывает без ``result.json``), новые файлы переносятся с заменой, ``result.json``
заменяется последним, и только потом удаляются файлы прошлого объёма, которых нет среди новых (с тем же именем они
уже перезаписаны). Каждый файл переносится атомарно: ``os.replace`` на том же диске, между дисками — копия во
временное имя ``.tmp-publish-*`` рядом с назначением и ``os.replace``. Поэтому прерванную публикацию можно
повторить тем же вызовом (исполнитель задач так и делает после перезапуска сервиса): уже перенесённые файлы
берутся из назначения, остатки временных копий удаляются.

``archive_h5(cfg, exp_id)`` — архивная копия ``<storage_dir>/<id>.h5`` из ``cfg.scan_path(exp_id)``, если её ещё
нет (как ``mv /fast/<id>.h5 /storage/`` в ноутбуке): копия во временное имя в том же каталоге и ``os.replace``.
"""
from __future__ import annotations

import datetime
import errno
import json
import logging
import os
import re
import shutil
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

from .config import Config

logger = logging.getLogger(__name__)

RESULT = 'result.json'
RECIPE = 'recipe.json'
HISTORY = 'history'
_TMP_PREFIX = '.tmp-publish-'
_RUN_ID_RE = re.compile(r'^[A-Za-z0-9_-]{1,64}$')
_ERROR_NOT_SAME_DEVICE = 17          # winerror: os.replace между дисками на Windows


# --- имена файлов результата -------------------------------------------------------------------------------

def safe_name(name: Any) -> bool:
    """Имя файла без каталогов и «..» (имена берутся из result.json — ему не доверяем)."""
    return (isinstance(name, str) and name not in ('', '.', '..') and '/' not in name and '\\' not in name
            and '\x00' not in name and os.path.basename(name) == name)


def safe_run_id(run_id: Any) -> bool:
    return isinstance(run_id, str) and bool(_RUN_ID_RE.match(run_id))


def _full_hx_name(raw: str, shape) -> Optional[str]:
    """``tomo.<имя>.1.hx`` полного объёма: в result.json его нет, имя восстанавливается по raw
    (``<имя>.<d0>_<d1>_<d2>.1.raw``, см. ``outputs.amira_raw_name``)."""
    try:
        suffix = '.{}_{}_{}.1.raw'.format(*[int(s) for s in shape])
    except (TypeError, ValueError):
        return None
    if not raw.endswith(suffix) or len(raw) == len(suffix):
        return None
    return 'tomo.{}.1.hx'.format(raw[:-len(suffix)])


def binned_files(doc: Dict[str, Any]) -> List[str]:
    """Файлы копий с биннингом из result.json: raw, raw.size, hx."""
    names: List[str] = []
    for b in doc.get('binned') or []:
        if not isinstance(b, dict):
            continue
        if b.get('raw'):
            names += [b['raw'], str(b['raw']) + '.size']
        if b.get('hx'):
            names.append(b['hx'])
    return [n for n in dict.fromkeys(names) if safe_name(n)]


def engine_files(doc: Dict[str, Any]) -> List[str]:
    """Все файлы объёма, созданные движком по result.json: полный объём (raw, raw.size, hx) и копии с биннингом."""
    names: List[str] = []
    vol = doc.get('volume') or {}
    raw = vol.get('file')
    if isinstance(raw, str) and raw:
        names += [raw, raw + '.size']
        hx = vol.get('hx') or _full_hx_name(raw, vol.get('shape'))
        if hx:
            names.append(hx)
    names += binned_files(doc)
    return [n for n in dict.fromkeys(names) if safe_name(n)]


def read_json(path: str) -> Dict[str, Any]:
    with open(path, encoding='utf-8') as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict):
        raise ValueError('{}: ожидался объект JSON'.format(path))
    return doc


def read_published(cfg: Config, exp_id: str) -> Optional[Dict[str, Any]]:
    """Опубликованный result.json или None (нет или не читается)."""
    path = os.path.join(cfg.reconstruction_dir(exp_id), RESULT)
    if not os.path.isfile(path):
        return None
    try:
        return read_json(path)
    except (OSError, ValueError) as exc:
        logger.warning('%s: result.json не читается: %s', exp_id, exc)
        return None


# --- файловые операции -------------------------------------------------------------------------------------

def _remove_quiet(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning('не удалось удалить %s: %s', path, exc)


def _is_cross_device(exc: OSError) -> bool:
    return exc.errno == errno.EXDEV or getattr(exc, 'winerror', None) == _ERROR_NOT_SAME_DEVICE


def _replace(src: str, dst: str) -> None:
    """os.replace; на Windows замена файла, который кто-то держит открытым (отдача файла, чтение среза), даёт
    PermissionError — несколько повторов (на Linux замена открытого файла не мешает читателю)."""
    attempts = 10 if sys.platform == 'win32' else 1
    for k in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if k == attempts - 1:
                raise
            time.sleep(0.2)


def _move_file(src: str, dst: str) -> None:
    """Перенести файл с заменой назначения атомарно: dst всегда либо прежний, либо новый целиком."""
    try:
        _replace(src, dst)
        return
    except OSError as exc:
        if not _is_cross_device(exc):
            raise
    tmp = os.path.join(os.path.dirname(dst), '{}{}-{}'.format(_TMP_PREFIX, uuid.uuid4().hex[:8],
                                                              os.path.basename(dst)))
    try:
        shutil.copyfile(src, tmp)
        _replace(tmp, dst)
    except BaseException:
        _remove_quiet(tmp)
        raise
    os.remove(src)


def _remove_stale_tmp(directory: str) -> None:
    """Остатки копий прерванной публикации (публикует один исполнитель — чужих временных файлов здесь нет)."""
    for name in os.listdir(directory):
        if name.startswith(_TMP_PREFIX):
            _remove_quiet(os.path.join(directory, name))


def _history_dir(dest: str, old: Dict[str, Any]) -> str:
    run_id = old.get('run_id')
    if not safe_run_id(run_id):
        run_id = 'unknown-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
    return os.path.join(dest, HISTORY, run_id)


def _save_history(dest: str, old: Dict[str, Any]) -> str:
    """Рецепт и метаданные прошлого запуска → history/<run_id>/ (копией: на месте они заменятся новыми)."""
    hist = _history_dir(dest, old)
    os.makedirs(hist, exist_ok=True)
    for name in (RECIPE, RESULT):
        src = os.path.join(dest, name)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(hist, name))
    return hist


# --- публикация ----------------------------------------------------------------------------------------------

def publish(cfg: Config, exp_id: str, run_dir: str) -> Dict[str, Any]:
    """Перенести результат; вернуть опубликованный result.json (dict)."""
    src_result = os.path.join(run_dir, RESULT)
    new = read_json(src_result)                          # FileNotFoundError: движок не дописал результат
    dest = cfg.reconstruction_dir(exp_id)
    os.makedirs(dest, exist_ok=True)
    _remove_stale_tmp(dest)

    new_files = engine_files(new)
    if not (new.get('volume') or {}).get('file') or not new_files:
        raise ValueError('{}: в result.json нет файлов объёма'.format(src_result))
    missing = [n for n in new_files
               if not os.path.isfile(os.path.join(run_dir, n)) and not os.path.isfile(os.path.join(dest, n))]
    if missing:
        raise FileNotFoundError('{}: нет файлов результата: {}'.format(run_dir, ', '.join(missing)))

    old = read_published(cfg, exp_id)
    if old is not None and old.get('run_id') != new.get('run_id'):
        hist = _save_history(dest, old)
        logger.info('%s: прошлый запуск %s → %s', exp_id, old.get('run_id'), hist)

    for name in new_files:
        src = os.path.join(run_dir, name)
        if os.path.isfile(src):                         # иначе перенесён прерванной публикацией
            _move_file(src, os.path.join(dest, name))
    if os.path.isfile(os.path.join(run_dir, RECIPE)):
        _move_file(os.path.join(run_dir, RECIPE), os.path.join(dest, RECIPE))
    _move_file(src_result, os.path.join(dest, RESULT))

    if old is not None:
        keep = set(new_files)
        for name in engine_files(old):
            if name not in keep:
                _remove_quiet(os.path.join(dest, name))

    try:
        os.rmdir(run_dir)                               # только пустой: engine.log и recipe.in.json остаются
    except OSError:
        pass
    logger.info('%s: опубликован запуск %s (%d файлов объёма)', exp_id, new.get('run_id'), len(new_files))
    return new


def archive_h5(cfg: Config, exp_id: str) -> bool:
    """Скопировать исходный HDF5 в архив, если копии нет. True — скопирован сейчас."""
    dst = os.path.join(cfg.storage_dir, exp_id + '.h5')
    if os.path.exists(dst):
        return False
    src = cfg.scan_path(exp_id)
    if not os.path.isfile(src):
        raise FileNotFoundError('нет исходного файла {}'.format(src))
    os.makedirs(cfg.storage_dir, exist_ok=True)
    tmp = os.path.join(cfg.storage_dir, '.{}.h5.tmp-{}'.format(exp_id, uuid.uuid4().hex[:8]))
    try:
        shutil.copyfile(src, tmp)
        st = os.stat(src)
        os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))    # как mv: время изменения — исходного файла
        os.replace(tmp, dst)
    except BaseException:
        _remove_quiet(tmp)
        raise
    logger.info('%s: архивная копия %s', exp_id, dst)
    return True
