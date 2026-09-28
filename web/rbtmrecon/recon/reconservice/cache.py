"""Предел кэша кропов на SSD (``cfg.fast_limit_gb``): вытеснение давно не использованных.

Кроп — пара ``<cfg.cache_dir(id)>/crop-<hash>.u16`` + ``.json`` (``reconengine.data.CropLoader``); время
использования — mtime ``.json``: CropLoader обновляет его при каждом попадании в кэш и пишет при создании.
Пока суммарный размер больше предела, удаляются кропы с самым старым временем использования, кроме тех, что
использовались за последние ``protect_s`` секунд (по умолчанию — TTL сессии: кроп открытой сессии, идущей задачи
или ещё записываемый). На Linux удаление файла, отображённого в память, безопасно (данные живут до закрытия), на
Windows занятый файл не удаляется — он пропускается.
"""
from __future__ import annotations

import glob
import logging
import os
import time
from typing import List, Optional, Tuple

from .config import Config

logger = logging.getLogger(__name__)

_GB = 1024 ** 3


def crop_entries(cfg: Config) -> List[Tuple[float, int, str, str]]:
    """(время использования, байт, путь .u16, путь .json) всех кропов в ``<fast_dir>/studio/*/cache``."""
    out = []
    for data_path in glob.glob(os.path.join(cfg.fast_dir, 'studio', '*', 'cache', 'crop-*.u16')):
        meta_path = os.path.splitext(data_path)[0] + '.json'
        try:
            size = os.path.getsize(data_path)
            used = os.path.getmtime(meta_path) if os.path.exists(meta_path) else os.path.getmtime(data_path)
        except OSError:
            continue
        out.append((used, size, data_path, meta_path))
    return out


def cleanup(cfg: Config, protect_s: Optional[float] = None, now: Optional[float] = None) -> List[str]:
    """Удалить старые кропы, пока кэш больше ``cfg.fast_limit_gb``. Возвращает удалённые пути .u16."""
    limit = float(cfg.fast_limit_gb) * _GB
    protect_s = cfg.session_ttl_s if protect_s is None else float(protect_s)
    now = time.time() if now is None else now
    entries = sorted(crop_entries(cfg))
    total = sum(e[1] for e in entries)
    removed = []
    for used, size, data_path, meta_path in entries:
        if total <= limit:
            break
        if now - used < protect_s:
            continue
        try:
            os.remove(data_path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning('кэш кропов: %s не удалён: %s', data_path, exc)
            continue
        try:
            os.remove(meta_path)
        except OSError:
            pass
        total -= size
        removed.append(data_path)
        logger.info('кэш кропов: удалён %s (%.1f ГБ, не использовался %.0f ч)', data_path, size / _GB,
                    (now - used) / 3600)
    if total > limit:
        logger.warning('кэш кропов %.1f ГБ больше предела %.1f ГБ: всё остальное используется', total / _GB,
                       limit / _GB)
    return removed


def cleanup_quiet(cfg: Config) -> None:
    """cleanup без исключений — для вызова после загрузки кропа и после задачи."""
    try:
        cleanup(cfg)
    except Exception:  # noqa: BLE001 — чистка кэша не должна ломать загрузку или задачу
        logger.exception('ошибка чистки кэша кропов')
