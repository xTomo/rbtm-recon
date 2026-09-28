"""Выбор вычислительного бэкенда (cupy на GPU или numpy на CPU) без импорта cupy на верхнем уровне.

Переменная окружения ``RECON_ENGINE_CPU=1`` принудительно включает numpy (тесты, машина без GPU).
GPU для процесса выбирается снаружи через ``CUDA_VISIBLE_DEVICES``.
"""
from __future__ import annotations

import contextlib
import os
from typing import Optional, Tuple

import numpy as np

_XP = None


def _try_cupy():
    if os.environ.get('RECON_ENGINE_CPU') == '1':
        return None
    try:
        import cupy  # noqa: WPS433 — ленивый импорт
        if cupy.cuda.runtime.getDeviceCount() < 1:
            return None
        return cupy
    except Exception:  # noqa: BLE001 — нет cupy, нет драйвера, заглушка в тестах
        return None


def get_xp():
    """Модуль массивов: cupy, если доступен GPU, иначе numpy. Результат кэшируется."""
    global _XP
    if _XP is None:
        _XP = _try_cupy() or np
    return _XP


def reset_backend() -> None:
    """Сбросить кэш выбора бэкенда (для тестов)."""
    global _XP
    _XP = None


def is_gpu(xp=None) -> bool:
    return (xp or get_xp()) is not np


def ndimage(xp=None):
    """scipy.ndimage или cupyx.scipy.ndimage под выбранный бэкенд."""
    if is_gpu(xp):
        import cupyx.scipy.ndimage as cndi  # noqa: WPS433
        return cndi
    import scipy.ndimage as ndi  # noqa: WPS433
    return ndi


def to_numpy(a) -> np.ndarray:
    """Перенести массив на CPU (для numpy — без копии)."""
    if isinstance(a, np.ndarray):
        return a
    get = getattr(a, 'get', None)
    return np.asarray(get() if callable(get) else a)


def mem_info() -> Optional[Tuple[int, int]]:
    """(свободно, всего) байт на текущем GPU или None на CPU."""
    xp = get_xp()
    if not is_gpu(xp):
        return None
    free, total = xp.cuda.runtime.memGetInfo()
    return int(free), int(total)


def free_memory() -> None:
    """Вернуть драйверу закэшированную память cupy."""
    xp = get_xp()
    if is_gpu(xp):
        xp.get_default_memory_pool().free_all_blocks()
        xp.get_default_pinned_memory_pool().free_all_blocks()


@contextlib.contextmanager
def gpu_lock(path: Optional[str]):
    """Эксклюзивная межпроцессная блокировка GPU (flock). На Windows и при path=None — без блокировки."""
    if not path:
        yield
        return
    try:
        import fcntl  # noqa: WPS433 — только Linux
    except ImportError:
        yield
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a+') as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def lock_busy(path: Optional[str]) -> Optional[bool]:
    """Занята ли блокировка gpu_lock(path) другим процессом. None — не определить (Windows, нет пути)."""
    if not path:
        return None
    try:
        import fcntl  # noqa: WPS433 — только Linux
    except ImportError:
        return None
    try:
        fh = open(path, 'a+')  # noqa: SIM115 — закрывается ниже
    except OSError:
        return None
    with fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return False


def device_name() -> Optional[str]:
    """Имя текущей карты или None на CPU."""
    xp = get_xp()
    if not is_gpu(xp):
        return None
    try:
        name = xp.cuda.runtime.getDeviceProperties(xp.cuda.Device().id).get('name')
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:  # noqa: BLE001 — только для отчёта
        return None
