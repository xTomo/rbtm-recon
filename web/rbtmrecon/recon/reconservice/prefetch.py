"""Предзагрузка исходного HDF5 в кэш страниц ОС, пока человек выбирает поле зрения (``POST /scans/<id>/prefetch``).

Исходники на сервере лежат на HDD (``/exp_src``), и «Загрузить область» упирается в чтение файла целиком: кроп
всё равно распаковывает почти весь файл (``reconengine.data.CropLoader``). Пока человек двигает рамку на шаге 1,
диск простаивает. Здесь один фоновый поток читает файл подряд блоками и выбрасывает прочитанное — данные остаются
в кэше страниц ядра, и загрузка потом берёт уже прочитанную часть из памяти, а с диска дочитывает остаток.

Правила:
- одна предзагрузка на сервис; запрос по другому скану останавливает текущую; повторный по тому же — только
  состояние (идущая или законченная не перезапускается);
- останавливается, как только сервис сам начинает читать исходники: загрузка области (``SessionManager``) или
  задача (``JobRunner``) — два читателя одного HDD мешают друг другу сильнее, чем помогают;
- не запускается, если файл больше доли ``mem_fraction`` доступной памяти (``MemAvailable`` из ``/proc/meminfo``):
  такой файл вытеснит сам себя из кэша раньше, чем его прочтут; где ``/proc/meminfo`` нет — без проверки;
- ``cfg.prefetch = False`` (``RECON_PREFETCH=0``) — выключена: запрос отвечает ``skipped``.

Состояние (``status()``, в ``/health`` и в ответе запроса): ``state`` — ``idle`` | ``running`` | ``done`` |
``stopped`` | ``skipped`` | ``error``, ``exp_id``, ``done_bytes``, ``total_bytes``, ``elapsed_s``, ``reason``.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional

from .config import Config

logger = logging.getLogger(__name__)

BLOCK = 16 << 20        # байт за одно чтение
MEM_FRACTION = 0.5      # доля MemAvailable, больше которой файл не предзагружается


def mem_available() -> Optional[int]:
    """MemAvailable из /proc/meminfo, байт; None — неизвестно (не Linux)."""
    try:
        with open('/proc/meminfo', encoding='ascii') as fh:
            for line in fh:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


class Prefetcher:
    def __init__(self, cfg: Config, scans, block: int = BLOCK, mem_fraction: float = MEM_FRACTION,
                 meminfo: Callable[[], Optional[int]] = mem_available, clock=time.monotonic):
        self.cfg = cfg
        self.scans = scans
        self.block = int(block)
        self.mem_fraction = float(mem_fraction)
        self.meminfo = meminfo
        self.clock = clock
        self._lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop: Optional[threading.Event] = None
        self._st: Dict[str, Any] = {'state': 'idle'}

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return self._status_locked()

    def start(self, exp_id: str) -> Dict[str, Any]:
        """Начать предзагрузку скана exp_id (FileNotFoundError, если его нет). Возвращает состояние."""
        with self._start_lock:          # два одновременных запроса по одному скану не перезапускают чтение
            return self._start(exp_id)

    def _start(self, exp_id: str) -> Dict[str, Any]:
        scan = self.scans.info(exp_id)
        path = scan.path
        with self._lock:
            if self._st.get('exp_id') == exp_id and self._st.get('state') in ('running', 'done'):
                return self._status_locked()
        if not self.cfg.prefetch:
            return self._set_final(exp_id, 'skipped', 'выключена (RECON_PREFETCH)')
        total = os.path.getsize(path)
        avail = self.meminfo()
        if avail is not None and total > self.mem_fraction * avail:
            return self._set_final(exp_id, 'skipped', 'файл {:.1f} ГБ больше {:.0%} доступной памяти ({:.1f} ГБ)'.format(
                total / 1e9, self.mem_fraction, avail / 1e9), total)
        self.stop('replaced')
        with self._lock:
            ev = threading.Event()
            self._stop = ev
            self._st = {'state': 'running', 'exp_id': exp_id, 'done_bytes': 0, 'total_bytes': total,
                        '_t0': self.clock()}
            t = threading.Thread(target=self._run, args=(exp_id, path, total, ev),
                                 name='recon-prefetch', daemon=True)
            self._thread = t
            t.start()
        logger.info('предзагрузка %s: %.1f ГБ', exp_id, total / 1e9)
        return self.status()

    def stop(self, reason: str, wait: float = 5.0) -> None:
        """Остановить идущую предзагрузку (причина видна в состоянии) и дождаться потока до wait секунд."""
        with self._lock:
            ev, t = self._stop, self._thread
            if ev is None or ev.is_set():
                return
            self._st['reason'] = reason
            ev.set()
        if t is not None and t is not threading.current_thread():
            t.join(wait)

    # --- внутреннее ---------------------------------------------------------------------------------------

    def _status_locked(self) -> Dict[str, Any]:
        st = dict(self._st)
        t0 = st.pop('_t0', None)
        if st.get('state') == 'running' and t0 is not None:
            st['elapsed_s'] = round(self.clock() - t0, 1)
        return st

    def _set_final(self, exp_id: str, state: str, reason: str, total: Optional[int] = None) -> Dict[str, Any]:
        self.stop('replaced')
        with self._lock:
            self._st = {'state': state, 'exp_id': exp_id, 'reason': reason, 'done_bytes': 0, 'total_bytes': total}
            return dict(self._st)

    def _run(self, exp_id: str, path: str, total: int, stop: threading.Event) -> None:
        t0 = self.clock()
        done = 0
        state, error = 'done', None
        try:
            buf = bytearray(self.block)
            view = memoryview(buf)
            with open(path, 'rb', buffering=0) as fh:
                if hasattr(os, 'posix_fadvise'):
                    os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)   # больше упреждающее чтение
                while not stop.is_set():
                    n = fh.readinto(view)
                    if not n:
                        break
                    done += n
                    with self._lock:
                        if self._stop is stop:
                            self._st['done_bytes'] = done
            if stop.is_set():
                state = 'stopped'
        except Exception as exc:  # noqa: BLE001 — предзагрузка только ускоряет, её ошибка не мешает работе
            state, error = 'error', '{}: {}'.format(type(exc).__name__, exc)
            logger.warning('предзагрузка %s: %s', exp_id, error)
        dt = self.clock() - t0
        with self._lock:
            if self._stop is stop:
                self._st.update(state=state, done_bytes=done, elapsed_s=round(dt, 1))
                self._st.pop('_t0', None)
                if error:
                    self._st['reason'] = error
                stop.set()
        logger.info('предзагрузка %s: %s, %.1f из %.1f ГБ за %.0f с (%.0f МБ/с)', exp_id, state, done / 1e9,
                    total / 1e9, dt, done / 1e6 / max(dt, 1e-3))
