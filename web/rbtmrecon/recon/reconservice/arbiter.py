"""«Последний запрос выигрывает» для интерактивных каналов.

Пока пользователь тянет ползунок центра, браузер шлёт запросы с растущим ``seq`` по одному каналу (например,
``(sid, 'slice')``). Считать каждый бессмысленно: запрос, для которого уже пришёл более новый ``seq``, должен
выйти как можно раньше — до захвата GPU и между стадиями вычисления::

    ticket = arbiter.begin((sid, 'slice'), seq)
    with session.compute_lock:        # на GPU сессии одновременно считается один запрос
        ticket.check()                # пока ждали, пришёл более новый — Superseded (HTTP 409)
        ...стадия...
        ticket.check()
        ...стадия...

``seq`` не обязателен: без него запрос не вытесняет другие и сам не вытесняется.
"""
from __future__ import annotations

import threading
from typing import Dict, Hashable, Optional


class Superseded(Exception):
    """Запрос устарел: по его каналу уже пришёл более новый."""


class Ticket:
    def __init__(self, arbiter: 'Arbiter', key: Hashable, seq: Optional[int]):
        self._arbiter = arbiter
        self.key = key
        self.seq = seq

    @property
    def current(self) -> bool:
        return self.seq is None or self._arbiter.latest(self.key) == self.seq

    def check(self) -> None:
        if not self.current:
            raise Superseded('канал {!r}: запрос {} устарел'.format(self.key, self.seq))


class Arbiter:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest: Dict[Hashable, int] = {}

    def begin(self, key: Hashable, seq: Optional[int]) -> Ticket:
        """Зарегистрировать запрос. seq меньше уже виденного — сразу Superseded."""
        if seq is None:
            return Ticket(self, key, None)
        seq = int(seq)
        with self._lock:
            cur = self._latest.get(key)
            if cur is not None and seq < cur:
                raise Superseded('канал {!r}: запрос {} старше {}'.format(key, seq, cur))
            self._latest[key] = seq
        return Ticket(self, key, seq)

    def latest(self, key: Hashable) -> Optional[int]:
        with self._lock:
            return self._latest.get(key)

    def forget(self, owner: Hashable) -> None:
        """Забыть каналы, у которых первый элемент ключа — owner (закрытая сессия)."""
        with self._lock:
            for k in [k for k in self._latest if isinstance(k, tuple) and k and k[0] == owner]:
                del self._latest[k]
