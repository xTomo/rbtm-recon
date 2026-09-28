"""Интерактивная сессия студии (``/sessions/*``) — одна на сервис, на GPU процесса сервиса.

Жизненный цикл:
- ``POST /sessions {exp_id, force?}`` — открыть. Если есть активная сессия другого пользователя — 409
  ``{error: 'busy', owner, exp_id, idle_s}``; с ``force: true`` старая закрывается (её владелец при следующем
  запросе получит 410 ``{error: 'taken_over', by}``). Тот же пользователь и тот же exp_id — возвращается
  существующая сессия; тот же пользователь и другой exp_id — старая закрывается, открывается новая.
- ``GET /sessions/<sid>`` — состояние: ``open`` → ``loading`` (progress, stage) → ``ready`` | ``error``; roi,
  владелец, простой, признак готовности кропа.
- ``POST /sessions/<sid>/load {roi}`` — 202; фоновый поток: ``CropLoader.load`` (кэш ``cfg.cache_dir(id)``,
  прогресс, отмена), затем dark/empty по кропу и сдвиги образца (advanced) → ``ready``. Новая загрузка
  отменяет текущую. ``POST .../load/cancel`` — отмена.
- ``POST /sessions/<sid>/ping`` — продлить; ``DELETE /sessions/<sid>`` — закрыть (отмена загрузки, освобождение
  памяти GPU, ``arbiter.forget(sid)``). Поток-«уборщик» закрывает сессию после ``cfg.session_ttl_s`` простоя.
- Запросы к чужой сессии — 403, к закрытой/неизвестной — 404 (410 при перехвате).

Превью (сессия в ``ready``, иначе 409 ``{error: 'not_ready', state}``); вычисления — в модуле ``preview``,
на GPU сессии одновременно считается один запрос (``Session.compute_lock``), каналы с ``seq`` —
через ``arbiter`` (устаревший — 409 ``superseded``). Ось в параметрах — в координатах детектора:
``center`` — столбец оси на строке ``row``, ``tilt`` — наклон, градусы (как ``model.Axis``); без них берётся
текущая ось сессии (после ``axis/auto`` — найденная, до — авто-ось считается при первом запросе).

| Метод и путь                                   | Ответ |
|------------------------------------------------|-------|
| GET  .../slice?row&center&tilt&rings&angles&region&max_px&seq | binary uint16 (h, w) среза; X-Meta: row, axis, rings, angles, n_angles, region, timings |
| POST .../axis/auto                             | JSON: axis, shift_x, alfa, углы пары 0°/180° |
| POST .../axis/scan {row, center?, tilt?, step, n, metric, region, seq} | binary uint16 (n, th, tw): фрагменты среза при центрах center + (i − n//2)·step, общее окно квантования; X-Meta: centers, metrics, best |
| POST .../axis/tilt {y_top, c_top, y_bottom, c_bottom} | JSON: axis (``axis.tilt_from_centers``) |
| GET  .../axis/diff?center&tilt&max_px          | binary uint16: ``axis.diff_view`` пары 0°/180° |
| GET  .../rings/preview?row&center&tilt&preset&region&max_px&seq | binary uint16 (2, h, w): без колец и с пресетом, общее окно |
| GET  .../repositioning                         | JSON: применимость, checkpoint-ы (угол, sy, sx), накопленные сдвиги, предупреждения |
| POST .../estimate {recipe}                     | JSON: ``pipeline.estimate`` + оценка времени, с/срез по замерам превью |

``region`` — ``x0,y0,x1,y1`` в пикселях среза (w×w, w — ширина ROI), для увеличенного фрагмента; ``rings`` —
пресет (``rings.PRESETS``, по умолчанию ``medium``); ``angles`` — ``fbp.ANGLE_MODES`` (по умолчанию
``first_180``); ``max_px`` — по умолчанию ``cfg.preview_max_px``.

Память: кроп — memmap uint16 на диске; полоса нормированных (и сдвинутых по образцу) строк вокруг строки превью
кэшируется, чтобы смена центра/наклона не нормировала кадры заново; её высота — запас под текущий наклон
(``axis.margin_rows``) с небольшим резервом, при большем наклоне пересчитывается. На 6 ГБ полоса для
~400–800 углов и ширины ~3000 — сотни МБ; кэш держится на GPU, при нехватке памяти — в RAM.
"""
from __future__ import annotations

import dataclasses
import threading
from typing import Any, Dict, Optional

from flask import Blueprint

from reconengine.model import CropData, ROI, ScanInfo

from .arbiter import Arbiter
from .config import Config

bp = Blueprint('sessions', __name__, url_prefix='/sessions')


@dataclasses.dataclass
class Session:
    id: str
    owner: str
    exp_id: str
    created: float
    last_seen: float
    state: str = 'open'                 # open | loading | ready | error | closed
    progress: float = 0.0
    stage: str = ''
    error: Optional[str] = None
    roi: Optional[ROI] = None
    scan: Optional[ScanInfo] = None
    crop: Optional[CropData] = None
    compute_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock, repr=False)
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict, repr=False)   # кэши preview


class SessionManager:
    def __init__(self, cfg: Config, scans):
        raise NotImplementedError

    arbiter: Arbiter

    def create(self, exp_id: str, owner: str, force: bool = False) -> Session:
        raise NotImplementedError

    def get(self, sid: str, owner: str) -> Session:
        raise NotImplementedError

    def close(self, sid: str, owner: Optional[str] = None) -> None:
        raise NotImplementedError

    def start_load(self, sid: str, owner: str, roi: ROI) -> None:
        raise NotImplementedError

    def summary(self) -> Dict[str, Any]:
        """Для /health: есть ли сессия, владелец, exp_id, состояние, простой."""
        raise NotImplementedError

    def start_reaper(self) -> None:
        raise NotImplementedError

    def stop_reaper(self) -> None:
        raise NotImplementedError
