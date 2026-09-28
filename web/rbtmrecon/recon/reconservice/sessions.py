"""Интерактивная сессия студии (``/sessions/*``) — одна на сервис, на GPU процесса сервиса.

Жизненный цикл:
- ``POST /sessions {exp_id, force?}`` — открыть. Если есть активная сессия другого пользователя — 409
  ``{error: 'busy', owner, exp_id, idle_s}``; с ``force: true`` старая закрывается (её владелец при следующем
  запросе получит 410 ``{error: 'taken_over', by}``). Тот же пользователь и тот же exp_id — возвращается
  существующая сессия; тот же пользователь и другой exp_id — старая закрывается, открывается новая.
  Пользователь (``X-Recon-User``) обязателен — без него 400.
- ``GET /sessions/<sid>`` — состояние: ``open`` → ``loading`` (progress, stage) → ``ready`` | ``error``; roi,
  владелец, простой, признак готовности кропа.
- ``POST /sessions/<sid>/load {roi}`` — 202; фоновый поток: ``CropLoader.load`` (кэш ``cfg.cache_dir(id)``,
  прогресс, отмена), затем dark/empty по кропу и сдвиги образца (advanced) → ``ready``. Новая загрузка
  отменяет текущую. ``POST .../load/cancel`` — отмена (состояние ``open``, stage ``canceled``).
- ``POST /sessions/<sid>/ping`` — продлить; ``DELETE /sessions/<sid>`` — закрыть (отмена загрузки, освобождение
  памяти GPU, ``arbiter.forget(sid)``). Поток-«уборщик» закрывает сессию после ``cfg.session_ttl_s`` простоя.
  Простой отсчитывается от последнего запроса владельца к сессии (любого, включая опрос состояния).
- Запросы к чужой сессии — 403, к закрытой/неизвестной — 404 (410 при перехвате).

Превью (сессия в ``ready``, иначе 409 ``{error: 'not_ready', state}``); вычисления — в модуле ``preview``,
на GPU сессии одновременно считается один запрос (``Session.compute_lock``), каналы с ``seq`` —
через ``arbiter`` (устаревший — 409 ``superseded``). Ось в параметрах — в координатах детектора:
``center`` — столбец оси на строке ``row``, ``tilt`` — наклон, градусы (как ``model.Axis``); без них берётся
текущая ось сессии (после ``axis/auto`` — найденная, до — авто-ось считается при первом запросе).
``row`` — строка детектора, по умолчанию ``roi.preview_row``.

| Метод и путь                                   | Ответ |
|------------------------------------------------|-------|
| GET  .../slice?row&center&tilt&rings&angles&region&max_px&seq | binary uint16 (h, w) среза; X-Meta: row, axis, rings, angles, n_angles, region, timings |
| POST .../axis/auto                             | JSON: axis, shift_x, alfa, углы пары 0°/180° |
| POST .../axis/scan {row, center?, tilt?, step, n, metric, region, seq} | binary uint16 (n, th, tw): фрагменты среза при центрах center + (i − n//2)·step, общее окно квантования; X-Meta: centers, metrics, best |
| POST .../axis/tilt {y_top, c_top, y_bottom, c_bottom} | JSON: axis (``axis.tilt_from_centers``) |
| POST .../axis/set {center, tilt, row?}         | JSON: axis — ось, заданная вручную (method ``manual``) |
| GET  .../axis/diff?center&tilt&max_px          | binary uint16: ``axis.diff_view`` пары 0°/180° |
| GET  .../rings/preview?row&center&tilt&preset&region&max_px&seq | binary uint16 (2, h, w): без колец и с пресетом, общее окно |
| GET  .../repositioning                         | JSON: применимость, checkpoint-ы (угол, sy, sx), накопленные сдвиги, предупреждения |
| POST .../recipe {center?, tilt?, row?, rings?, angles?, slices?, xy_roi?, pixel_size_mm?} | JSON: полный рецепт по состоянию сессии (ROI кропа, ось, размер пикселя) — для POST /jobs |
| POST .../estimate {recipe}                     | JSON: ``pipeline.estimate`` + оценка времени, с/срез по замерам превью |

``region`` — ``x0,y0,x1,y1`` в пикселях среза (w×w, w — ширина ROI), для увеличенного фрагмента; ``rings`` —
пресет (``rings.PRESETS``, по умолчанию ``medium``); ``angles`` — ``fbp.ANGLE_MODES`` (по умолчанию
``first_180``); ``max_px`` — по умолчанию ``cfg.preview_max_px``.

Подробности:
- ``axis/scan``: по умолчанию step = 1 px, n = 9, metric = ``grad``, region — квадрат 256 px с наибольшей
  энергией краёв (``preview.structured_region``); кандидаты сдвигаются в частотной области;
  принимает и ``rings``, ``angles``, ``max_px``. Метрика считается по показываемым фрагментам (см. ``preview``).
  В ``axis/diff`` центр — на строке ``row`` (по умолчанию ``roi.preview_row``).
- ``axis/tilt`` и ``axis/set`` делают ось текущей осью сессии (как ``axis/auto``; её же берёт ``recipe`` без
  center/tilt и отдаёт ``GET /sessions/<sid>`` при восстановлении страницы); ``slice`` и прочие с явными
  center/tilt текущую ось не меняют.
- ``estimate`` не требует ``ready``: без замеров превью ``time: null``.

Память: кроп — memmap uint16 на диске; полоса нормированных (и сдвинутых по образцу) строк вокруг строки превью
кэшируется, чтобы смена центра/наклона не нормировала кадры заново; её высота — запас под текущий наклон
(``axis.margin_rows``) с небольшим резервом, при большем наклоне пересчитывается. На 6 ГБ полоса для
~400–800 углов и ширины ~3000 — сотни МБ; кэш держится на GPU, при нехватке памяти — в RAM.
"""
from __future__ import annotations

import collections
import dataclasses
import logging
import threading
import time
import uuid
from typing import Any, Dict, Optional

from flask import Blueprint, current_app, jsonify, request

from reconengine import axis as axis_mod
from reconengine import data, fbp, gpu, rings
from reconengine import recipe as recipe_mod
from reconengine.model import Cancelled, CropData, ROI, ScanInfo, check_cancel

from . import auth, binary, cache, preview
from .arbiter import Arbiter
from .config import Config
from .scans import arg_float, arg_int, pixel_size_json

logger = logging.getLogger(__name__)

bp = Blueprint('sessions', __name__, url_prefix='/sessions')

#: доля прогресса загрузки на кроп (остальное — dark/empty и сдвиги образца)
_P_CROP = 0.85
#: сколько перехваченных сессий помнить (их владельцам — 410 вместо 404)
_TAKEN_KEEP = 32
_MAX_PX_RANGE = (16, 16384)


class SessionError(Exception):
    """Отказ по сессии: HTTP-статус и тело ответа ``{error, ...}``."""

    def __init__(self, status: int, error: str, **body):
        super().__init__(error)
        self.status = int(status)
        self.body = dict(error=error, **body)


@bp.errorhandler(SessionError)
def _session_error(e: SessionError):
    return jsonify(e.body), e.status


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

    # extra: 'ctx' — preview.Context (в ready), 'pixel_size' — PixelSize, 'cancel' — Event текущей загрузки,
    # 'load_gen' — номер загрузки (поток устаревшей загрузки не трогает состояние)

    def ctx(self) -> Optional[preview.Context]:
        return self.extra.get('ctx')


def _check_owner(owner: Optional[str]) -> str:
    owner = (owner or '').strip()
    if not owner:
        raise ValueError('не указан пользователь (X-Recon-User)')
    return owner


class SessionManager:
    """Одна интерактивная сессия на сервис. Все поля сессии меняются под ``self._lock``; вычисления превью — под
    ``Session.compute_lock``; загрузка — в фоновом потоке (одновременно идёт одна: новая ждёт отменённую, иначе
    два потока писали бы один файл кэша кропа)."""

    def __init__(self, cfg: Config, scans, clock=time.time):
        self.cfg = cfg
        self.scans = scans
        self.arbiter = Arbiter()
        self.clock = clock
        self._lock = threading.RLock()
        self._session: Optional[Session] = None
        self._taken: 'collections.OrderedDict[str, str]' = collections.OrderedDict()   # sid → кто перехватил
        self._loader: Optional[threading.Thread] = None
        self._reaper: Optional[threading.Thread] = None
        self._stop = threading.Event()

    arbiter: Arbiter

    # --- открытие, доступ, закрытие ------------------------------------------------------------------------

    def create(self, exp_id: str, owner: str, force: bool = False) -> Session:
        owner = _check_owner(owner)
        scan = self.scans.info(exp_id)                 # FileNotFoundError → 404
        ps = self.scans.pixel_size(exp_id)             # до замка: запрос к storage до 5 с
        closed = False
        with self._lock:
            now = self.clock()
            cur = self._session
            if cur is not None:
                if cur.owner == owner and cur.exp_id == exp_id:
                    cur.last_seen = now
                    return cur
                if cur.owner != owner and not force:
                    raise SessionError(409, 'busy', owner=cur.owner, exp_id=cur.exp_id,
                                       idle_s=round(now - cur.last_seen, 1))
                self._close_locked(cur, by=owner if cur.owner != owner else None)
                closed = True
            s = Session(id=uuid.uuid4().hex, owner=owner, exp_id=exp_id, created=now, last_seen=now, scan=scan)
            s.extra['pixel_size'] = ps
            self._session = s
        if closed:
            gpu.free_memory()
        logger.info('сессия %s: %s открыл %s', s.id[:8], owner, exp_id)
        return s

    def get(self, sid: str, owner: str) -> Session:
        with self._lock:
            s = self._session
            if s is not None and s.id == sid:
                if s.owner != owner:
                    raise SessionError(403, 'forbidden', owner=s.owner)
                s.last_seen = self.clock()
                return s
            if sid in self._taken:
                raise SessionError(410, 'taken_over', by=self._taken[sid])
        raise SessionError(404, 'not_found')

    def close(self, sid: str, owner: Optional[str] = None) -> None:
        with self._lock:
            s = self._session
            if s is None or s.id != sid:
                if sid in self._taken:
                    raise SessionError(410, 'taken_over', by=self._taken[sid])
                raise SessionError(404, 'not_found')
            if owner is not None and s.owner != owner:
                raise SessionError(403, 'forbidden', owner=s.owner)
            self._close_locked(s)
        gpu.free_memory()

    def _close_locked(self, s: Session, by: Optional[str] = None, reason: str = 'closed') -> None:
        """Закрыть сессию (под self._lock): отмена загрузки, кэши превью и кроп отпускаются, каналы arbiter-а
        забываются. by — кто перехватил (владельцу потом 410)."""
        s.state = 'closed'
        ev = s.extra.get('cancel')
        if ev is not None:
            ev.set()
        ctx = s.extra.pop('ctx', None)
        if ctx is not None:
            ctx.release()
        s.crop = None
        if self._session is s:
            self._session = None
        self.arbiter.forget(s.id)
        if by is not None:
            self._taken[s.id] = by
            while len(self._taken) > _TAKEN_KEEP:
                self._taken.popitem(last=False)
        logger.info('сессия %s (%s, %s) закрыта: %s', s.id[:8], s.owner, s.exp_id,
                    'перехвачена ' + by if by else reason)

    # --- загрузка ------------------------------------------------------------------------------------------

    def start_load(self, sid: str, owner: str, roi: ROI) -> None:
        s = self.get(sid, owner)
        roi.validate(s.scan.height, s.scan.width)
        with self._lock:
            if s.state == 'closed':
                raise SessionError(404, 'not_found')
            old = s.extra.get('cancel')
            if old is not None:
                old.set()
            ctx = s.extra.pop('ctx', None)
            if ctx is not None:
                ctx.release()
            cancel = threading.Event()
            gen = int(s.extra.get('load_gen', 0)) + 1
            s.extra.update(cancel=cancel, load_gen=gen)
            s.state, s.progress, s.stage, s.error = 'loading', 0.0, 'queued', None
            s.roi, s.crop = roi, None
            prev = self._loader
            t = threading.Thread(target=self._load, args=(s, roi, cancel, gen, prev),
                                 name='recon-load-{}'.format(s.id[:8]), daemon=True)
            self._loader = t
            t.start()
        if ctx is not None:
            gpu.free_memory()                 # полоса прежнего кропа
        logger.info('сессия %s: загрузка ROI %s', s.id[:8], roi.to_dict())

    def cancel_load(self, sid: str, owner: str) -> Session:
        s = self.get(sid, owner)
        with self._lock:
            ev = s.extra.get('cancel')
            if s.state == 'loading' and ev is not None:
                ev.set()
                s.stage = 'canceling'
        return s

    def _load(self, s: Session, roi: ROI, cancel: threading.Event, gen: int,
              prev: Optional[threading.Thread]) -> None:
        def current() -> bool:
            return s.extra.get('load_gen') == gen and s.state != 'closed'

        def progress(a: float, b: float):
            def cb(frac: float, stage: str) -> None:
                with self._lock:
                    if current() and not cancel.is_set():
                        s.progress = a + (b - a) * min(max(float(frac), 0.0), 1.0)
                        s.stage = stage
            return cb

        t0 = time.time()
        try:
            if prev is not None:
                prev.join()
            check_cancel(cancel)
            loader = data.CropLoader(s.scan, self.cfg.cache_dir(s.exp_id))
            crop = loader.load(roi, progress=progress(0.0, _P_CROP), cancel=cancel, workers=self.cfg.workers)
            check_cancel(cancel)
            ps = s.extra.get('pixel_size')
            ctx = preview.build_context(s.scan, crop, ps.value_mm, progress=progress(_P_CROP, 1.0), cancel=cancel)
            with self._lock:
                ok = current() and not cancel.is_set()
                if ok:
                    s.crop = crop
                    s.extra['ctx'] = ctx
                    s.state, s.progress, s.stage = 'ready', 1.0, 'ready'
            if not ok:
                ctx.release()
                check_cancel(cancel)
                return
            logger.info('сессия %s: кроп готов за %.1f с', s.id[:8], time.time() - t0)
            cache.cleanup_quiet(self.cfg)             # новый кроп мог превысить предел кэша на /fast
        except Cancelled:
            with self._lock:
                if current():
                    s.state, s.progress, s.stage = 'open', 0.0, 'canceled'
            logger.info('сессия %s: загрузка отменена', s.id[:8])
        except Exception as exc:  # noqa: BLE001 — ошибка загрузки видна в состоянии сессии
            logger.exception('сессия %s: ошибка загрузки', s.id[:8])
            with self._lock:
                if current():
                    s.state, s.stage = 'error', 'error'
                    s.error = '{}: {}'.format(type(exc).__name__, exc)

    def wait_loaded(self, timeout: Optional[float] = None) -> bool:
        """Дождаться окончания текущего потока загрузки (для тестов и остановки сервиса)."""
        t = self._loader
        if t is None:
            return True
        t.join(timeout)
        return not t.is_alive()

    # --- /health, уборщик ----------------------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Для /health: есть ли сессия, владелец, exp_id, состояние, простой."""
        with self._lock:
            s = self._session
            if s is None:
                return {'active': False}
            return {'active': True, 'owner': s.owner, 'exp_id': s.exp_id, 'state': s.state,
                    'progress': round(s.progress, 3), 'stage': s.stage,
                    'idle_s': round(self.clock() - s.last_seen, 1)}

    def reap(self, now: Optional[float] = None) -> bool:
        """Закрыть сессию, простаивающую дольше cfg.session_ttl_s. True — закрыта."""
        now = self.clock() if now is None else now
        with self._lock:
            s = self._session
            if s is None or now - s.last_seen <= self.cfg.session_ttl_s:
                return False
            self._close_locked(s, reason='простой {:.0f} с'.format(now - s.last_seen))
        gpu.free_memory()
        return True

    def start_reaper(self) -> None:
        if self._reaper is not None and self._reaper.is_alive():
            return
        self._stop.clear()
        interval = max(1.0, min(30.0, self.cfg.session_ttl_s / 10.0))

        def loop():
            while not self._stop.wait(interval):
                try:
                    self.reap()
                except Exception:  # noqa: BLE001 — уборщик не должен умирать
                    logger.exception('уборщик сессий')

        self._reaper = threading.Thread(target=loop, name='recon-session-reaper', daemon=True)
        self._reaper.start()

    def stop_reaper(self) -> None:
        """Остановить уборщика; заодно отменить загрузку (сервис останавливается)."""
        self._stop.set()
        t = self._reaper
        if t is not None:
            t.join(timeout=5)
        self._reaper = None
        with self._lock:
            s = self._session
            ev = s.extra.get('cancel') if s is not None else None
            if ev is not None:
                ev.set()


# --- эндпоинты ---------------------------------------------------------------------------------------------

def _mgr() -> SessionManager:
    return current_app.extensions['recon'].sessions


def _cfg() -> Config:
    return current_app.extensions['recon'].cfg


def _user() -> str:
    return auth.current_user()


def _body() -> Dict[str, Any]:
    body = request.get_json(silent=True)
    if body is None:
        return {}
    if not isinstance(body, dict):
        raise ValueError('тело запроса должно быть JSON-объектом')
    return body


def session_json(s: Session, now: Optional[float] = None) -> Dict[str, Any]:
    now = _mgr().clock() if now is None else now
    ctx = s.ctx()
    ps = s.extra.get('pixel_size')
    sc = s.scan
    return {
        'id': s.id, 'owner': s.owner, 'exp_id': s.exp_id, 'state': s.state,
        'progress': round(float(s.progress), 4), 'stage': s.stage, 'error': s.error,
        'roi': s.roi.to_dict() if s.roi is not None else None,
        'created': s.created, 'idle_s': round(max(0.0, now - s.last_seen), 1),
        'crop_ready': bool(s.state == 'ready' and ctx is not None),
        'scan': None if sc is None else {'height': sc.height, 'width': sc.width, 'n_frames': sc.n_frames,
                                         'n_data': int(len(sc.data_idx)), 'advanced': bool(sc.is_advanced),
                                         'fingerprint': sc.fingerprint},
        'pixel_size': pixel_size_json(ps) if ps is not None else None,
        'axis': ctx.axis.to_dict() if ctx is not None and ctx.axis is not None else None,
        'warnings': list(ctx.warnings) if ctx is not None else [],
    }


def _ready(s: Session) -> preview.Context:
    ctx = s.ctx()
    if s.state != 'ready' or ctx is None:
        raise SessionError(409, 'not_ready', state=s.state)
    return ctx


def _compute(sid: str, channel: Optional[str], seq, fn):
    """Вычисление превью: сессия владельца в ready, «последний выигрывает» по каналу (sid, channel) при заданном
    seq, один запрос на GPU сессии. fn(ctx, check) → результат."""
    mgr = _mgr()
    user = _user()
    s = mgr.get(sid, user)
    _ready(s)
    ticket = mgr.arbiter.begin((sid, channel or '-'), seq if channel else None)
    with s.compute_lock:
        ticket.check()
        s = mgr.get(sid, user)          # пока ждали — сессию могли закрыть или перезагрузить
        ctx = _ready(s)
        return fn(ctx, ticket.check)


def _seq(src) -> Optional[int]:
    return arg_int(src, 'seq')


def _region(src, name: str = 'region'):
    """'x0,y0,x1,y1' (строка из query) или список из JSON; None — не задан (проверка по ширине — в preview)."""
    v = src.get(name) if src is not None else None
    if v is None or v == '':
        return None
    parts = v.split(',') if isinstance(v, str) else v
    try:
        vals = [int(float(p)) for p in parts]
    except (TypeError, ValueError):
        raise ValueError('region: ожидается x0,y0,x1,y1, получено {!r}'.format(v)) from None
    if len(vals) != 4:
        raise ValueError('region: ожидается x0,y0,x1,y1, получено {!r}'.format(v))
    return tuple(vals)


def _str(src, name: str, default: str) -> str:
    v = src.get(name) if src is not None else None
    return default if v is None or v == '' else str(v)


def _max_px(src) -> int:
    return arg_int(src, 'max_px', _cfg().preview_max_px, *_MAX_PX_RANGE)


def _axis_args(ctx: preview.Context, src, check):
    row = arg_int(src, 'row', ctx.roi.preview_row)
    ctx.crop_row(row)
    ax = ctx.resolve_axis(row, arg_float(src, 'center'), arg_float(src, 'tilt', lo=-45.0, hi=45.0), check)
    return row, ax


@bp.post('')
def create():
    body = _body()
    exp_id = auth.valid_exp_id(body.get('exp_id'))
    s = _mgr().create(exp_id, _user(), force=bool(body.get('force')))
    return jsonify(session_json(s))


@bp.get('/<sid>')
def status(sid):
    return jsonify(session_json(_mgr().get(sid, _user())))


@bp.delete('/<sid>')
def close(sid):
    _mgr().close(sid, _user())
    return jsonify({'ok': True})


@bp.post('/<sid>/ping')
def ping(sid):
    s = _mgr().get(sid, _user())
    return jsonify({'ok': True, 'state': s.state, 'ttl_s': _cfg().session_ttl_s})


@bp.post('/<sid>/load')
def load(sid):
    roi_d = _body().get('roi')
    if not isinstance(roi_d, dict):
        raise ValueError('нужен roi {x0, x1, y0, y1, preview_row?}')
    try:
        roi = ROI.from_dict(roi_d)
    except (KeyError, TypeError) as exc:
        raise ValueError('некорректный roi: {!r}'.format(exc)) from None
    mgr = _mgr()
    mgr.start_load(sid, _user(), roi)
    return jsonify(session_json(mgr.get(sid, _user()))), 202


@bp.post('/<sid>/load/cancel')
def load_cancel(sid):
    return jsonify(session_json(_mgr().cancel_load(sid, _user())))


@bp.get('/<sid>/slice')
def slice_(sid):
    a = request.args

    def fn(ctx, check):
        row, ax = _axis_args(ctx, a, check)
        return ctx.slice(row, ax, _str(a, 'rings', preview.DEFAULT_RINGS), _str(a, 'angles', preview.DEFAULT_ANGLES),
                         _region(a), check, exact=a.get('exact', '') in ('1', 'true'))

    img, meta = _compute(sid, 'slice', _seq(a), fn)
    return binary.array_response(img, meta=meta, max_px=_max_px(a))


@bp.post('/<sid>/axis/auto')
def axis_auto(sid):
    return jsonify(_compute(sid, None, None, lambda ctx, check: ctx.auto_axis(check)))


@bp.post('/<sid>/axis/scan')
def axis_scan(sid):
    b = _body()

    def fn(ctx, check):
        row, ax = _axis_args(ctx, b, check)
        return ctx.center_scan(row, ax, step=arg_float(b, 'step', 1.0), n=arg_int(b, 'n', 9, 1, preview.SCAN_MAX_N),
                               metric=_str(b, 'metric', 'grad'), region=_region(b),
                               preset=_str(b, 'rings', preview.DEFAULT_RINGS),
                               angle_mode=_str(b, 'angles', preview.DEFAULT_ANGLES), check=check)

    frags, meta = _compute(sid, 'scan', _seq(b), fn)
    lo, hi = preview.pooled_window(frags)
    return binary.array_response(frags, lo=lo, hi=hi, meta=meta, max_px=_max_px(b))


@bp.post('/<sid>/axis/tilt')
def axis_tilt(sid):
    b = _body()
    vals = {}
    for k in ('y_top', 'c_top', 'y_bottom', 'c_bottom'):
        vals[k] = arg_float(b, k)
        if vals[k] is None:
            raise ValueError('нужны y_top, c_top, y_bottom, c_bottom')
    ax = axis_mod.tilt_from_centers(**vals)

    def fn(ctx, check):
        ctx.axis = ax
        return {'axis': ax.to_dict(), 'shift_x': axis_mod.to_crop_params(ax, ctx.roi)[0],
                'alfa': axis_mod.to_crop_params(ax, ctx.roi)[1]}

    return jsonify(_compute(sid, None, None, fn))


@bp.post('/<sid>/axis/set')
def axis_set(sid):
    """Ось, заданная вручную: ``{center, tilt, row}`` — столбец оси на строке детектора row и наклон, градусы.
    Становится осью сессии (превью без center/tilt, рецепт, восстановление страницы); сброс — ``axis/auto``."""
    b = _body()
    if arg_float(b, 'center') is None or arg_float(b, 'tilt', lo=-45.0, hi=45.0) is None:
        raise ValueError('нужны center и tilt')

    def fn(ctx, check):
        _, ax = _axis_args(ctx, b, check)
        ctx.axis = ax
        return {'axis': ax.to_dict()}

    return jsonify(_compute(sid, None, None, fn))


@bp.get('/<sid>/axis/diff')
def axis_diff(sid):
    a = request.args

    def fn(ctx, check):
        row, ax = _axis_args(ctx, a, check)
        return ctx.diff(ax), {'row': row, 'axis': ax.to_dict()}

    img, meta = _compute(sid, 'diff', _seq(a), fn)
    return binary.array_response(img, meta=meta, max_px=_max_px(a))


@bp.get('/<sid>/rings/preview')
def rings_preview(sid):
    a = request.args

    def fn(ctx, check):
        row, ax = _axis_args(ctx, a, check)
        return ctx.rings_preview(row, ax, _str(a, 'preset', preview.DEFAULT_RINGS),
                                 _str(a, 'angles', preview.DEFAULT_ANGLES), _region(a), check)

    stack, meta = _compute(sid, 'rings', _seq(a), fn)
    return binary.array_response(stack, meta=meta, max_px=_max_px(a))    # окно по обоим срезам — общее


@bp.get('/<sid>/repositioning')
def repositioning(sid):
    s = _mgr().get(sid, _user())
    return jsonify(_ready(s).repositioning_info())


@bp.post('/<sid>/recipe')
def make_recipe(sid):
    """Полный рецепт по состоянию сессии — для ``POST /jobs`` (браузеру не нужно знать схему рецепта).

    Тело (всё необязательно): ``center``/``tilt``/``row`` — ось (как у slice; без них — текущая ось сессии, при
    первом запросе авто-ось), ``rings`` — пресет, ``angles`` — режим углов, ``slices`` — [z0, z1) строк детектора
    (по умолчанию весь ROI), ``xy_roi``, ``pixel_size_mm`` — размер пикселя, введённый пользователем (иначе
    найденный для скана, с источником). ROI — загруженного кропа. Рецепт проверяется по кадру скана."""
    b = _body()
    s = _mgr().get(sid, _user())
    preset = _str(b, 'rings', preview.DEFAULT_RINGS)
    if preset not in rings.PRESETS:
        raise ValueError('неизвестный пресет колец: {}'.format(preset))
    angles = _str(b, 'angles', preview.DEFAULT_ANGLES)
    if angles not in fbp.ANGLE_MODES:
        raise ValueError('неизвестный режим углов: {}'.format(angles))
    user_ps = arg_float(b, 'pixel_size_mm', None, lo=1e-6, hi=10.0)

    def fn(ctx, check):
        if b.get('center') is None and b.get('tilt') is None:
            ax = ctx.current_axis(check)
        else:
            _, ax = _axis_args(ctx, b, check)
        ps = s.extra['pixel_size']
        r = recipe_mod.default_recipe(s.exp_id, s.scan.fingerprint, ctx.roi, user_ps or ps.value_mm,
                                      'user' if user_ps else ps.source, s.scan.is_advanced)
        r.pixel_size['user_edited'] = user_ps is not None
        r.author = s.owner
        r.axis = ax
        r.rings = {'preset': preset, 'params': None}
        r.recon['angles'] = angles
        if b.get('slices') is not None:
            z = b['slices']
            if not (isinstance(z, (list, tuple)) and len(z) == 2):
                raise ValueError('slices: нужно [z0, z1]')
            r.recon['slices'] = [int(z[0]), int(z[1])]
        if b.get('xy_roi') is not None:
            r.recon['xy_roi'] = dict(b['xy_roi'])
        r.provenance['steps'] = {'fov': 'checked',
                                 'axis': 'auto' if ax.method == 'auto' else 'checked',
                                 'rings': 'checked' if 'rings' in b else 'auto',
                                 'run': 'checked'}
        d = recipe_mod.to_dict(r)
        r = recipe_mod.from_dict(d)                      # проверка типов (xy_roi, slices из тела)
        recipe_mod.validate(r, s.scan.height, s.scan.width)
        return recipe_mod.to_dict(r)

    return jsonify(_compute(sid, None, None, fn))


@bp.post('/<sid>/estimate')
def estimate(sid):
    s = _mgr().get(sid, _user())
    rd = _body().get('recipe')
    if not isinstance(rd, dict):
        raise ValueError('нужен recipe')
    try:
        r = recipe_mod.from_dict(rd)
    except (KeyError, TypeError) as exc:
        raise ValueError('некорректный рецепт: {!r}'.format(exc)) from None
    recipe_mod.validate(r, s.scan.height, s.scan.width)
    ctx = s.ctx() if s.state == 'ready' else None
    rate = current_app.extensions['recon'].jobs.recent_rate()
    return jsonify(preview.estimate(r, s.scan, ctx, rate))
