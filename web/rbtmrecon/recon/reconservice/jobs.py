"""Очередь задач реконструкции (``/jobs/*``): Mongo ``autotom.jobs``, один исполнитель, GPU задач.

Документ задачи::

    {_id: str (uuid hex), exp_id, user, name (имя образца для файлов), recipe (dict), recipe_sha256,
     status: queued | running | publishing | done | error | canceled | interrupted,
     progress (0..1), stage, created, started, finished (UTC datetime), run_id, error, warnings,
     result (краткое: volume, binned, timings), cancel_requested: bool}

Эндпоинты:

| Метод и путь                 | Ответ |
|------------------------------|-------|
| POST /jobs {recipe, name?}   | 201 JSON задачи. Рецепт проверяется (``recipe.from_dict`` + ``recipe.validate`` по скану); активная (queued/running/publishing) задача по этому exp_id — 409 |
| GET  /jobs?exp_id=&limit=    | JSON: список задач, новые первыми (limit ≤ 200, по умолчанию 50) |
| GET  /jobs/<id>              | JSON задачи |
| GET  /jobs/<id>/log          | text/plain: хвост лога движка (``<run_dir>/engine.log``) |
| POST /jobs/<id>/cancel       | JSON: queued → canceled сразу; running → cancel_requested, процесс получает SIGTERM |

В JSON задачи даты — ISO 8601 с часовым поясом (UTC), кроме ``_id`` есть ``id`` (то же значение); в списке
``GET /jobs`` рецепта нет (он в ``GET /jobs/<id>``) — список опрашивается часто. ``GET /jobs/<id>/log?lines=N``
(по умолчанию 200, до 5000) — пустой текст, если лога ещё (queued) или уже (canceled/interrupted) нет. 409 от
``POST /jobs``: ``{error, job_id}`` активной задачи. Отмена задачи в publishing не выполняется (перенос файлов
не прерывается), ответ — задача как есть.

Исполнитель (поток ``JobRunner``):
- при старте задачи в running/publishing помечаются ``interrupted`` (сервис перезапускали посреди работы);
- цикл: старейшая queued-задача → запуск; нет задач и ``cfg.legacy_queue`` — один шаг старой очереди ноутбуков
  (``tomo_worker.process_once`` под ``gpu_lock(cfg.gpu_lock)``); иначе пауза ``cfg.job_poll_s``;
- запуск: ``run_dir = cfg.runs_dir(exp)/<run_id>``, рецепт в ``run_dir/recipe.in.json``, процесс
  ``[cfg.job_python, '-m', 'reconengine', 'run', <скан>, '--recipe', …, '--out', run_dir, '--cache',
  cfg.cache_dir(exp), '--name', name, '--progress-json', '--gpu-lock', cfg.gpu_lock]`` с
  ``CUDA_VISIBLE_DEVICES=cfg.job_gpu`` и ``PYTHONPATH`` = каталог с ``reconengine``; stderr — в
  ``run_dir/engine.log``, строки-JSON ``{"progress": f, "stage": s}`` обновляют задачу (не чаще раза в секунду);
- отмена: ``cancel_requested`` проверяется раз в секунду → SIGTERM, через 30 с — SIGKILL; код 130 — canceled;
- код 0 → ``publishing`` → ``publish.publish`` → ``done``; затем ``publish.archive_h5`` (ошибка — предупреждение);
- иначе ``error`` с последними строками лога.
Статусы дублируются в ``autotom.tomoobjects`` (документы ``{obj_id, status, date}``, как ``tomo_queue``):
reconstructing / done / error: … / canceled — старая страница очереди показывает и новые задачи.

Уточнения к исполнителю:
- процессу передаётся ``--run-id <run_id>``: тот же id в result.json, поэтому задача, каталог запуска и
  ``history/<run_id>`` публикации связаны одним id; ``--gpu-lock`` — только если ``cfg.gpu_lock`` задан;
  stdout процесса (итоговый JSON) идёт в тот же ``engine.log``;
- статус canceled определяется по факту запроса отмены, а не по коду выхода: на Windows процессу шлётся Ctrl+Break
  (движок отменяет запуск так же, как по SIGTERM), а без общей консоли — TerminateProcess без очистки; после
  отмены и прерывания ``run_dir`` удаляется исполнителем (после ошибки — остаётся с логом для разбора);
- ``stop()`` (остановка сервиса) перестаёт брать задачи, прерывает запущенную (SIGTERM, через 30 с — SIGKILL),
  помечает её ``interrupted`` (в tomoobjects — ``error: interrupted``), удаляет её ``run_dir`` и ждёт поток;
  повторный вызов безопасен. Шаг старой очереди (ноутбук) и публикацию ``stop()`` не прерывает;
- задача, прерванная в publishing, при старте публикуется заново (публикация повторяема, см. ``publish``), если
  ``result.json`` ещё в ``run_dir``; если уже опубликован её ``run_id`` — она ``done``;
- шаг старой очереди берёт блокировку GPU, только если в очереди есть задание (иначе простаивающий сервис
  ждал бы блокировку, пока Jupyter держит GPU).
"""
from __future__ import annotations

import collections
import datetime
import json
import logging
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from flask import Blueprint, Response, current_app, jsonify, request
from pymongo import ASCENDING, DESCENDING, ReturnDocument

import reconengine
from reconengine import gpu
from reconengine import recipe as recipe_mod

from . import auth, cache, publish
from .config import Config

logger = logging.getLogger(__name__)

bp = Blueprint('jobs', __name__, url_prefix='/jobs')

ACTIVE = ('queued', 'running', 'publishing')
FINAL = ('done', 'error', 'canceled', 'interrupted')
#: каталог, где лежит пакет reconengine: cwd и PYTHONPATH процесса задачи
ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(reconengine.__file__)))

EXIT_CANCELLED = 130
KILL_AFTER_S = 30.0          # после SIGTERM — SIGKILL
_DB_PROGRESS_S = 1.0         # прогресс в Mongo — не чаще
_CANCEL_POLL_S = 1.0         # cancel_requested в Mongo — не реже
_WAIT_STEP_S = 0.2           # шаг ожидания процесса
_TAIL_LINES = 20             # строк лога в поле error
_LOG_READ_BYTES = 1 << 20    # хвост лога для /log читается не дальше 1 МБ от конца
_ERROR_CHARS = 4000
_SHORT_ERROR_CHARS = 150     # как у старой очереди: 'error: ' + кратко
#: имя образца попадает в имена файлов: без разделителей путей и служебных символов
_NAME_BAD = re.compile(r'[\x00-\x1f/\\:*?"<>|]')


class JobConflict(Exception):
    """По эксперименту уже есть активная задача (409)."""

    def __init__(self, message: str, job_id: str):
        super().__init__(message)
        self.job_id = job_id


def _utcnow() -> datetime.datetime:
    # Mongo хранит UTC без пояса: наивное время в UTC (pymongo так же возвращает его)
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def _iso(value: datetime.datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=datetime.timezone.utc)
    return value.isoformat()


def public(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Документ задачи → JSON API: даты в ISO 8601 (UTC), ``id`` рядом с ``_id``."""
    out = {k: (_iso(v) if isinstance(v, datetime.datetime) else v) for k, v in doc.items()}
    out['id'] = out.get('_id')
    return out


def sample_name(name: Optional[str], exp_id: str) -> str:
    """Имя образца для файлов объёма (пробелы движок заменит на «_»); пустое — exp_id."""
    if name is None:
        return exp_id
    if not isinstance(name, str):
        raise ValueError('name должен быть строкой')
    name = name.strip()
    if not name:
        return exp_id
    if len(name) > 100 or _NAME_BAD.search(name) or name.startswith('.'):
        raise ValueError('некорректное имя образца {!r}: без / \\ : * ? " < > |, без точки в начале, '
                         'до 100 символов'.format(name))
    return name


def _parse_progress(line: str) -> Optional[Tuple[float, str]]:
    """Строка ``{"progress": f, "stage": s}`` от ``reconengine run --progress-json`` → (f, s)."""
    if not line.startswith('{"progress"'):
        return None
    try:
        d = json.loads(line)
        return float(d['progress']), str(d.get('stage') or '')
    except (ValueError, KeyError, TypeError):
        return None


def _short_error(lines: List[str], rc: Optional[int]) -> str:
    """Кратко для tomoobjects: последняя строка вида «Исключение: текст» или просто последняя строка."""
    for line in reversed(lines):
        s = line.strip()
        if re.match(r'^[A-Za-z_][\w.]*(Error|Exception|Cancelled|Interrupt)\b', s):
            return s[:_SHORT_ERROR_CHARS]
    for line in reversed(lines):
        if line.strip():
            return line.strip()[:_SHORT_ERROR_CHARS]
    return 'код выхода {}'.format(rc)


def _rmtree(path: str) -> None:
    """Удалить каталог запуска; на Windows файлы убитого процесса освобождаются не мгновенно — повторы."""
    for k in range(10):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            if k == 9:
                logger.warning('не удалось удалить %s: %s', path, exc)
                return
            time.sleep(0.2)


def _popen_kwargs() -> Dict[str, Any]:
    """Процесс задачи — в своей группе: сигналы терминала сервиса до него не доходят, отмену шлёт исполнитель;
    на Windows группа нужна для Ctrl+Break (os.kill с CTRL_BREAK_EVENT шлёт его всей группе)."""
    if os.name == 'nt':
        return {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
    return {'start_new_session': True}


def _terminate(proc: subprocess.Popen) -> None:
    """Попросить процесс движка остановиться: SIGTERM; на Windows — Ctrl+Break (обработчик движка отменяет запуск),
    а если консоли нет — TerminateProcess (без очистки: run_dir удалит исполнитель)."""
    try:
        if os.name == 'nt':
            try:
                os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
                return
            except OSError:
                proc.terminate()
        else:
            proc.send_signal(signal.SIGTERM)
    except (OSError, ProcessLookupError):       # уже завершился
        pass


class _EngineOutput(threading.Thread):
    """Читает объединённые stdout/stderr движка: всё — в engine.log, строки-JSON прогресса — в ``progress``,
    остальные непустые строки — в хвост для поля error."""

    def __init__(self, stream, log):
        super().__init__(name='recon-job-output', daemon=True)
        self.stream = stream
        self.log = log
        self._lock = threading.Lock()
        self._progress: Optional[Tuple[float, str]] = None
        self._tail: Deque[str] = collections.deque(maxlen=_TAIL_LINES)

    def run(self) -> None:
        try:
            for raw in iter(self.stream.readline, b''):
                try:
                    self.log.write(raw)
                    self.log.flush()
                except (OSError, ValueError):
                    pass
                line = raw.decode('utf-8', 'replace').rstrip('\r\n')
                p = _parse_progress(line)
                with self._lock:
                    if p is not None:
                        self._progress = p
                    elif line.strip():
                        self._tail.append(line)
        except (OSError, ValueError):        # поток закрыт
            pass

    def progress(self) -> Optional[Tuple[float, str]]:
        with self._lock:
            return self._progress

    def tail(self) -> List[str]:
        with self._lock:
            return list(self._tail)


def _legacy_worker():
    """Модуль старой очереди ноутбуков. Импорт ленивый: tomo_worker тянет nbformat и tomotools2, а tomo_queue при
    импорте создаёт клиент Mongo из conf.py (в тестах подменяется)."""
    import tomo_worker  # noqa: WPS433
    return tomo_worker


class JobRunner:
    """Исполнитель: одна задача за раз. ``run_once()`` — один шаг (тесты зовут его напрямую), поток ``start()`` —
    цикл над ним."""

    kill_after_s = KILL_AFTER_S
    #: сколько stop() ждёт завершения текущей задачи и потока
    stop_timeout_s = KILL_AFTER_S + 15.0
    #: пустая старая очередь проверяется не чаще (прежний воркер спал 10 с)
    legacy_poll_s = 10.0

    def __init__(self, svc: 'JobService'):
        self.svc = svc
        self.cfg: Config = svc.cfg
        self._stopping = threading.Event()
        self._wake = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._thread: Optional[threading.Thread] = None
        self._recovered = False
        self._legacy = None                   # модуль tomo_worker; False — недоступен
        self._legacy_next = 0.0               # раньше этого времени (monotonic) старую очередь не проверять
        self.current: Optional[Dict[str, str]] = None   # {id, exp_id, run_dir} выполняемой задачи

    # --- поток --------------------------------------------------------------------------------------------

    def wake(self) -> None:
        self._wake.set()

    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.alive():
            return
        self._stopping.clear()
        self._thread = threading.Thread(target=self._loop, name='recon-jobs', daemon=True)
        self._thread.start()

    def stop(self, timeout: Optional[float] = None) -> None:
        """Не брать новые задачи; запущенную прервать (interrupted); дождаться потока. Повторный вызов безопасен."""
        self._stopping.set()
        self._wake.set()
        timeout = self.stop_timeout_s if timeout is None else timeout
        deadline = time.monotonic() + timeout
        if not self._idle.wait(timeout):
            logger.warning('текущий шаг очереди не завершился за %.0f с', timeout)
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(max(0.0, deadline - time.monotonic()))
            if t.is_alive():
                logger.warning('поток очереди задач не остановился')
            else:
                self._thread = None

    def _loop(self) -> None:
        while not self._stopping.is_set():
            took = False
            try:
                took = self.run_once()
            except Exception:  # noqa: BLE001 — поток очереди не должен умирать (Mongo недоступна и т.п.)
                logger.exception('ошибка в цикле задач')
            if not took and not self._stopping.is_set():
                self._wake.wait(self.cfg.job_poll_s)
                self._wake.clear()

    # --- шаг ----------------------------------------------------------------------------------------------

    def run_once(self) -> bool:
        """Один шаг: восстановление после рестарта (первый раз), старейшая queued-задача, иначе шаг старой очереди.
        True — что-то выполнено."""
        self._idle.clear()
        try:
            if self._stopping.is_set():
                return False
            if not self._recovered:
                self.recover()
            job = self._claim()
            if job is None:
                return self._legacy_step() if self.cfg.legacy_queue else False
            self._execute(job)
            return True
        finally:
            self._idle.set()

    def _claim(self) -> Optional[Dict[str, Any]]:
        if self._stopping.is_set():
            return None
        return self.svc.coll.find_one_and_update(
            {'status': 'queued'},
            {'$set': {'status': 'running', 'started': _utcnow(), 'run_id': uuid.uuid4().hex, 'progress': 0.0,
                      'stage': 'start'}},
            sort=[('created', ASCENDING)], return_document=ReturnDocument.AFTER)

    def recover(self) -> int:
        """Задачи running/publishing прошлого процесса: publishing с неопубликованным результатом — публикуется
        заново, остальные → interrupted (run_dir удаляется). Возвращает число обработанных задач."""
        n = 0
        for doc in list(self.svc.coll.find({'status': {'$in': ['running', 'publishing']}})):
            if self.current and doc['_id'] == self.current['id']:
                continue
            n += 1
            if doc['status'] == 'publishing' and self._resume_publish(doc):
                continue
            self._interrupt(doc, 'сервис перезапущен во время задачи')
        self._recovered = True
        return n

    def _run_dir(self, doc: Dict[str, Any]) -> Optional[str]:
        run_id = doc.get('run_id')
        if not publish.safe_run_id(run_id) or not auth.EXP_ID_RE.match(str(doc.get('exp_id', ''))):
            return None
        return os.path.join(self.cfg.runs_dir(doc['exp_id']), run_id)

    def _resume_publish(self, doc: Dict[str, Any]) -> bool:
        run_dir = self._run_dir(doc)
        if run_dir is None:
            return False
        published = publish.read_published(self.cfg, doc['exp_id'])
        if published is not None and published.get('run_id') == doc['run_id']:
            logger.info('задача %s: результат уже опубликован до перезапуска', doc['_id'])
            self._done(doc, published)
            return True
        if not os.path.isfile(os.path.join(run_dir, publish.RESULT)):
            return False
        logger.info('задача %s: публикация прервана перезапуском — повтор', doc['_id'])
        try:
            self._publish(doc, run_dir)
        except Exception as exc:  # noqa: BLE001
            logger.exception('задача %s: повторная публикация не удалась', doc['_id'])
            self._fail(doc, 'публикация: {}: {}'.format(type(exc).__name__, exc))
        return True

    # --- выполнение ---------------------------------------------------------------------------------------

    def _execute(self, job: Dict[str, Any]) -> None:
        job_id, exp_id = job['_id'], job['exp_id']
        run_dir = self._run_dir(job)
        self.current = {'id': job_id, 'exp_id': exp_id, 'run_dir': run_dir}
        logger.info('задача %s (%s): запуск %s', job_id, exp_id, job['run_id'])
        if self.svc.on_source_read is not None:
            self.svc.on_source_read('job')
        self.svc.tomo_status(exp_id, 'reconstructing', job_id)
        try:
            reason, rc, tail = self._run_engine(job, run_dir)
            if reason == 'interrupted':
                self._interrupt(job, 'сервис остановлен во время задачи')
            elif reason == 'canceled':
                self._cancelled(job, run_dir)
            elif rc == 0:
                self._publish(job, run_dir)
            else:
                text = '\n'.join(tail)
                error = 'движок завершился с кодом {}'.format(rc) + ('\n' + text if text else '')
                self._fail(job, error[-_ERROR_CHARS:], _short_error(tail, rc))
        except Exception as exc:  # noqa: BLE001 — задача завершается ошибкой, исполнитель живёт дальше
            logger.exception('задача %s: сбой', job_id)
            self._fail(job, '{}: {}'.format(type(exc).__name__, exc))
        finally:
            self.current = None
            cache.cleanup_quiet(self.cfg)             # задача могла создать кроп сверх предела кэша

    def _engine_cmd(self, job: Dict[str, Any], scan_path: str, recipe_path: str, run_dir: str) -> List[str]:
        cmd = [self.cfg.job_python, '-m', 'reconengine', 'run', scan_path, '--recipe', recipe_path, '--out', run_dir,
               '--cache', self.cfg.cache_dir(job['exp_id']), '--name', job.get('name') or job['exp_id'],
               '--run-id', job['run_id'], '--progress-json']
        if self.cfg.gpu_lock:
            cmd += ['--gpu-lock', self.cfg.gpu_lock]
        return cmd

    def _engine_env(self) -> Dict[str, str]:
        env = dict(os.environ)
        env['PYTHONPATH'] = ENGINE_ROOT + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUNBUFFERED'] = '1'
        if self.cfg.job_gpu not in (None, ''):
            env['CUDA_VISIBLE_DEVICES'] = str(self.cfg.job_gpu)
        return env

    def _run_engine(self, job: Dict[str, Any], run_dir: str) -> Tuple[Optional[str], Optional[int], List[str]]:
        """Запустить движок и дождаться. (reason, код выхода, хвост вывода); reason — None | canceled | interrupted."""
        os.makedirs(run_dir, exist_ok=True)
        recipe_path = os.path.join(run_dir, 'recipe.in.json')
        recipe_mod.save(recipe_mod.from_dict(job['recipe']), recipe_path)
        scan_path = self.svc.scans.path(job['exp_id'])
        cmd = self._engine_cmd(job, scan_path, recipe_path, run_dir)
        with open(os.path.join(run_dir, 'engine.log'), 'ab') as log:
            log.write('# задача {}, запуск {}: {}\n'.format(job['_id'], job['run_id'], ' '.join(cmd)).encode('utf-8'))
            log.flush()
            proc = subprocess.Popen(cmd, cwd=ENGINE_ROOT, env=self._engine_env(), stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **_popen_kwargs())
            out = _EngineOutput(proc.stdout, log)
            out.start()
            try:
                reason = self._watch(job['_id'], proc, out)
            finally:
                if proc.poll() is None:              # исключение в самом исполнителе: процесс не оставляем
                    proc.kill()
                proc.wait()
                out.join(10.0)
                proc.stdout.close()
        return reason, proc.returncode, out.tail()

    def _watch(self, job_id: str, proc: subprocess.Popen, out: _EngineOutput) -> Optional[str]:
        """Ждать процесс: прогресс в Mongo (не чаще раза в секунду), отмена (раз в секунду или по wake), остановка
        сервиса. Возвращает причину остановки или None, если процесс завершился сам."""
        reason: Optional[str] = None
        t_sent = t_db = 0.0
        t_cancel = -_CANCEL_POLL_S
        killed = woke = False
        written = None
        while True:
            running = proc.poll() is None
            now = time.monotonic()
            if running:
                if reason is None:
                    if self._stopping.is_set():
                        reason = 'interrupted'
                    elif woke or now - t_cancel >= _CANCEL_POLL_S:
                        t_cancel = now
                        if self.svc.cancel_requested(job_id):
                            reason = 'canceled'
                    if reason is not None:
                        logger.info('задача %s: %s — остановка процесса %d', job_id, reason, proc.pid)
                        _terminate(proc)
                        t_sent = now
                elif not killed and now - t_sent >= self.kill_after_s:
                    logger.warning('задача %s: процесс %d не завершился за %.0f с — SIGKILL', job_id, proc.pid,
                                   self.kill_after_s)
                    proc.kill()
                    killed = True
            p = out.progress()
            if reason is None and p is not None and p != written and (not running or now - t_db >= _DB_PROGRESS_S):
                written, t_db = p, now
                try:
                    self.svc.set_fields(job_id, {'progress': round(p[0], 4), 'stage': p[1]})
                except Exception:  # noqa: BLE001 — сбой Mongo не должен прерывать расчёт
                    logger.exception('задача %s: прогресс не записан', job_id)
            if not running:
                return reason
            woke = self._wake.wait(_WAIT_STEP_S)
            self._wake.clear()

    # --- завершение ---------------------------------------------------------------------------------------

    def _publish(self, job: Dict[str, Any], run_dir: str) -> None:
        job_id, exp_id = job['_id'], job['exp_id']
        self.svc.set_fields(job_id, {'status': 'publishing', 'stage': 'publish'})
        doc = publish.publish(self.cfg, exp_id, run_dir)
        self._done(job, doc)
        try:
            if publish.archive_h5(self.cfg, exp_id):
                logger.info('задача %s: архивная копия %s.h5 создана', job_id, exp_id)
        except Exception as exc:  # noqa: BLE001 — результат уже опубликован
            logger.exception('задача %s: архивная копия не создана', job_id)
            self.svc.push_warning(job_id, 'архивная копия {}.h5 не создана: {}: {}'.format(
                exp_id, type(exc).__name__, exc))

    def _done(self, job: Dict[str, Any], doc: Dict[str, Any]) -> None:
        warnings = list(job.get('warnings') or [])
        warnings += [w for w in doc.get('warnings') or [] if w not in warnings]
        result = {k: doc.get(k) for k in ('run_id', 'volume', 'binned', 'timings')}
        self.svc.set_fields(job['_id'], {'status': 'done', 'finished': _utcnow(), 'progress': 1.0, 'stage': 'done',
                                         'result': result, 'warnings': warnings})
        self.svc.tomo_status(job['exp_id'], 'done', job['_id'])
        logger.info('задача %s (%s): готово', job['_id'], job['exp_id'])

    def _fail(self, job: Dict[str, Any], error: str, short: Optional[str] = None) -> None:
        self.svc.set_fields(job['_id'], {'status': 'error', 'finished': _utcnow(), 'error': error})
        if not short:
            short = error.strip().splitlines()[-1][:_SHORT_ERROR_CHARS] if error.strip() else 'ошибка'
        self.svc.tomo_status(job['exp_id'], 'error: {}'.format(short), job['_id'])
        logger.error('задача %s (%s): ошибка: %s', job['_id'], job['exp_id'], short)

    def _cancelled(self, job: Dict[str, Any], run_dir: Optional[str]) -> None:
        if run_dir:
            _rmtree(run_dir)
        self.svc.set_fields(job['_id'], {'status': 'canceled', 'finished': _utcnow(), 'stage': None})
        self.svc.tomo_status(job['exp_id'], 'canceled', job['_id'])
        logger.info('задача %s (%s): отменена', job['_id'], job['exp_id'])

    def _interrupt(self, job: Dict[str, Any], error: str) -> None:
        run_dir = self._run_dir(job)
        if run_dir:
            _rmtree(run_dir)
        self.svc.set_fields(job['_id'], {'status': 'interrupted', 'finished': _utcnow(), 'error': error})
        self.svc.tomo_status(job['exp_id'], 'error: interrupted', job['_id'])
        logger.warning('задача %s (%s): прервана (%s)', job['_id'], job['exp_id'], error)

    # --- старая очередь -----------------------------------------------------------------------------------

    def _legacy_module(self):
        if self._legacy is None:
            try:
                # ноутбук должен считать на GPU задач: tomo_worker берёт его из RECON_JOB_GPU (как и Config)
                if self.cfg.job_gpu not in (None, '') and not os.environ.get('RECON_JOB_GPU'):
                    os.environ['RECON_JOB_GPU'] = str(self.cfg.job_gpu)
                self._legacy = _legacy_worker()
            except Exception:  # noqa: BLE001 — без nbformat/tomotools2 старая очередь просто не обслуживается
                logger.exception('старая очередь ноутбуков недоступна')
                self._legacy = False
        return self._legacy or None

    def _legacy_step(self) -> bool:
        """Один шаг старой очереди: задание есть — под блокировкой GPU ``process_once()``. Пустую очередь
        проверяем не чаще раза в ``legacy_poll_s`` (как прежний воркер: tomo_queue пишет в лог каждую проверку)."""
        if time.monotonic() < self._legacy_next:
            return False
        tw = self._legacy_module()
        if tw is None:
            return False
        try:
            if tw.get_rec_queue_next_obj() is None:
                self._legacy_next = time.monotonic() + self.legacy_poll_s
                return False
            with gpu.gpu_lock(self.cfg.gpu_lock):
                return bool(tw.process_once())
        except Exception:  # noqa: BLE001 — как worker_loop: ошибка шага не роняет исполнитель
            logger.exception('ошибка шага старой очереди')
            self._legacy_next = time.monotonic() + self.legacy_poll_s
            return False


class JobService:
    """Задачи в Mongo и исполнитель. Методы вызываются из потоков Flask одновременно с исполнителем."""

    def __init__(self, cfg: Config, db, scans):
        self.cfg = cfg
        self.db = db
        self.scans = scans
        self._create_lock = threading.Lock()
        self.runner = JobRunner(self)
        #: вызывается с причиной перед запуском задачи (ServiceState: остановить предзагрузку исходников)
        self.on_source_read: Optional[Callable[[str], None]] = None

    @property
    def coll(self):
        return self.db['jobs']

    @property
    def tomo(self):
        return self.db['tomoobjects']

    # --- API ----------------------------------------------------------------------------------------------

    def create(self, recipe: Dict[str, Any], user: str, name: Optional[str] = None) -> Dict[str, Any]:
        if not isinstance(recipe, dict):
            raise ValueError('recipe должен быть объектом JSON')
        try:
            r = recipe_mod.from_dict(recipe)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError('recipe: некорректная структура ({}: {})'.format(type(exc).__name__, exc))
        exp_id = r.input.get('exp_id')
        if not isinstance(exp_id, str) or not auth.EXP_ID_RE.match(exp_id) or '..' in exp_id:
            raise ValueError('recipe: некорректный input.exp_id: {!r}'.format(exp_id))
        name = sample_name(name, exp_id)
        info = self.scans.info(exp_id)                   # FileNotFoundError → 404
        try:
            recipe_mod.validate(r, info.height, info.width)
        except (KeyError, TypeError) as exc:
            raise ValueError('recipe: некорректная структура ({}: {})'.format(type(exc).__name__, exc))
        warnings = []
        fp = r.input.get('fingerprint')
        if fp and fp != info.fingerprint:
            warnings.append('отпечаток файла скана не совпадает с рецептом: файл изменился после настройки')
        if not r.author:
            r.author = user
        doc = {
            '_id': uuid.uuid4().hex, 'exp_id': exp_id, 'user': user, 'name': name,
            'recipe': recipe_mod.to_dict(r), 'recipe_sha256': recipe_mod.sha256(r),
            'status': 'queued', 'progress': 0.0, 'stage': None,
            'created': _utcnow(), 'started': None, 'finished': None, 'run_id': None,
            'error': None, 'warnings': warnings, 'result': None, 'cancel_requested': False,
        }
        with self._create_lock:
            active = self.coll.find_one({'exp_id': exp_id, 'status': {'$in': list(ACTIVE)}}, {'_id': 1, 'status': 1})
            if active is not None:
                raise JobConflict('по {} уже есть активная задача {} ({})'.format(
                    exp_id, active['_id'], active['status']), active['_id'])
            self.coll.insert_one(doc)
        logger.info('задача %s (%s) поставлена в очередь: %s', doc['_id'], exp_id, user)
        self.runner.wake()
        return public(doc)

    def _doc(self, job_id: str, projection=None) -> Dict[str, Any]:
        doc = self.coll.find_one({'_id': str(job_id)}, projection)
        if doc is None:
            raise FileNotFoundError('задача {} не найдена'.format(job_id))
        return doc

    def get(self, job_id: str) -> Dict[str, Any]:
        return public(self._doc(job_id))

    def list(self, exp_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        limit = min(max(int(limit), 1), 200)
        query = {'exp_id': exp_id} if exp_id else {}
        cur = self.coll.find(query, {'recipe': 0}).sort('created', DESCENDING).limit(limit)
        return [public(d) for d in cur]

    def cancel(self, job_id: str, user: str) -> Dict[str, Any]:
        job_id = str(job_id)
        doc = self.coll.find_one_and_update(
            {'_id': job_id, 'status': 'queued'},
            {'$set': {'status': 'canceled', 'finished': _utcnow(), 'cancel_requested': True, 'canceled_by': user}},
            return_document=ReturnDocument.AFTER)
        if doc is not None:
            # задача не запускалась: в tomoobjects о ней ничего не писали — и отмену не пишем
            logger.info('задача %s отменена до запуска: %s', job_id, user)
            return public(doc)
        doc = self.coll.find_one_and_update(
            {'_id': job_id, 'status': 'running'},
            {'$set': {'cancel_requested': True, 'canceled_by': user}},
            return_document=ReturnDocument.AFTER)
        if doc is not None:
            logger.info('задача %s: запрошена отмена: %s', job_id, user)
            self.runner.wake()
            return public(doc)
        return self.get(job_id)                  # publishing или завершена — как есть (нет — 404)

    def log_tail(self, job_id: str, lines: int = 200) -> str:
        """Последние строки engine.log задачи; пустая строка — лога нет."""
        doc = self._doc(job_id, {'exp_id': 1, 'run_id': 1})
        run_dir = self.runner._run_dir(doc)
        if run_dir is None:
            return ''
        path = os.path.join(run_dir, 'engine.log')
        try:
            with open(path, 'rb') as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - _LOG_READ_BYTES))
                data = fh.read()
        except FileNotFoundError:
            return ''
        text = data.decode('utf-8', 'replace').splitlines()
        if size > _LOG_READ_BYTES and text:
            text = text[1:]                          # первая строка обрезана
        return '\n'.join(text[-max(1, int(lines)):]) + '\n' if text else ''

    def recent_rate(self, last: int = 5) -> Optional[Dict[str, float]]:
        """Скорость по последним выполненным задачам для оценки времени новой: медиана секунд реконструкции на
        (срез · ширина² · угол) — FBP и выравнивание растут как площадь среза и число углов; и медиана секунд
        подготовки (dark/empty, сдвиги образца, ось). None — выполненных задач с замерами нет (или нет Mongo)."""
        try:
            docs = list(self.coll.find({'status': 'done', 'result.timings.recon_s': {'$exists': True}},
                                       {'result': 1, 'recipe.fov': 1}).sort('finished', DESCENDING).limit(last))
        except Exception:  # noqa: BLE001 — оценка не обязательна
            return None
        unit, prepare = [], []
        for d in docs:
            t = (d.get('result') or {}).get('timings') or {}
            shape = ((d.get('result') or {}).get('volume') or {}).get('shape') or []
            fov = (d.get('recipe') or {}).get('fov') or {}
            try:
                nz, w, n_ang = int(shape[0]), int(fov['x1']) - int(fov['x0']), int(t['n_angles'])
                unit.append(float(t['recon_s']) / (nz * w * w * n_ang))
                prepare.append(float(t.get('prepare_s', 0.0)))
            except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
                continue
        if not unit:
            return None
        return {'recon_s_per_slice_px2_angle': statistics.median(unit), 'prepare_s': statistics.median(prepare),
                'jobs': len(unit)}

    def summary(self) -> Dict[str, Any]:
        """Для /health: число queued, текущая задача (id, exp_id, progress, stage)."""
        try:
            queued = self.coll.count_documents({'status': 'queued'})
            cur = self.coll.find_one({'status': {'$in': ['running', 'publishing']}},
                                     {'exp_id': 1, 'status': 1, 'progress': 1, 'stage': 1})
        except Exception as exc:  # noqa: BLE001 — /health не должен падать без Mongo
            return {'error': '{}: {}'.format(type(exc).__name__, exc), 'runner': self.runner.alive()}
        current = None
        if cur is not None:
            current = {'id': cur['_id'], 'exp_id': cur.get('exp_id'), 'status': cur.get('status'),
                       'progress': cur.get('progress'), 'stage': cur.get('stage')}
        return {'queued': queued, 'current': current, 'runner': self.runner.alive(),
                'legacy_queue': bool(self.cfg.legacy_queue)}

    def start(self) -> None:
        self.runner.start()

    def stop(self) -> None:
        self.runner.stop()

    # --- для исполнителя ----------------------------------------------------------------------------------

    def set_fields(self, job_id: str, fields: Dict[str, Any]) -> None:
        self.coll.update_one({'_id': job_id}, {'$set': fields})

    def push_warning(self, job_id: str, warning: str) -> None:
        self.coll.update_one({'_id': job_id}, {'$push': {'warnings': warning}})

    def cancel_requested(self, job_id: str) -> bool:
        try:
            doc = self.coll.find_one({'_id': job_id}, {'cancel_requested': 1})
        except Exception:  # noqa: BLE001 — Mongo недоступна: задача продолжается
            logger.exception('задача %s: не удалось проверить отмену', job_id)
            return False
        return bool(doc and doc.get('cancel_requested'))

    def tomo_status(self, exp_id: str, status: str, job_id: str) -> None:
        """Статус для старой страницы очереди — документом как у ``tomo_queue.set_object_status``."""
        try:
            self.tomo.insert_one({'obj_id': exp_id, 'status': status, 'date': datetime.datetime.now(),
                                  'job_id': job_id})
        except Exception:  # noqa: BLE001 — старая страница не должна ломать задачу
            logger.exception('%s: статус в tomoobjects не записан', exp_id)


# --- эндпоинты -----------------------------------------------------------------------------------------------

def _svc() -> JobService:
    return current_app.extensions['recon'].jobs


@bp.errorhandler(JobConflict)
def _conflict(e: JobConflict):
    return jsonify({'error': str(e), 'job_id': e.job_id}), 409


@bp.post('')
def create_job():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ValueError('ожидается JSON {recipe, name?}')
    job = _svc().create(body.get('recipe'), auth.current_user(), body.get('name'))
    return jsonify(job), 201


@bp.get('')
def list_jobs():
    exp_id = request.args.get('exp_id') or None
    if exp_id is not None:
        exp_id = auth.valid_exp_id(exp_id)
    limit = request.args.get('limit', 50)
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        raise ValueError('limit должен быть целым')
    return jsonify(_svc().list(exp_id, limit))


@bp.get('/<job_id>')
def get_job(job_id):
    return jsonify(_svc().get(job_id))


@bp.get('/<job_id>/log')
def job_log(job_id):
    lines = request.args.get('lines', 200)
    try:
        lines = min(max(int(lines), 1), 5000)
    except (TypeError, ValueError):
        raise ValueError('lines должен быть целым')
    return Response(_svc().log_tail(job_id, lines), mimetype='text/plain; charset=utf-8')


@bp.post('/<job_id>/cancel')
def cancel_job(job_id):
    return jsonify(_svc().cancel(job_id, auth.current_user()))
