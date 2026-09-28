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
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from flask import Blueprint

from .config import Config

bp = Blueprint('jobs', __name__, url_prefix='/jobs')

ACTIVE = ('queued', 'running', 'publishing')


class JobService:
    def __init__(self, cfg: Config, db, scans):
        raise NotImplementedError

    def create(self, recipe: Dict[str, Any], user: str, name: Optional[str] = None) -> Dict[str, Any]:
        raise NotImplementedError

    def get(self, job_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    def list(self, exp_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def cancel(self, job_id: str, user: str) -> Dict[str, Any]:
        raise NotImplementedError

    def summary(self) -> Dict[str, Any]:
        """Для /health: число queued, текущая задача (id, exp_id, progress, stage)."""
        raise NotImplementedError

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError
