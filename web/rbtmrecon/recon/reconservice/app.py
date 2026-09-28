"""Сборка Flask-приложения сервиса.

    gunicorn -w 1 -k gthread --threads 8 -b 0.0.0.0:5560 --timeout 600 'reconservice.app:create_app()'

Фоновые потоки (очередь задач, закрытие простаивающей сессии) запускаются в create_app — при ``-w 1`` без
``--preload`` это происходит в единственном рабочем процессе gunicorn.
"""
from __future__ import annotations

import logging
from typing import Optional

from flask import Flask, current_app, jsonify
from werkzeug.exceptions import HTTPException

from reconengine import __version__ as ENGINE_VERSION
from reconengine import gpu
from reconengine.model import Cancelled

from . import auth
from .arbiter import Superseded
from .config import Config

logger = logging.getLogger(__name__)


class ServiceState:
    """Всё состояние процесса: реестр сканов, интерактивная сессия, очередь задач."""

    def __init__(self, cfg: Config, mongo_client=None):
        from . import jobs, scans, sessions  # noqa: WPS433 — модули тянут движок
        self.cfg = cfg
        self._mongo = mongo_client
        self.scans = scans.ScanRegistry(cfg)
        self.sessions = sessions.SessionManager(cfg, self.scans)
        self.jobs = jobs.JobService(cfg, self.db, self.scans)

    @property
    def mongo(self):
        if self._mongo is None:
            from pymongo import MongoClient  # noqa: WPS433
            # connect=False: процесс стартует и без Mongo (после перезагрузки сервера Mongo может подняться позже)
            self._mongo = MongoClient(self.cfg.mongodb_uri, serverSelectionTimeoutMS=5000, connect=False)
        return self._mongo

    @property
    def db(self):
        return self.mongo[self.cfg.mongo_db]

    def start(self) -> None:
        self.jobs.start()
        self.sessions.start_reaper()

    def stop(self) -> None:
        self.jobs.stop()
        self.sessions.stop_reaper()


def state() -> ServiceState:
    return current_app.extensions['recon']


def create_app(config: Optional[Config] = None, *, mongo_client=None, start_threads: bool = True) -> Flask:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    cfg = config or Config.from_env()
    if not cfg.token:
        logger.warning('RECON_TOKEN не задан: сервис отвечает 503 на всё, кроме /health')
    app = Flask(__name__)
    app.config['RECON'] = cfg
    app.config['JSON_AS_ASCII'] = False
    app.json.ensure_ascii = False
    st = ServiceState(cfg, mongo_client)
    app.extensions['recon'] = st

    app.before_request(auth.require_token)

    from . import jobs, results, scans, sessions  # noqa: WPS433
    app.register_blueprint(scans.bp)
    app.register_blueprint(sessions.bp)
    app.register_blueprint(jobs.bp)
    app.register_blueprint(results.bp)

    @app.get('/health')
    def health():
        s = state()
        return jsonify({
            'ok': True,
            'engine': ENGINE_VERSION,
            'token_configured': bool(s.cfg.token),
            'gpu': {'name': gpu.device_name(), 'mem': gpu.mem_info()},
            'gpu_lock_busy': gpu.lock_busy(s.cfg.gpu_lock),
            'session': s.sessions.summary(),
            'jobs': s.jobs.summary(),
        })

    _register_errors(app)
    if start_threads:
        st.start()
    return app


def _register_errors(app: Flask) -> None:
    @app.errorhandler(HTTPException)
    def http_error(e: HTTPException):
        if e.response is not None:          # abort(make_response(...)) — ответ уже собран
            return e.response
        return jsonify({'error': e.description}), e.code

    @app.errorhandler(Superseded)
    def superseded(e):
        return jsonify({'error': 'superseded', 'detail': str(e)}), 409

    @app.errorhandler(Cancelled)
    def cancelled(e):
        return jsonify({'error': 'cancelled'}), 409

    @app.errorhandler(FileNotFoundError)
    def not_found(e):
        return jsonify({'error': str(e) or 'не найдено'}), 404

    @app.errorhandler(ValueError)
    def bad_request(e):
        return jsonify({'error': str(e)}), 400

    @app.errorhandler(Exception)
    def internal(e):
        logger.exception('необработанная ошибка')
        return jsonify({'error': '{}: {}'.format(type(e).__name__, e)}), 500
