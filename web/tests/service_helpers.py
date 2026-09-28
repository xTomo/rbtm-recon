"""Сборка тестового recon-service: каталоги во временной папке, Mongo — mongomock, без фоновых потоков.

    app, client, cfg = make_service(tmp_path)
    write_scan(cfg, 'exp1', simple_scan())          # <exp_src>/exp1.h5
    r = client.get('/scans/exp1/info', headers=HEADERS)
"""
import os

import mongomock

from engine_scans import write_h5
from reconservice import binary
from reconservice.app import create_app
from reconservice.config import Config

TOKEN = 'test-token'
USER = 'alice'
HEADERS = {'X-Recon-Token': TOKEN, 'X-Recon-User': USER}


def headers(user=USER):
    return {'X-Recon-Token': TOKEN, 'X-Recon-User': user}


def make_config(tmp_path, **overrides) -> Config:
    base = dict(token=TOKEN, exp_src=str(tmp_path / 'exp_src'), fast_dir=str(tmp_path / 'fast'),
                storage_dir=str(tmp_path / 'storage'), mongodb_uri='mongodb://unused', job_gpu=None,
                gpu_lock=None, job_poll_s=0.05, legacy_queue=False, session_ttl_s=1800.0, workers=2,
                storage_server='http://127.0.0.1:9/')     # порт 9 (discard): storage «недоступен» быстро
    base.update(overrides)
    cfg = Config(**base)
    for d in (cfg.exp_src, cfg.fast_dir, cfg.storage_dir):
        os.makedirs(d, exist_ok=True)
    return cfg


def make_service(tmp_path, start_threads=False, mongo_client=None, **overrides):
    cfg = make_config(tmp_path, **overrides)
    mongo = mongo_client or mongomock.MongoClient()
    app = create_app(cfg, mongo_client=mongo, start_threads=start_threads)
    app.testing = True
    return app, app.test_client(), cfg


def write_scan(cfg: Config, exp_id: str, ss, **kw) -> str:
    """Записать скан туда, где его ищет сервис: <exp_src>/<id>/before_processing/<id>.h5."""
    path = cfg.scan_path(exp_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return write_h5(ss, path, exp_id=exp_id, **kw)


def decode(response):
    """binary-ответ → (массив в физических единицах, meta)."""
    return binary.decode(response.get_data(), dict(response.headers))
