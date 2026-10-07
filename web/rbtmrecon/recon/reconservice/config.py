"""Настройки сервиса из окружения."""
from __future__ import annotations

import dataclasses
import os
import sys
from typing import Mapping, Optional


def _default_mongo_uri() -> str:
    try:
        from conf import MONGODB_URI  # noqa: WPS433 — conf.py пишется в образ при сборке
        return MONGODB_URI
    except Exception:  # noqa: BLE001
        return 'mongodb://web_database_1:27017'


@dataclasses.dataclass
class Config:
    #: общий секрет с rbtm-web; пустой — сервис отвечает 503 на всё, кроме /health
    token: str = ''
    exp_src: str = '/exp_src'                      # исходные HDF5 (<id>.h5), только чтение
    fast_dir: str = '/fast'                        # SSD: кэш кропов, обзоры, запуски
    storage_dir: str = '/storage'                  # результаты (<id>/reconstruction/), архив <id>.h5
    mongodb_uri: str = dataclasses.field(default_factory=_default_mongo_uri)
    mongo_db: str = 'autotom'
    storage_server: str = 'http://rbtmstorage_server_1:5006/'   # документ эксперимента (размер пикселя)
    job_gpu: Optional[str] = '0'                   # CUDA_VISIBLE_DEVICES для процесса задачи
    gpu_lock: Optional[str] = '/fast/.gpu0.lock'   # flock, общий с Jupyter и старой очередью
    job_python: str = sys.executable               # интерпретатор для python -m reconengine
    job_poll_s: float = 2.0                        # период опроса очереди
    legacy_queue: bool = True                      # обслуживать старую очередь ноутбуков (до этапа 7)
    session_ttl_s: float = 1800.0                  # сессия закрывается после стольких секунд простоя
    fast_limit_gb: float = 300.0                   # предел кэша кропов в /fast (LRU по времени доступа)
    workers: int = 8                               # потоков распаковки HDF5
    preview_max_px: int = 1400                     # сторона превью, до которой ужимается картинка
    prefetch: bool = True                          # предзагрузка исходного HDF5 в кэш ОС после обзора (prefetch.py)

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> 'Config':
        env = os.environ if env is None else env
        d = cls()

        def get(name, default, conv=str):
            v = env.get(name)
            if v is None or v == '':
                return default
            return conv(v)

        def flag(v: str) -> bool:
            return v.strip().lower() not in ('0', 'false', 'no', 'off')

        return cls(
            token=get('RECON_TOKEN', d.token),
            exp_src=get('RECON_EXP_SRC', d.exp_src),
            fast_dir=get('RECON_FAST', d.fast_dir),
            storage_dir=get('RECON_STORAGE', d.storage_dir),
            mongodb_uri=get('MONGODB_URI', d.mongodb_uri),
            mongo_db=get('RECON_MONGO_DB', d.mongo_db),
            storage_server=get('RECON_STORAGE_SERVER', d.storage_server),
            job_gpu=get('RECON_JOB_GPU', d.job_gpu),
            gpu_lock=get('RECON_GPU_LOCK', d.gpu_lock),
            job_python=get('RECON_JOB_PYTHON', d.job_python),
            job_poll_s=get('RECON_JOB_POLL_S', d.job_poll_s, float),
            legacy_queue=get('RECON_LEGACY_QUEUE', d.legacy_queue, flag),
            session_ttl_s=get('RECON_SESSION_TTL_S', d.session_ttl_s, float),
            fast_limit_gb=get('RECON_FAST_LIMIT_GB', d.fast_limit_gb, float),
            workers=get('RECON_WORKERS', d.workers, int),
            preview_max_px=get('RECON_PREVIEW_MAX_PX', d.preview_max_px, int),
            prefetch=get('RECON_PREFETCH', d.prefetch, flag),
        )

    # --- пути ---------------------------------------------------------------------------------------------

    def scan_path(self, exp_id: str) -> str:
        """Исходный HDF5 эксперимента (не проверяет существование): раскладка rbtm-storage
        ``data/experiments/<id>/before_processing/<id>.h5``, смонтированная как ``/exp_src``."""
        return os.path.join(self.exp_src, exp_id, 'before_processing', exp_id + '.h5')

    def fast_exp_dir(self, exp_id: str) -> str:
        """``/fast/studio/<id>`` — не внутри ``/fast/<id>/``: тот каталог принадлежит ноутбуку, который в конце
        копирует его целиком в ``/storage/<id>/reconstruction`` (``cp -rT``) и удаляет (``rm -rf``)."""
        return os.path.join(self.fast_dir, 'studio', exp_id)

    def cache_dir(self, exp_id: str) -> str:
        return os.path.join(self.fast_exp_dir(exp_id), 'cache')

    def view3d_dir(self, exp_id: str) -> str:
        """Кэш объёма 3D-вида результата (``results.volume3d``) — вне ``cache/``: предел кэша кропов его не считает."""
        return os.path.join(self.fast_exp_dir(exp_id), 'view3d')

    def runs_dir(self, exp_id: str) -> str:
        return os.path.join(self.fast_exp_dir(exp_id), 'runs')

    def storage_exp_dir(self, exp_id: str) -> str:
        return os.path.join(self.storage_dir, exp_id)

    def reconstruction_dir(self, exp_id: str) -> str:
        """Каталог результата — тот же, куда копирует ноутбук (``/storage/<id>/reconstruction``)."""
        return os.path.join(self.storage_exp_dir(exp_id), 'reconstruction')
