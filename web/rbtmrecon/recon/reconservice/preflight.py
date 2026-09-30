"""Проверка окружения recon-service после выкатки: всё ли на месте, чтобы студия работала.

Запуск в контейнере сервиса (рабочий каталог ``/rbtm/recon``, окружение контейнера)::

    docker exec -w /rbtm/recon web_reconstructor_1 /opt/conda/envs/xrecon/bin/python -m reconservice.preflight [--exp <id>]

Проверяет по порядку и печатает ``OK`` / ``WARN`` / ``FAIL`` по пунктам:
- версии библиотек (numpy, scipy, scikit-image, h5py, cupy, astra, flask, gunicorn, pymongo);
- настройки: токен задан, каталоги /exp_src, /fast, /storage есть, в /fast/studio и /storage можно писать
  (временный файл создаётся и удаляется);
- GPU сессии (CUDA_VISIBLE_DEVICES процесса): cupy, ядра cupyx (медиана — кольца), БПФ, astra FBP_CUDA;
- GPU задач (RECON_JOB_GPU) — то же в отдельном процессе, как ``python -m reconengine run``;
- Mongo (``autotom.jobs``) и storage (``storage/experiments/get`` — только чтение);
- ``--exp <id>``: исходный HDF5 скана, формат v2, углы, размер пикселя с источником;
- работающий сервис на :5560 — ``/health`` (если запущен).

Ничего не меняет, кроме временных файлов проверки записи. Код выхода 1, если есть FAIL."""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import time
import uuid
from typing import List, Optional, Tuple

from .config import Config

_results: List[Tuple[str, str, str]] = []


def _report(status: str, what: str, detail: str = '') -> None:
    detail = ' '.join(str(detail).split())
    if len(detail) > 240:
        detail = detail[:237] + '…'
    _results.append((status, what, detail))
    print('{:4s}  {}{}'.format(status, what, (' — ' + detail) if detail else ''), flush=True)


def _version(mod: str) -> Optional[str]:
    try:
        from importlib.metadata import version  # noqa: WPS433
        return version({'skimage': 'scikit-image', 'cupy': 'cupy'}.get(mod, mod))
    except Exception:  # noqa: BLE001
        pass
    try:
        return str(getattr(importlib.import_module(mod), '__version__', '?'))
    except Exception:  # noqa: BLE001
        return None


def check_versions() -> None:
    for mod in ('numpy', 'scipy', 'skimage', 'h5py', 'flask', 'gunicorn', 'pymongo', 'requests'):
        v = _version(mod)
        _report('OK' if v else 'FAIL', 'библиотека {}'.format(mod), v or 'не установлена')
    for mod in ('cupy', 'astra'):
        try:
            importlib.import_module(mod)
            _report('OK', 'библиотека {}'.format(mod), _version(mod) or '?')
        except Exception as exc:  # noqa: BLE001
            _report('FAIL', 'библиотека {}'.format(mod), '{}: {}'.format(type(exc).__name__, exc))


def _writable(path: str) -> Tuple[bool, str]:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, '.preflight-{}'.format(uuid.uuid4().hex[:8]))
        with open(probe, 'wb') as fh:
            fh.write(b'ok')
        os.remove(probe)
        return True, path
    except OSError as exc:
        return False, '{}: {}'.format(path, exc)


def check_config(cfg: Config) -> None:
    _report('OK' if cfg.token else 'FAIL', 'RECON_TOKEN', 'задан ({} символов)'.format(len(cfg.token)) if cfg.token
            else 'пуст — сервис отвечает 503 на всё, кроме /health')
    for name, path in (('исходники (RECON_EXP_SRC)', cfg.exp_src), ('SSD (RECON_FAST)', cfg.fast_dir),
                       ('хранилище (RECON_STORAGE)', cfg.storage_dir)):
        _report('OK' if os.path.isdir(path) else 'FAIL', name, path)
    for name, path in (('запись в /fast/studio', os.path.join(cfg.fast_dir, 'studio')),
                       ('запись в хранилище', cfg.storage_dir)):
        ok, detail = _writable(path)
        _report('OK' if ok else 'FAIL', name, detail)
    lock_dir = os.path.dirname(cfg.gpu_lock) if cfg.gpu_lock else None
    if lock_dir:
        ok, detail = _writable(lock_dir)
        _report('OK' if ok else 'FAIL', 'блокировка GPU задач (RECON_GPU_LOCK)', cfg.gpu_lock if ok else detail)
    cache = os.environ.get('CUPY_CACHE_DIR')
    _report('OK' if cache else 'WARN', 'кэш ядер cupy (CUPY_CACHE_DIR)',
            cache or 'не задан — ядра компилируются заново после каждой пересборки (первые запросы медленные)')
    _report('OK', 'GPU сессии / задач (CUDA_VISIBLE_DEVICES / RECON_JOB_GPU)',
            '{} / {}'.format(os.environ.get('CUDA_VISIBLE_DEVICES', 'не задан'), cfg.job_gpu))


#: проверка одного GPU: cupy, cupyx-медиана (кольца), БПФ (сглаживание, быстрый сдвиг центра), astra FBP_CUDA
_GPU_PROBE = r'''
import time, numpy as np
t = time.time()
import cupy as cp
from cupyx.scipy import ndimage as cnd
dev = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
name = dev['name'].decode() if isinstance(dev['name'], bytes) else dev['name']
free, total = cp.cuda.runtime.memGetInfo()
a = cp.random.random((64, 128, 256), dtype=cp.float32)
m = cnd.median_filter(a, (1, 1, 21))
f = cp.fft.irfft2(cp.fft.rfft2(a, axes=(1, 2)), s=a.shape[1:], axes=(1, 2))
assert float(cp.abs(f - a).max()) < 1e-3
import astra
assert astra.use_cuda(), 'astra без CUDA'
from reconengine import fbp
sino = np.random.random((1, 90, 128)).astype('float32')
rec = fbp.recon_rows(sino, np.arange(0, 180, 2.0), 1.0, backend='astra')
part = fbp.recon_rows(sino, np.arange(0, 180, 2.0), 1.0, backend='astra', region=(10, 20, 60, 70))
assert rec.shape == (1, 128, 128) and np.abs(part[0] - rec[0, 20:70, 10:60]).max() < 1e-3 * np.ptp(rec)
print('{} · свободно {:.1f} из {:.1f} ГБ · cupy+cupyx+БПФ+astra FBP за {:.1f} с'.format(
    name, free / 2**30, total / 2**30, time.time() - t))
'''


def _gpu_probe(env_gpu: Optional[str]) -> Tuple[bool, str]:
    env = dict(os.environ)
    if env_gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(env_gpu)
    try:
        p = subprocess.run([sys.executable, '-c', _GPU_PROBE], env=env, capture_output=True, text=True,
                           timeout=600, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    except subprocess.TimeoutExpired:
        return False, 'нет ответа за 600 с'
    out = (p.stdout or '').strip().splitlines()
    if p.returncode == 0 and out:
        return True, out[-1]
    err = (p.stderr or '').strip().splitlines()
    return False, err[-1] if err else 'код {}'.format(p.returncode)


def check_gpus(cfg: Config) -> None:
    ok, detail = _gpu_probe(None)
    _report('OK' if ok else 'FAIL', 'GPU сессии (CUDA_VISIBLE_DEVICES={})'.format(
        os.environ.get('CUDA_VISIBLE_DEVICES', '—')), detail)
    if cfg.job_gpu is not None:
        ok, detail = _gpu_probe(cfg.job_gpu)
        _report('OK' if ok else 'FAIL', 'GPU задач (RECON_JOB_GPU={})'.format(cfg.job_gpu), detail)


def check_mongo(cfg: Config) -> None:
    try:
        import pymongo  # noqa: WPS433
        client = pymongo.MongoClient(cfg.mongodb_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        n = client[cfg.mongo_db]['jobs'].estimated_document_count()
        _report('OK', 'Mongo', '{} · {}.jobs: {} документов'.format(cfg.mongodb_uri, cfg.mongo_db, n))
    except Exception as exc:  # noqa: BLE001
        _report('FAIL', 'Mongo', '{}: {}: {}'.format(cfg.mongodb_uri, type(exc).__name__, exc))


def check_storage(cfg: Config) -> None:
    try:
        import requests  # noqa: WPS433
        r = requests.post(cfg.storage_server + 'storage/experiments/get', json={'_id': '__preflight__'}, timeout=5)
        _report('OK' if r.status_code == 200 else 'WARN', 'storage (документы экспериментов)',
                '{} → {}'.format(cfg.storage_server, r.status_code))
    except Exception as exc:  # noqa: BLE001
        _report('WARN', 'storage (документы экспериментов)',
                '{}: {} — размер пикселя будет браться без документа'.format(cfg.storage_server, type(exc).__name__))


def check_scan(cfg: Config, exp_id: str) -> None:
    path = cfg.scan_path(exp_id)
    if not os.path.isfile(path):
        _report('FAIL', 'скан {}'.format(exp_id), 'нет файла {}'.format(path))
        return
    try:
        from reconengine import data  # noqa: WPS433
        t = time.time()
        scan = data.open_scan(path, exp_id)
        a = scan.angles[scan.data_idx]
        _report('OK', 'скан {}'.format(exp_id), '{}×{} · {} кадров · углы {:.1f}…{:.1f}° · {} · {:.1f} с'.format(
            scan.width, scan.height, scan.n_frames, float(a.min()), float(a.max()),
            'advanced' if scan.is_advanced else 'обычный', time.time() - t))
    except Exception as exc:  # noqa: BLE001
        _report('FAIL', 'скан {}'.format(exp_id), '{}: {}'.format(type(exc).__name__, exc))
        return
    try:
        from .scans import ScanRegistry  # noqa: WPS433
        reg = ScanRegistry(cfg)
        ps = reg.pixel_size(exp_id)
        _report('OK' if ps.source != 'default' else 'WARN', 'размер пикселя {}'.format(exp_id),
                '{:.5g} мм, источник {}{}'.format(ps.value_mm, ps.source,
                                                  ('; ' + '; '.join(ps.warnings)) if ps.warnings else ''))
    except Exception as exc:  # noqa: BLE001
        _report('WARN', 'размер пикселя {}'.format(exp_id), '{}: {}'.format(type(exc).__name__, exc))


def check_service(cfg: Config) -> None:
    try:
        import requests  # noqa: WPS433
        h = requests.get('http://localhost:5560/health', timeout=5).json()
        gpu = (h.get('gpu') or {}).get('name')
        _report('OK' if h.get('ok') and h.get('token_configured') else 'FAIL', 'сервис :5560 /health',
                'ok={} token={} gpu={} очередь={}'.format(h.get('ok'), h.get('token_configured'), gpu,
                                                          (h.get('jobs') or {}).get('queued')))
    except Exception as exc:  # noqa: BLE001
        _report('WARN', 'сервис :5560 /health', 'не отвечает ({}) — для проверки до запуска это нормально'.format(
            type(exc).__name__))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog='python -m reconservice.preflight', description=__doc__.split('\n')[0])
    ap.add_argument('--exp', help='проверить и этот скан (id эксперимента)')
    ap.add_argument('--no-gpu', action='store_true', help='пропустить проверки GPU')
    args = ap.parse_args(argv)
    import logging  # noqa: WPS433
    logging.basicConfig(level=logging.ERROR)      # предупреждения модулей сервиса — в отчёт, не в поток
    _results.clear()
    cfg = Config.from_env()
    print('recon-service preflight · {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')), flush=True)
    check_versions()
    check_config(cfg)
    if not args.no_gpu:
        check_gpus(cfg)
    check_mongo(cfg)
    check_storage(cfg)
    if args.exp:
        check_scan(cfg, args.exp)
    check_service(cfg)
    fails = sum(1 for s, _, _ in _results if s == 'FAIL')
    warns = sum(1 for s, _, _ in _results if s == 'WARN')
    print('итог: {} FAIL, {} WARN, {} OK'.format(fails, warns, len(_results) - fails - warns), flush=True)
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
