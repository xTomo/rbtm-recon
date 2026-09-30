"""Сведения о скане и обзор для шага «Поле зрения» — без загрузки всех данных (``/scans/<exp_id>/...``).

Всё читается из исходного ``cfg.scan_path(exp_id)`` движком (``reconengine.data``): структура — мгновенно,
обзор (dark/empty + N углов с биннингом) — 2–3 с на кадре 5056×2968, дальше из кэша.

Эндпоинты (все требуют токен; exp_id проверяется ``auth.valid_exp_id``):

| Метод и путь                          | Ответ |
|---------------------------------------|-------|
| GET /scans/<id>/info                  | JSON: форма, число кадров по режимам, advanced, диапазон углов, размер пикселя с источником и предупреждениями, fingerprint |
| GET /scans/<id>/overview              | JSON: предложенный ROI (координаты детектора), углы, где объект выходит за ROI, биннинг, форма обзора, углы выборки, размер пикселя |
| GET /scans/<id>/outside?x0&x1&y0&y1  | JSON: углы выборки, на которых объект выходит за столбцы рамки (для рамки, подвинутой пользователем) |
| GET /scans/<id>/envelope              | binary uint16 (h/b, w/b): огибающая max(−ln T) по углам выборки |
| GET /scans/<id>/thumbs                | binary uint16 (k, h/b, w/b): −ln T кадров выборки; X-Meta: angles, indices |
| GET /scans/<id>/sample/<k>            | binary uint16 (h/b, w/b): k-й кадр выборки (−ln T) |
| POST /scans/<id>/prefetch             | 202, JSON: состояние предзагрузки исходного HDF5 в кэш ОС (``reconservice.prefetch``; rbtm-web вызывает после обзора, если пользователь может загружать) |
| GET /scans/<id>/sinogram?row=&n=      | binary uint16 (n, W): строка детектора row по n углам (по умолчанию 90), кадры с наименьшей стоимостью распаковки в каждом угловом интервале; X-Meta: angles |

Параметры обзора по умолчанию: n=16 углов (``?n=`` до 64), bin=4. Кэш: в памяти (последние несколько сканов) и
``<cfg.fast_exp_dir(id)>/overview-<fingerprint[:16]>-n<n>-b<bin>.npz`` — второй запрос и рестарт сервиса не читают HDF5.
Размер пикселя: ``pixelsize.resolve(документ storage, metadata HDF5)``; документ берётся запросом
``POST <cfg.storage_server>storage/experiments/get`` с ``{"_id": exp_id}`` (Content-Type: application/json,
таймаут 5 с); недоступность storage — не ошибка: без документа и с предупреждением в ответе.

Окно квантования: огибающая и синограмма — персентили самого массива; thumbs и sample — общее окно −ln T всех
кадров выборки (``OverviewData.window``), чтобы кадры выборки не «мигали» при листании.
Синограмма: −ln T с dark/empty этой же строки без биннинга (медианы по нескольким самым дешёвым dark- и
начальным empty-кадрам, как в ``data.sample_overview``) — всё одним проходом ``ChunkSampler.read_frames``.

Потокобезопасность (gunicorn gthread, 8 потоков): общий замок защищает только словари кэшей; долгие вычисления
идут под замком своего ключа (скан, обзор, строка синограммы, документ storage), поэтому два одновременных запроса
одного обзора считают его один раз, а разные сканы — параллельно.
"""
from __future__ import annotations

import collections
import dataclasses
import logging
import math
import os
import threading
import time
from typing import Any, Dict, Hashable, List, Mapping, Optional, Tuple

import numpy as np
import requests
from flask import Blueprint, current_app, jsonify, request

from reconengine import data, pipeline, pixelsize, preprocess
from reconengine.model import Overview, ROI, ScanInfo
from reconengine.pixelsize import PixelSize

from . import auth, binary
from .config import Config

logger = logging.getLogger(__name__)

bp = Blueprint('scans', __name__, url_prefix='/scans')

OVERVIEW_N, OVERVIEW_MAX_N = 16, 64
OVERVIEW_BIN, OVERVIEW_MAX_BIN = 4, 16
SINO_N, SINO_MAX_N = 90, 360
#: dark- и empty-кадров на опорную строку синограммы (как n_dark/n_empty в sample_overview)
SINO_REF_FRAMES = 3
#: T обрезается снизу до eps (как в preprocess.envelope)
_EPS = 1e-3
_MEM_INFOS, _MEM_OVERVIEWS, _MEM_SINOS = 64, 4, 8
#: документ storage: удачный ответ живёт 5 мин, неудача — 30 с (не ждать таймаут на каждом запросе)
_DOC_TTL_S, _DOC_FAIL_TTL_S = 300.0, 30.0
_STORAGE_TIMEOUT_S = 5
_NPZ_FORMAT = 'rbtm-recon-overview/1'


@dataclasses.dataclass
class OverviewData:
    """Обзор скана с производными: огибающая, предложенный ROI, углы, где объект выходит за ROI."""
    overview: Overview
    envelope: np.ndarray            # float32 (h/b, w/b)
    roi: ROI                        # координаты полного кадра
    angles_outside: List[float]
    window: Tuple[float, float] = (0.0, 1.0)   # общее окно −ln T кадров выборки (персентили 0,1 и 99,9)


# --- параметры запросов (общие с sessions) -----------------------------------------------------------------

def _number(src: Optional[Mapping[str, Any]], name: str, default, conv, lo=None, hi=None):
    v = src.get(name) if src is not None else None
    if v is None or v == '':
        return default
    if isinstance(v, bool):
        raise ValueError('{}: ожидается число, получено {!r}'.format(name, v))
    try:
        v = conv(v)
    except (TypeError, ValueError):
        raise ValueError('{}: ожидается {}, получено {!r}'.format(
            name, 'целое' if conv is int else 'число', v)) from None
    if conv is float and not math.isfinite(v):
        raise ValueError('{}: ожидается конечное число, получено {!r}'.format(name, v))
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise ValueError('{} = {} вне [{}, {}]'.format(name, v, lo, hi))
    return v


def arg_int(src: Optional[Mapping[str, Any]], name: str, default: Optional[int] = None,
            lo: Optional[int] = None, hi: Optional[int] = None) -> Optional[int]:
    """Целый параметр из request.args или JSON-тела; пустой — default; нечисло и выход за [lo, hi] — ValueError."""
    return _number(src, name, default, int, lo, hi)


def arg_float(src: Optional[Mapping[str, Any]], name: str, default: Optional[float] = None,
              lo: Optional[float] = None, hi: Optional[float] = None) -> Optional[float]:
    """Вещественный параметр (конечный), как arg_int."""
    return _number(src, name, default, float, lo, hi)


# --- реестр ------------------------------------------------------------------------------------------------

def _minus_log_t(frames: np.ndarray, dark: np.ndarray, empty: np.ndarray) -> np.ndarray:
    """−ln T кадров обзора, как в огибающей (preprocess.envelope)."""
    return preprocess._minus_log_t(frames, dark, empty, _EPS)


def _sample_window(ov: Overview) -> Tuple[float, float]:
    """Общее окно −ln T кадров выборки: персентили 0,1 и 99,9 (по каждому второму пикселю — хватает с запасом)."""
    if ov.samples.shape[0] == 0:
        return 0.0, 1.0
    t = _minus_log_t(ov.samples[:, ::2, ::2], ov.dark[::2, ::2], ov.empty[::2, ::2])
    lo, hi = np.percentile(t, [0.1, 99.9])
    lo, hi = float(lo), float(hi)
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def _put(cache: 'collections.OrderedDict', key, value, limit: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)


class ScanRegistry:
    """Потокобезопасный кэш сведений о сканах. ScanInfo — по (путь, размер, mtime); обзор — см. модуль."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._lock = threading.Lock()                     # только словари ниже
        self._key_locks: Dict[Hashable, threading.Lock] = {}
        self._infos: 'collections.OrderedDict[str, Tuple[tuple, ScanInfo]]' = collections.OrderedDict()
        self._overviews: 'collections.OrderedDict[tuple, OverviewData]' = collections.OrderedDict()
        self._sinos: 'collections.OrderedDict[tuple, Tuple[np.ndarray, np.ndarray]]' = collections.OrderedDict()
        self._docs: Dict[str, Tuple[float, Optional[Dict[str, Any]], Optional[str]]] = {}

    def _key_lock(self, key: Hashable) -> threading.Lock:
        """Замок вычисления по ключу: одновременные запросы одного и того же ждут первого, а не считают заново."""
        with self._lock:
            lk = self._key_locks.get(key)
            if lk is None:
                lk = self._key_locks[key] = threading.Lock()
            return lk

    # --- скан ----------------------------------------------------------------------------------------------

    def path(self, exp_id: str) -> str:
        """Путь к исходному HDF5; FileNotFoundError, если файла нет."""
        p = self.cfg.scan_path(exp_id)
        if not os.path.isfile(p):
            raise FileNotFoundError('скан {} не найден'.format(exp_id))
        return p

    def info(self, exp_id: str) -> ScanInfo:
        p = self.path(exp_id)
        st = os.stat(p)
        key = (p, int(st.st_size), int(st.st_mtime_ns))
        with self._lock:
            hit = self._infos.get(exp_id)
            if hit is not None and hit[0] == key:
                self._infos.move_to_end(exp_id)
                return hit[1]
        with self._key_lock(('info', exp_id)):
            with self._lock:
                hit = self._infos.get(exp_id)
                if hit is not None and hit[0] == key:
                    return hit[1]
            scan = data.open_scan(p, exp_id)
            with self._lock:
                _put(self._infos, exp_id, (key, scan), _MEM_INFOS)
        return scan

    # --- обзор ---------------------------------------------------------------------------------------------

    def overview_path(self, exp_id: str, scan: ScanInfo, n: int, bin: int) -> str:
        return os.path.join(self.cfg.fast_exp_dir(exp_id),
                            'overview-{}-n{}-b{}.npz'.format(scan.fingerprint[:16], int(n), int(bin)))

    def overview(self, exp_id: str, n: int = OVERVIEW_N, bin: int = OVERVIEW_BIN) -> OverviewData:
        n = arg_int({'n': n}, 'n', OVERVIEW_N, 1, OVERVIEW_MAX_N)
        bin = arg_int({'bin': bin}, 'bin', OVERVIEW_BIN, 1, OVERVIEW_MAX_BIN)
        scan = self.info(exp_id)
        key = (scan.fingerprint, n, bin)
        with self._lock:
            hit = self._overviews.get(key)
            if hit is not None:
                self._overviews.move_to_end(key)
                return hit
        with self._key_lock(('overview',) + key):
            with self._lock:
                hit = self._overviews.get(key)
                if hit is not None:
                    return hit
            path = self.overview_path(exp_id, scan, n, bin)
            od = self._load_npz(path, scan)
            if od is None:
                t0 = time.time()
                od = self._compute_overview(scan, n, bin)
                logger.info('обзор %s (n=%d, bin=%d): %.2f с', exp_id, n, bin, time.time() - t0)
                self._save_npz(path, scan, od)
            with self._lock:
                _put(self._overviews, key, od, _MEM_OVERVIEWS)
        return od

    def _compute_overview(self, scan: ScanInfo, n: int, bin: int) -> OverviewData:
        ov = data.sample_overview(scan, n=n, bin=bin, workers=self.cfg.workers)
        env = preprocess.envelope(ov)
        roi = preprocess.suggest_roi(env, bin, scan.height, scan.width)
        outside = preprocess.angles_outside(ov, roi)
        return OverviewData(overview=ov, envelope=env, roi=roi, angles_outside=outside, window=_sample_window(ov))

    def _save_npz(self, path: str, scan: ScanInfo, od: OverviewData) -> None:
        """Атомарно (временный файл + os.replace); ошибка записи — только предупреждение (обзор уже в памяти)."""
        ov = od.overview
        tmp = '{}.tmp-{}-{}'.format(path, os.getpid(), threading.get_ident())
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            r = od.roi
            with open(tmp, 'wb') as fh:
                np.savez(fh, format=np.array(_NPZ_FORMAT), fingerprint=np.array(scan.fingerprint),
                         bin=np.int64(ov.bin), dark=ov.dark, empty=ov.empty, samples=ov.samples,
                         sample_idx=np.asarray(ov.sample_idx, dtype=np.int64),
                         sample_angles=np.asarray(ov.sample_angles, dtype=np.float64),
                         full_shape=np.array([ov.full_height, ov.full_width], dtype=np.int64),
                         envelope=od.envelope,
                         roi=np.array([r.x0, r.x1, r.y0, r.y1, r.preview_row], dtype=np.int64),
                         angles_outside=np.asarray(od.angles_outside, dtype=np.float64),
                         window=np.asarray(od.window, dtype=np.float64))
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning('обзор не сохранён в %s: %s', path, exc)
            try:
                os.remove(tmp)
            except OSError:
                pass

    @staticmethod
    def _load_npz(path: str, scan: ScanInfo) -> Optional[OverviewData]:
        """Обзор из npz, если файл есть и сделан для этого скана; иначе None (битый файл — предупреждение)."""
        if not os.path.isfile(path):
            return None
        try:
            with np.load(path, allow_pickle=False) as z:
                if str(z['format']) != _NPZ_FORMAT or str(z['fingerprint']) != scan.fingerprint:
                    return None
                x0, x1, y0, y1, prow = (int(v) for v in z['roi'])
                fh, fw = (int(v) for v in z['full_shape'])
                ov = Overview(bin=int(z['bin']), dark=z['dark'], empty=z['empty'], samples=z['samples'],
                              sample_idx=z['sample_idx'], sample_angles=z['sample_angles'],
                              full_height=fh, full_width=fw)
                lo, hi = (float(v) for v in z['window'])
                return OverviewData(overview=ov, envelope=z['envelope'], roi=ROI(x0, x1, y0, y1, prow),
                                    angles_outside=[float(a) for a in z['angles_outside']], window=(lo, hi))
        except Exception as exc:  # noqa: BLE001 — битый/чужой файл: посчитать заново
            logger.warning('обзор %s не прочитан (%s) — считаем заново', path, exc)
            return None

    # --- синограмма строки ---------------------------------------------------------------------------------

    def sinogram(self, exp_id: str, row: int, n: int = SINO_N) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(−ln T float32 (k, W), углы (k,), индексы timeline (k,)) строки детектора row по k ≤ n углам."""
        scan = self.info(exp_id)
        row = arg_int({'row': row}, 'row', None, 0, scan.height - 1)
        n = arg_int({'n': n}, 'n', SINO_N, 1, SINO_MAX_N)
        key = (scan.fingerprint, row, n)
        with self._lock:
            hit = self._sinos.get(key)
            if hit is not None:
                self._sinos.move_to_end(key)
                return hit
        with self._key_lock(('sino',) + key):
            with self._lock:
                hit = self._sinos.get(key)
                if hit is not None:
                    return hit
            idx = data.pick_sample_indices(scan, n)
            if idx.size == 0:
                raise ValueError('{}: в скане нет data-кадров'.format(exp_id))
            with data.ChunkSampler(scan) as sampler:
                dark_sel = data._cheapest(sampler, scan.dark_idx, SINO_REF_FRAMES)
                empty_sel = data._cheapest(sampler, data.initial_empty_indices(scan), SINO_REF_FRAMES)
                if empty_sel.size == 0:
                    raise ValueError('{}: нет empty-кадров'.format(exp_id))
                all_idx = np.concatenate([dark_sel, empty_sel, idx])
                rows = sampler.read_frames(all_idx, rows=(row, row + 1), workers=self.cfg.workers)
            rows = rows[:, 0, :].astype(np.float32)
            nd, ne = len(dark_sel), len(empty_sel)
            dark = np.median(rows[:nd], axis=0) if nd else np.zeros(rows.shape[1], np.float32)
            empty = np.median(rows[nd:nd + ne], axis=0)
            sino = _minus_log_t(rows[nd + ne:], dark, empty).astype(np.float32)
            res = (sino, np.asarray(scan.angles, dtype=np.float64)[idx], idx)
            with self._lock:
                _put(self._sinos, key, res, _MEM_SINOS)
        return res

    # --- размер пикселя ------------------------------------------------------------------------------------

    def _storage_doc(self, exp_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """(документ эксперимента из storage или None, текст ошибки или None). Кэшируется (см. _DOC_TTL_S)."""
        def cached():
            with self._lock:
                hit = self._docs.get(exp_id)
            if hit is None:
                return None
            ttl = _DOC_TTL_S if hit[2] is None else _DOC_FAIL_TTL_S
            return hit if time.monotonic() - hit[0] < ttl else None

        hit = cached()
        if hit is not None:
            return hit[1], hit[2]
        with self._key_lock(('doc', exp_id)):
            hit = cached()
            if hit is not None:
                return hit[1], hit[2]
            doc, err = None, None
            try:
                r = requests.post(self.cfg.storage_server + 'storage/experiments/get', json={'_id': exp_id},
                                  timeout=_STORAGE_TIMEOUT_S)
                r.raise_for_status()
                body = r.json()
                if isinstance(body, list):
                    body = body[0] if body else None
                if isinstance(body, dict):
                    doc = body
                else:
                    err = 'в storage нет документа эксперимента'
                    logger.warning('документ %s из storage: %s', exp_id, err)
            except Exception as exc:  # noqa: BLE001 — storage недоступен/ответ не JSON: работаем без документа
                # пользователю — коротко (текст идёт в предупреждение размера пикселя), подробности — в лог
                logger.warning('документ %s из storage: %s: %s', exp_id, type(exc).__name__, exc)
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    err = 'storage ответил {}'.format(exc.response.status_code)
                elif isinstance(exc, requests.RequestException):
                    err = 'нет связи со storage'
                else:
                    err = 'некорректный ответ storage'
            with self._lock:
                self._docs[exp_id] = (time.monotonic(), doc, err)
        return doc, err

    def pixel_size(self, exp_id: str, user_value: Optional[float] = None) -> PixelSize:
        scan = self.info(exp_id)
        if user_value is not None:
            return pixelsize.resolve(None, scan.metadata, user_value)
        doc, err = self._storage_doc(exp_id)
        ps = pixelsize.resolve(doc, scan.metadata)
        if err:
            ps.warnings.append('документ эксперимента из storage недоступен ({}) — размер пикселя выбран '
                               'без него'.format(err))
        return ps

    # --- JSON ----------------------------------------------------------------------------------------------

    def info_json(self, exp_id: str) -> Dict[str, Any]:
        """Тело ответа /info."""
        scan = self.info(exp_id)
        ps = self.pixel_size(exp_id)
        a = np.asarray(scan.angles, dtype=np.float64)[scan.data_idx]
        angles = None
        has_pair = False
        if a.size:
            uniq = np.unique(a)
            step = np.diff(uniq)
            step = step[step > 1e-6]
            angles = {'min': float(a.min()), 'max': float(a.max()), 'range': scan.data_angle_range,
                      'step': float(np.median(step)) if step.size else 0.0}
            try:
                pipeline.pair_0_180(a)
                has_pair = True
            except ValueError:
                has_pair = False
        return {
            'exp_id': exp_id,
            'shape': [scan.n_frames, scan.height, scan.width],
            'n_frames': scan.n_frames, 'height': scan.height, 'width': scan.width, 'dtype': scan.dtype,
            'frames': {'dark': int(len(scan.dark_idx)), 'empty': int(len(scan.empty_idx)),
                       'data': int(len(scan.data_idx)), 'data_check': int(len(scan.check_idx))},
            'advanced': bool(scan.is_advanced), 'series_length': int(scan.series_length),
            'empty_period': int(scan.empty_period),
            'angles': angles, 'pair_0_180': has_pair,
            'pixel_size': pixel_size_json(ps),
            'fingerprint': scan.fingerprint, 'file_size': int(os.path.getsize(scan.path)),
            'chunk_frames': int(scan.chunk_frames), 'fast_path': bool(scan.fast_path),
        }


def pixel_size_json(ps: PixelSize) -> Dict[str, Any]:
    return {'value_mm': float(ps.value_mm), 'source': ps.source, 'warnings': list(ps.warnings)}


# --- эндпоинты ---------------------------------------------------------------------------------------------

def _registry() -> ScanRegistry:
    return current_app.extensions['recon'].scans


def _overview_args(exp_id: str) -> Tuple[ScanRegistry, OverviewData]:
    reg = _registry()
    n = arg_int(request.args, 'n', OVERVIEW_N, 1, OVERVIEW_MAX_N)
    b = arg_int(request.args, 'bin', OVERVIEW_BIN, 1, OVERVIEW_MAX_BIN)
    return reg, reg.overview(exp_id, n, b)


@bp.get('/<exp_id>/info')
def info(exp_id):
    exp_id = auth.valid_exp_id(exp_id)
    return jsonify(_registry().info_json(exp_id))


@bp.get('/<exp_id>/overview')
def overview(exp_id):
    exp_id = auth.valid_exp_id(exp_id)
    reg, od = _overview_args(exp_id)
    ov = od.overview
    return jsonify({
        'exp_id': exp_id,
        'roi': od.roi.to_dict(),
        'angles_outside': [float(a) for a in od.angles_outside],
        'bin': int(ov.bin), 'n': int(ov.samples.shape[0]),
        'shape': [int(s) for s in ov.samples.shape],
        'full_shape': [int(ov.full_height), int(ov.full_width)],
        'sample_angles': [float(a) for a in ov.sample_angles],
        'sample_indices': [int(i) for i in ov.sample_idx],
        'window': [float(v) for v in od.window],
        'pixel_size': pixel_size_json(reg.pixel_size(exp_id)),
    })


@bp.get('/<exp_id>/outside')
def outside(exp_id):
    """Углы выборки обзора, на которых объект выходит за столбцы ROI (в строках ROI) — для рамки, которую
    пользователь подвинул: ``?x0&x1&y0&y1`` в координатах полного кадра (``n``, ``bin`` — как у обзора)."""
    exp_id = auth.valid_exp_id(exp_id)
    _, od = _overview_args(exp_id)
    ov = od.overview
    a = request.args
    roi = ROI(arg_int(a, 'x0', od.roi.x0), arg_int(a, 'x1', od.roi.x1), arg_int(a, 'y0', od.roi.y0),
              arg_int(a, 'y1', od.roi.y1))
    roi.validate(ov.full_height, ov.full_width)
    angles = [float(v) for v in preprocess.angles_outside(ov, roi)]
    hit = set(angles)
    return jsonify({'roi': roi.to_dict(), 'angles_outside': angles,
                    'indices': [k for k, v in enumerate(ov.sample_angles) if float(v) in hit]})


@bp.get('/<exp_id>/envelope')
def envelope(exp_id):
    exp_id = auth.valid_exp_id(exp_id)
    _, od = _overview_args(exp_id)
    return binary.array_response(od.envelope, meta={'bin': int(od.overview.bin)})


@bp.get('/<exp_id>/thumbs')
def thumbs(exp_id):
    exp_id = auth.valid_exp_id(exp_id)
    _, od = _overview_args(exp_id)
    ov = od.overview
    t = _minus_log_t(ov.samples, ov.dark, ov.empty)
    meta = {'angles': [round(float(a), 4) for a in ov.sample_angles],
            'indices': [int(i) for i in ov.sample_idx], 'bin': int(ov.bin)}
    return binary.array_response(t, lo=od.window[0], hi=od.window[1], meta=meta)


@bp.get('/<exp_id>/sample/<int:k>')
def sample(exp_id, k):
    exp_id = auth.valid_exp_id(exp_id)
    _, od = _overview_args(exp_id)
    ov = od.overview
    if not 0 <= k < ov.samples.shape[0]:
        raise ValueError('кадр выборки {} вне [0, {})'.format(k, ov.samples.shape[0]))
    t = _minus_log_t(ov.samples[k], ov.dark, ov.empty)
    meta = {'k': k, 'angle': float(ov.sample_angles[k]), 'index': int(ov.sample_idx[k]), 'bin': int(ov.bin)}
    return binary.array_response(t, lo=od.window[0], hi=od.window[1], meta=meta)


@bp.post('/<exp_id>/prefetch')
def prefetch(exp_id):
    """Начать предзагрузку исходного HDF5 в кэш ОС (``reconservice.prefetch``) — 202 и состояние."""
    exp_id = auth.valid_exp_id(exp_id)
    return jsonify(current_app.extensions['recon'].prefetch.start(exp_id)), 202


@bp.get('/<exp_id>/sinogram')
def sinogram(exp_id):
    exp_id = auth.valid_exp_id(exp_id)
    reg = _registry()
    scan = reg.info(exp_id)
    row = arg_int(request.args, 'row', scan.height // 2, 0, scan.height - 1)
    n = arg_int(request.args, 'n', SINO_N, 1, SINO_MAX_N)
    sino, angles, _ = reg.sinogram(exp_id, row, n)
    # углы округлены: X-Meta до 360 чисел не должен упираться в лимиты заголовков прокси
    meta = {'row': row, 'n': int(sino.shape[0]), 'angles': [round(float(a), 3) for a in angles]}
    return binary.array_response(sino, meta=meta)
