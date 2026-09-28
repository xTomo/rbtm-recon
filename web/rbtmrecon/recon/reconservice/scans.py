"""Сведения о скане и обзор для шага «Поле зрения» — без загрузки всех данных (``/scans/<exp_id>/...``).

Всё читается из исходного ``<exp_src>/<exp_id>.h5`` движком (``reconengine.data``): структура — мгновенно,
обзор (dark/empty + N углов с биннингом) — 2–3 с на кадре 5056×2968, дальше из кэша.

Эндпоинты (все требуют токен; exp_id проверяется ``auth.valid_exp_id``):

| Метод и путь                          | Ответ |
|---------------------------------------|-------|
| GET /scans/<id>/info                  | JSON: форма, число кадров по режимам, advanced, диапазон углов, размер пикселя с источником и предупреждениями, fingerprint |
| GET /scans/<id>/overview              | JSON: предложенный ROI (координаты детектора), углы, где объект выходит за ROI, биннинг, форма обзора, углы выборки, размер пикселя |
| GET /scans/<id>/envelope              | binary uint16 (h/b, w/b): огибающая max(−ln T) по углам выборки |
| GET /scans/<id>/thumbs                | binary uint16 (k, h/b, w/b): −ln T кадров выборки; X-Meta: angles, indices |
| GET /scans/<id>/sample/<k>            | binary uint16 (h/b, w/b): k-й кадр выборки (−ln T) |
| GET /scans/<id>/sinogram?row=&n=      | binary uint16 (n, W): строка детектора row по n углам (по умолчанию 90), кадры с наименьшей стоимостью распаковки в каждом угловом интервале; X-Meta: angles |

Параметры обзора по умолчанию: n=16 углов (``?n=`` до 64), bin=4. Кэш: в памяти (последние несколько сканов) и
``<cfg.fast_exp_dir(id)>/overview-<fingerprint[:16]>-n<n>-b<bin>.npz`` — второй запрос и рестарт сервиса не читают HDF5.
Размер пикселя: ``pixelsize.resolve(документ storage, metadata HDF5)``; документ берётся запросом
``POST <cfg.storage_server>storage/experiments/get`` с ``{"_id": exp_id}`` (Content-Type: application/json,
таймаут 5 с); недоступность storage — не ошибка: без документа и с предупреждением в ответе.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

import numpy as np
from flask import Blueprint

from reconengine.model import Overview, ROI, ScanInfo
from reconengine.pixelsize import PixelSize

from .config import Config

bp = Blueprint('scans', __name__, url_prefix='/scans')


@dataclasses.dataclass
class OverviewData:
    """Обзор скана с производными: огибающая, предложенный ROI, углы, где объект выходит за ROI."""
    overview: Overview
    envelope: np.ndarray            # float32 (h/b, w/b)
    roi: ROI                        # координаты полного кадра
    angles_outside: List[float]


class ScanRegistry:
    """Потокобезопасный кэш сведений о сканах. ScanInfo — по (путь, размер, mtime); обзор — см. модуль."""

    def __init__(self, cfg: Config):
        raise NotImplementedError

    def path(self, exp_id: str) -> str:
        """Путь к исходному HDF5; FileNotFoundError, если файла нет."""
        raise NotImplementedError

    def info(self, exp_id: str) -> ScanInfo:
        raise NotImplementedError

    def overview(self, exp_id: str, n: int = 16, bin: int = 4) -> OverviewData:
        raise NotImplementedError

    def pixel_size(self, exp_id: str, user_value: Optional[float] = None) -> PixelSize:
        raise NotImplementedError

    def info_json(self, exp_id: str) -> Dict[str, Any]:
        """Тело ответа /info."""
        raise NotImplementedError
