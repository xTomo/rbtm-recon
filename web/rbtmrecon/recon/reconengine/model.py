"""Общие типы движка.

Соглашения:
- координаты — пиксели полного кадра детектора: строка ``y`` (0 — верх), столбец ``x`` (0 — слева);
- прямоугольники полуоткрытые: ``[x0, x1) × [y0, y1)``;
- режимы кадров в timeline HDF5 v2: 0 dark, 1 empty, 2 data, 3 data_check.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Dict, Optional

import numpy as np

MODE_DARK = 0
MODE_EMPTY = 1
MODE_DATA = 2
MODE_DATA_CHECK = 3

ProgressFn = Callable[[float, str], None]


def no_progress(frac: float, stage: str) -> None:  # noqa: ARG001
    """Callback прогресса по умолчанию."""


class Cancelled(Exception):
    """Операция отменена через cancel-событие."""


def check_cancel(cancel) -> None:
    """Бросить Cancelled, если событие отмены выставлено (cancel может быть None)."""
    if cancel is not None and cancel.is_set():
        raise Cancelled()


@dataclasses.dataclass
class ROI:
    """Поле зрения на детекторе: столбцы [x0, x1), строки (срезы) [y0, y1) и строка превью."""
    x0: int
    x1: int
    y0: int
    y1: int
    preview_row: Optional[int] = None

    def __post_init__(self):
        self.x0, self.x1, self.y0, self.y1 = int(self.x0), int(self.x1), int(self.y0), int(self.y1)
        if self.preview_row is None:
            self.preview_row = (self.y0 + self.y1) // 2
        self.preview_row = int(self.preview_row)

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def validate(self, height: int, width: int) -> None:
        """Проверить, что ROI лежит в кадре height×width и не пуст; иначе ValueError."""
        if not (0 <= self.x0 < self.x1 <= width):
            raise ValueError('ROI x [{}, {}) вне кадра шириной {}'.format(self.x0, self.x1, width))
        if not (0 <= self.y0 < self.y1 <= height):
            raise ValueError('ROI y [{}, {}) вне кадра высотой {}'.format(self.y0, self.y1, height))
        if not (self.y0 <= self.preview_row < self.y1):
            raise ValueError('строка превью {} вне [{}, {})'.format(self.preview_row, self.y0, self.y1))

    def to_dict(self) -> Dict[str, int]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ROI':
        return cls(d['x0'], d['x1'], d['y0'], d['y1'], d.get('preview_row'))


@dataclasses.dataclass
class Axis:
    """Ось вращения в координатах детектора.

    ``center_x`` — столбец оси на строке ``y_ref``; ``tilt_deg`` — наклон оси к вертикали:
    столбец оси на строке y равен ``center_x + tan(tilt_deg) · (y − y_ref)``.
    """
    center_x: float
    y_ref: float
    tilt_deg: float = 0.0
    method: str = 'auto'

    def center_at(self, y: float) -> float:
        return self.center_x + math.tan(math.radians(self.tilt_deg)) * (y - self.y_ref)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Axis':
        return cls(float(d['center_x']), float(d['y_ref']), float(d.get('tilt_deg', 0.0)), d.get('method', 'auto'))


@dataclasses.dataclass
class ScanInfo:
    """Сведения о скане HDF5 v2, прочитанные без распаковки изображений."""
    path: str
    exp_id: str
    n_frames: int
    height: int
    width: int
    dtype: str
    chunk_frames: int            # кадров в чанке images/all (первая ось формы чанка)
    compression: Optional[str]   # 'gzip' или None
    shuffle: bool
    fast_path: bool              # True — чанки можно читать напрямую (только gzip, без shuffle/прочих фильтров)
    angles: np.ndarray           # (N,) float64, градусы, в порядке timeline
    modes: np.ndarray            # (N,) uint8
    frame_numbers: np.ndarray    # (N,) int64
    dark_idx: np.ndarray         # индексы timeline, по возрастанию
    empty_idx: np.ndarray
    data_idx: np.ndarray
    check_idx: np.ndarray
    is_advanced: bool
    series_length: int
    empty_period: int
    metadata: Dict[str, Any]     # metadata/* со строками, декодированными в str
    fingerprint: str             # sha256 структуры файла (форма, чанки, фильтры, размер, timeline, mapping)

    @property
    def data_angle_range(self) -> float:
        """Размах углов data-кадров, градусы."""
        a = self.angles[self.data_idx]
        return float(a.max() - a.min()) if len(a) else 0.0


@dataclasses.dataclass
class Overview:
    """Прореженная выборка кадров для шага «Поле зрения».

    Все изображения — float32, уменьшены биннингом ``bin`` (среднее bin×bin), в отсчётах детектора.
    """
    bin: int
    dark: np.ndarray             # (h, w) медиана нескольких dark-кадров
    empty: np.ndarray            # (h, w) медиана нескольких кадров начальной empty-серии
    samples: np.ndarray          # (k, h, w) data-кадры выборки
    sample_idx: np.ndarray       # (k,) индексы timeline
    sample_angles: np.ndarray    # (k,) градусы
    full_height: int
    full_width: int


@dataclasses.dataclass
class CropData:
    """Кроп всех кадров скана в порядке timeline, uint16, лежит memmap-файлом в кэше."""
    roi: ROI
    frames: np.ndarray           # (N, roi.height, roi.width) uint16 (np.memmap)
    path: str                    # путь к .u16
    fingerprint: str             # fingerprint скана, из которого сделан кроп
