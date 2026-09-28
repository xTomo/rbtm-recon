"""Предобработка: огибающая и авто-ROI по обзору, dark/empty по кропу, нормировка слоями, репозиционирование.

Нормировка повторяет ``tomotools4.normalize_projections`` / ``normalize_projections_with_timeline``:
    empty и data после вычитания dark обрезаются снизу до 1, d = log(empty) − log(data),
    затем ``safe_median`` (медиана 3×3 для выбросов), затем clip(d, 0, ∞).
Для advanced-экспериментов empty для кадра интерполируется по frame_number между начальной и
периодическими empty-сериями (семантика ``tomotools4._interpolate_empty`` после исправления: у начальной серии
свой ``initial_empty_fnumber``).
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .model import CropData, Overview, ROI, ScanInfo


def envelope(ov: Overview, eps: float = 1e-3) -> np.ndarray:
    """max по выборке углов от −ln T, T = (I − D) / (E − D), T обрезается снизу до eps. float32 (h, w)."""
    raise NotImplementedError


def suggest_roi(env: np.ndarray, bin: int, full_height: int, full_width: int,
                margin_frac: float = 0.03, k_sigma: float = 6.0) -> ROI:
    """Предложить ROI по огибающей: маска объекта = env > фон + k_sigma·шум (MAD); столбцы/строки, где доля маски
    заметна, расширяются на margin_frac ширины/высоты кадра и обрезаются по кадру. Координаты — полного кадра.
    Если объект не найден — весь кадр."""
    raise NotImplementedError


def angles_outside(ov: Overview, roi: ROI, k_sigma: float = 6.0) -> List[float]:
    """Углы выборки, на которых маска объекта (−ln T по отдельному кадру) выходит за столбцы [x0, x1)."""
    raise NotImplementedError


@dataclasses.dataclass
class DarkEmpty:
    """Опорные кадры по кропу (строки rows кропа, если заданы при расчёте). float32."""
    dark: np.ndarray                               # (h, w)
    initial_empty: np.ndarray                      # (h, w), dark вычтен
    initial_empty_fnumber: int
    periodic_empties: List[np.ndarray]             # K × (h, w), dark вычтен (только advanced)
    periodic_empty_fnumbers: List[int]


def dark_empty_from_crop(scan: ScanInfo, crop: CropData,
                         rows: Optional[Tuple[int, int]] = None) -> DarkEmpty:
    """Медианы dark, начальной empty-серии и периодических empty-серий по кропу (строки rows кропа, если заданы).

    Разбиение empty на серии — как в ``hdf5_v2.load_tomo_data_advanced_v2`` (первые series_length — начальная,
    далее по series_length подряд), frame_number серии — номер её первого кадра."""
    raise NotImplementedError


def empty_for_frame(de: DarkEmpty, frame_number: int) -> np.ndarray:
    """Empty для кадра: линейная интерполяция по frame_number между соседними сериями; до первой периодической —
    между начальной и первой; после последней — последняя; без периодических — начальная."""
    raise NotImplementedError


def normalize_slab(frames, frame_numbers: Sequence[int], de: DarkEmpty, xp=None, median3: bool = True):
    """Нормировать слой кадров (n, s, w) uint16 → xp.float32 −ln(T) как в tomotools4 (см. модуль).
    de должен быть посчитан для тех же строк. Работает на xp (cupy или numpy)."""
    raise NotImplementedError


def repositioning_shifts(scan: ScanInfo, crop: CropData, de: DarkEmpty) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Сдвиги образца после каждой периодической вставки (advanced): (углы checkpoint-ов, sy, sx), K элементов.
    Порт ``tomotools4.measure_repositioning_shifts`` (debug=False): пара «первый data_check после вставки k ↔
    последний data-кадр с тем же углом до вставки», фазовая корреляция по нормированным кадрам.
    Checkpoint без пары — сдвиг 0 и предупреждение в лог. Не-advanced — пустые массивы."""
    raise NotImplementedError


def segment_index(frame_numbers: Sequence[int], periodic_empty_fnumbers: Sequence[int]) -> np.ndarray:
    """Номер сегмента для каждого кадра: 0 до первой периодической вставки, k после k-й."""
    raise NotImplementedError


def cumulative_shifts(sy: np.ndarray, sx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Абсолютные сдвиги сегментов 0..K: cum[0] = 0, cum[k] = sum(shifts[:k]); NaN трактуется как 0."""
    raise NotImplementedError
