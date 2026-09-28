"""Ось вращения.

В рецепте ось хранится в координатах детектора (``model.Axis``: столбец ``center_x`` на строке ``y_ref``,
наклон ``tilt_deg``), чтобы правка ROI её не портила. Для кропа ось переводится в параметры
``(shift_x, alfa)`` старой функции ``tomotools4.transform_image`` (сдвиг по x, затем поворот вокруг центра
кропа ``((h−1)/2, (w−1)/2)``, order=3, mode='nearest'): после преобразования ось вертикальна и проходит через
столбец ``(w−1)/2`` кропа — на этом держится и старый ноутбук, и FBP без смещения центра.

Выравнивание слоя строк (``align_rows``) — то же преобразование, но только для нужных выходных строк: из-за
наклона выходная строка берёт данные из входных строк в пределах ``margin_rows``.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .model import Axis, ROI


def to_crop_params(axis: Axis, roi: ROI) -> Tuple[float, float]:
    """(shift_x, alfa) для transform_image на кропе roi, эквивалентные оси axis."""
    raise NotImplementedError


def from_crop_params(shift_x: float, alfa: float, roi: ROI, method: str = 'auto') -> Axis:
    """Обратное преобразование: Axis в координатах детектора (y_ref — строка центра кропа)."""
    raise NotImplementedError


def margin_rows(alfa: float, width: int, extra: int = 8) -> int:
    """Запас входных строк сверху и снизу для align_rows при повороте на alfa: ceil(width/2·|tan alfa|) + extra."""
    raise NotImplementedError


def align_rows(frames, in_row0: int, out_rows: Tuple[int, int], shift_x: float, alfa: float,
               crop_height: int, xp=None):
    """Выровнять слой: frames (n, s_in, w) — строки кропа [in_row0, in_row0 + s_in); вернуть (n, s_out, w) для
    строк кропа out_rows = [r0, r1), совпадающих с transform_image на полном кропе высоты crop_height
    (в пределах интерполяции). Требует, чтобы вход покрывал out_rows ± margin_rows (иначе ValueError)."""
    raise NotImplementedError


def auto_axis(img0: np.ndarray, img180: np.ndarray, roi: ROI) -> Axis:
    """Авто-ось по нормированным кадрам кропа при ~0° и ~180° (img180 НЕ отражён): поиск (shift, alfa)
    как ``tomotools4.find_axis_correction`` (Powell от начального приближения по X центра масс, целевая —
    ‖T(im0, s, a) − T(flip(im180), −s, −a)‖²), результат переводится в Axis."""
    raise NotImplementedError


def center_metric(slice_img: np.ndarray, kind: str = 'entropy') -> float:
    """Метрика резкости среза внутри вписанного круга: 'entropy' (энтропия гистограммы, меньше — лучше) или
    'tv' (полная вариация, больше — резче). Возвращает значение, у которого меньше = лучше."""
    raise NotImplementedError


def center_scan(sino_row: np.ndarray, angles_deg: np.ndarray, centers: Sequence[float], crop_center: float,
                pixel_size: float, recon_fn, metric: str = 'entropy') -> Tuple[List[np.ndarray], np.ndarray]:
    """Перебор центра для одной строки. sino_row (n, w) — строка, уже выровненная по текущей оси (ось в
    crop_center = (w−1)/2); для центра c строка сдвигается на (crop_center − c) по x и восстанавливается
    recon_fn(sino, angles, pixel_size) → (w, w). Возвращает (срезы, метрики)."""
    raise NotImplementedError


def tilt_from_centers(y_top: float, c_top: float, y_bottom: float, c_bottom: float, method: str = 'tilt') -> Axis:
    """Ось по центрам на двух строках детектора."""
    raise NotImplementedError


def diff_view(img0: np.ndarray, img180: np.ndarray, shift_x: float, alfa: float) -> np.ndarray:
    """Вспомогательный вид: T(img0, s, a) − flip(T(img180, s, a)) (как «Показать совмещение» в ноутбуке)."""
    raise NotImplementedError
