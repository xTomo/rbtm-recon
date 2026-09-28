"""Выбор размера пикселя детектора с указанием источника.

Источники (по убыванию приоритета): значение, введённое пользователем; ``pixel_size`` из документа Mongo;
известная модель детектора (:data:`DETECTOR_PIXEL_MM`); ``pixel_size`` из метаданных HDF5 (если это не
«технический» дефолт, который storage писал, когда модель детектора была не известна); значение по
умолчанию (:data:`DEFAULT_MM`) с предупреждением.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional

#: Известные модели детекторов → размер пикселя, мм (как ``PIXEL_SIZES`` в rbtm-drivers
#: ``drivers/Detector/Detector.py``). Список расширяемый.
DETECTOR_PIXEL_MM: Dict[str, float] = {
    'MH110XC-KK-FA': 0.009,
    'MJ150XR-GP-FA-GO': 0.00425,
}

#: Значение, которое rbtm-storage писало в metadata/pixel_size по умолчанию, когда модель детектора
#: не была известна на момент съёмки. Встретив его в HDF5, считаем размер пикселя «не заданным»
#: (оно же — настоящий размер пикселя MJ150XR, поэтому модель детектора проверяется раньше).
WRITER_DEFAULT_MM = 0.00425

#: Значение по умолчанию, если размер пикселя не удалось определить ни одним из способов.
DEFAULT_MM = 0.00425

#: Порог расхождения источников, после которого выдаётся предупреждение (доля от значения).
_MISMATCH_TOL = 0.01


@dataclasses.dataclass
class PixelSize:
    """Размер пикселя с источником и предупреждениями, накопленными при выборе."""
    value_mm: float
    source: str
    warnings: List[str] = dataclasses.field(default_factory=list)


def _positive_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def resolve(mongo_doc: Optional[Dict[str, Any]], hdf5_metadata: Optional[Dict[str, Any]],
           user_value: Optional[float] = None) -> PixelSize:
    """Выбрать размер пикселя.

    Порядок: ``user_value`` → ``mongo_doc['pixel_size']`` (> 0) → модель детектора
    (``mongo_doc['detector_model']`` или ``hdf5_metadata['detector_model']``) →
    ``hdf5_metadata['pixel_size']`` (если это не :data:`WRITER_DEFAULT_MM`) → :data:`DEFAULT_MM`
    с предупреждением. Отдельное предупреждение — если mongo и модель детектора расходятся > 1 %.
    """
    warnings: List[str] = []
    mongo_doc = mongo_doc or {}
    hdf5_metadata = hdf5_metadata or {}

    user = _positive_or_none(user_value)
    if user is not None:
        return PixelSize(user, 'user', warnings)

    mongo_value = _positive_or_none(mongo_doc.get('pixel_size'))

    detector_model = mongo_doc.get('detector_model') or hdf5_metadata.get('detector_model')
    detector_value = DETECTOR_PIXEL_MM.get(detector_model) if detector_model else None

    if mongo_value is not None and detector_value is not None:
        if abs(mongo_value - detector_value) > _MISMATCH_TOL * detector_value:
            warnings.append(
                'размер пикселя из mongo ({:.5f} мм) расходится с моделью детектора {!r} '
                '({:.5f} мм) более чем на 1%'.format(mongo_value, detector_model, detector_value))

    if mongo_value is not None:
        return PixelSize(mongo_value, 'mongo', warnings)

    if detector_value is not None:
        return PixelSize(detector_value, 'detector', warnings)

    hdf5_value = _positive_or_none(hdf5_metadata.get('pixel_size'))
    if hdf5_value is not None and not math.isclose(hdf5_value, WRITER_DEFAULT_MM, rel_tol=1e-9, abs_tol=1e-12):
        return PixelSize(hdf5_value, 'hdf5', warnings)

    warnings.append('размер пикселя не известен, взято значение по умолчанию 4,25 мкм')
    return PixelSize(DEFAULT_MM, 'default', warnings)
