"""Подавление колец (полос в синограмме) — ``tomo.remove_stripe.remove_all_stripe`` (метод Vo, cupy).

Пресеты: 'off' — без обработки; 'medium' — нынешние параметры ноутбука (snr=3, la_size=61, sm_size=21);
'weak' и 'strong' — предварительные, подбираются на реальных данных (этап 4 плана).
На CPU (numpy) реальная функция недоступна (модуль tomo.remove_stripe импортирует cupy) — в тестах
она подменяется заглушкой.
"""
from __future__ import annotations

from typing import Dict, Optional

PRESETS: Dict[str, Optional[Dict[str, float]]] = {
    'off': None,
    'weak': {'snr': 4.0, 'la_size': 41, 'sm_size': 11},
    'medium': {'snr': 3.0, 'la_size': 61, 'sm_size': 21},
    'strong': {'snr': 2.0, 'la_size': 81, 'sm_size': 31},
}


def resolve(preset: str = 'medium', params: Optional[Dict[str, float]] = None) -> Optional[Dict[str, float]]:
    """Параметры remove_all_stripe: явные params важнее пресета; None — не обрабатывать."""
    if params:
        return {'snr': float(params['snr']), 'la_size': int(params['la_size']), 'sm_size': int(params['sm_size'])}
    if preset not in PRESETS:
        raise ValueError('неизвестный пресет колец: {}'.format(preset))
    p = PRESETS[preset]
    return None if p is None else {'snr': float(p['snr']), 'la_size': int(p['la_size']), 'sm_size': int(p['sm_size'])}


def apply(sino_rows, params: Optional[Dict[str, float]], xp=None):
    """Обработать слой синограмм (s, n, w) → той же формы. params=None — вернуть как есть.
    remove_all_stripe ждёт форму [n, s, w] (проекции × строки × столбцы)."""
    if not params:
        return sino_rows
    from tomo.remove_stripe import remove_all_stripe  # noqa: WPS433 — ленивый импорт (cupy)
    from .gpu import get_xp
    xp = xp or get_xp()
    tomo = xp.ascontiguousarray(xp.swapaxes(xp.asarray(sino_rows, dtype=xp.float32), 0, 1))
    out = remove_all_stripe(tomo, snr=params['snr'], la_size=int(params['la_size']),
                            sm_size=int(params['sm_size']), dim=1)
    return xp.swapaxes(out, 0, 1)
