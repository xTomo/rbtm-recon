"""Шум среза и подбор σ сглаживания без эталона — по двум независимым половинам углов (чётные и нечётные).

Срезы R_ч и R_н по чётным и нечётным выбранным углам — один и тот же объект x плюс независимые шумы n_ч, n_н
(у каждого дисперсия 2v, где v — дисперсия шума среза по всем углам; FBP усредняет вдвое больше проекций). Отсюда:

- шум среза по всем углам: σ = std((R_ч − R_н)/2) — в этой разности нет объекта (``noise_sigma``; вместо std —
  1,4826·MAD, чтобы края объекта, где половины расходятся из-за редкой выборки углов, не завышали оценку);
- ошибка линейного фильтра D (Noise2Noise): E|D(R_ч) − R_н|² = |Dx − x|² + var(D n_ч) + 2v, потому что n_н не
  зависит от D(R_ч). Вычитая 2v = |R_ч − R_н|²/2, получаем ошибку фильтра на половине углов; var(D n_ч) =
  |D(R_ч) − D(R_н)|²/2, поэтому смещение |Dx − x|² = ошибка − var(D n_ч), а ошибка того же фильтра на ВСЕХ углах —
  смещение + var(D n_ч)/2 (``filter_mse``). Берётся среднее двух симметричных оценок (ч→н и н→ч).

Так σ подбирается под шум полного набора углов, а не половины (по половине оптимум сдвинут к большему σ).
Половины выборки углов реже вдвое: на краях объекта и за радиусом, где углов мало, их различие — не только шум,
поэтому оценки шума слегка завышены (подбор σ — тоже немного к сглаживанию). Сравнение вариантов честное: у всех
одно и то же смещение оценки.

Метрика ``preview.noise_level`` (MAD лапласиана) видит только самые высокие частоты и после сглаживания показывает
снижение шума в разы больше настоящего (af443cef: лапласиан — в 16 раз, по половинам — в 2,8 раза) — поэтому шум
в сравнении вариантов теперь по половинам.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from . import fbp

#: Сетка σ автоподбора (0 — без сглаживания); пределы — smoothing.SIGMA_RANGE.
AUTO_SIGMAS = (0.0, 0.7, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0)
#: Минимум ошибки по σ пологий: берётся наименьшая σ с ошибкой не больше (1 + AUTO_TOLERANCE)·минимум — заметно
#: резче почти без потери (af443cef: 3,5 вместо 4; 79d1ba3e: 2 вместо 2,5).
AUTO_TOLERANCE = 0.05


def split_halves(angles_deg: np.ndarray, mode: str = 'first_180') -> Tuple[np.ndarray, np.ndarray]:
    """Индексы чётных и нечётных (по порядку съёмки) среди углов, выбранных ``fbp.select_angles``, поровну."""
    idx = np.where(fbp.select_angles(np.asarray(angles_deg), mode))[0]
    k = idx.size // 2
    if k < 2:
        raise ValueError('мало углов для оценки шума по половинам: {}'.format(idx.size))
    return idx[0::2][:k], idx[1::2][:k]


def robust_std(a) -> float:
    """1,4826·MAD — σ гауссова шума, устойчиво к редким выбросам (краям объекта)."""
    a = np.asarray(a, dtype='float64').ravel()
    if a.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


def noise_sigma(r_even, r_odd) -> float:
    """Шум среза по всем углам (те же единицы) по срезам двух половин."""
    return robust_std((np.asarray(r_even, dtype='float64') - np.asarray(r_odd, dtype='float64')) / 2)


def _ms(a, b) -> float:
    d = np.asarray(a, dtype='float64') - np.asarray(b, dtype='float64')
    return float(np.mean(d * d))


def filter_mse(d_even, d_odd, r_even, r_odd) -> Tuple[float, float]:
    """(оценка среднего квадрата ошибки среза по всем углам после фильтра D, дисперсия его шума) по срезам половин
    без фильтра (r_*) и с ним (d_*). Для D = тождество ошибка ≈ v — дисперсия шума без фильтра."""
    s2_half = _ms(r_even, r_odd) / 2                          # 2v: шум среза по половине углов
    err_half = 0.5 * (_ms(d_even, r_odd) + _ms(d_odd, r_even)) - s2_half
    n_half = _ms(d_even, d_odd) / 2                           # var(D n) на половине углов
    bias2 = max(err_half - n_half, 0.0)
    return bias2 + n_half / 2, n_half / 2


def rmse(mse: float) -> float:
    return math.sqrt(max(mse, 0.0))


def pick(errors, tolerance: float = AUTO_TOLERANCE) -> Tuple[int, int]:
    """(выбранный, наилучший) индексы по ошибкам вариантов, упорядоченных от слабого сглаживания к сильному:
    выбранный — первый с ошибкой ≤ (1 + tolerance)·минимум."""
    e = [float(v) for v in errors]
    if not e:
        raise ValueError('нет вариантов')
    best = min(range(len(e)), key=e.__getitem__)
    lim = e[best] * (1.0 + tolerance)
    return next(j for j, v in enumerate(e) if v <= lim), best
