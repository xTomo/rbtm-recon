"""Подавление колец (полос в синограмме).

Два метода, применяются по очереди:

- ``vo`` — ``tomo.remove_stripe.remove_all_stripe`` (метод Vo, cupy): крупные, «мёртвые» и нестабильные полосы;
- ``fft`` — фильтр Raven (1998) в спектре синограммы: в строках нулевой и низших ``v`` угловых частот (составляющие,
  почти постоянные по углу — полосы) гасятся высокие частоты по x фильтром Баттерворта порядка ``n``; отсечка задана
  периодом ``period`` (px): гасятся полосы уже ~period пикселей при любой ширине синограммы (u = ширина / period;
  для af443cef period 50 ≈ u 45 algotom). Широкую размытую часть полосы фильтр оставляет — её берёт Vo. Как и другие
  фильтры колец, приглушает мелкие детали у самой оси (почти не движутся с углом и похожи на полосы).
  ``v`` ≥ 2 убирает и полосы, медленно меняющиеся по углу (неполные кольца, дуги); при ``v`` ≥ 4 у центра среза
  проступает «звёздочка» радиальных штрихов.

Порядок относительно сдвигов кадров (сдвиги образца после вставок, компенсация смещения). Полоса — дефект столбца
детектора; синограмма здесь уже выровнена, и каждая проекция сдвинута на свой ``frame_sx`` — полоса перестаёт быть
вертикальной, и методы её не узнают (остаются дуги). Поэтому (версия 2): проекции возвращаются на место детектора
(сдвиг на −frame_sx), по ним считается поправка «после − до» и прибавляется к выровненной синограмме со сдвигом
обратно. Это то же, что подавлять кольца до сдвигов кадров, но без лишних строк (наклон оси при этом не мешает:
поворот одинаков для всех кадров). Замер на af443cef и 79d1ba3e (метрика дуг колец у оси): Vo после сдвигов — 72 %
от «выкл», Vo с возвратом сдвигов — 61 %, Vo + БПФ (u 45, v 2) — ~40 %; на глаз кольца у оси уходят.

Версии (рецепт ``rings.version``): 1 — прежние пресеты (только Vo, без учёта сдвигов кадров; рецепты, записанные
до версии 2, считаются так же, как тогда); 2 — :data:`PRESETS`.

На CPU (numpy) реальный Vo недоступен (``tomo.remove_stripe`` импортирует cupy) — в тестах он подменяется заглушкой;
БПФ-фильтр работает и на numpy.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

VERSION = 2
VERSIONS = (1, 2)

_VO = {
    'weak': {'snr': 4.0, 'la_size': 41, 'sm_size': 11},
    'medium': {'snr': 3.0, 'la_size': 61, 'sm_size': 21},
    'strong': {'snr': 2.0, 'la_size': 81, 'sm_size': 31},
}
#: Версия 1 (рецепты до 01.10.2026): только Vo, параметры ноутбука для 'medium'.
PRESETS_V1: Dict[str, Optional[Dict[str, float]]] = {'off': None, **_VO}
#: Версия 2: Vo (кроме «слабо») и БПФ-фильтр; поправка считается на проекциях без сдвигов кадров.
PRESETS: Dict[str, Optional[Dict[str, Any]]] = {
    'off': None,
    'weak': {'vo': None, 'fft': {'period': 50, 'n': 8, 'v': 1}},
    'medium': {'vo': dict(_VO['medium']), 'fft': {'period': 50, 'n': 8, 'v': 2}},
    'strong': {'vo': dict(_VO['strong']), 'fft': {'period': 50, 'n': 8, 'v': 3}},
}
DESCRIPTIONS = {
    'off': 'без подавления колец',
    'weak': 'БПФ-фильтр полос (период 50 px, v 1)',
    'medium': 'Vo (snr 3, окна 61/21) + БПФ-фильтр (период 50 px, v 2)',
    'strong': 'Vo (snr 2, окна 81/31) + БПФ-фильтр (период 50 px, v 3)',
}


def _vo_params(p: Dict[str, Any]) -> Dict[str, float]:
    return {'snr': float(p['snr']), 'la_size': int(p['la_size']), 'sm_size': int(p['sm_size'])}


def _fft_params(p: Dict[str, Any]) -> Dict[str, Any]:
    period = float(p['period'])
    if not period > 1:
        raise ValueError('fft.period должен быть > 1 px: {}'.format(period))
    return {'period': period, 'n': int(p['n']), 'v': int(p['v'])}


def resolve(preset: str = 'medium', params: Optional[Dict[str, Any]] = None,
            version: int = VERSION) -> Optional[Dict[str, Any]]:
    """Параметры подавления колец: ``{'vo': {...} | None, 'fft': {...} | None, 'unshift': bool}``; None — не
    обрабатывать. Явные params важнее пресета: версия 1 — ``{snr, la_size, sm_size}`` (Vo), версия 2 —
    ``{vo: {...} | None, fft: {...} | None}``."""
    if version not in VERSIONS:
        raise ValueError('неизвестная версия колец: {}'.format(version))
    if version == 1:
        if params:
            return {'vo': _vo_params(params), 'fft': None, 'unshift': False}
        if preset not in PRESETS_V1:
            raise ValueError('неизвестный пресет колец: {}'.format(preset))
        p = PRESETS_V1[preset]
        return None if p is None else {'vo': _vo_params(p), 'fft': None, 'unshift': False}
    if params:
        src = params
    else:
        if preset not in PRESETS:
            raise ValueError('неизвестный пресет колец: {}'.format(preset))
        src = PRESETS[preset]
        if src is None:
            return None
    vo, ff = src.get('vo'), src.get('fft')
    if not vo and not ff:
        return None
    return {'vo': _vo_params(vo) if vo else None, 'fft': _fft_params(ff) if ff else None, 'unshift': True}


# --- методы ------------------------------------------------------------------------------------------------------

def _vo(x, p: Dict[str, float], xp):
    from tomo.remove_stripe import remove_all_stripe  # noqa: WPS433 — ленивый импорт (cupy)
    tomo = xp.ascontiguousarray(xp.swapaxes(x, 0, 1))                 # remove_all_stripe: [проекции, строки, столбцы]
    out = remove_all_stripe(tomo, snr=p['snr'], la_size=int(p['la_size']), sm_size=int(p['sm_size']), dim=1)
    return xp.swapaxes(xp.asarray(out, dtype=xp.float32), 0, 1)


def stripe_fft(x, period: float = 50.0, n: int = 8, v: int = 2, xp=None):
    """Фильтр Raven (1998) для слоя синограмм (s, углы, w): как ``algotom.prep.removal.remove_stripe_based_fft``
    (дополнение средним по углу и крайними столбцами, 2D БПФ), но строки |k_угол| ≤ v умножаются на
    1 / (1 + (k_x / u)^(2n)), k_x — номер частоты по x, u = ширина с дополнением / period, и берётся действительная
    часть."""
    from .gpu import get_xp  # noqa: WPS433
    xp = xp or get_xp()
    s = xp.asarray(x, dtype=xp.float32)
    ns, na, w = s.shape
    pad = min(150, int(0.1 * min(na, w)))
    if pad > 0:
        top = xp.broadcast_to(s.mean(axis=1, keepdims=True), (ns, pad, w))
        s = xp.concatenate([top, s, top], axis=1)
        s = xp.concatenate([xp.repeat(s[:, :, :1], pad, axis=2), s, xp.repeat(s[:, :, -1:], pad, axis=2)], axis=2)
    N, M = s.shape[1], s.shape[2]
    F = xp.fft.fft2(s, axes=(1, 2))
    k = xp.fft.fftfreq(M) * M
    u = M / float(period)
    win = (1.0 / (1.0 + (k / u) ** (2 * int(n)))).astype(xp.complex64)
    v = int(max(0, min(v, N // 2 - 1)))
    F[:, :v + 1, :] *= win
    if v > 0:
        F[:, N - v:, :] *= win
    out = xp.fft.ifft2(F, axes=(1, 2)).real
    return xp.ascontiguousarray(out[:, pad:N - pad, pad:M - pad] if pad > 0 else out).astype(xp.float32)


def shift_angles(x, d, xp=None):
    """Сдвиг каждой проекции слоя (s, углы, w) вдоль x на d[угол] пикселей (d > 0 — вправо, как ndimage.shift) в
    частотной области; края продолжаются крайними значениями (дополнение до степени двойки ≥ 2w)."""
    from .gpu import get_xp  # noqa: WPS433
    xp = xp or get_xp()
    a = xp.asarray(x, dtype=xp.float32)
    w = a.shape[-1]
    size = 1 << int(math.ceil(math.log2(2 * w)))
    left = (size - w) // 2
    f = xp.fft.rfft(xp.pad(a, [(0, 0), (0, 0), (left, size - w - left)], mode='edge'), axis=-1)
    kk = xp.fft.rfftfreq(size)
    dd = xp.asarray(np.asarray(d, dtype='float64'))
    f *= xp.exp(-2j * math.pi * dd[None, :, None] * kk[None, None, :]).astype(xp.complex64)
    return xp.fft.irfft(f, n=size, axis=-1)[..., left:left + w].astype(xp.float32)


def _clean(x, params: Dict[str, Any], xp):
    out = x
    if params.get('vo'):
        out = _vo(out, params['vo'], xp)
    if params.get('fft'):
        f = params['fft']
        out = stripe_fft(out, f['period'], f['n'], f['v'], xp=xp)
    return out


def apply(sino_rows, params: Optional[Dict[str, Any]], xp=None, frame_dx=None):
    """Обработать слой синограмм (s, n, w) → той же формы. params=None — вернуть как есть. frame_dx — сдвиги кадров по
    x (n,), уже внесённые в синограмму: при ``params['unshift']`` поправка считается на проекциях, возвращённых на
    место детектора (см. модуль)."""
    if not params:
        return sino_rows
    from .gpu import get_xp  # noqa: WPS433
    xp = xp or get_xp()
    x = xp.asarray(sino_rows, dtype=xp.float32)
    d = None
    if params.get('unshift') and frame_dx is not None:
        d = np.asarray(frame_dx, dtype='float64')
        if d.shape != (x.shape[1],):
            raise ValueError('frame_dx: {} значений на {} проекций'.format(d.shape, x.shape[1]))
        if not np.any(np.abs(d) > 1e-3):
            d = None
    if d is None:
        return _clean(x, params, xp)
    base = shift_angles(x, -d, xp)                                    # проекции на месте детектора
    return x + shift_angles(_clean(base, params, xp) - base, d, xp)
