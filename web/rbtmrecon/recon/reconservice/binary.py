"""Бинарные ответы для просмотрщика.

Тело — сырые little-endian байты массива (C-порядок). Заголовки:
- ``X-Shape``  — размеры через запятую (``h,w`` или ``k,h,w``);
- ``X-Dtype``  — ``uint16`` | ``float32`` | ``uint8``;
- ``X-Scale``, ``X-Offset`` — для квантованных данных: физическое значение = код · scale + offset;
- ``X-Meta``   — JSON (ASCII) со сведениями о запросе: углы, центр, окно, коэффициент уменьшения и т.п.

Превью квантуется в uint16 по окну [lo, hi] (по умолчанию персентили 0,1 и 99,9), значения вне окна
обрезаются: вдвое меньше байт, чем float32, и точности для отображения с запасом.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

import numpy as np
from flask import Response

_UINT16_MAX = 65535


def downsample(arr: np.ndarray, max_px: Optional[int]) -> Tuple[np.ndarray, int]:
    """Уменьшить две последние оси целым коэффициентом (среднее по блокам), чтобы сторона была ≤ max_px.
    Возвращает (массив, коэффициент); края, не кратные коэффициенту, отбрасываются."""
    a = np.asarray(arr)
    if not max_px or max(a.shape[-2:]) <= max_px:
        return a, 1
    f = int(np.ceil(max(a.shape[-2:]) / float(max_px)))
    h, w = a.shape[-2] // f * f, a.shape[-1] // f * f
    v = a[..., :h, :w].reshape(a.shape[:-2] + (h // f, f, w // f, f))
    return v.mean(axis=(-3, -1), dtype='float64').astype('float32'), f


def quantize(arr: np.ndarray, lo: Optional[float] = None, hi: Optional[float] = None
             ) -> Tuple[np.ndarray, float, float]:
    """float → uint16 по окну [lo, hi]: (коды, scale, offset), значение = код · scale + offset."""
    a = np.asarray(arr, dtype='float32')
    finite = a[np.isfinite(a)]
    if lo is None or hi is None:
        if finite.size:
            plo, phi = np.percentile(finite, [0.1, 99.9])
        else:
            plo, phi = 0.0, 1.0
        lo = float(plo) if lo is None else float(lo)
        hi = float(phi) if hi is None else float(hi)
    if not hi > lo:
        hi = lo + 1.0
    scale = (hi - lo) / _UINT16_MAX
    codes = np.clip(np.round((np.nan_to_num(a, nan=lo) - lo) / scale), 0, _UINT16_MAX).astype('<u2')
    return codes, float(scale), float(lo)


def array_response(arr: Any, *, quantized: bool = True, lo: Optional[float] = None, hi: Optional[float] = None,
                   meta: Optional[Dict[str, Any]] = None, max_px: Optional[int] = None,
                   status: int = 200) -> Response:
    """Ответ с массивом. quantized=True — float → uint16 (см. модуль), иначе как есть (float32/uint16/uint8)."""
    from reconengine.gpu import to_numpy  # noqa: WPS433 — cupy → numpy

    a = np.asarray(to_numpy(arr))
    meta = dict(meta or {})
    a, factor = downsample(a, max_px)
    if factor > 1:
        meta['downsample'] = factor
    headers = {}
    if quantized:
        a, scale, offset = quantize(a, lo, hi)
        headers['X-Scale'] = repr(scale)
        headers['X-Offset'] = repr(offset)
        dtype = 'uint16'
    else:
        if a.dtype == np.float64:
            a = a.astype('float32')
        if a.dtype not in (np.dtype('float32'), np.dtype('uint16'), np.dtype('uint8')):
            raise ValueError('неподдерживаемый тип массива: {}'.format(a.dtype))
        a = a.astype(a.dtype.newbyteorder('<'), copy=False)
        dtype = a.dtype.name
    headers['X-Shape'] = ','.join(str(int(s)) for s in a.shape)
    headers['X-Dtype'] = dtype
    headers['X-Meta'] = json.dumps(meta, ensure_ascii=True, separators=(',', ':'), default=_json_default)
    headers['Access-Control-Expose-Headers'] = 'X-Shape, X-Dtype, X-Scale, X-Offset, X-Meta'
    return Response(np.ascontiguousarray(a).tobytes(), status=status, mimetype='application/octet-stream',
                    headers=headers)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError('{} не сериализуется в JSON'.format(type(o).__name__))


def decode(body: bytes, headers: Dict[str, str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Обратное преобразование (для тестов и клиентов на Python): (массив в физических единицах, meta)."""
    shape = tuple(int(s) for s in headers['X-Shape'].split(',') if s)
    dtype = np.dtype(headers['X-Dtype']).newbyteorder('<')
    a = np.frombuffer(body, dtype=dtype).reshape(shape)
    if 'X-Scale' in headers:
        a = a.astype('float32') * np.float32(float(headers['X-Scale'])) + np.float32(float(headers['X-Offset']))
    return a, json.loads(headers.get('X-Meta') or '{}')
