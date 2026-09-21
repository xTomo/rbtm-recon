"""Заглушки GPU/ноутбучных зависимостей для тестов recon.

В тестовом окружении нет cupy / astra / tomopy / ipywidgets, поэтому до импорта
модулей из ``web/rbtmrecon/recon`` мы подставляем в ``sys.modules`` минимальные
заглушки, реализованные поверх numpy/scipy/matplotlib.

Также добавляет каталог ``web/rbtmrecon/recon`` в ``sys.path``, чтобы
``import tomotools4`` / ``import hdf5_v2`` работали так же, как в контейнере.
"""
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.dirname(HERE)
RECON_DIR = os.path.join(WEB_DIR, 'rbtmrecon', 'recon')
WEBRECON_DIR = os.path.join(WEB_DIR, 'rbtmwebrecon', 'webrecon')
SCRIPTS_DIR = os.path.join(WEB_DIR, 'scripts')

for _p in (RECON_DIR, WEBRECON_DIR, SCRIPTS_DIR, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _make_module(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


# --- cupy → numpy -----------------------------------------------------------
if 'cupy' not in sys.modules:
    _cupy = _make_module('cupy')

    class _FakeGPUArray(np.ndarray):
        """np.ndarray с методом .get(), как у cupy.ndarray."""

        def get(self):
            return np.asarray(self).view(np.ndarray)

    def _as_gpu(a, *args, **kwargs):
        return np.asarray(a, *args, **kwargs).view(_FakeGPUArray)

    def _cupy_getattr(item, _np=np):
        try:
            return getattr(_np, item)
        except AttributeError as exc:  # pragma: no cover - диагностика
            raise AttributeError('cupy stub has no {}'.format(item)) from exc

    _cupy.__getattr__ = _cupy_getattr
    _cupy.ndarray = _FakeGPUArray
    _cupy.asnumpy = lambda a: np.asarray(a).view(np.ndarray)
    _cupy.asarray = _as_gpu
    _cupy.asanyarray = _as_gpu

# --- cupyx.scipy.ndimage → scipy.ndimage ------------------------------------
if 'cupyx' not in sys.modules:
    import scipy.ndimage as _ndi

    _cupyx = _make_module('cupyx')
    _cupyx_scipy = _make_module('cupyx.scipy')
    _cupyx_ndi = _make_module('cupyx.scipy.ndimage')

    def _as_gpu_result(fn):
        """cupyx-функции возвращают cupy.ndarray — сохраняем метод .get()."""
        def wrapper(*args, **kwargs):
            res = fn(*args, **kwargs)
            gpu_array = sys.modules['cupy'].ndarray
            return np.asarray(res).view(gpu_array)
        wrapper.__name__ = getattr(fn, '__name__', 'wrapped')
        return wrapper

    _cupyx_ndi.median_filter = _as_gpu_result(_ndi.median_filter)
    _cupyx_ndi.shift = _as_gpu_result(_ndi.shift)
    _cupyx_ndi.rotate = _as_gpu_result(_ndi.rotate)
    _cupyx.scipy = _cupyx_scipy
    _cupyx_scipy.ndimage = _cupyx_ndi

# --- tomo.recon.astra_utils -------------------------------------------------
if 'tomo' not in sys.modules:
    _tomo = _make_module('tomo')
    _tomo.__path__ = []
    _tomo_recon = _make_module('tomo.recon')
    _tomo_recon.__path__ = []
    _astra_utils = _make_module('tomo.recon.astra_utils')

    def _astra_recon_2d_parallel(sino, angles, *args, **kwargs):
        raise RuntimeError('astra is not available in the test environment')

    _astra_utils.astra_recon_2d_parallel = _astra_recon_2d_parallel
    _tomo_recon.astra_utils = _astra_utils
    _tomo.recon = _tomo_recon

    _remove_stripe = _make_module('tomo.remove_stripe')
    _remove_stripe.remove_all_stripe = lambda data, *a, **k: data
    _tomo.remove_stripe = _remove_stripe

# --- tqdm.notebook → tqdm ---------------------------------------------------
if 'tqdm.notebook' not in sys.modules:
    import tqdm as _tqdm

    _tqdm_nb = _make_module('tqdm.notebook')
    _tqdm_nb.tqdm = _tqdm.tqdm
    _tqdm_nb.trange = _tqdm.trange

# --- pylab → matplotlib.pyplot (Agg) ---------------------------------------
if 'pylab' not in sys.modules:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as _plt

    sys.modules['pylab'] = _plt
