"""Восстановление срезов по синограммам (параллельный пучок).

Синограмма строки — (n_углов, w), ось вращения в столбце (w−1)/2 (после axis.align_rows).
Бэкенды:
- 'astra' — ``tomo.recon.astra_utils.astra_recon_2d_parallel(sino, angles, [['FBP_CUDA']])``, как в
  ``tomotools4.recon_2d_parallel`` (те же соглашения об углах и ориентации);
- 'cpu'   — ``skimage.transform.iradon`` (фильтр ramp), для тестов и машин без GPU; ориентация и масштаб
  приводятся к astra (проверяется на сервере).
Результат делится на pixel_size (мм) → коэффициент ослабления в 1/мм, как сейчас.

Выбор углов (``select_angles``):
- 'first_180'   — углы с (a − a.min) < 180 (текущее поведение);
- 'full_halves' — наибольшее кратное 180° число полуоборотов от начала скана (для 0–360° — все углы);
  результат FBP масштабируется на 180 / (180·k), чтобы значения не зависели от числа полуоборотов.
"""
from __future__ import annotations

import numpy as np

ANGLE_MODES = ('first_180', 'full_halves')


def select_angles(angles_deg: np.ndarray, mode: str = 'first_180') -> np.ndarray:
    """Булева маска выбранных углов и (через атрибут результата не нужно) — см. halves_count."""
    raise NotImplementedError


def halves_count(angles_deg: np.ndarray, mode: str = 'first_180') -> int:
    """Число полуоборотов, которое покрывают выбранные углы (1 для first_180)."""
    raise NotImplementedError


def recon_slice(sino: np.ndarray, angles_deg: np.ndarray, pixel_size: float,
                backend: str = 'auto', angle_mode: str = 'first_180') -> np.ndarray:
    """Срез (w, w) float32 по синограмме (n, w). backend: 'auto' (astra, если доступна, иначе cpu), 'astra', 'cpu'."""
    raise NotImplementedError


def recon_rows(sino_rows: np.ndarray, angles_deg: np.ndarray, pixel_size: float,
               backend: str = 'auto', angle_mode: str = 'first_180') -> np.ndarray:
    """Слой срезов: sino_rows (s, n, w) → (s, w, w) float32."""
    raise NotImplementedError
