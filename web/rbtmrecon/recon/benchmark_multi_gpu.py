"""benchmark_multi_gpu.py
========================
Сравнение производительности однопроцессорной и многопроцессорной (multi-GPU)
реконструкции томографии на 3D-фантоме Шеппа-Логана.

Использование
-------------
    python benchmark_multi_gpu.py [--size 256] [--angles 200] [--num-gpus 2]

Аргументы
---------
--size      : линейный размер фантома (default: 256; для полного теста: 1000)
--angles    : количество углов проекций в диапазоне [0°, 180°) (default: 200)
--num-gpus  : количество GPU для multi-GPU теста (default: 2)
--no-cgls   : использовать только FBP_CUDA (быстрее, но ниже качество)
--out-dir   : каталог для PNG-срезов и CSV-отчёта (default: ./benchmark_out)

Метрики
-------
* RMSE  — корень среднеквадратической ошибки по всему объёму
* SSIM  — структурное сходство по 3 центральным срезам (XY, XZ, YZ), среднее
* Speedup = t_single / t_multi

ВАЖНО: Запускайте из каталога rbtm-recon/web/rbtmrecon/recon/,
       где находятся tomotools2.py и пакет tomo/.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Защита точки входа — обязательна для spawn-процессов на Windows
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # При spawn-старте дочерние процессы импортируют этот модуль, но
    # __name__ != '__main__', поэтому основной код не выполняется дважды.
    pass


def _build_sinogram(phantom: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Создаёт синограмму (H, N_angles, W) методом прямого проецирования ASTRA.

    phantom   : shape (H, W, W), float32
    angles_deg: shape (N_angles,), градусы
    """
    import tomo.recon.astra_utils as astra_utils

    H, W, _ = phantom.shape
    N = len(angles_deg)
    sinogram = np.empty((H, N, W), dtype='float32')

    print(f"  Прямое проецирование фантома {H}×{W}×{W} при {N} углах…")
    t0 = time.perf_counter()
    for i in range(H):
        # astra_fp_2d_parallel: (W, W) → (N_angles, W)
        sinogram[i] = astra_utils.astra_fp_2d_parallel(phantom[i], angles_deg)
        if i % max(1, H // 10) == 0:
            pct = 100 * i / H
            print(f"    проекция среза {i}/{H} ({pct:.0f}%)")
    dt = time.perf_counter() - t0
    print(f"  Прямое проецирование завершено за {dt:.1f} с")
    return sinogram


def _recon_single_gpu(sinogram: np.ndarray, angles_deg: np.ndarray,
                       pixel_size: float, use_cgls: bool,
                       gpu_id: int = 0) -> tuple[np.ndarray, float]:
    """Однопроцессорная реконструкция на одном GPU.

    Возвращает (rec_vol, elapsed_seconds).
    """
    import tomo.recon.astra_utils as astra_utils

    H, N, W = sinogram.shape
    rec_vol = np.empty((H, W, W), dtype='float32')

    method = [['FBP_CUDA'], ['CGLS_CUDA', 10]] if use_cgls else [['FBP_CUDA']]

    print(f"  Однопроцессорная реконструкция {H} срезов на GPU {gpu_id}…")
    t0 = time.perf_counter()
    for i in range(H):
        rec = astra_utils.astra_recon_2d_parallel(
            sinogram[i], angles_deg, method, gpu_id=gpu_id
        )
        rec_vol[i] = rec / pixel_size
        if i % max(1, H // 10) == 0:
            pct = 100 * i / H
            print(f"    срез {i}/{H} ({pct:.0f}%)")
    elapsed = time.perf_counter() - t0
    print(f"  Однопроцессорная реконструкция завершена за {elapsed:.1f} с")
    return rec_vol, elapsed


def _recon_multi_gpu(sinogram: np.ndarray, angles_deg: np.ndarray,
                      pixel_size: float, num_gpus: int) -> tuple[np.ndarray, float]:
    """Multi-GPU реконструкция через recon_volume_multi_gpu (shared memory).

    Возвращает (rec_vol, elapsed_seconds).
    """
    # Импорт здесь, чтобы не мешать spawn-процессам
    from tomotools2 import recon_volume_multi_gpu

    H, N, W = sinogram.shape
    rec_vol = np.empty((H, W, W), dtype='float32')

    print(f"  Multi-GPU реконструкция {H} срезов на {num_gpus} GPU…")
    t0 = time.perf_counter()
    recon_volume_multi_gpu(sinogram, angles_deg, pixel_size, rec_vol, num_gpus=num_gpus)
    elapsed = time.perf_counter() - t0
    print(f"  Multi-GPU реконструкция завершена за {elapsed:.1f} с")
    return rec_vol, elapsed


# ---------------------------------------------------------------------------
# Метрики качества
# ---------------------------------------------------------------------------

def compute_rmse(rec: np.ndarray, ref: np.ndarray) -> float:
    """Корень среднеквадратической ошибки по всему объёму."""
    diff = rec.astype('float64') - ref.astype('float64')
    return float(np.sqrt(np.mean(diff ** 2)))


def compute_ssim_slice(rec_slice: np.ndarray, ref_slice: np.ndarray) -> float:
    """SSIM для одного 2D среза (реализация без scikit-image)."""
    r = rec_slice.astype('float64')
    t = ref_slice.astype('float64')

    mu_r, mu_t = r.mean(), t.mean()
    sig_r  = r.std()
    sig_t  = t.std()
    sig_rt = np.mean((r - mu_r) * (t - mu_t))

    # Константы стабилизации (L = диапазон значений)
    L  = max(t.max() - t.min(), 1e-8)
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    num   = (2 * mu_r * mu_t + C1) * (2 * sig_rt + C2)
    denom = (mu_r ** 2 + mu_t ** 2 + C1) * (sig_r ** 2 + sig_t ** 2 + C2)
    return float(num / denom)


def compute_ssim_volume(rec: np.ndarray, ref: np.ndarray) -> float:
    """Среднее SSIM по трём центральным срезам (XY, XZ, YZ)."""
    H, W0, W1 = rec.shape
    cx, cy, cz = H // 2, W0 // 2, W1 // 2
    ssim_xy = compute_ssim_slice(rec[cx],    ref[cx])
    ssim_xz = compute_ssim_slice(rec[:, cy], ref[:, cy])
    ssim_yz = compute_ssim_slice(rec[:, :, cz], ref[:, :, cz])
    return float(np.mean([ssim_xy, ssim_xz, ssim_yz]))


# ---------------------------------------------------------------------------
# Сохранение результатов
# ---------------------------------------------------------------------------

def save_slices_png(rec_single: np.ndarray, rec_multi: np.ndarray,
                     phantom: np.ndarray, out_dir: Path) -> None:
    """Сохраняет PNG с тремя центральными срезами для каждого из трёх объёмов."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib не установлен, PNG не будут сохранены")
        return

    H = phantom.shape[0]
    W = phantom.shape[1]
    cx, cy, cz = H // 2, W // 2, W // 2

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    titles_row = ['Фантом (эталон)', 'Single-GPU', 'Multi-GPU']
    slices_col = [
        ('XY (z=центр)',   [p[cx]          for p in (phantom, rec_single, rec_multi)]),
        ('XZ (y=центр)',   [p[:, cy]       for p in (phantom, rec_single, rec_multi)]),
        ('YZ (x=центр)',   [p[:, :, cz]    for p in (phantom, rec_single, rec_multi)]),
    ]

    for col_idx, (col_title, imgs) in enumerate(slices_col):
        vmin = float(np.percentile(imgs[0], 1))
        vmax = float(np.percentile(imgs[0], 99))
        for row_idx, (title, img) in enumerate(zip(titles_row, imgs)):
            ax = axes[row_idx][col_idx]
            im = ax.imshow(img, cmap='viridis', vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f'{title}\n{col_title}')
            ax.axis('off')

    plt.tight_layout()
    out_path = out_dir / 'benchmark_slices.png'
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"  PNG срезов сохранён: {out_path}")


def save_csv_report(results: dict, out_dir: Path) -> None:
    """Сохраняет таблицу результатов в CSV."""
    out_path = out_dir / 'benchmark_results.csv'
    fieldnames = list(results.keys())
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(results)
    print(f"  CSV-отчёт сохранён: {out_path}")


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def run_benchmark(size: int = 256,
                   n_angles: int = 200,
                   num_gpus: int = 2,
                   use_cgls: bool = True,
                   out_dir: str = './benchmark_out') -> None:
    """Запускает полный бенчмарк и выводит результаты.

    Параметры
    ----------
    size      : линейный размер фантома (size × size × size)
    n_angles  : количество углов в [0°, 180°)
    num_gpus  : количество GPU для multi-GPU теста
    use_cgls  : если True — FBP_CUDA + CGLS_CUDA×10, иначе только FBP_CUDA
    out_dir   : каталог для артефактов
    """
    from tomopy.misc.phantom import shepp3d

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pixel_size = 9e-3  # мм

    print("=" * 60)
    print(f"BENCHMARK: size={size}, angles={n_angles}, num_gpus={num_gpus}")
    print(f"           use_cgls={use_cgls}, pixel_size={pixel_size} мм")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Генерация фантома
    # ------------------------------------------------------------------
    print("\n[1/5] Генерация фантома Шеппа-Логана…")
    t0 = time.perf_counter()
    phantom_raw = shepp3d(size)                     # (size, size, size), float64
    phantom = phantom_raw.astype('float32')
    del phantom_raw
    mem_gb = phantom.nbytes / 1024**3
    print(f"  Фантом: shape={phantom.shape}, dtype={phantom.dtype}, "
          f"размер={mem_gb:.2f} GB, за {time.perf_counter()-t0:.1f} с")

    angles_deg = np.linspace(0, 180, n_angles, endpoint=False).astype('float32')

    # ------------------------------------------------------------------
    # 2. Прямое проецирование (создание синограммы)
    # ------------------------------------------------------------------
    print("\n[2/5] Прямое проецирование…")
    sinogram = _build_sinogram(phantom, angles_deg)
    sino_gb = sinogram.nbytes / 1024**3
    print(f"  Синограмма: shape={sinogram.shape}, размер={sino_gb:.2f} GB")

    # ------------------------------------------------------------------
    # 3. Single-GPU реконструкция
    # ------------------------------------------------------------------
    print("\n[3/5] Single-GPU реконструкция…")
    rec_single, t_single = _recon_single_gpu(
        sinogram, angles_deg, pixel_size, use_cgls, gpu_id=0
    )

    # ------------------------------------------------------------------
    # 4. Multi-GPU реконструкция
    # ------------------------------------------------------------------
    print(f"\n[4/5] Multi-GPU реконструкция ({num_gpus} GPU)…")
    if num_gpus < 2:
        print("  ПРОПУЩЕНО (num_gpus < 2)")
        rec_multi  = rec_single
        t_multi    = t_single
        speedup    = 1.0
    else:
        rec_multi, t_multi = _recon_multi_gpu(sinogram, angles_deg, pixel_size, num_gpus)
        speedup = t_single / t_multi if t_multi > 0 else float('inf')

    # ------------------------------------------------------------------
    # 5. Метрики качества
    # ------------------------------------------------------------------
    print("\n[5/5] Вычисление метрик качества…")

    rmse_single = compute_rmse(rec_single, phantom)
    rmse_multi  = compute_rmse(rec_multi,  phantom)
    ssim_single = compute_ssim_volume(rec_single, phantom)
    ssim_multi  = compute_ssim_volume(rec_multi,  phantom)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"  {'Параметр':<30} {'Single-GPU':>12} {'Multi-GPU':>12}")
    print(f"  {'-'*54}")
    print(f"  {'Время реконструкции, с':<30} {t_single:>12.1f} {t_multi:>12.1f}")
    print(f"  {'Ускорение':<30} {'—':>12} {speedup:>12.2f}x")
    print(f"  {'RMSE':<30} {rmse_single:>12.6f} {rmse_multi:>12.6f}")
    print(f"  {'SSIM (среднее 3 срезов)':<30} {ssim_single:>12.4f} {ssim_multi:>12.4f}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Сохранение артефактов
    # ------------------------------------------------------------------
    results = {
        'size':          size,
        'n_angles':      n_angles,
        'num_gpus':      num_gpus,
        'use_cgls':      use_cgls,
        't_single_s':    round(t_single, 3),
        't_multi_s':     round(t_multi,  3),
        'speedup':       round(speedup,  3),
        'rmse_single':   round(rmse_single, 8),
        'rmse_multi':    round(rmse_multi,  8),
        'ssim_single':   round(ssim_single, 6),
        'ssim_multi':    round(ssim_multi,  6),
    }

    print()
    save_csv_report(results, out_path)
    save_slices_png(rec_single, rec_multi, phantom, out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Бенчмарк single-GPU vs multi-GPU реконструкции томографии'
    )
    parser.add_argument('--size',     type=int,  default=256,
                        help='Линейный размер фантома (default: 256)')
    parser.add_argument('--angles',   type=int,  default=200,
                        help='Количество углов (default: 200)')
    parser.add_argument('--num-gpus', type=int,  default=2,
                        help='Количество GPU (default: 2)')
    parser.add_argument('--no-cgls',  action='store_true',
                        help='Только FBP_CUDA (без CGLS_CUDA)')
    parser.add_argument('--out-dir',  type=str,  default='./benchmark_out',
                        help='Каталог для результатов (default: ./benchmark_out)')
    return parser.parse_args()


if __name__ == '__main__':
    # Обязательная защита для корректной работы multiprocessing.spawn на Windows
    args = _parse_args()
    run_benchmark(
        size=args.size,
        n_angles=args.angles,
        num_gpus=args.num_gpus,
        use_cgls=not args.no_cgls,
        out_dir=args.out_dir,
    )
