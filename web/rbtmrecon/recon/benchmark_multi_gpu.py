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

    Пустые срезы (норма синограммы < порога) реконструируются только через FBP,
    чтобы избежать NaN в CGLS при нулевом правом векторе.
    """
    import tomo.recon.astra_utils as astra_utils

    H, N, W = sinogram.shape
    rec_vol = np.zeros((H, W, W), dtype='float32')

    method_full = [['FBP_CUDA'], ['CGLS_CUDA', 10]] if use_cgls else [['FBP_CUDA']]
    method_fbp  = [['FBP_CUDA']]

    # Порог: срез считается пустым, если L2-норма синограммы < 1e-6 от максимума
    sino_norms  = np.linalg.norm(sinogram.reshape(H, -1), axis=1)  # (H,)
    norm_thresh = sino_norms.max() * 1e-6

    print(f"  Однопроцессорная реконструкция {H} срезов на GPU {gpu_id}…")
    print(f"  Порог пустого среза: {norm_thresh:.4g}, "
          f"пустых: {int((sino_norms <= norm_thresh).sum())}/{H}")
    t0 = time.perf_counter()
    for i in range(H):
        method = method_fbp if sino_norms[i] <= norm_thresh else method_full
        rec = astra_utils.astra_recon_2d_parallel(
            sinogram[i], angles_deg, method, gpu_id=gpu_id
        )
        rec_vol[i] = rec / pixel_size
        if i % max(1, H // 10) == 0:
            pct = 100 * i / H
            print(f"    срез {i}/{H} ({pct:.0f}%), "
                  f"sino_norm={sino_norms[i]:.3g}, "
                  f"rec min={rec_vol[i].min():.3g} max={rec_vol[i].max():.3g}")
    elapsed = time.perf_counter() - t0

    nan_count = int(np.isnan(rec_vol).sum())
    inf_count = int(np.isinf(rec_vol).sum())
    if nan_count or inf_count:
        warnings.warn(f"Single-GPU: {nan_count} NaN, {inf_count} Inf — заменены нулями")
        np.nan_to_num(rec_vol, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Однопроцессорная реконструкция завершена за {elapsed:.1f} с")
    return rec_vol, elapsed


def _recon_multi_gpu(sinogram: np.ndarray, angles_deg: np.ndarray,
                      pixel_size: float, num_gpus: int,
                      use_cgls: bool = True) -> tuple[np.ndarray, float]:
    """Multi-GPU реконструкция через recon_volume_multi_gpu (shared memory + memmap).

    Возвращает (rec_vol, elapsed_seconds).
    """
    # Импорт здесь, чтобы не мешать spawn-процессам
    from tomotools2 import recon_volume_multi_gpu

    H, N, W = sinogram.shape
    rec_vol = np.empty((H, W, W), dtype='float32')

    print(f"  Multi-GPU реконструкция {H} срезов на {num_gpus} GPU "
          f"({'FBP+CGLS' if use_cgls else 'FBP'})…")
    t0 = time.perf_counter()
    recon_volume_multi_gpu(
        sinogram, angles_deg, pixel_size, rec_vol,
        num_gpus=num_gpus, use_cgls=use_cgls,
    )
    elapsed = time.perf_counter() - t0

    nan_count = int(np.isnan(rec_vol).sum())
    inf_count = int(np.isinf(rec_vol).sum())
    if nan_count or inf_count:
        warnings.warn(f"Multi-GPU: {nan_count} NaN, {inf_count} Inf — заменены нулями")
        np.nan_to_num(rec_vol, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Multi-GPU реконструкция завершена за {elapsed:.1f} с")
    return rec_vol, elapsed


# ---------------------------------------------------------------------------
# Метрики качества
# ---------------------------------------------------------------------------

def compute_rmse(rec: np.ndarray, ref: np.ndarray) -> float:
    """Нормализованная RMSE по всему объёму.

    NRMSE = RMSE / (max(ref) - min(ref)), что позволяет сравнивать объёмы
    с разными масштабами (напр., когда pixel_size != 1).
    """
    diff = rec.astype('float64') - ref.astype('float64')
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    signal_range = float(ref.max()) - float(ref.min())
    if signal_range < 1e-12:
        return float('nan')
    return rmse / signal_range


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

# ---------------------------------------------------------------------------
# Собственная реализация фантома Шеппа-Логана (совместима с NumPy 2.0)
# ---------------------------------------------------------------------------

# Параметры эллипсоидов: (ρ, a, b, c, x0, y0, z0, φ_deg)
# Источник: Kak & Slaney "Principles of CT", Table 3.1
_SHEPP_LOGAN_PARAMS = np.array([
    # ρ      a      b      c     x0     y0     z0    φ
    [ 1.0,  0.69,  0.92,  0.9,   0.0,  0.0,  0.0,   0.0],
    [-0.8,  0.6624,0.874, 0.88,  0.0,  0.0,  0.0,   0.0],
    [-0.2,  0.11,  0.31,  0.22,  0.22, 0.0, -0.25, -18.0],
    [-0.2,  0.16,  0.41,  0.28, -0.22, 0.0, -0.25,  18.0],
    [ 0.1,  0.21,  0.25,  0.41,  0.0,  0.35,-0.25,   0.0],
    [ 0.1,  0.046, 0.046, 0.05,  0.0,  0.1, -0.25,   0.0],
    [ 0.1,  0.046, 0.046, 0.05,  0.0, -0.1, -0.25,   0.0],
    [-0.02, 0.046, 0.023, 0.05, -0.08,-0.605, 0.0,   0.0],
    [-0.02, 0.023, 0.023, 0.02,  0.06,-0.605, 0.0,   0.0],
    [ 0.02, 0.023, 0.046, 0.02,  0.06,-0.605, 0.0,   0.0],
], dtype='float64')


def _make_shepp_logan_3d(size: int) -> np.ndarray:
    """Генерирует 3D фантом Шеппа-Логана размером (size, size, size).

    Совместима с NumPy 2.0 (не использует np.lib.index_tricks).
    Возвращает float32 массив.

    Оптимизации по памяти и скорости
    ---------------------------------
    * Послойный обход по Z: единовременно создаются только два 2D-массива
      yy и xx размером size×size×4 байт = ~8 MB при size=1000.
      Полные 3D meshgrid потребовали бы 3 × 1000³ × 8 байт ≈ 24 GB.
    * Результат накапливается сразу в float32.
    * Early-exit per ellipsoid: если zr² > 1, данный эллипсоид
      пропускается без создания 2D-маски.
    * Предвычисляются sin/cos и параметры для каждого эллипсоида.
    """
    half = (size - 1) / 2.0
    coords = (np.arange(size, dtype='float32') - half) / half   # [-1, 1]

    # 2D сетки y и x — создаются один раз, переиспользуются для всех z-срезов
    yy, xx = np.meshgrid(coords, coords, indexing='ij')          # (size, size) float32

    # Предвычисляем параметры эллипсоидов один раз
    ellipsoids = []
    for row in _SHEPP_LOGAN_PARAMS:
        rho          = float(row[0])
        a, b, c      = float(row[1]), float(row[2]), float(row[3])
        x0, y0, z0   = float(row[4]), float(row[5]), float(row[6])
        phi          = np.deg2rad(float(row[7]))
        cos_p, sin_p = float(np.cos(phi)), float(np.sin(phi))
        ellipsoids.append((rho, a, b, c, x0, y0, z0, cos_p, sin_p))

    phantom   = np.zeros((size, size, size), dtype='float32')
    slice_buf = np.empty((size, size), dtype='float32')

    for zi, zval in enumerate(coords):
        slice_buf[:] = 0.0
        for rho, a, b, c, x0, y0, z0, cos_p, sin_p in ellipsoids:
            zr2 = ((float(zval) - z0) / c) ** 2
            if zr2 > 1.0:              # z вне эллипсоида — пропускаем
                continue
            dx = xx - x0
            dy = yy - y0
            xr = (cos_p * dx + sin_p * dy) / a
            yr = (-sin_p * dx + cos_p * dy) / b
            mask = xr * xr + yr * yr <= (1.0 - zr2)
            slice_buf[mask] += rho
        np.clip(slice_buf, 0, None, out=slice_buf)
        phantom[zi] = slice_buf

    return phantom


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
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Для синтетического бенчмарка используем pixel_size=1.0:
    # ASTRA astra_fp/FBP работает в безразмерных единицах пикселей,
    # деление на реальный pixel_size (9e-3 мм) даёт значения ~100x больше фантома
    # и может приводить к расходимости CGLS и NaN в метриках.
    pixel_size = 1.0

    print("=" * 60)
    print(f"BENCHMARK: size={size}, angles={n_angles}, num_gpus={num_gpus}")
    print(f"           use_cgls={use_cgls}, pixel_size={pixel_size} мм")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Генерация фантома
    # ------------------------------------------------------------------
    print("\n[1/5] Генерация фантома Шеппа-Логана…")
    t0 = time.perf_counter()
    phantom = _make_shepp_logan_3d(size)            # (size, size, size), float32
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
        rec_multi, t_multi = _recon_multi_gpu(sinogram, angles_deg, pixel_size, num_gpus,
                                               use_cgls=use_cgls)
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
