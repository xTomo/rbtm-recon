# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# # %load_ext autoreload
# # %autoreload 2

# %%
# automatic mode
# %matplotlib inline

# %%
import logging
import os
import configparser
import time
from pathlib import Path

import pylab as plt
import numpy as np
from plumbum import local
from plumbum.cmd import cp, rm, mv, ls

logger = logging.getLogger()
logger.setLevel(logging.WARN)

logging.getLogger('asyncio').setLevel(logging.CRITICAL)

from tomotools4 import (
    STORAGE_SERVER,
    # I/O
    mkdir_p, get_tomoobject_info, get_experiment_hdf5,
    load_tomo_data, load_recon_config, save_recon_config,
    persistent_array,
    # Advanced experiment support
    is_advanced_experiment, load_tomo_data_advanced, AdvancedTomoData,
    # Preprocessing
    normalize_projections, normalize_projections_with_timeline,
    remove_stripes_sinogram,
    # Repositioning correction
    measure_repositioning_shifts, apply_repositioning_correction,
    analyze_repositioning_accuracy,
    # Source drift analysis
    analyze_source_drift,
    # Axis correction
    find_axis_correction, apply_axis_correction,
    # Reconstruction
    recon_2d_parallel,
    # Visualization
    tqdm, show_exp_data, show_frames_with_border,
    show_center_of_mass, show_reconstruction_cuts,
    preview_axis_correction, create_axis_search_widget,
    disable_output_scrolling,
    # Volume utilities
    save_amira,
)

import ipywidgets

plt.rcParams['figure.facecolor'] = 'white'

# %%
# disable_output_scrolling()

# %%
config = configparser.ConfigParser()
config.read('tomo.ini')
experiment_id = config['SAMPLE']['_id']
data_dir = '/fast/'
storage_dir = '/storage/'
exp_src_dir = '/exp_src'

pixel_size = float(config['SAMPLE'].get('pixel_size', 4.25e-3))
print(f"pixel_size = {pixel_size} mm")

tmp_dir = os.path.join(data_dir, experiment_id)

tomo_info = get_tomoobject_info(experiment_id, STORAGE_SERVER)
tomo_info

# %% [markdown]
# # Загрузка экспериментальных данных

# %%
data_file = get_experiment_hdf5(experiment_id, data_dir,
                                os.path.join(exp_src_dir, experiment_id, 'before_processing'),
                                STORAGE_SERVER)
mkdir_p(tmp_dir)

# Автодетекция типа эксперимента ПЕРЕД загрузкой
_is_advanced = is_advanced_experiment(data_file)
print(f"Формат эксперимента: {'advanced (AdvancedExperiment)' if _is_advanced else 'standard (Experiment)'}")

if _is_advanced:
    # Advanced: загружаем сразу AdvancedTomoData (один проход чтения)
    print("Загружаем расширенные данные (initial/periodic empty, data_check)...")
    adv_data = load_tomo_data_advanced(data_file, tmp_dir)
    print(f"  series_length      = {adv_data.series_length}")
    print(f"  periodic empties   = {len(adv_data.periodic_empties)}")
    print(f"  data frames        = {adv_data.data_images.shape[0]}")
    print(f"  data_check frames  = {adv_data.data_check_images.shape[0]}")
    
    # Для совместимости с остальным кодом
    all_empties = [adv_data.initial_empty] + adv_data.periodic_empties
    empty_beam = np.median(np.stack(all_empties, axis=0), axis=0).astype('float32')
    empty_beam[empty_beam < 1] = 1
    data_images = adv_data.data_images
    data_angles = adv_data.data_angles
    
    print("\nАнализ дрейфа источника:")
    analyze_source_drift(adv_data)
else:
    # Standard: обычная загрузка
    adv_data = None
    empty_beam, data_images, data_angles = load_tomo_data(data_file, tmp_dir)
    empty_beam[empty_beam < 1] = 1

show_exp_data(empty_beam, data_images)

# %% [markdown]
# # Конфигурация реконструкции

# %%
recon_config = load_recon_config(tmp_dir, storage_dir, experiment_id)
recon_config['sample'] = tomo_info
recon_config['pixel_size'] = pixel_size
recon_config

# %% [markdown]
# # ROI — область интереса

# %%
if 'roi' in recon_config:
    print("Read from ini file")
    x_min = int(recon_config['roi']['x_min'])
    x_max = int(recon_config['roi']['x_max'])
    y_min = int(recon_config['roi']['y_min'])
    y_max = int(recon_config['roi']['y_max'])
else:
    x_min, x_max = 0, data_images.shape[2] - 1
    y_min, y_max = 0, data_images.shape[1] - 1

print("x_min, x_max, y_min, y_max = ", x_min, x_max, y_min, y_max)

tmp_img = data_images[::10]
ff = ipywidgets.interact_manual(
    show_frames_with_border,
    data_images=ipywidgets.fixed(tmp_img),
    empty_beam=ipywidgets.fixed(empty_beam),
    data_angles=ipywidgets.fixed(data_angles),
    image_id=ipywidgets.IntSlider(min=0, max=len(tmp_img) - 1, step=1, value=0),
    x_min=ipywidgets.IntSlider(min=0, max=data_images.shape[2], step=1, value=x_min),
    x_max=ipywidgets.IntSlider(min=0, max=data_images.shape[2], step=1, value=x_max),
    y_min=ipywidgets.IntSlider(min=0, max=data_images.shape[1], step=1, value=y_min),
    y_max=ipywidgets.IntSlider(min=0, max=data_images.shape[1], step=1, value=y_max),
)

# %%
try:
    if 'x_min' in ff.widget.kwargs:  # type: ignore[union-attr]
        x_min = ff.widget.kwargs['x_min']  # type: ignore[union-attr]
        x_max = ff.widget.kwargs['x_max']  # type: ignore[union-attr]
        y_min = ff.widget.kwargs['y_min']  # type: ignore[union-attr]
        y_max = ff.widget.kwargs['y_max']  # type: ignore[union-attr]
except AttributeError:
    pass  # ff.widget.kwargs недоступен — используем значения по умолчанию

x_min = int(x_min); x_max = int(x_max)
y_min = int(y_min); y_max = int(y_max)

recon_config['roi'] = {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}

show_frames_with_border(data_images, empty_beam, data_angles, 0, x_min, x_max, y_min, y_max)

# %% [markdown]
# # Коррекция позиционирования объекта (только для Advanced эксперимента)
#
# Измеряем сдвиг позиции объекта после каждого возврата на стол
# (кросс-корреляция data vs data_check при одном угле).
# Коррекция sub-pixel сдвига применяется ПОСЛЕ нормировки.

# %%
checkpoint_angles = np.array([])
shifts_y = np.array([])
shifts_x = np.array([])

if adv_data is not None:
    print("Измерение сдвигов позиционирования...")
    checkpoint_angles, shifts_y, shifts_x = measure_repositioning_shifts(
        adv_data, x_min, x_max, y_min, y_max, debug=True)

    print("\nРезультаты:")
    for k in range(len(shifts_y)):
        print(f"  Checkpoint {k:2d} (угол {checkpoint_angles[k]:6.1f}°): "
              f"shift_y={shifts_y[k]:+.3f} пикс., shift_x={shifts_x[k]:+.3f} пикс.")

# %%
# Визуализация точности позиционирования
if adv_data is not None and len(shifts_y) > 0:
    analyze_repositioning_accuracy(
        adv_data, checkpoint_angles, shifts_y, shifts_x,
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

# %% [markdown]
# # Нормировка проекций

# %%
data_images_crop = data_images[:, y_min:y_max, x_min:x_max]
empty_beam_crop = empty_beam[y_min:y_max, x_min:x_max]
# %xdel data_images

# %%
# Advanced: нормировка с интерполяцией empty между checkpoint-ами.
# Standard: стандартная нормировка единым empty.
use_timeline_normalization = adv_data is not None  # True для advanced
print(f"Нормировка: {'timeline (interpolated empty)' if use_timeline_normalization else 'standard'}")

if use_timeline_normalization:
    normalize_projections_with_timeline(
        data_images_crop, adv_data, x_min, x_max, y_min, y_max)
else:
    normalize_projections(data_images_crop, empty_beam_crop)

# %%
# Применить коррекцию позиционирования к нормированным data_images_crop
# ВАЖНО: выполняется ПОСЛЕ нормировки
apply_repositioning = adv_data is not None and len(shifts_y) > 0  # можно вручную отключить

if apply_repositioning:
    print("Применяем коррекцию позиционирования к нормированным данным...")
    apply_repositioning_correction(
        data_images_crop, adv_data.data_numbers, adv_data, shifts_y, shifts_x)
    print("Коррекция позиционирования применена.")
else:
    print("Коррекция позиционирования пропущена.")

# %% [markdown]
# # Центр масс

# %%
show_center_of_mass(data_images_crop)

# %% [markdown]
# # Автоматический поиск смещения оси

# %%
shift_x, alfa = find_axis_correction(data_images_crop, data_angles)
print(f"shift_x={shift_x:.4f}  alfa={alfa:.4f}")

recon_config['axis_corr'] = {'shift_x': shift_x, 'alfa': alfa}

sinogram_fixed = apply_axis_correction(data_images_crop, shift_x, alfa)

preview_axis_correction(sinogram_fixed, data_angles, remove_rings=True)
manual_axis_search = False

# %% [markdown]
# # Ручной поиск смещения и поворота

# %%
manual_axis_search = True
ui, shift_text, angle_text = create_axis_search_widget(
    sinogram_fixed, data_images_crop, data_angles, shift_x, alfa
)
display(ui)

# %%
if manual_axis_search:
    shift_x, alfa = shift_text.value, angle_text.value
    sinogram_fixed = apply_axis_correction(data_images_crop, shift_x, alfa)
    preview_axis_correction(sinogram_fixed, data_angles)

recon_config['axis_corr'] = {'shift_x': shift_x, 'alfa': alfa}
# %xdel data_images_crop

# %% [markdown]
# # Удаление колец

# %%
remove_stripes_sinogram(sinogram_fixed)
preview_axis_correction(sinogram_fixed, data_angles, remove_rings=False)

# %% [markdown]
# # Реконструкция

# %%
raw_file_name = (f"{tomo_info['specimen']}"
                 f".{sinogram_fixed.shape[0]}_{sinogram_fixed.shape[2]}_{sinogram_fixed.shape[2]}"
                 f".1.raw")
rec_vol, _ = persistent_array(
    os.path.join(tmp_dir, raw_file_name),
    dtype=np.float32, force_create=False,
    shape=(sinogram_fixed.shape[0], sinogram_fixed.shape[2], sinogram_fixed.shape[2]),
)

# %%
t0 = time.time()
print(sinogram_fixed.shape)
t_angles = (data_angles - data_angles.min()) < 180  # убираем углы >180
for i in tqdm(range(sinogram_fixed.shape[0])):
    rec_vol[i] = recon_2d_parallel(sinogram_fixed[i][t_angles], data_angles[t_angles], pixel_size)

rec_vol.flush()
print(time.time() - t0)

# %xdel sinogram_fixed

# %% [markdown]
# # Просмотр срезов

# %%
show_reconstruction_cuts(rec_vol)

# %% [markdown]
# # Сохранение

# %%
save_amira(rec_vol, tmp_dir, tomo_info['specimen'], 1, pixel_size)

# %%
save_amira(rec_vol, tmp_dir, tomo_info['specimen'], 4, pixel_size)

# %%
save_recon_config(recon_config, tmp_dir)

# %%
recon_config

# %%
mkdir_p(os.path.join(storage_dir, experiment_id))

# %%
storage_exp_dir = Path(storage_dir) / experiment_id
tmp_path = Path(tmp_dir)

# cp['tomo.ini', storage_exp_dir]()
with local.env(LD_LIBRARY_PATH=''):
    cp['-rT', tmp_path, storage_exp_dir / 'reconstruction']()
    mv[Path(data_dir) / (experiment_id + '.h5'), storage_dir]()
rm['-rf', tmp_path]()
print(ls['-lhaR', storage_exp_dir]())

# %%
get_ipython().kernel.do_shutdown(restart=False)

# %% [markdown]
# # Changelog:
# * 4.0 (2025.05)
#  - поддержка AdvancedExperiment (advanced=True формат данных)
#  - is_advanced_experiment() — автодетекция формата HDF5
#  - load_tomo_data_advanced() — раздельная загрузка initial/periodic empty и data_check
#  - analyze_source_drift() — визуализация дрейфа рентгеновской трубки
#  - measure_repositioning_shifts() — измерение сдвига позиции объекта после возврата
#  - apply_repositioning_correction() — sub-pixel коррекция до нормировки
#  - analyze_repositioning_accuracy() — визуализация точности позиционирования
#  - normalize_projections_with_timeline() — нормировка с интерполяцией empty
#  - backward compatible: standard эксперимент работает без изменений
# * 3.1 (2025.05)
#  - рефакторинг: весь вычислительный код перенесён в tomotools2.py
#  - ноутбук содержит только параметры и вызовы функций
# * 3.0 (2025.03.31)
#  - back to swap file
#  - grand cleanup
#  - axis search rewrite
