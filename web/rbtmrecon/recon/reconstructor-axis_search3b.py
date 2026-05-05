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
# #jupytext --to notebook reconstructor.py
# manual mode
# #%matplotlib notebook

# automatic mode
# %matplotlib inline

# %%
# Отключаем сворачивание вывода при большом количестве изображений
from IPython.display import display, Javascript
display(Javascript("""
    // Убираем класс у всех уже свёрнутых ячеек
    document.querySelectorAll('.jp-Cell.jp-mod-outputsScrolled').forEach(function(el) {
        el.classList.remove('jp-mod-outputsScrolled');
    });

    // MutationObserver: следим за всеми ячейками и снимаем класс,
    // как только JupyterLab пытается его добавить
    if (window._noScrollObserver) {
        window._noScrollObserver.disconnect();
    }
    window._noScrollObserver = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                var el = mutation.target;
                if (el.classList.contains('jp-mod-outputsScrolled')) {
                    el.classList.remove('jp-mod-outputsScrolled');
                }
            }
        });
    });
    document.querySelectorAll('.jp-Cell').forEach(function(cell) {
        window._noScrollObserver.observe(cell, { attributes: true, attributeFilter: ['class'] });
    });
"""))

# %%
import logging

logger = logging.getLogger()
logger.setLevel(logging.WARN)
import time
import os
import configparser

import pylab as plt
import numpy as np

import scipy.ndimage as ndi

# import imreg_dft as ird
# import my_imreg_dft as ird

from tomotools2 import (STORAGE_SERVER, safe_median, recon_2d_parallel, get_tomoobject_info, get_experiment_hdf5,
                       mkdir_p, show_exp_data, load_tomo_data, tqdm, persistent_array, get_angles_at_180_deg, save_amira, show_frames_with_border,
                       preview_axis_correction, reshape_volume)

import ipywidgets

plt.rcParams['figure.facecolor'] = 'white'

# %%
# # settings for docker

config = configparser.ConfigParser()
config.read('tomo.ini')
experiment_id = config['SAMPLE']['_id']
data_dir = '/fast/'
storage_dir = '/storage/'
exp_src_dir = '/exp_src'

tmp_dir = os.path.join(data_dir, experiment_id)

tomo_info = get_tomoobject_info(experiment_id, STORAGE_SERVER)
tomo_info

# %%
pixel_size = 9.0e-3 # pixel size in mm

# %%
if os.path.exists("rec_config.ini"): # in current dir
    config_file = "rec_config.ini"
elif os.path.exists(os.path.join(tmp_dir, 'rec_config.ini')):  # in tmp dir
    config_file = os.path.join(tmp_dir, 'rec_config.ini')
elif os.path.exists(os.path.join(storage_dir, experiment_id, 'reconstruction', 'rec_config.ini')):
    config_file = os.path.join(storage_dir, experiment_id, 'reconstruction', 'rec_config.ini')
else:
    config_file = None

if config_file is not None:
    recon_conf = configparser.ConfigParser()
    recon_conf.read(config_file)
    recon_config = recon_conf._sections
    del recon_conf
else:
    recon_config = {}

recon_config['sample'] = tomo_info
recon_config['pixel_size'] = pixel_size
recon_config

# %% [markdown]
# # Loading experimental data

# %%
data_file = get_experiment_hdf5(experiment_id, data_dir,
                                os.path.join(exp_src_dir, experiment_id, 'before_processing'),
                                STORAGE_SERVER)

mkdir_p(tmp_dir)

empty_beam, data_images, data_angles = load_tomo_data(data_file, tmp_dir)
empty_beam[empty_beam<1] = 1

show_exp_data(empty_beam, data_images)

# %%
if 'roi' in recon_config:
    print("Read from ini file")
    x_min, x_max, y_min, y_max = (int(recon_config['roi']['x_min']),
                                  int(recon_config['roi']['x_max']),
                                  int(recon_config['roi']['y_min']),
                                  int(recon_config['roi']['y_max']))
else:
    x_min, x_max, y_min, y_max = 0, data_images.shape[2]-1, 0, data_images.shape[1]-1

print("x_min, x_max, y_min, y_max = ", x_min, x_max, y_min, y_max)

tmp_img = data_images[::100]
ff = ipywidgets.interact_manual(show_frames_with_border, data_images=ipywidgets.fixed(tmp_img),
                                empty_beam=ipywidgets.fixed(empty_beam),
                                data_angles=ipywidgets.fixed(data_angles),
                                image_id=ipywidgets.IntSlider(min=0, max=len(tmp_img) - 1, step=1, value=0),
                                x_min=ipywidgets.IntSlider(min=0, max=data_images.shape[2], step=1, value=x_min),
                                x_max=ipywidgets.IntSlider(min=0, max=data_images.shape[2], step=1, value=x_max),
                                y_min=ipywidgets.IntSlider(min=0, max=data_images.shape[1], step=1, value=y_min),
                                y_max=ipywidgets.IntSlider(min=0, max=data_images.shape[1], step=1, value=y_max)
                                )

# %%
try:
    if 'x_min' in ff.widget.kwargs:
        x_min = ff.widget.kwargs['x_min']
        x_max = ff.widget.kwargs['x_max']
        y_min = ff.widget.kwargs['y_min']
        y_max = ff.widget.kwargs['y_max']
except AttributeError:
    pass  # ff.widget.kwargs недоступен — используем значения по умолчанию
    
x_min = int(x_min); x_max = int(x_max); y_min = int(y_min); y_max = int(y_max); 

recon_config['roi'] = {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max};

show_frames_with_border(data_images, empty_beam, data_angles, 0, x_min, x_max, y_min, y_max)

# %%
data_images_crop = data_images[:, y_min:y_max, x_min:x_max]
empty_beam_crop = empty_beam[y_min:y_max, x_min:x_max]
# %xdel data_images

# don't check non unique files
# from tomotools import group_data
# uniq_data_images, uniq_angles = group_data(data_images_crop, data_angles, tmp_dir)
# uniq_data_images, uniq_angles = data_images_crop, data_angles[()]

#nornalize images
te = empty_beam_crop
te[te < 1] = 1
log_te = np.log(te)
# for di in tqdm(range(uniq_data_images.shape[0])):
for di in tqdm(range(data_images_crop.shape[0])):
    td = data_images_crop[di]
    td[td < 1] = 1

    d = log_te - np.log(td)
    d = safe_median(d)
    d[d < 0] = 0
    data_images_crop[di] = d

# %% [markdown]
# # Find center of the mass

# %%
cxy = [ndi.center_of_mass(data_images_crop[i]) for i in range(data_images_crop.shape[0])]
cxy = np.asarray(cxy)

plt.figure(figsize=(6,6))
plt.scatter(cxy[:,1],cxy[:,0], c=range(cxy.shape[0]), cmap='viridis')
plt.grid()
plt.show()

# %% [markdown]
# # Автоматический поиск смещения

# %%
from tomo.remove_stripe import remove_stripe_ti, remove_all_stripe
import cupy as cp
import cupyx.scipy.ndimage as cndi

def transform_image(im, shift_x, angle):
    imcu = cp.asarray(im)
    imcu = cndi.shift(imcu, [0, shift_x], order=3, mode='nearest')
    imcu = cndi.rotate(imcu, angle, order=3, reshape=False, mode='nearest')
    return imcu.get()

from scipy import optimize

position_0, position_180 = get_angles_at_180_deg(data_angles)

data_0_orig = data_images_crop[position_0[0]]
data_180_orig = np.fliplr(data_images_crop[position_180[0]])

def calc_grad(im):
    grad = np.gradient(im)
    return((grad[0]**2+grad[1]**2))**0.5
    
def loss(im):
    res = (im**2).sum()
    return res

def objective_function(shift_angle, img0, img1):
    shift, angle = shift_angle
    tmp_im = transform_image(img0, shift, angle) - transform_image(img1, -shift, -angle)
    return loss(tmp_im)
    

im0 = data_0_orig / (data_0_orig**2).sum()**0.5
im1 = data_180_orig / (data_180_orig**2).sum()**0.5

initial_shift = (ndi.center_of_mass(im0)[0] - ndi.center_of_mass(im1)[0]) / 2

result = optimize.minimize(
    objective_function,
    np.array([initial_shift,0]),
    args=(im0, im1),
    method='Powell',
    options = {'return_all': True}
)

shift_x, angle = result.x

print(shift_x,  angle)

tmp_im = transform_image(im0, shift_x, angle) - transform_image(im1, -shift_x, -angle)

plt.figure()
plt.imshow(tmp_im, cmap=plt.cm.seismic)
plt.colorbar()
plt.show()



shift_x, alfa = shift_x, angle
tr_dict = {"scale": 1, "angle": alfa, "tvec": (0, shift_x)}

recon_config['axis_corr'] = {'shift_x': shift_x,
                             'alfa': alfa,
                            #  'angle_180': data_angles[position_180],
                            #  'angle_0': data_angles[position_0]
                             }

sinogram_fixed = np.zeros((data_images_crop.shape[1], 
                           data_images_crop.shape[0], 
                           data_images_crop.shape[2]),
                         dtype='float32')

for i in tqdm(range(data_images_crop.shape[0])):
    sinogram_fixed[:,i,:] = transform_image(data_images_crop[i], shift_x, alfa)

preview_axis_correction(sinogram_fixed, data_angles, remove_rings=True)
manual_axis_search = False


# %% [markdown]
# # Ручной поиск смещения и поворота

# %%
manual_axis_search = True
p_0 = get_angles_at_180_deg(data_angles)[0][0]
p_180 = get_angles_at_180_deg(data_angles)[1][0]
ang_0, ang_180 = data_angles[p_0], data_angles[p_180]
im_0, im_180 = data_images_crop[p_0],data_images_crop[p_180]

def show_alignment(shift, angle):
    """Быстрый просмотр совмещения: только 2 кадра (0° и 180°)."""
    t_im_0 = transform_image(im_0, shift, angle)
    t_im_180 = transform_image(im_180, shift, angle)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    im1 = ax1.imshow(t_im_0 - np.fliplr(t_im_180), cmap=plt.cm.seismic)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_title('Разность (0° − flip(180°))')

    im2 = ax2.imshow(t_im_0, cmap=plt.cm.viridis)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_title('Кадр 0°')

    plt.tight_layout()
    plt.show()


def apply_and_reconstruct(shift, angle):
    """Медленная часть: заполнение всего синограма и предпросмотр реконструкции."""
    for i in tqdm(range(data_images_crop.shape[0])):
        sinogram_fixed[:, i, :] = transform_image(data_images_crop[i], shift, angle)

    preview_axis_correction(sinogram_fixed, data_angles, remove_rings=True)


import ipywidgets as widgets
from IPython.display import display

# Создаем виджеты
shift_slider = widgets.FloatSlider(
    min=-200, max=200, step=0.05, value=shift_x,
    description='Shift:', layout=widgets.Layout(width='400px')
)
shift_text = widgets.FloatText(
    value=shift_x, step=0.05,
    layout=widgets.Layout(width='100px')
)

angle_slider = widgets.FloatSlider(
    min=-3., max=3, step=0.001, value=alfa,
    description='Angle:', layout=widgets.Layout(width='400px')
)
angle_text = widgets.FloatText(
    value=alfa, step=0.001,
    layout=widgets.Layout(width='100px')
)

# Синхронизация слайдера и текстового поля
widgets.jslink((shift_slider, 'value'), (shift_text, 'value'))
widgets.jslink((angle_slider, 'value'), (angle_text, 'value'))

# Две кнопки
btn_show = widgets.Button(
    description='Показать совмещение', button_style='info',
    layout=widgets.Layout(width='220px')
)
btn_apply = widgets.Button(
    description='Применить + реконструкция', button_style='primary',
    layout=widgets.Layout(width='250px')
)
output = widgets.Output(layout=widgets.Layout(border='none'))


def on_show_click(b):
    with output:
        output.clear_output(wait=True)
        show_alignment(shift_slider.value, angle_slider.value)


def on_apply_click(b):
    with output:
        output.clear_output(wait=True)
        apply_and_reconstruct(shift_slider.value, angle_slider.value)


btn_show.on_click(on_show_click)
btn_apply.on_click(on_apply_click)

# Компоновка
shift_box = widgets.HBox([shift_slider, shift_text])
angle_box = widgets.HBox([angle_slider, angle_text])
buttons_box = widgets.HBox([btn_show, btn_apply])
ui = widgets.VBox([shift_box, angle_box, buttons_box, output])

display(ui)

# %%
if manual_axis_search:
    shift_x, alfa = shift_text.value, angle_text.value

    for i in tqdm(range(data_images_crop.shape[0])):
        sinogram_fixed[:,i,:] = transform_image(data_images_crop[i], shift_x, alfa)
    preview_axis_correction(sinogram_fixed, data_angles)

recon_config['axis_corr'] = {'shift_x': shift_x,
                         'alfa': alfa,
                         }
# %xdel data_images_crop

# %% [markdown]
# # Удаление колец

# %%
indexes = range(sinogram_fixed.shape[0])
num_subrrays = len(indexes) // 48 + 1

for subarr in tqdm(np.array_split(indexes, num_subrrays)):
    t = sinogram_fixed[subarr]
    t = remove_all_stripe(cp.asanyarray(t.swapaxes(0,1))).get().swapaxes(0,1)
    sinogram_fixed[subarr] = t

preview_axis_correction(sinogram_fixed, data_angles, remove_rings=False)

# %%
raw_file_name = f"{tomo_info['specimen']}.{sinogram_fixed.shape[0]}_{sinogram_fixed.shape[2]}_{sinogram_fixed.shape[2]}.1.raw"
rec_vol, _ = persistent_array(os.path.join(tmp_dir, raw_file_name),
                              dtype=np.float32, force_create=False,
                              shape=(sinogram_fixed.shape[0], sinogram_fixed.shape[2], sinogram_fixed.shape[2]))

# %%
# multi 2d case
t0 = time.time()
print(sinogram_fixed.shape)
t_angles = (data_angles - data_angles.min()) < 180  # remove angles >180
for i in tqdm(range(0, sinogram_fixed.shape[0])):
    sino = sinogram_fixed[i]
    # sino[sino < 0] = 0
    # sino = np.power(sino, bh_corr)  # BH!
    t = recon_2d_parallel(sino[t_angles], data_angles[t_angles], pixel_size)
    rec_vol[i] = t
    
rec_vol.flush()
print(time.time() - t0)

# %%
# %xdel sinogram_fixed 

# %%
for j in range(2):
    N = 20  # number of cuts
    for i in range(N):
        plt.figure(figsize=(10, 8))
        data = rec_vol.take(i * rec_vol.shape[j] // N, axis=j)
        plt.imshow(data, cmap=plt.cm.viridis,
                   vmin=np.maximum(0, np.percentile(data[:], 10)),
                   vmax=np.percentile(data[:], 99.9))
        plt.axis('image')
        plt.title(i * rec_vol.shape[j] // N)
        plt.colorbar()
        plt.show()

# %%
save_amira(rec_vol, tmp_dir, tomo_info['specimen'], 1, pixel_size)

# %%
save_amira(rec_vol, tmp_dir, tomo_info['specimen'], 4, pixel_size)

# %%
recon_config


# %%
cfg = configparser.ConfigParser()
for key in ['roi', 'corr','axis_corr']:
    if key in recon_config:
        cfg[key] = recon_config[key]

with open(os.path.join(tmp_dir, 'rec_config.ini'), 'w') as configfile:
    cfg.write(configfile)

# %%
# os.path.join(tmp_dir, 'rec_config.ini')

# %%
# files_to_remove = glob(os.path.join(tmp_dir, '*.tmp'))
# files_to_remove

# %%
# for fr in files_to_remove:
#     try:
#         os.remove(os.path.join(tmp_dir, fr))
#     except:
#         pass
#     try:
#         os.remove(os.path.join(tmp_dir, fr + '.size'))
#     except:
#         pass

# %%
mkdir_p(os.path.join(storage_dir, experiment_id))

# %%
# # !cp 'tomo.ini'  {os.path.join(storage_dir, experiment_id)}

# %%
# !rm -rf {tmp_dir}/*.size

# %%
# !unset LD_LIBRARY_PATH; cp -r {tmp_dir} {os.path.join(storage_dir, experiment_id, 'reconstruction')}

# %%
# !rm -rf {tmp_dir}

# %%
# !unset LD_LIBRARY_PATH; mv {os.path.join(data_dir, experiment_id+'.h5')} {storage_dir}

# %%
# !ls -lha {storage_dir+'/'+experiment_id}

# %%
# %reset -sf

# %% [markdown]
# # Changelog:
# * 3.0 (2025.03.31)
#  - back to swap file
#  - grand cleanup
#  - axis search rewrite
