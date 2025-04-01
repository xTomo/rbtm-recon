# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
# #jupytext --to notebook reconstructor.py
# manual mode
# #%matplotlib notebook

# automatic mode
# %matplotlib inline

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
                       preview_axis_correction)

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
if os.path.exists("rec_config.ini"): # in current dir
    config_file = "rec_config.ini"
elif os.path.exists(os.path.join(tmp_dir, 'rec_config.ini')):  # in tmp dir
    config_file = os.path.join(tmp_dir, 'rec_config.ini')
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
    x_min, x_max, y_min, y_max = (recon_config['roi']['x_min'],
                                  recon_config['roi']['x_max'],
                                  recon_config['roi']['y_min'],
                                  recon_config['roi']['y_max'])
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
except:
    pass
    
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
from tomopy.prep.stripe import remove_stripe_ti
import cupy as cp
import cupyx.scipy.ndimage as cndi

def transfrom_image(im, shift_x, angle):
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
    tmp_im = transfrom_image(im0, shift, angle) - transfrom_image(im1, -shift, -angle)
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

tmp_im = transfrom_image(im0, shift_x, angle) - transfrom_image(im1, -shift_x, -angle)

plt.figure()
plt.imshow(tmp_im, cmap=plt.cm.seismic)
plt.colorbar()
plt.show()



shift_x, alfa = shift_x, angle
tr_dict = {"scale": 1, "angle": alfa, "tvec": (0, shift_x)}

recon_config['axis_corr'] = {'shift_x': shift_x,
                             'alfa': alfa,
                             'angle_180': data_angles[position_180],
                             'angle_0': data_angles[position_0]
                             }

sinogram_fixed = np.zeros((data_images_crop.shape[1], 
                           data_images_crop.shape[0], 
                           data_images_crop.shape[2]),
                         dtype='float32')

for i in tqdm(range(data_images_crop.shape[0])):
    sinogram_fixed[:,i,:] = transfrom_image(data_images_crop[i], shift_x, alfa)

preview_axis_correction(sinogram_fixed, data_angles)


# %% [markdown]
# # Ручной поиск смещения и поворота

# %%
p_0 = get_angles_at_180_deg(data_angles)[0][0]
p_180 = get_angles_at_180_deg(data_angles)[1][0]
ang_0, ang_180 = data_angles[p_0], data_angles[p_180]
im_0, im_180 = data_images_crop[p_0],data_images_crop[p_180]

def find_shift_angle(shift, angle):
    t_im_0 = transfrom_image(im_0, shift, angle)
    t_im_180 = transfrom_image(im_180, shift, angle)

    # t_im_0 = fix_porj(t_im_0)
    # t_im_180 = fix_porj(t_im_180)
    
    plt.figure(figsize = (12,8))
    plt.subplot(121)
    plt.imshow(t_im_0-np.fliplr(t_im_180), cmap=plt.cm.seismic)
    plt.colorbar(orientation='vertical')
    plt.subplot(122)
    plt.imshow(t_im_0, cmap=plt.cm.viridis)
    plt.colorbar(orientation='vertical')
    plt.show()

ff = ipywidgets.interact_manual(find_shift_angle, 
                                shift=ipywidgets.FloatSlider(min=-200, max=200, step=0.05, value=shift_x, readout_format='.2f',),
                                angle=ipywidgets.FloatSlider(min=-3., max=3, step=0.001, value=alfa, readout_format='.3f',),
                                )

# %%
if 'shift' in ff.widget.kwargs:
    shift_x, alfa = ff.widget.kwargs['shift'], ff.widget.kwargs['angle']
    
    for i in tqdm(range(data_images_crop.shape[0])):
        sinogram_fixed[:,i,:] = transfrom_image(data_images_crop[i], shift_x, alfa)
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
    t = remove_stripe_ti(t.swapaxes(0,1), ).swapaxes(0,1)
    sinogram_fixed[subarr] = t

# preview_axis_correction(sinogram_fixed, data_angles)

# %%
raw_file_name = tomo_info['specimen'] + '.1'
rec_vol, _ = persistent_array(os.path.join(tmp_dir, raw_file_name + '.raw'),
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
    t = recon_2d_parallel(sino[t_angles], data_angles[t_angles])
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
save_amira(rec_vol, tmp_dir, tomo_info['specimen'], 1)

# %%
save_amira(rec_vol, tmp_dir, tomo_info['specimen'], 4)

# %%
recon_config

# %%
# with h5py.File(os.path.join(tmp_dir, 'tomo_rec.' + tomo_info['specimen'] + '.h5'), 'w') as h5f:
#     h5f.create_dataset('Reconstruction', data=rec_vol, chunks=True,
#                        compression='lzf')
#     recursively_save_dict_contents_to_group(h5f, '/recon_config/', recon_config)

# %%
import k3d
from tomotools import reshape_volume

# %%
resize = int(np.power(np.prod(rec_vol.shape) / 1e7, 1. / 3))
print(resize)
small_rec = reshape_volume(rec_vol, 10)

# %%
volume = k3d.volume(
    small_rec.astype(np.float32),
    #     alpha_coef=1000,
    #     shadow='dynamic',
    #     samples=600,
    #     shadow_res=128,
    #     shadow_delay=50,
    color_range=[np.percentile(small_rec, 10), np.percentile(small_rec, 99.9)],
    color_map=(np.array(k3d.colormaps.matplotlib_color_maps.jet).reshape(-1, 4)).astype(np.float32),
    compression_level=4
)
size = small_rec.shape
volume.transform.bounds = [-size[2] / 2, size[2] / 2,
                           -size[1] / 2, size[1] / 2,
                           -size[0] / 2, size[0] / 2]

plot = k3d.plot(camera_auto_fit=True)
plot += volume
plot.lighting = 2
plot.display()

# %%
plot.fetch_snapshot()
with open('./tomo_3d.html', 'w') as fp:
    fp.write(plot.snapshot)

# %%
cfg = configparser.ConfigParser()
cfg['roi'] = recon_config['roi']
cfg['corr'] = recon_config['corr']
cfg['axis_corr'] = recon_config['axis_corr']
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