# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

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
from glob import glob

import pylab as plt
import numpy as np

import h5py

import scipy.optimize
import scipy.ndimage

import imreg_dft as ird
from tomopy.prep.stripe import remove_all_stripe
from tomopy.recon.rotation import find_center_vo

from tomotools import (STORAGE_SERVER, safe_median, recon_2d_parallel, get_tomoobject_info, get_experiment_hdf5,
                       mkdir_p, show_exp_data, load_tomo_data, group_data, tqdm, find_roi,
                       persistent_array, get_angles_at_180_deg, test_rec, save_amira, show_frames_with_border,
                       recursively_save_dict_contents_to_group)

import ipywidgets

# %%
plt.rcParams['figure.facecolor'] = 'white'

# %%
# # settings for docker

config = configparser.ConfigParser()
config.read('tomo.ini')
experiment_id = config['SAMPLE']['_id']
data_dir = '/fast/'
storage_dir = '/storage/'
exp_src_dir = '/exp_src'

# %%
tomo_info = get_tomoobject_info(experiment_id, STORAGE_SERVER)
tomo_info

# %%
if os.path.exists("rec_config.ini"):
    recon_conf = configparser.ConfigParser()
    recon_conf.read("rec_config.ini")
    recon_config = recon_conf._sections
    del recon_conf
else:
    recon_config = {}
recon_config

# %%
recon_config['sample'] = tomo_info

# %% [markdown]
# # Loading experimental data

# %%
data_file = get_experiment_hdf5(experiment_id, data_dir,
                                os.path.join(exp_src_dir, experiment_id, 'before_processing'),
                                STORAGE_SERVER)

tmp_dir = os.path.join(data_dir, experiment_id)
mkdir_p(tmp_dir)

empty_beam, data_images, data_angles = load_tomo_data(data_file, tmp_dir)
show_exp_data(empty_beam, data_images)

# %%
if 'roi' in recon_config:
    print("Read from ini file")
    x_min, x_max, y_min, y_max = (recon_config['roi']['x_min'],
                                  recon_config['roi']['x_max'],
                                  recon_config['roi']['y_min'],
                                  recon_config['roi']['y_max'])
else:

    x_min, x_max, y_min, y_max = find_roi(data_images, empty_beam, data_angles)
    # x_min, x_max, y_min, y_max = 865, 3123, 324, 2443
print("x_min, x_max, y_min, y_max = ", x_min, x_max, y_min, y_max)


# %%
ff = ipywidgets.interact_manual(show_frames_with_border, data_images=ipywidgets.fixed(data_images),
                                empty_beam=ipywidgets.fixed(empty_beam),
                                data_angles=ipywidgets.fixed(data_angles),
                                image_id=ipywidgets.IntSlider(min=0, max=len(data_angles) - 1, step=1, value=0),
                                x_min=ipywidgets.IntSlider(min=0, max=data_images.shape[1], step=1, value=x_min),
                                x_max=ipywidgets.IntSlider(min=0, max=data_images.shape[1], step=1, value=x_max),
                                y_min=ipywidgets.IntSlider(min=0, max=data_images.shape[2], step=1, value=y_min),
                                y_max=ipywidgets.IntSlider(min=0, max=data_images.shape[2], step=1, value=y_max)
                                )

# %%
try:
    if ff.widget.kwargs['x_min'] is not None:
        x_min = ff.widget.kwargs['x_min']
    if ff.widget.kwargs['x_max'] is not None:
        x_max = ff.widget.kwargs['x_max']
    if ff.widget.kwargs['y_min'] is not None:
        y_min = ff.widget.kwargs['y_min']
    if ff.widget.kwargs['y_max'] is not None:
        y_max = ff.widget.kwargs['y_max']
except KeyError:
    pass

# %%
x_min = int(x_min)
x_max = int(x_max)
y_min = int(y_min)
y_max = int(y_max)

# %%
recon_config['roi'] = {'x_min': x_min,
                       'x_max': x_max,
                       'y_min': y_min,
                       'y_max': y_max}

# %%
show_frames_with_border(data_images, empty_beam, data_angles, 0, x_min, x_max, y_min, y_max)

# %%
data_images_crop, _ = persistent_array(os.path.join(tmp_dir, 'data_images_crop.tmp'),
                                       shape=(len(data_angles), x_max - x_min, y_max - y_min),
                                       dtype='float32')

data_images_crop[:] = data_images[:, x_min:x_max, y_min:y_max]
empty_beam_crop = empty_beam[x_min:x_max, y_min:y_max]


# %%
uniq_data_images, uniq_angles = group_data(data_images_crop, data_angles, tmp_dir)

# %%
sinogram, _ = persistent_array(os.path.join(tmp_dir, 'sinogram.tmp'), shape=uniq_data_images.shape,
                               dtype='float32')
te = np.asarray(empty_beam_crop)
te[te < 1] = 1
log_te = np.log(te)
for di in tqdm(range(uniq_data_images.shape[0])):
    td = uniq_data_images[di]
    td[td < 1] = 1

    d = log_te - np.log(td)
    d = safe_median(d)
    d[d < 0] = 0
    sinogram[di] = d
# ne.evaluate('-log(uniq_data_images)', out=sinogram);

# %%
position_0, position_180 = get_angles_at_180_deg(uniq_angles)
# position_0 +=3
# position_180 +=1
print(position_0, position_180)
print(uniq_angles[position_0], uniq_angles[position_180], uniq_angles[position_180] - uniq_angles[position_0])

data_0_orig = np.rot90(sinogram[position_0])
data_180_orig = np.fliplr(np.rot90(sinogram[position_180]))
filt = scipy.signal.gaussian(data_0_orig.shape[0], data_0_orig.shape[0] / 2)
# plt.plot(filt)

data_0 = (data_0_orig.T * filt).T
data_180 = (data_180_orig.T * filt).T

# data_0 = data_0_orig
# data_180 = data_180_orig
# data_0 = np.pad(data_0, 100, mode='constant')
# data_180 = np.pad(data_180, 100, mode='constant')

transorm_result = ird.similarity(data_0, data_180, order=1, numiter=4,
                                 constraints={"scale": (1., 0),
                                              "angle": (0, 5.),
                                              "ty": (0, 0)})

fig = plt.figure(figsize=(12, 12))
ird.imshow(data_0, data_180, transorm_result['timg'], fig=fig)
plt.show()

transorm_result

# %%
shift_x = -transorm_result['tvec'][1] / 2.
alfa = - transorm_result['angle'] / 2
tr_dict = {"scale": 1, "angle": alfa, "tvec": (0, shift_x)}

# %%
recon_config['axis_corr'] = {'shift_x': shift_x,
                             'alfa': alfa,
                             'angle_180': uniq_angles[position_180],
                             'angle_0': uniq_angles[position_0]
                             }

# %%
plt.gray()
plt.figure(figsize=(12, 12))
im_max = np.max([np.max(data_0), np.max(data_180)])
plt.subplot(221)
plt.imshow(data_0_orig, vmin=0, vmax=im_max, cmap=plt.cm.gray_r)
plt.axis('tight')
plt.title('a')
plt.xlabel('Каналы детектора')
plt.ylabel('Каналы детектора')
cbar = plt.colorbar()
cbar.set_label('Поглощение, усл.ед.', rotation=90)

plt.subplot(222)
plt.imshow(data_180_orig, vmin=0, vmax=im_max, cmap=plt.cm.gray_r)
plt.axis('tight')
plt.title('б')
plt.xlabel('Каналы детектора')
plt.ylabel('Каналы детектора')
cbar = plt.colorbar()
cbar.set_label('Поглощение, усл.ед.', rotation=90)

plt.subplot(223)
plt.imshow(data_0_orig - data_180_orig, vmin=-im_max / 2, vmax=im_max / 2, cmap=plt.cm.seismic)
plt.axis('tight')
plt.title('в')
plt.xlabel('Каналы детектора')
plt.ylabel('Каналы детектора')
cbar = plt.colorbar()
cbar.set_label('Поглощение, усл.ед.', rotation=90)

tt_180 = np.fliplr(ird.imreg.transform_img_dict(np.fliplr(data_180_orig), tr_dict, order=1))
tt_0 = ird.imreg.transform_img_dict(data_0_orig, tr_dict, order=1)

plt.subplot(224)
plt.imshow(tt_0 - tt_180, vmin=-im_max / 10, vmax=im_max / 10, cmap=plt.cm.seismic)
plt.axis('tight')
plt.title('г')
plt.xlabel('Каналы детектора')
plt.ylabel('Каналы детектора')
cbar = plt.colorbar()
cbar.set_label('Поглощение, усл.ед.', rotation=90)

# %%
tim = ird.imreg.transform_img_dict(data_0_orig, tr_dict)
sinogram_fixed, _ = persistent_array(os.path.join(tmp_dir, 'sinogram_fixed.tmp'),
                                     shape=(sinogram.shape[0], tim.shape[1], tim.shape[0]),
                                     dtype='float32', force_create=True)

# fix axis tlit
for i in tqdm(range(sinogram_fixed.shape[0])):
    sinogram_fixed[i] = np.rot90(
        ird.imreg.transform_img_dict(np.rot90(sinogram[i]), tr_dict, order=2, bgval=0),
        -1)

# %%
preview_slice_number = int(sinogram_fixed.shape[-1] // 2)

# %%
s1_angles = uniq_angles
s1 = np.require(sinogram_fixed[:, :, preview_slice_number],
                dtype=np.float32, requirements=['C'])
test_rec(s1, uniq_angles, 2)

# %%
rot_center = find_center_vo(s1, uniq_angles)
print(rot_center)

# %%
center_shift = np.rint((rot_center - s1.shape[1] / 2.) / 2.)
print(center_shift)

# %%
shift_corr = 2  # change this for turning -2, -1, 0, 1, 2
s2 = ird.imreg.transform_img_dict(s1, {'tvec': (0, -center_shift + shift_corr), 'scale': 1, 'angle': 0})
test_rec(s2, uniq_angles, 2)

# %%
# experimental fix axis tlit
# for i in tqdm(range(sinogram_fixed.shape[0])):
#     sinogram_fixed[i] = ird.imreg.transform_img_dict(sinogram_fixed[i],
#                                                      {'tvec': (-center_shift + shift_corr, 0), 'scale': 1, 'angle': 0},
#                                                      order=2, bgval=0)

# %%
tmp_sinogram = s2[np.argsort(uniq_angles)]
ring_corr = remove_all_stripe(tmp_sinogram[:, None, :], 10, 11, 5)
ring_corr = np.squeeze(ring_corr)
plt.figure(figsize=(15, 8))
plt.subplot(131)
plt.imshow(tmp_sinogram, cmap=plt.cm.viridis, interpolation='bilinear')
plt.axis('tight')
cbar = plt.colorbar()
cbar.set_label('Пропускание, усл.ед.', rotation=90)
plt.title('Синограмма без коррекции')
plt.subplot(132)
plt.imshow(ring_corr, cmap=plt.cm.viridis, interpolation='bilinear')
plt.axis('tight')
cbar = plt.colorbar()
cbar.set_label('Пропускание, усл.ед.', rotation=90)
plt.title('Синограмма с коррекцией колец')

plt.subplot(133)
plt.imshow(tmp_sinogram - ring_corr, cmap=plt.cm.viridis, interpolation='bilinear')
plt.axis('tight')
cbar = plt.colorbar()
cbar.set_label('Пропускание, усл.ед.', rotation=90)
plt.title('Разница');

s1_angles = np.sort(uniq_angles)
s1 = np.require(ring_corr,
                dtype=np.float32, requirements=['C'])
# s1[np.isnan(s1)] = 0
test_rec(s1, s1_angles, 2)

# %%
# #uncomment to fix rings
# step = 50
# for i in tqdm(range(np.int(np.ceil(sinogram_fixed.shape[1]/step)))):
#     start = i*step
#     stop = np.min([(i+1)*step, sinogram_fixed.shape[1]])
#     sinogram_fixed[:,start:stop,:] = remove_all_stripe(sinogram_fixed[:,start:stop,:],10, 11, 5)

# %%
s1_angles = uniq_angles
s1 = np.require(sinogram_fixed[:, :, preview_slice_number - 0],
                dtype=np.float32, requirements=['C'])
test_rec(s1, uniq_angles, 2)

# %%
del data_0_orig, data_180_orig, data_images_crop, data_images
del sinogram, sinogram_fixed, uniq_angles, uniq_data_images

# %%
files_to_remove = glob(os.path.join(tmp_dir, '*.tmp'))
files_to_remove = [f for f in files_to_remove if f.split('/')[-1] not in [
    'uniq_angles.tmp', 'sinogram_fixed.tmp']]

for fr in files_to_remove:
    try:
        os.remove(os.path.join(tmp_dir, fr))
    except:
        pass
    try:
        os.remove(os.path.join(tmp_dir, fr + '.size'))
    except:
        pass

# %%
uniq_angles, _ = persistent_array(os.path.join(tmp_dir, 'uniq_angles.tmp'),
                                  shape=None, force_create=False,
                                  dtype='float32')
s1, _ = persistent_array(os.path.join(tmp_dir, 'sinogram_fixed.tmp'),
                         shape=None, force_create=False,
                         dtype='float32')

rec_vol, _ = persistent_array(os.path.join(tmp_dir, 'rec.tmp'),
                              dtype=np.float32, force_create=False,
                              shape=(s1.shape[-1], s1.shape[1], s1.shape[1]))


# %%
def find_optimal_bh(sino, angles, show_polt=True):
    def calc_radon_inv(sinogram):
        return sinogram.sum(axis=-1)

    def radon_metrics(sinogram):
        radon_inv = calc_radon_inv(sinogram)
        radon_inv = radon_inv / radon_inv.mean()
        std = np.std(radon_inv)
        res = std
        return res

    sino[sino < 0] = 0
    opt_func = lambda x: radon_metrics(np.power(sino, x))

    optimal_gamma = scipy.optimize.minimize(opt_func, [1.0, ], method='Nelder-Mead')

    if show_polt:
        xr = np.arange(1, np.max([3, optimal_gamma.x * 1.1]), 0.1)
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.title('Optimal bh')
        plt.plot(xr, [opt_func(x) for x in xr])
        plt.plot([optimal_gamma.x, ], opt_func(optimal_gamma.x), 'ro')
        plt.grid()
        plt.subplot(122)
        plt.title('Radon invatiant')
        plt.plot(angles[np.argsort(angles)], calc_radon_inv(sino)[np.argsort(angles)])  # TODO: sort for angles order
        plt.grid()
        plt.show()

    return optimal_gamma.x


# %%
# preview
need_optimal_bh = False
if need_optimal_bh:
    sino = s1[..., preview_slice_number].copy()
    optimal_gamma = find_optimal_bh(sino, uniq_angles)
    print(optimal_gamma)
    bh_corr = optimal_gamma
else:
    bh_corr = 1

# sss = s1[..., preview_slice_number]
# t_angles = (uniq_angles - uniq_angles.min()) < 180  # remove angles >180
#
# s4 = sss.copy()
# s4[s4 < 0] = 0
# s4 = np.power(s4, bh_corr)
#
# rec_slice = recon_2d_parallel(s4[t_angles], uniq_angles[t_angles])
#
# plt.figure(figsize=(10, 8))
# plt.imshow(safe_median(rec_slice),
#            vmin=0, vmax=np.percentile(rec_slice, 95) * 1.2, cmap=plt.cm.viridis)
# plt.axis('equal')
# plt.colorbar()
# plt.show()
#
# plt.figure(figsize=(10, 5))
# plt.plot(safe_median(rec_slice)[rec_slice.shape[0]//2, :], '-o', ms=3, lw=2.0)
# plt.grid()
# plt.show()

# %%
recon_config['corr'] = {'bh': bh_corr}

# %%
# multi 2d case
t = time.time()
print(s1.shape)
angles = np.array(uniq_angles)
for i in tqdm(range(0, s1.shape[-1])):
    sino = s1[:, :, i].copy()
    sino[sino < 0] = 0
    sino = np.power(sino, bh_corr)  # BH!
    t_angles = (uniq_angles - uniq_angles.min()) < 180  # remove angles >180
    rec_vol[i] = recon_2d_parallel(sino[t_angles], angles[t_angles])
print(time.time() - t)

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
save_amira(rec_vol, tmp_dir, tomo_info['specimen'], 3)

# %%
recon_config

# %%
with h5py.File(os.path.join(tmp_dir, 'tomo_rec.' + tomo_info['specimen'] + '.h5'), 'w') as h5f:
    h5f.create_dataset('Reconstruction', data=rec_vol, chunks=True,
                       compression='lzf')
    recursively_save_dict_contents_to_group(h5f, '/recon_config/', recon_config)

# %%
import k3d
from tomotools import reshape_volume

# %%
resize = int(np.power(np.prod(rec_vol.shape) / 1e7, 1. / 3))
print(resize)
small_rec = reshape_volume(rec_vol, resize)

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
resize = int(np.power(np.prod(rec_vol.shape) / 1e6, 1. / 3))
print(resize)
small_rec = reshape_volume(rec_vol, resize)
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
files_to_remove = glob(os.path.join(tmp_dir, '*.tmp'))
files_to_remove

# %%
for fr in files_to_remove:
    try:
        os.remove(os.path.join(tmp_dir, fr))
    except:
        pass
    try:
        os.remove(os.path.join(tmp_dir, fr + '.size'))
    except:
        pass

# %%
mkdir_p(os.path.join(storage_dir, experiment_id))

# %%
# # !cp 'tomo.ini'  {os.path.join(storage_dir, experiment_id)}

# %%
# !cp -r {tmp_dir} {storage_dir}

# %%
# !rm -rf {tmp_dir}

# %%
# !mv {os.path.join(data_dir, experiment_id+'.h5')} {storage_dir}

# %%
# !ls -lha {storage_dir+'/'+experiment_id}

# %%
# %reset -sf

# %% [markdown]
# # Changelog:
# * 2.5.1 (2021.04.12)
#  - fix crash in indexes converting to int
#  - change interpolation of rotation
#  - axis rotation manual mode added
# * 2.5 (2021.03.01)
#  - fiх shift and tilt detection
#  - add ring correction from tomopy
#  - read roi from ini file
# * 2.4 (2020.12.08)
#  - 3d render
# * 2.3 (2020.11.23)
#  - Auto roi
#  - Searching shift with phase correlation
# * 2.2а (2020.04.28)
#  - Add auto bh option
#  - Remove sinogram normalization
#  - move a lot of fuctions in tomotools
#  - add 2 gaussian fitting
#  - save reconstruction parameters to file
# * 2.1а (2020.03.18)
#  - Add local files loading
#  - Improving poriosity support
#  - ENH: 180 deg search
# * 2.0d (2019.04.17-2019.05.06)
#  - Adding dask support
#  - Many code refactorings for semiautomatic runs
#  - Allow manual borders selections
# * 2.0.b1 (2019.04.03)
#  - bug fixing
#  - try remove bad frames
# * 1.6.2 (2019.02.11)
#  - fixing object detection
# * 1.6.1 (2018.11.19)
#  - exdend borbers range (mean to percentile)
# * 1.6 (2018.11.08)
#  - change algorithm of object detection with gaussian fitting
#  - add y-clipping to remove sample holder
#  - change algorithm of axis searching
#  - change hdf5 compression to lzf
#  - changing 3d visualisation
#  - replace log_process to tqdm
# * 1.5 (2018.09.11)
#  - saving full tomography volume
#  - deleting temporary files as soon as possible
#  - change thresshold in object detection (1/6 -> 1/5)
# * 1.4 (2018.08.23)
#  - Fix: correct resized volume serialization (smooth instead cherry picking)
#  - New: 3D visualisation
#  - Fix: sinogram shifting aftee rotation axis fix
#  - Update: Searching rotation axis
# * 1.3 (2018.07.03)
#  - Update graphics
#  - Update axis search algorithms
# * 1.2 (2018.06.04)
#  - Change threshold
# * 1.1 (2018.03.14)
#  - Add NLM filtering
# * 1.0 (2017.02.01)
#  - First automation version.

# %%