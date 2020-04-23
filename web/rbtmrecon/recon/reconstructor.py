# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.0
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
logger.setLevel(logging.INFO)
import time
import os
import configparser
from glob import glob

import pylab as plt
import numpy as np

import h5py

import cv2

import numexpr as ne

import scipy.optimize
import scipy.ndimage

from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.transform import resize

from tomotools import STORAGE_SERVER, safe_median, recon_2d_parallel, get_tomoobject_info, get_experiment_hdf5, \
    mkdir_p, show_exp_data, load_create_mm, load_tomo_data, find_good_frames, group_data, correct_rings, tqdm, \
    get_angles_at_180_deg, smooth, cv_rotate, find_axis_posiotion, test_rec, save_amira

import ipywidgets

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
# TODO: store this in ini file
x_min, x_max, y_min, y_max = 600, 2320, 100, 2550

# %%
def show_frames_with_border(image_id, x_min, x_max, y_min, y_max):
    angles_sorted_ind = np.argsort(data_angles)
    t_image = data_images[angles_sorted_ind[image_id]].T
    plt.figure(figsize=(15, 10))
    plt.subplot(121)
    plt.imshow(t_image, cmap=plt.cm.gray)
    plt.axis('equal')
    plt.hlines([y_min, y_max], x_min, x_max, 'r')
    plt.vlines([x_min, x_max], y_min, y_max, 'g')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.subplot(122)
    plt.imshow(t_image[y_min:y_max, x_min:x_max], cmap=plt.cm.gray)
    plt.show()
    print("x_min, x_max, y_min, y_max = {}, {}, {}, {}".format(x_min, x_max, y_min, y_max))


ff = ipywidgets.interact_manual(show_frames_with_border,
                                image_id=ipywidgets.IntSlider(min=0, max=len(data_angles), step=1, value=0),
                                x_min=ipywidgets.IntSlider(min=0, max=data_images.shape[1], step=1, value=x_min),
                                x_max=ipywidgets.IntSlider(min=0, max=data_images.shape[1], step=1, value=x_max),
                                y_min=ipywidgets.IntSlider(min=0, max=data_images.shape[2], step=1, value=y_min),
                                y_max=ipywidgets.IntSlider(min=0, max=data_images.shape[2], step=1, value=y_max)
                                )

# %%
try:
    x_min = ff.widget.kwargs['x_min']
    x_max = ff.widget.kwargs['x_max']
    y_min = ff.widget.kwargs['y_min']
    y_max = ff.widget.kwargs['y_max']
except KeyError:
    pass

# %%
data_images_crop, _ = load_create_mm(os.path.join(tmp_dir, 'data_images_crop.tmp'),
                                     shape=(len(data_angles), x_max - x_min, y_max - y_min),
                                     dtype='float32')
for i in range(len(data_angles)):
    data_images_crop[i] = data_images[i, x_min:x_max, y_min:y_max]

empty_beam = empty_beam[x_min:x_max, y_min:y_max]

# %%
good_frames = find_good_frames(data_images_crop, data_angles)

# %% [markdown]
# # Remove bad frames

# %%
data_images_good, _ = load_create_mm(os.path.join(tmp_dir, 'data_images_good.tmp'),
                                     shape=(len(good_frames), data_images_crop.shape[1], data_images_crop.shape[2]),
                                     dtype='float32')

# TODO: Profile this code. In case if no bad frames, just skip it
for i in tqdm(range(len(good_frames))):
    data_images_good[i] = data_images_crop[good_frames[i]]

data_angles = data_angles[good_frames]

# %%
uniq_data_images, uniq_angles = group_data(data_images_good, data_angles, tmp_dir)

# %%
# normalize data frames and calculate sinograms
empty_masked = safe_median(empty_beam)
for di in tqdm(range(uniq_data_images.shape[0])):
    t = uniq_data_images[di]
    t = t / empty_beam
    t[t < 1e-8] = 1e-8
    t[t > 1] = 1
    uniq_data_images[di] = safe_median(t)

# del empty_masked

# %%
sinogram, _ = load_create_mm(os.path.join(tmp_dir, 'sinogram.tmp'), shape=uniq_data_images.shape,
                             dtype='float32')
ne.evaluate('-log(uniq_data_images)', out=sinogram);

# %%
plt.gray()
plt.figure(figsize=(7, 5))
s = sinogram[np.argsort(uniq_angles), :, int(sinogram.shape[-1] // 2)]
plt.imshow(s, interpolation='bilinear')
plt.axis('tight')
cbar = plt.colorbar()
cbar.set_label('Пропускание, усл.ед.', rotation=90)
plt.title('Синограмма без коррекции')

# %%
rc_level = 10

# %%
tmp_sinogram = sinogram[np.argsort(uniq_angles), :, int(sinogram.shape[-1] // 2)]
ring_corr = correct_rings(tmp_sinogram, rc_level)
plt.figure(figsize=(8, 8))
plt.imshow(ring_corr, cmap=plt.cm.viridis, interpolation='nearest')
plt.axis('tight')
plt.colorbar(orientation='horizontal')

# %%
for s in tqdm(range(sinogram.shape[1])):
    sinogram[:, s, :] = correct_rings(sinogram[:, s, :], rc_level)

# %%
tmp_sinogram = sinogram[np.argsort(uniq_angles), :, int(sinogram.shape[-1] // 2)]

plt.figure(figsize=(8, 8))
plt.imshow(tmp_sinogram, cmap=plt.cm.viridis, interpolation='nearest')
plt.axis('tight')
plt.colorbar(orientation='horizontal')

# %%
position_0, position_180 = get_angles_at_180_deg(uniq_angles)

posiotion_180_sorted = np.argwhere(np.isclose(position_180, np.argsort(uniq_angles)))[0][0]
print(posiotion_180_sorted)
posiotions_to_check = np.argsort(uniq_angles)[
                      posiotion_180_sorted - 3:np.min(
                          [posiotion_180_sorted + 5, len(uniq_angles) - 1])]  # TODO: check ranges
print(uniq_angles[posiotions_to_check])

# %%
data_0_orig = np.rot90(sinogram[position_0]).copy()
data_0 = cv2.medianBlur(data_0_orig, 3)
data_0 = smooth(data_0)

# %%
plt.figure(figsize=(8, 8))
plt.imshow(smooth(data_0_orig))
plt.colorbar()

# %%
opt_func_values = []
for position_180 in posiotions_to_check:
    print(uniq_angles[position_180])
    data_0_orig = np.rot90(sinogram[position_0]).copy()
    data_180_orig = np.rot90(sinogram[position_180]).copy()
    data_0 = cv2.medianBlur(data_0_orig, 3)
    data_180 = cv2.medianBlur(data_180_orig, 3)

    data_0 = smooth(data_0)
    data_180 = smooth(data_180)

    res = find_axis_posiotion(data_0, data_180)
    opt_func_values.append(res['fun'])
    print(res)
    # alfa, shift_x, shift_y = res.x[0]/10, int(res.x[1]), int(res.x[2])//10

    alfa, shift_x, shift_y = res.x[0], int(np.floor(res.x[1])), 0

    if shift_x >= 0:
        t_180 = data_180_orig[:, shift_x:]
        t_0 = data_0_orig[:, shift_x:]
    else:
        t_180 = data_180_orig[:, :shift_x]
        t_0 = data_0_orig[:, :shift_x]

    if shift_y > 0:
        t_180 = t_180[shift_y:, :]
        t_0 = t_0[:-shift_y, :]
    elif shift_y < 0:
        t_180 = t_180[:shift_y, :]
        t_0 = t_0[-shift_y:, :]

    tt_180 = np.fliplr(cv_rotate(t_180, alfa))
    tt_0 = cv_rotate(t_0, alfa)

    plt.figure(figsize=(7, 7))
    plt.imshow(tt_180 - tt_0, cmap=plt.cm.viridis)
    plt.title('a={}, sx={} sy={}'.format(alfa, shift_x, shift_y))
    plt.colorbar()
    plt.show()

# %%
plt.figure()
plt.plot(uniq_angles[posiotions_to_check], opt_func_values)
plt.grid()
new_position_180 = posiotions_to_check[np.argmin(opt_func_values)]
print(new_position_180)

# %%
uniq_angles_orig = uniq_angles.copy()
uniq_angles *= 180. / uniq_angles[new_position_180]
position_0, position_180 = get_angles_at_180_deg(uniq_angles)

# %%
print(uniq_angles[position_180])
data_0_orig = np.rot90(sinogram[position_0]).copy()
data_180_orig = np.rot90(sinogram[position_180]).copy()
data_0 = cv2.medianBlur(data_0_orig, 3)
data_180 = cv2.medianBlur(data_180_orig, 3)

data_0 = smooth(data_0)
data_180 = smooth(data_180)

res = find_axis_posiotion(data_0, data_180)
# opt_func_values.append(res['fun'])
print(res)

# TODO: FIX shift_y
alfa, shift_x, shift_y = res.x[0], int(np.floor(res.x[1])), 0

if shift_x >= 0:
    t_180 = data_180_orig[:, shift_x:]
    t_0 = data_0_orig[:, shift_x:]
else:
    t_180 = data_180_orig[:, :shift_x]
    t_0 = data_0_orig[:, :shift_x]

if shift_y > 0:
    t_180 = t_180[shift_y:, :]
    t_0 = t_0[:-shift_y, :]
elif shift_y < 0:
    t_180 = t_180[:shift_y, :]
    t_0 = t_0[-shift_y:, :]

tt_180 = np.fliplr(cv_rotate(t_180, alfa))
tt_0 = cv_rotate(t_0, alfa)

plt.figure(figsize=(8, 8))
plt.imshow(tt_180 - tt_0, cmap=plt.cm.viridis)
plt.title('a={}, sx={} sy={}'.format(alfa, shift_x, shift_y))
plt.colorbar()
plt.show()

# %%
plt.gray()
plt.figure(figsize=(8, 8))
im_max = np.max([np.max(data_0_orig), np.max(data_180_orig)])
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
plt.imshow(data_0_orig - np.fliplr(data_180_orig), vmin=-im_max / 2, vmax=im_max / 2, cmap=plt.cm.gray_r)
plt.axis('tight')
plt.title('в')
plt.xlabel('Каналы детектора')
plt.ylabel('Каналы детектора')
cbar = plt.colorbar()
cbar.set_label('Поглощение, усл.ед.', rotation=90)

plt.subplot(224)
plt.imshow(1.0 * (tt_180 - tt_0), vmin=-im_max / 2, vmax=im_max / 2, cmap=plt.cm.gray_r)
plt.axis('tight')
plt.title('г')
plt.xlabel('Каналы детектора')
plt.ylabel('Каналы детектора')
cbar = plt.colorbar()
cbar.set_label('Поглощение, усл.ед.', rotation=90)

# %%
sinogram_fixed, _ = load_create_mm(os.path.join(tmp_dir, 'sinogram_fixed.tmp'),
                                   shape=(
                                       sinogram.shape[0], sinogram.shape[1] + abs(shift_x),
                                       sinogram.shape[2]),
                                   dtype='float32', force_create=True)

# fix axis tlit
for i in tqdm(range(sinogram.shape[0])):
    t = sinogram[i].copy()

    t_angle = uniq_angles[i]
    t = cv_rotate(t, alfa)

    if shift_x > 0:
        sinogram_fixed[i, :-shift_x] = t
    else:
        sinogram_fixed[i, -shift_x:] = t

# %%
s1_angles = uniq_angles
s1 = np.require(sinogram_fixed[:, :, int(sinogram_fixed.shape[-1] // 3)],
                dtype=np.float32, requirements=['C'])

# %%
test_rec(s1, s1_angles)

# %%
plt.figure(figsize=(7, 7))

plt.imshow(s1[np.argsort(uniq_angles)], interpolation='bilinear', cmap=plt.cm.gray_r)
plt.axis('tight')
cbar = plt.colorbar()
cbar.set_label('Пропускание, усл.ед.', rotation=90)
plt.title('Синограмма без коррекции')
plt.xlabel('Номер канала детектора')
plt.ylabel('Номер угла поворота')

# %%
# TODO: check mu physical value
sinogram_fixed_median = np.median(sinogram_fixed.sum(axis=-1).sum(axis=-1))
corr_factor = sinogram_fixed.sum(axis=-1).sum(axis=-1) / sinogram_fixed_median

# %%
# #TODO: fix bad data
# for i in range(len(sinogram_fixed)):
#     sinogram_fixed[i] = sinogram_fixed[i] / corr_factor[i]

# %%
s2 = np.require(sinogram_fixed[:, :, int(sinogram_fixed.shape[-1] // 2)],
                dtype=np.float32, requirements=['C'])

# %%
s2 = (s1.T / s1.sum(axis=-1) * s1.sum(axis=-1).mean()).T
test_rec(s1, uniq_angles)
test_rec(s2, uniq_angles)

# %%
del data_0_orig, data_180_orig, data_images_good, data_images_crop, data_images
del sinogram, sinogram_fixed, uniq_angles, uniq_angles_orig, uniq_data_images

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
uniq_angles, _ = load_create_mm(os.path.join(tmp_dir, 'uniq_angles.tmp'),
                                shape=None, force_create=False,
                                dtype='float32')
s1, _ = load_create_mm(os.path.join(tmp_dir, 'sinogram_fixed.tmp'),
                       shape=None, force_create=False,
                       dtype='float32')

rec_vol, _ = load_create_mm(os.path.join(tmp_dir, 'rec.tmp'),
                            dtype=np.float32, force_create=False,
                            shape=(s1.shape[-1], s1.shape[1], s1.shape[1]))


# %%
def calc_raddon_inv(sinogram):
    return sinogram.sum(axis=-1)


def radon_metrics(sinogram):
    radon_inv = calc_raddon_inv(sinogram)
    radon_inv = radon_inv / radon_inv.mean()
    std = np.std(radon_inv)
    res = std
    return res


sino = s1[..., int(s1.shape[-1] // 2)].copy()
sino[sino < 0] = 0
opt_func = lambda x: radon_metrics(np.power(sino, x))

optimal_gamma = scipy.optimize.minimize(opt_func, [1.0, ], method='Nelder-Mead')
print(optimal_gamma)

# radon_inv = calc_raddon_inv(np.power(sino, optimal_gamma['x']))
xr = np.arange(1, 3, 0.1)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(xr, [opt_func(x) for x in xr])
plt.plot([optimal_gamma.x, ], opt_func(optimal_gamma.x), 'ro')
plt.grid()
plt.subplot(122)
plt.plot(calc_raddon_inv(sino))  # TODO: sort for angles order
plt.grid()
plt.show()

# %%
# # %%timeit
# preview
bh_corr = optimal_gamma.x
sss = s1[..., int(s1.shape[-1] // 2)]
t_angles = (uniq_angles - uniq_angles.min()) <= 180  # remove angles >180
s4 = sss.copy()

s4[s4 < 0] = 0
s4 = np.power(s4, bh_corr)

rec_slice = recon_2d_parallel(s4[t_angles], uniq_angles[t_angles])

print('rec_slice.shape=', rec_slice.shape)

plt.figure(figsize=(10, 8))
plt.imshow(safe_median(rec_slice),
           vmin=0, vmax=np.percentile(rec_slice, 95) * 1.2, cmap=plt.cm.viridis)
plt.axis('equal')
plt.colorbar()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(safe_median(rec_slice)[870, :], '-o', ms=3, lw=2.0)
plt.grid()
plt.show()

# %%
# multi 2d case
t = time.time()
print(s1.shape)
angles = np.array(uniq_angles)
for i in tqdm(range(0, s1.shape[-1])):
    sino = s1[:, :, i].copy()
    sino[sino < 0] = 0
    sino = np.power(sino, bh_corr)  # BH!
    t_angles = (uniq_angles - uniq_angles.min()) <= 180  # remove angles >180
    rec_vol[i] = recon_2d_parallel(sino[t_angles], angles[t_angles])
print(time.time() - t)

# %%
rec_vol_filtered = rec_vol

# %%
for i in range(10):
    plt.figure(figsize=(8, 8))
    plt.imshow(rec_vol_filtered[i * rec_vol_filtered.shape[0] // 10], cmap=plt.cm.viridis, vmin=0)
    plt.axis('equal')
    plt.title(i * i * rec_vol_filtered.shape[0] // 10)
    plt.colorbar()
    plt.show()

# %%
for i in range(10):
    plt.figure(figsize=(8, 8))
    plt.imshow(rec_vol_filtered[:, i * rec_vol_filtered.shape[1] // 10, :], cmap=plt.cm.viridis, vmin=0)
    plt.axis('equal')
    plt.title(i * i * rec_vol_filtered.shape[0] // 10)
    plt.colorbar()
    plt.show()

# %%
for i in range(10):
    plt.figure(figsize=(8, 8))
    plt.imshow(rec_vol_filtered[:, :, i * rec_vol_filtered.shape[2] // 10], cmap=plt.cm.viridis, vmin=0)
    plt.axis('equal')
    plt.title(i * i * rec_vol_filtered.shape[0] // 10)
    plt.colorbar()
    plt.show()

# %%
noisy = rec_vol_filtered[int(rec_vol_filtered.shape[0] * 0.5)].astype('float64')
noisy = resize(noisy, (noisy.shape[0] // 1, noisy.shape[1] // 1))
# noisy = rec_vol_filtered[int(rec_vol_filtered.shape[0]*0.75)][::1,::1]
sigma_est = np.mean(estimate_sigma(noisy, multichannel=False))
print("estimated noise standard deviation = {}".format(sigma_est))

patch_kw = dict(patch_size=7,  # 5x5 patches
                patch_distance=15,  # 13x13 search area
                multichannel=False)

# 1 algorithm
denoise = denoise_nl_means(noisy, h=1.5 * sigma_est, fast_mode=True,
                           **patch_kw)

# 2 algorithm
denoise_fast = denoise_nl_means(noisy, h=0.8 * sigma_est, fast_mode=True,
                                **patch_kw)

plt.figure(figsize=(6, 12))
plt.subplot(311)
plt.imshow(noisy, interpolation='bilinear')
plt.axis('off')
plt.colorbar()
plt.title('noisy')

plt.subplot(312)
plt.imshow(denoise, interpolation='bilinear')
plt.axis('off')
plt.colorbar()
plt.title('non-local means\n(1)')

plt.subplot(313)
plt.imshow(denoise_fast, interpolation='bilinear')
plt.axis('off')
plt.colorbar()
plt.title('non-local means\n(2)')

plt.show()

plt.figure(figsize=(8, 8))
plt.subplot(321)
plt.imshow(noisy, interpolation='bilinear')
plt.axis('off')
plt.colorbar()
plt.title('noisy')

plt.subplot(322)
plt.hist(noisy.ravel(), bins=100);
plt.grid()

plt.subplot(323)
plt.imshow(denoise, interpolation='bilinear')
plt.axis('off')
plt.colorbar()
plt.title('non-local means\n(1)')

plt.subplot(324)
plt.hist(denoise.ravel(), bins=100);
plt.grid()

plt.subplot(325)
plt.imshow(denoise_fast, interpolation='bilinear')
plt.axis('off')
plt.colorbar()
plt.title('non-local means\n(2)')

plt.subplot(326)
plt.hist(denoise_fast.ravel(), bins=100);
plt.grid()

plt.show()

# %%
save_amira(rec_vol_filtered, tmp_dir, tomo_info['specimen'], 3)

# %%
with h5py.File(os.path.join(tmp_dir, 'tomo_rec.h5'), 'w') as h5f:
    h5f.create_dataset('Reconstruction', data=rec_vol_filtered, chunks=True,
                       compression='lzf')

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

# %% [markdown]
# # Changelog:
# * 2.2а (2020.03.18)
#  - Add auto bh option
#  - Remove sinogram normalization
#  - move a lot of fuctions in tomotools
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
