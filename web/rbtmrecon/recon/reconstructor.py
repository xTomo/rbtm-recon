# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
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

from .tomotools import (STORAGE_SERVER, safe_median, recon_2d_parallel, get_tomoobject_info, get_experiment_hdf5,
                        mkdir_p, show_exp_data, load_tomo_data, find_good_frames, group_data, correct_rings, tqdm,
                        persistent_array,
                        get_angles_at_180_deg, test_rec, save_amira, show_frames_with_border)

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

# %%
recon_config = {'sample': tomo_info}

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
def find_roi(data_images, empty_beam):
    te = np.asarray(empty_beam)
    te[te < 1] = 1
    x_mins = []
    x_maxs = []
    y_mins = []
    y_maxs = [te.shape[1]]
    for ia in tqdm(np.argsort(data_angles)[::len(data_angles) // 8]):
        td = np.asarray(data_images[ia])
        td[td < 1] = 1

        d = np.log(te) - np.log(td)
        d[d < 0] = 0
        q = d > np.percentile(np.asarray(d[::2, ::2]), 10)
        mask = scipy.ndimage.binary_opening(q, np.ones((9, 9), dtype=int))

        x_mask = np.argwhere(np.sum(mask, axis=1) > 10)  # np.percentile(mask, 99.9, axis=1)
        x_min = np.min(x_mask)
        x_max = np.max(x_mask)

        y_mask = np.argwhere(np.sum(mask, axis=0) > 10)  # np.percentile(mask, 99.9, axis=1)
        y_min = np.min(y_mask)
        y_max = np.max(y_mask)

        x_mins.append(x_min)
        y_mins.append(y_min)
        x_maxs.append(x_max)
        y_maxs.append(y_max)
    #         plt.figure(figsize=(10,10))
    #         plt.imshow(d, vmin=-1, vmax=2, cmap=plt.cm.gray)
    #         plt.hlines([x_min, x_max], y_min, y_max, 'g')
    #         plt.vlines([y_min, y_max], x_min, x_max, 'r')
    #         plt.imshow(mask, cmap=plt.cm.gray)
    #         plt.colorbar()
    #         plt.show()
    x_min = np.maximum(0, np.min(x_mins) - 50)
    y_min = np.maximum(0, np.min(y_mins) - 50)
    x_max = np.minimum(te.shape[0] - 1, np.max(x_maxs) + 50)
    y_max = np.minimum(te.shape[1] - 1, np.max(y_maxs) + 50)

    return x_min, x_max, y_min, y_max


# %%
# TODO: store this in ini file
x_min, x_max, y_min, y_max = find_roi(data_images, empty_beam)
print("x_min, x_max, y_min, y_max = ", x_min, x_max, y_min, y_max)

# %%
ff = ipywidgets.interact_manual(show_frames_with_border, data_images=ipywidgets.fixed(data_images),
                                empty_beam=ipywidgets.fixed(empty_beam),
                                data_angles=ipywidgets.fixed(data_angles),
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
empty_beam = empty_beam[x_min:x_max, y_min:y_max]

# %%
good_frames = find_good_frames(data_images_crop, data_angles)

# %% [markdown]
# # Remove bad frames

# %%
data_images_good, _ = persistent_array(os.path.join(tmp_dir, 'data_images_good.tmp'),
                                       shape=(len(good_frames), data_images_crop.shape[1], data_images_crop.shape[2]),
                                       dtype='float32')

# TODO: Profile this code. In case if no bad frames, just skip it
# for i in tqdm(range(len(good_frames))):
data_images_good[:] = data_images_crop[good_frames]

data_angles = data_angles[good_frames]

# %%
uniq_data_images, uniq_angles = group_data(data_images_good, data_angles, tmp_dir)

# %%
sinogram, _ = persistent_array(os.path.join(tmp_dir, 'sinogram.tmp'), shape=uniq_data_images.shape,
                               dtype='float32')
te = np.asarray(empty_beam)
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
rc_level = 10

# %%
preview_slice_number = int(sinogram.shape[-1] // 2)
tmp_sinogram = sinogram[np.argsort(uniq_angles), :, preview_slice_number]
ring_corr = correct_rings(tmp_sinogram, rc_level)

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

# %%
for s in tqdm(range(sinogram.shape[1])):
    sinogram[:, s, :] = correct_rings(sinogram[:, s, :], rc_level)

# %%
position_0, position_180 = get_angles_at_180_deg(uniq_angles)
print(position_0, position_180)
print(uniq_angles[position_0], uniq_angles[position_180])

data_0_orig = np.rot90(sinogram[position_0]).copy()
data_180_orig = np.fliplr(np.rot90(sinogram[position_180]).copy())
data_0 = data_0_orig
data_180 = data_180_orig
transorm_result = ird.similarity(data_0, data_180, order=1, numiter=5, constraints={'scale': (1., 0)})

# %%
fig = plt.figure(figsize=(12, 12))
ird.imshow(data_0, data_180, transorm_result['timg'], fig=fig)
plt.show()

# %%
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
plt.imshow(data_0, vmin=0, vmax=im_max, cmap=plt.cm.gray_r)
plt.axis('tight')
plt.title('a')
plt.xlabel('Каналы детектора')
plt.ylabel('Каналы детектора')
cbar = plt.colorbar()
cbar.set_label('Поглощение, усл.ед.', rotation=90)

plt.subplot(222)
plt.imshow(data_180, vmin=0, vmax=im_max, cmap=plt.cm.gray_r)
plt.axis('tight')
plt.title('б')
plt.xlabel('Каналы детектора')
plt.ylabel('Каналы детектора')
cbar = plt.colorbar()
cbar.set_label('Поглощение, усл.ед.', rotation=90)

plt.subplot(223)
plt.imshow(data_0 - data_180, vmin=-im_max / 2, vmax=im_max / 2, cmap=plt.cm.seismic)
plt.axis('tight')
plt.title('в')
plt.xlabel('Каналы детектора')
plt.ylabel('Каналы детектора')
cbar = plt.colorbar()
cbar.set_label('Поглощение, усл.ед.', rotation=90)

tt_180 = np.fliplr(ird.imreg.transform_img_dict(np.fliplr(data_180), tr_dict, order=1))
tt_0 = ird.imreg.transform_img_dict(data_0, tr_dict, order=1)

plt.subplot(224)
plt.imshow(tt_0 - tt_180, vmin=-im_max / 2, vmax=im_max / 2, cmap=plt.cm.seismic)
plt.axis('tight')
plt.title('г')
plt.xlabel('Каналы детектора')
plt.ylabel('Каналы детектора')
cbar = plt.colorbar()
cbar.set_label('Поглощение, усл.ед.', rotation=90)

# %%
tim = ird.imreg.transform_img_dict(data_0, tr_dict)
sinogram_fixed, _ = persistent_array(os.path.join(tmp_dir, 'sinogram_fixed.tmp'),
                                     shape=(sinogram.shape[0], tim.shape[1], tim.shape[0]),
                                     dtype='float32', force_create=True)

# fix axis tlit
for i in tqdm(range(sinogram.shape[0])):
    sinogram_fixed[i] = np.rot90(
        ird.imreg.transform_img_dict(np.rot90(sinogram[i]), tr_dict, order=1),
        -1)

# %%
s1_angles = uniq_angles
s1 = np.require(sinogram_fixed[:, :, preview_slice_number],
                dtype=np.float32, requirements=['C'])

# %%
test_rec(s1, uniq_angles)

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
# s2 = np.require(sinogram_fixed[:, :, preview_slice_number],
#                 dtype=np.float32, requirements=['C'])

# %%
del data_0_orig, data_180_orig, data_images_good, data_images_crop, data_images
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

sss = s1[..., preview_slice_number]
t_angles = (uniq_angles - uniq_angles.min()) < 180  # remove angles >180

s4 = sss.copy()
s4[s4 < 0] = 0
s4 = np.power(s4, bh_corr)

rec_slice = recon_2d_parallel(s4[t_angles], uniq_angles[t_angles])

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
recon_config['corr'] = {'bh': bh_corr}

# %%
# from scipy.optimize import curve_fit
# from scipy.signal import medfilt


# def gauss(x, *p):
#     A, mu, sigma = p
#     return np.abs(A) * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))


# def gauss2(x, *p):
#     return gauss(x, p[0], p[1], p[2]) + gauss(x, p[3], p[4], p[5])


# def optimize_2gaussian(rec, mask):
#     hist, bin_edges = np.histogram(rec[mask], bins=1000)
#     bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

#     p0 = [1.e3, 0.0, 0.01,
#           1.e3, 0.3, 0.01]  # A1, mu1, sigma1, A2, mu2, sigma2

#     coeff, var_matrix = curve_fit(gauss2, bin_centres, hist, p0=p0)

#     coeff_g, var_matrix_g = curve_fit(gauss, bin_centres, hist, p0=[1., 0., 1.])

#     # Get the fitted curve
#     hist_fit = gauss2(bin_centres, *coeff)
#     hist_fit1 = gauss(bin_centres, coeff[0], coeff[1], coeff[2])
#     hist_fit2 = gauss(bin_centres, coeff[3], coeff[4], coeff[5])

#     hist_fit_g = gauss(bin_centres, *coeff_g)

#     plt.figure(figsize=(12, 10))
#     plt.subplot(311)
#     plt.title('Histogram. bh_corr= {:.3f}'.format(bh_corr))
#     plt.plot(bin_centres, hist, label='experiment data')
#     plt.plot(bin_centres, hist_fit, lw=2, label='Sum of 2 gausians fit. L2={:.1f}'.format(
#         np.mean(np.sqrt(np.sum((hist - hist_fit) ** 2)))))
#     plt.plot(bin_centres, hist_fit1, label='1st gausians')
#     plt.plot(bin_centres, hist_fit2, label='2nd gausians')
#     plt.plot(bin_centres, hist_fit_g, 'k', lw=2, label='Single Gaussian. L2={:.1f}'.format(
#         np.mean(np.sqrt(np.sum((hist - hist_fit_g) ** 2)))))
#     plt.grid()
#     plt.legend()

#     plt.subplot(312)
#     plt.title('Central cut')
#     plt.plot(medfilt(rec[rec.shape[0] // 2], 9))
#     plt.plot(medfilt(rec[:, rec.shape[1] // 2], 9))
#     plt.grid()

#     plt.subplot(337)
#     t = mask
#     ds = 500
#     plt.imshow(t[t.shape[0] // 2 - ds:t.shape[0] // 2 + ds,
#                t.shape[1] // 2 - ds:t.shape[1] // 2 + ds],
#                cmap=plt.cm.viridis)

#     plt.subplot(338)
#     t = rec * mask
#     ds = 500
#     plt.imshow(safe_median(t[
#                            t.shape[0] // 2 - ds:t.shape[0] // 2 + ds,
#                            t.shape[1] // 2 - ds:t.shape[1] // 2 + ds]),
#                cmap=plt.cm.viridis)

#     plt.subplot(339)
#     t = rec * mask
#     ds = 200
#     plt.imshow(safe_median(t[
#                            t.shape[0] // 2 - ds:t.shape[0] // 2 + ds,
#                            t.shape[1] // 2 - ds:t.shape[1] // 2 + ds]),
#                cmap=plt.cm.viridis)

#     plt.show()


# def create_circle_mask(x, y, r, size):
#     X, Y = np.meshgrid(np.arange(size), np.arange(size))
#     X = X - x
#     Y = Y - y
#     R = np.sqrt(X ** 2 + Y ** 2)
#     mask = R < r
#     return mask


# mask = create_circle_mask(870, 870, 470, rec_slice.shape[0])

# sss = s1[..., int(s1.shape[-1] // 2)]
# t_angles = (uniq_angles - uniq_angles.min()) < 180  # remove angles >180

# for bh_corr_t in np.arange(1, 5, 0.5):
#     print(bh_corr_t)
#     s4 = sss.copy()
#     s4[s4 < 0] = 0
#     s4 = np.power(s4, bh_corr_t)
#     s4 = s4 / np.mean(s4) * np.mean(sss)

#     rec_slice = recon_2d_parallel(s4[t_angles], uniq_angles[t_angles])
#     optimize_2gaussian(rec_slice, mask)

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
rec_vol_filtered = rec_vol

# %%
for j in range(3):
    for i in range(10):
        plt.figure(figsize=(10, 8))
        plt.imshow(rec_vol_filtered.take(i * rec_vol_filtered.shape[j] // 10, axis=j),
                   cmap=plt.cm.viridis, vmin=0)
        plt.axis('image')
        plt.title(i * i * rec_vol_filtered.shape[j] // 10)
        plt.colorbar()
        plt.show()

# %%
save_amira(rec_vol_filtered, tmp_dir, tomo_info['specimen'], 3)

# %%
recon_config


# %%
def save_dict_to_hdf5(dic, filename):
    """
    ....
    """
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, int, float, np.int32, np.int64, np.float32, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save {} {} type'.format(item, type(item)))


def load_dict_from_hdf5(filename):
    """
    ....
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')


def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans


# %%
with h5py.File(os.path.join(tmp_dir, 'tomo_rec.' + tomo_info['specimen'] + '.h5'), 'w') as h5f:
    h5f.create_dataset('Reconstruction', data=rec_vol_filtered, chunks=True,
                       compression='lzf')
    recursively_save_dict_contents_to_group(h5f, '/recon_config/', recon_config)

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
