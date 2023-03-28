# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# # %load_ext autoreload
# # %autoreload

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
import scipy.ndimage as ndi

import imreg_dft as ird
from tomopy.prep.stripe import remove_all_stripe
from tomopy.recon.rotation import find_center_vo

from tomotools import (STORAGE_SERVER, safe_median, recon_2d_parallel, get_tomoobject_info, get_experiment_hdf5,
                       mkdir_p, show_exp_data, load_tomo_data, tqdm, find_roi,
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
ff = ipywidgets.interact_manual(show_frames_with_border, data_images=ipywidgets.fixed(data_images[::10]),
                                empty_beam=ipywidgets.fixed(empty_beam),
                                data_angles=ipywidgets.fixed(data_angles),
                                image_id=ipywidgets.IntSlider(min=0, max=len(data_angles) // 10 - 1, step=1, value=0),
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
# don't check non unique files
from tomotools import group_data
uniq_data_images, uniq_angles = group_data(data_images_crop, data_angles, tmp_dir)
# uniq_data_images, uniq_angles = data_images_crop, data_angles[()]

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
cxy = [ndi.center_of_mass(sinogram[i]) for i in range(sinogram.shape[0])]
cxy = np.asarray(cxy)
plt.figure(figsize=(6,6))
plt.plot(uniq_angles, cxy[:,1], 'o')
plt.grid()
plt.show()

# %%
from tomotools import astra_utils
from tomopy.prep.stripe import remove_stripe_ti
from scipy.optimize import minimize_scalar, minimize, curve_fit
from skimage.measure import LineModelND, ransac
import cupy as cp
import cupyx.scipy.ndimage as cndi

def transfrom_image(im, shift_x, angle):
    imcu = cp.asarray(im)
    imcu = cndi.rotate(imcu, angle, order=2, reshape=False, mode='nearest')
    imcu = cndi.shift(imcu, [shift_x, 0], order=2, mode='nearest')
    return imcu.get()


# %%
def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1]) - 50

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def recon_2d_parallel_nonorm(sino, angles):  # used for axis search
    rec = astra_utils.astra_recon_2d_parallel(sino[angles<180], angles[angles<180], ['FBP_CUDA'])
    return rec

def fp_2d_parallel(sample, angles):
    rec = astra_utils.astra_fp_2d_parallel(sample, angles)
    return rec

def shift_sino(sino2d, shift):
    tmp_sino = scipy.ndimage.shift(sino2d, [0,shift], order=2, mode='constant')
    return tmp_sino

def calc_loss(sino2d, angles, shift, return_data=True):
    expand = 4*int(np.abs(shift))
    shifted_sino = np.pad(sino2d, [[0,0], [expand,expand]])
    
    shifted_sino = shift_sino(shifted_sino, shift)
    recon = recon_2d_parallel_nonorm(shifted_sino, angles)
    
    if not expand == 0:
        shifted_sino = shifted_sino[:, expand:-expand]
        recon = recon[expand:-expand, expand:-expand]
    
#     print(sino2d.shape, shifted_sino.shape, recon.shape, shift)
    
    mask = create_circular_mask(sino2d.shape[1],sino2d.shape[1])
    sino_back = fp_2d_parallel(recon*mask, angles)
    loss = np.mean((sino_back-shifted_sino)**4)
    
    if return_data:
        return loss, recon, sino_back, shifted_sino
    else:
        return loss

def linear_func(x, a, b):
    return a * x + b
    
def axis_search2(sinogram_mem, uniq_angles_mem, debug=False):
    n_slices = 10
    shift_points = []
    losses = []
    shift_0 = 0
    ds = sinogram_mem.shape[2]//n_slices//2
    for slice_idx in tqdm(range(1, n_slices)):
        slice_numb = sinogram_mem.shape[2]*slice_idx//n_slices
        
        sino2d = np.mean(sinogram_mem[:, :, slice_numb-ds:slice_numb+ds+1], axis=-1)
        angles = uniq_angles_mem
        indexes = np.argsort(angles)
        angles = angles[indexes]
        sino2d = sino2d[indexes,:]

        l_calc_loss = lambda shift: calc_loss(sino2d, angles, shift, return_data=False)
        optim = minimize_scalar(l_calc_loss, method='brent',
#                                 bounds=(sino2d.shape[1]//4, sino2d.shape[1]//4*3),
                                options = {'xtol':0.1})
        shift = optim['x']
        shift_points.append([slice_numb, shift])
        
        l, recon, sino_back, shifted_sino = calc_loss(sino2d, angles, shift , return_data=True)
        
        losses.append(l)
        if debug:    
            sino2d = remove_stripe_ti(sino2d[:, None, :])
            sino2d = np.squeeze(sino2d)
            l, recon, sino_back, shifted_sino = calc_loss(sino2d, angles, shift , return_data=True)
            plt.figure(figsize=(10,10))
            plt.title(optim['x'])
            plt.imshow(recon, vmin = np.percentile(recon,10), vmax = np.percentile(recon,99.9))
            #         plt.colorbar()
            plt.axis('tight')
            plt.show()
            plt.figure(figsize=(12,6))
            plt.subplot(121)
            plt.imshow(shifted_sino)
            plt.yticks(range(shifted_sino.shape[0])[::50], angles[::50])
            plt.axis('tight')
            plt.colorbar()
            plt.subplot(122)
            plt.imshow(sino_back-shifted_sino, cmap=plt.cm.seismic)
            plt.axis('tight')
            plt.colorbar()
            plt.show()
            
            
    
    #calculate shift and roteation angle
    shift_points = np.asarray(shift_points)
    
    xdata = shift_points[:,0]
    ydata = shift_points[:,1]

    popt, pcov = curve_fit(linear_func, xdata, ydata)
    
    shift_points = np.asarray(shift_points)
    
    #ransac
    data = np.column_stack([xdata, ydata])
    model = LineModelND()
    model.estimate(data)

    # robustly fit line only using inlier data with RANSAC algorithm
    model_robust, inliers = ransac(data, LineModelND, min_samples=4,
                                   residual_threshold=0.1, max_trials=10000)
    
    params = model_robust.params[1][1], model_robust.predict_y([0,])[0]
    
    if debug:
        plt.figure()
        plt.plot(xdata,ydata,'o')
        plt.plot(xdata, linear_func(xdata, *params), 'r-',
                 label='ransac fit: a=%5.3f, b=%5.3f' % tuple(params))

        plt.plot(xdata, linear_func(xdata, *popt), '--',
                 label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
        plt.grid()
        plt.legend()
        plt.show()
        
                       
    angle = np.mean([-np.rad2deg(np.arctan(params[0])), -np.rad2deg(np.arctan(popt[0]))])
    shift_x = np.mean([params[1], popt[1]])

#     angle = -np.rad2deg(np.arctan(popt[0]))
#     shift_x = popt[1]
    
#     angle = -np.rad2deg(np.arctan(params[0]))
#     shift_x = params[1]
    
    return angle, shift_x

def preview_axis_correction(sinogram_mem, uniq_angles_mem):
    n_slices = 10
    start_slice = sinogram_mem.shape[2]//n_slices
    for slice_idx in tqdm(range(1, n_slices)):
        slice_numb = start_slice*slice_idx
        sino2d = np.mean(sinogram_mem[:, :, slice_numb:slice_numb+1], axis=-1)
        angles = uniq_angles_mem
#         sino2d = ndi.median_filter(sino2d, 3)
        sino2d = remove_stripe_ti(sino2d[:, None, :])
        sino2d = np.squeeze(sino2d)
        l, recon, sino_back, tmp_sino = calc_loss(sino2d, angles, 0 , return_data=True)
        plt.figure(figsize=(10,10))
        plt.imshow(recon, vmin = np.percentile(recon,10), vmax = np.percentile(recon,99.9))
        plt.axis('tight')
        plt.show()
        
    plt.figure(figsize=(10,10))
    plt.imshow((sinogram_mem[0]-np.flipud(sinogram_mem[np.argmax(uniq_angles_mem)])).T,
              cmap=plt.cm.seismic)
    plt.axis('tight')
    plt.colorbar()


# %%
sinogram_mem, _ = persistent_array(os.path.join(tmp_dir, 'sinogram_fixed.tmp'),
                                     shape=(np.sum(uniq_angles<=180), sinogram.shape[1],sinogram.shape[2]),
                                   dtype='float32', force_create=True)

for iid, i in enumerate(np.argwhere(uniq_angles<=180)):
    sinogram_mem[iid] = sinogram[i]
    
uniq_angles_mem = uniq_angles[uniq_angles<=180]

res_angle = 0
res_shift_x = 0

for i in tqdm(range(5)):
    angle, shift_x = axis_search2(sinogram_mem, uniq_angles_mem, debug=True)
    res_angle += angle
    res_shift_x += shift_x
    
    print(res_angle, res_shift_x)
      
    for iid, i in enumerate(np.argwhere(uniq_angles<=180)):
        sinogram_mem[iid] = sinogram[i]

    for i in tqdm(range(sinogram_mem.shape[0])):
        sinogram_mem[i] = transfrom_image(sinogram_mem[i], res_shift_x, res_angle)
        
    plt.figure(figsize=(10,10))
    plt.imshow((sinogram_mem[0]-np.flipud(sinogram_mem[np.argmax(uniq_angles_mem)])).T,
          cmap=plt.cm.seismic)
    plt.axis('tight')
    plt.colorbar()
    plt.show()

preview_axis_correction(sinogram_mem, uniq_angles_mem)

# %%
# remove rings
for i in tqdm(range(sinogram_mem.shape[0])):
    t = sinogram_mem[i].copy()
#     t = ndi.median_filter(t, 3)
    t = np.squeeze(remove_stripe_ti(t[:, None, :]))
    sinogram_mem[i] = t

# %%
recon_config['axis_corr'] = {'shift_x': res_shift_x,
                             'alfa': res_angle
                             }

# %%
del uniq_angles
uniq_angles, _ = persistent_array(os.path.join(tmp_dir, 'group_data_angles.tmp'),
                                  shape=uniq_angles_mem.shape, force_create=True,
                                  dtype='float32')
uniq_angles[:] = uniq_angles_mem

# %%
del data_images_crop, data_images, data_angles
del sinogram, uniq_angles_mem, uniq_data_images, sinogram_mem

# %%
files_to_remove = glob(os.path.join(tmp_dir, '*.tmp'))
files_to_remove = [f for f in files_to_remove if f.split('/')[-1] not in [
    'group_data_angles.tmp', 'sinogram_fixed.tmp']]

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
uniq_angles, _ = persistent_array(os.path.join(tmp_dir, 'group_data_angles.tmp'),
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
# * 2.6.1 (2023.01.31)
#  - new axis search more stable
# * 2.6 (2023.01.12)
#  - new axis search based on reprojection
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
