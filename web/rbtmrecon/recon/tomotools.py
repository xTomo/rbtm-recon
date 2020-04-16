import json
import logging
import os
import shutil
import time
from urllib.request import urlretrieve

import cv2
import dask.array as da
import h5py
import numpy as np
import pylab as plt
import requests
import scipy.ndimage
import scipy.optimize
from skimage.metrics import normalized_root_mse
from tqdm import tqdm_notebook as tqdm

import tomo.recon.astra_utils as astra_utils

# STORAGE_SERVER = "http://10.0.7.153:5006/"
STORAGE_SERVER = "http://rbtmstorage_server_1:5006/"


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)


def get_experiment_hdf5(experiment_id, output_dir, experiment_files_dir=None, storage_server=STORAGE_SERVER):
    data_file = os.path.join(output_dir, experiment_id + '.h5')
    logging.info('Output experiment HDF5 file: {}'.format(data_file))

    # check if file exist and can be read
    if os.path.isfile(data_file):
        try:
            with h5py.File(data_file, 'r') as h5f:
                pass
        except Exception as e:
            raise e
        else:
            logging.info('File exists. Use local copy')
            return data_file

    if experiment_files_dir is None:
        # download file
        hdf5_url = storage_server + 'storage/experiments/{}.h5'.format(
            experiment_id)
        logging.info('Downloading file: {}'.format(hdf5_url))

        remaining_download_tries = 5

        while remaining_download_tries > 0:
            try:
                urlretrieve(hdf5_url, filename=data_file)
                logging.info('Successfully downloaded: {}'.format(hdf5_url))
                time.sleep(0.1)
            except Exception as e:
                logging.warning("error downloading {}  on trial no {}: {}".format(
                    hdf5_url, 6 - remaining_download_tries, e))
                remaining_download_tries = remaining_download_tries - 1
                continue
            else:
                break
    else:
        # copy local file
        src_file = os.path.join(experiment_files_dir, experiment_id + '.h5')
        logging.info('Copyng local  file: {}'.format(src_file))
        shutil.copy(src_file, data_file)

    return data_file


def get_tomoobject_info(experiment_id, storage_server=STORAGE_SERVER):
    exp_info = json.dumps(({"_id": experiment_id}))
    experiment = requests.post(storage_server + 'storage/experiments/get',
                               exp_info, timeout=1000)
    experiment_info = json.loads(experiment.content)[0]
    return experiment_info


def get_mm_shape(data_file):
    if os.path.exists(data_file + '.size'):
        res = np.loadtxt(data_file + '.size').astype('uint16')
        if res.ndim > 0:
            return tuple(res)
        else:
            return res,
    else:
        return None


def load_create_mm(data_file, shape, dtype, force_create=True):
    if force_create:
        logging.info('Force create')
    elif os.path.exists(data_file):
        mm_shape = get_mm_shape(data_file)
        if (shape is None) and (mm_shape is not None):
            res = np.memmap(data_file, dtype=dtype, mode='r+', shape=mm_shape)
            logging.info('Loading existing file: {}'.format(data_file))
            return res, True
        elif (np.array(shape) == mm_shape).all():
            res = np.memmap(data_file, dtype=dtype, mode='r+', shape=shape)
            logging.info('Loading existing file: {}'.format(data_file))
            return res, True
        else:
            logging.info('Shape missmatch.')

    logging.info('Creating new file: {}'.format(data_file))
    res = np.memmap(data_file, dtype=dtype, mode='w+', shape=shape)
    np.savetxt(data_file + '.size', res.shape, fmt='%5u')
    return res, False


def get_frame_group(data_file, group_name, mmap_file_dir):
    with h5py.File(data_file, 'r') as h5f:
        images_count = len(h5f[group_name])
        images = None
        file_number = 0
        angles = None
        for k, v in tqdm(h5f[group_name].items()):
            if images is None:
                mm_shape = (images_count, v.shape[1], v.shape[0])
                images, is_images_exists = load_create_mm(
                    os.path.join(mmap_file_dir, 'group_' +
                                 group_name + '.tmp'),
                    shape=mm_shape, dtype='float32')

            if angles is None:
                angles, is_angles_exists = load_create_mm(
                    os.path.join(
                        mmap_file_dir, 'group_' + group_name + '_angles.tmp'),
                    shape=(images_count,), dtype='float32')

            if is_images_exists and is_angles_exists:
                logging.info(
                    'Images and angles in group {} found. Skip it'.format(
                        group_name))
                break

            attributes = json.loads(v.attrs[list(v.attrs)[0]])[0]
            angles[file_number] = attributes['frame']['object']['angle position']
            tmp_image = np.flipud(v.value.astype('float32').swapaxes(0, 1))
            images[file_number] = tmp_image
            file_number = file_number + 1

    return images, angles


def safe_median(data):
    m_data = cv2.medianBlur(data, 3)
    mask = np.abs(m_data - data) > 0.1 * np.abs(data)
    res = data.copy()
    res[mask] = m_data[mask]
    return res


def recon_2d_parallel(sino, angles):
    rec = astra_utils.astra_recon_2d_parallel(sino, angles, ['FBP_CUDA', ['CGLS_CUDA', 10]])
    pixel_size = 9e-3
    return rec / pixel_size


def show_exp_data(empty_beam, data_images):
    max_intensity = np.percentile(empty_beam[:], 90)
    plt.figure(figsize=(8, 12))
    plt.subplot(211)
    plt.imshow(empty_beam.T, vmin=0, vmax=max_intensity, cmap=plt.cm.gray, interpolation='bilinear')
    cbar = plt.colorbar()
    cbar.set_label('Интенсивность, усл.ед.', rotation=90)
    plt.title('Прямой пучок')
    plt.xlabel('Номер канала детектора')
    plt.ylabel('Номер канала детектора')

    plt.subplot(212)
    plt.imshow(data_images[0].T, vmin=0, vmax=max_intensity, cmap=plt.cm.gray, interpolation='bilinear')
    cbar = plt.colorbar()
    cbar.set_label('Интенсивность, усл.ед.', rotation=90)
    plt.title('Изображение объекта')
    plt.xlabel('Номер канала детектора')
    plt.ylabel('Номер канала детектора')
    plt.show()


def load_tomo_data(data_file, tmp_dir):
    empty_images, _ = get_frame_group(data_file, 'empty', tmp_dir)
    dark_images, _ = get_frame_group(data_file, 'dark', tmp_dir)

    empty_image = np.median(empty_images, axis=0)
    dark_image = np.median(dark_images, axis=0)

    empty_beam = empty_image - dark_image

    # Загружаем кадры с даннымии
    # TODO: добавить поддержку, когда много кадров на одном угле
    data_images, data_angles = get_frame_group(data_file, 'data', tmp_dir)

    data_images_clear = da.from_array(data_images, chunks=(1, 1024, 1024)) - dark_image
    return empty_beam, data_images_clear, data_angles


# TODO: Profile this function
def find_good_frames(data_images, data_angles):
    intensity = data_images.mean(axis=-1).mean(axis=-1)

    intensity_mask = (intensity < 1.2 * intensity.mean()) * (intensity > 0.8 * intensity.mean())  # dorp bad points
    good_frames = np.arange(len(intensity))[intensity_mask]

    intensity_t = intensity[good_frames]
    data_angles_t = data_angles[good_frames]

    plt.figure(figsize=(8, 5))
    plt.plot(data_angles[np.argsort(data_angles)],
             intensity[np.argsort(data_angles)],
             label='Before filtering')

    plt.hlines(np.median(intensity, axis=0), 0, np.max(data_angles), 'r', label='Reference value')

    plt.plot(data_angles_t[np.argsort(data_angles_t)],
             intensity_t[np.argsort(data_angles_t)],
             'g', label='After filtering')

    plt.xlabel('Angle')
    plt.ylabel('Frame mean intensity')
    plt.grid()
    plt.legend(loc=0)
    plt.show()
    return good_frames


def group_data(data_images, data_angles, mmap_file_dir):
    uniq_angles, _ = load_create_mm(
        os.path.join(mmap_file_dir, 'uniq_angles.tmp'),
        shape=(len(list(set(data_angles))),),
        dtype='float32', force_create=True)
    uniq_angles[:] = list(set(data_angles))

    uniq_data_images, _ = load_create_mm(
        os.path.join(mmap_file_dir, 'uniq_data_images.tmp'),
        shape=(len(uniq_angles), data_images.shape[1], data_images.shape[2]),
        dtype='float32', force_create=True)

    for ua_id, ua in tqdm(list(enumerate(uniq_angles))):
        indexes = np.argwhere(data_angles == uniq_angles[ua_id])
        if len(indexes) > 1:
            tmp_images = data_images[indexes]
            tmp_images = np.squeeze(tmp_images)
            mean_image = np.mean(tmp_images, axis=0)
            uniq_data_images[ua_id] = mean_image
        else:
            uniq_data_images[ua_id] = data_images[indexes]
    return uniq_data_images, uniq_angles


def correct_rings(sino0, level):
    def get_my_b(level):
        t = np.mean(sino0, axis=0)
        gt = scipy.ndimage.filters.gaussian_filter1d(t, level / 2.)
        return gt - t

    def get_my_a(level):
        my_b = get_my_b(level)
        return np.mean(my_b) / my_b.shape[0]

    my_a = get_my_a(level)
    my_b = get_my_b(level)

    res = sino0.copy()
    if not level == 0:
        res += sino0 * my_a + my_b

    return res


# # build frames for video
# images_dir = os.path.join(tmp_dir,'images')
# tomotools.mkdir_p(images_dir)
# im_max=np.max(sinogram)
# im_min=np.min(sinogram)
# print(im_min, im_max)
# for ia, a in tomotools.log_progress(list(enumerate(np.argsort(uniq_angles)))):
# #     print('{:34}'.format(ia))
#     plt.imsave(os.path.join(images_dir,'prj_{:03}.png'.format(ia)),
#                np.rot90(sinogram[a],3), vmin=im_min, vmax=im_max,
#                cmap=plt.cm.gray_r)

# # !cd {images_dir} && avconv -r 10 -i "prj_%03d.png" -b:v 1000k prj.avi
# # !cd {images_dir} && rm prj.mp4


def cv_rotate(x, angle):
    """
    Rotate square array using OpenCV2 around center of the array
    :param x: 2d numpy array
    :param angle: angle in degrees
    :return: rotated array
    """
    x_center = tuple(
        np.array((x.shape[1], x.shape[0]), dtype='float32') / 2.0 - 0.5)
    rot_mat = cv2.getRotationMatrix2D(x_center, angle, 1.0)
    xro = cv2.warpAffine(
        x, rot_mat, (x.shape[1], x.shape[0]), flags=cv2.INTER_LINEAR)
    return xro


def smooth(x):
    return x - scipy.ndimage.filters.gaussian_filter(x, 50) + scipy.ndimage.filters.gaussian_filter(x, 10)


def find_axis_posiotion(image_0, image_180):
    def corr(x):
        alfa = x[0]
        shift_x = int(x[1])
        if shift_x >= 0:
            t_180 = image_180[:, shift_x:]
            t_0 = image_0[:, shift_x:]
        else:
            t_180 = image_180[:, :shift_x]
            t_0 = image_0[:, :shift_x]

        tt_180 = np.fliplr(cv_rotate(t_180, alfa))
        tt_180 = cv2.medianBlur(tt_180, 3)  # *t_mask
        tt_0 = cv_rotate(t_0, alfa)
        tt_0 = cv2.medianBlur(tt_0, 3)  # *t_mask

        res = normalized_root_mse(tt_0, tt_180)

        return res

    s180 = image_180.sum(axis=0)
    r180 = np.flipud(np.arange(len(s180)))
    p180 = (s180 * r180).sum() / s180.sum()

    s0 = image_0.sum(axis=0)
    r0 = np.arange(len(s0))
    p0 = (s0 * r0).sum() / s0.sum()

    x0 = [1., 0.5 * (p0 - p180)]

    left = x0[1] - 200
    right = x0[1] + 200
    qq = [corr([0, q]) for q in np.arange(left, right)]
    min_pos = left + np.argmin(qq)
    if min_pos == left or min_pos == right:
        position_found = False
    else:
        position_found = True

    plt.figure()
    plt.plot(np.arange(left, right), qq)
    plt.grid()
    plt.show()

    while not position_found:
        if min_pos == left:
            right = left
            left = right - 200
        elif min_pos == right:
            left = right
            right = left + 200

        qq = [corr([0, q]) for q in np.arange(left, right)]
        min_pos = left + np.argmin(qq)
        if min_pos == left or min_pos == right:
            position_found = False
        else:
            position_found = True

        plt.figure()
        plt.plot(np.arange(left, right), qq)
        plt.grid()
        plt.show()

    shift_0 = min_pos
    x0 = [1., shift_0],
    res = scipy.optimize.minimize(corr, x0, method='Powell')
    return res


# seraching opposite frames (0 and 180 deg)
def get_angles_at_180_deg(uniq_angles):
    array_0 = np.asarray(uniq_angles) % 360
    cross_array = np.zeros((len(array_0), len(array_0)))
    for i in range(1, len(array_0)):
        cross_array[i] = np.roll(array_0, i)

    pos = np.argmin(np.abs(cross_array + 180 - array_0) % 360)
    print(pos)
    position_180 = pos % len(array_0)
    position_0 = (pos - position_180) // len(array_0)
    print(position_0, position_180)
    return position_0, position_180


def test_rec(s1, uniq_angles):
    plt.figure(figsize=(7, 7))
    plt.imshow(s1[np.argsort(uniq_angles)], interpolation='bilinear', cmap=plt.cm.gray_r)
    plt.colorbar()
    plt.show()

    bh_corr = 1.0
    t_angles = (uniq_angles - uniq_angles.min()) <= 180  # remove angles >180
    rec_slice = recon_2d_parallel(s1[t_angles], uniq_angles[t_angles] * np.pi / 180)

    plt.figure(figsize=(10, 8))
    plt.imshow(safe_median(rec_slice),
               vmin=0, vmax=np.percentile(rec_slice, 95) * 1.2, cmap=plt.cm.viridis)
    plt.axis('equal')
    plt.colorbar()
    plt.show()


def reshape_volume(volume, reshape):
    res = np.zeros([s // reshape for s in volume.shape], dtype='float32')
    xs, ys, zs = [s * reshape for s in res.shape]
    for x, y, z in np.ndindex(reshape, reshape, reshape):
        res += volume[x:xs:reshape, y:ys:reshape, z:zs:reshape]
    return res / reshape ** 3


def save_amira(in_array, out_path, name, reshape=3):
    data_path = str(out_path)
    os.makedirs(data_path, exist_ok=True)
    with open(os.path.join(data_path, name + '.raw'), 'wb') as amira_file:
        reshaped_vol = reshape_volume(in_array, reshape)
        reshaped_vol.tofile(amira_file)
        file_shape = reshaped_vol.shape
        with open(os.path.join(data_path, 'tomo.' + name + '.hx'), 'w') as af:
            af.write('# Amira Script\n')
            af.write('remove -all\n')
            af.write(r'[ load -raw ${SCRIPTDIR}/' + name + '.raw little xfastest float 1 ' +
                     str(file_shape[2]) + ' ' + str(file_shape[1]) + ' ' + str(file_shape[0]) +
                     ' 0 ' + str(file_shape[2] - 1) + ' 0 ' + str(file_shape[1] - 1) + ' 0 ' + str(file_shape[0] - 1) +
                     ' ] setLabel ' + name + '\n')

# %%
# import ipyvolume as ipv

# %%
# ipv.figure()
# ipv.volshow(reshape_volume(rec_vol_filtered,10),
#             max_shape=1024,
#             extent=[[0, rec_vol_filtered.shape[2]*9e-3],
#                    [0, rec_vol_filtered.shape[1]*9e-3],
#                    [0, rec_vol_filtered.shape[0]*9e-3]]
#            )
# ipv.xlim(0, rec_vol_filtered.shape[2]*9e-3)
# ipv.xlabel('mm')
# ipv.ylim(0, rec_vol_filtered.shape[1]*9e-3)
# ipv.ylabel('mm')
# ipv.zlim(0, rec_vol_filtered.shape[0]*9e-3)
# ipv.zlabel('mm')
# ipv.squarelim()
# # ipv.show()
# ipv.save(os.path.join(tmp_dir,'tomo.html'))
