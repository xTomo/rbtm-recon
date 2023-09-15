import json
import logging
import os
import shutil
import time
from urllib.request import urlretrieve

import cv2
import h5py
import numpy as np
import pylab as plt
import requests
import scipy.ndimage
import scipy.optimize
from tqdm.notebook import tqdm  # noqa

import tomo.recon.astra_utils as astra_utils  # noqa

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
        except OSError as e:  # Seams file is damaged
            logging.info('Deleting damaged file: {}'.format(data_file))
            os.remove(data_file)

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


# def persistent_array(data_file, shape, dtype, force_create=True):
#     if force_create:
#         logging.info('Force create')
#         logging.info('Creating new file: {}'.format(data_file))
#         h5f = h5py.File(data_file, 'w')
#         res = h5f.create_dataset('data', dtype=dtype,
#                                  shape=shape)
#         return res, False

#     elif os.path.exists(data_file):
#         h5f = h5py.File(data_file, 'r+')
#         res = h5f['data']
#         logging.info('Loading existing file: {}'.format(data_file))
#         return res, True


def persistent_array(data_file, shape, dtype, force_create=True):
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


# def persistent_array(data_file, shape, dtype, force_create=True):
#     compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
#     if force_create:
#         logging.info('Force create')
#         logging.info('Creating new file: {}'.format(data_file))
#         res = zarr.open(data_file, dtype=dtype, mode='w',
#                         shape=shape)
#         return res, False
#
#     elif os.path.exists(data_file):
#         res = zarr.open(data_file, dtype=dtype, mode='a',
#                         shape=shape)
#         logging.info('Loading existing file: {}'.format(data_file))
#         return res, True


def get_frame_group(data_file, group_name, mmap_file_dir, dark_image):
    with h5py.File(data_file, 'r') as h5f:
        images_count = len(h5f[group_name])
        images = None
        file_number = 0
        angles = None
        for k, v in tqdm(h5f[group_name].items()):
            if images is None:
                mm_shape = (images_count, v.shape[1], v.shape[0])
                images, is_images_exists = persistent_array(
                    os.path.join(mmap_file_dir, 'group_' +
                                 group_name + '.tmp'),
                    shape=mm_shape, dtype='float32')

            if angles is None:
                angles, is_angles_exists = persistent_array(
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
            tmp_image = np.rot90(v[()])
            images[file_number] = tmp_image - dark_image
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
    empty_images, _ = get_frame_group(data_file, 'empty', tmp_dir, 0)
    dark_images, _ = get_frame_group(data_file, 'dark', tmp_dir, 0)

    empty_image = np.median(empty_images, axis=0)
    dark_image = np.median(dark_images, axis=0)

    empty_beam = empty_image - dark_image

    # Загружаем кадры с даннымии
    # TODO: добавить поддержку, когда много кадров на одном угле
    data_images, data_angles = get_frame_group(data_file, 'data', tmp_dir, dark_image)
    return empty_beam, data_images, data_angles


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
    uniq_angles, _ = persistent_array(
        os.path.join(mmap_file_dir, 'uniq_angles.tmp'),
        shape=(len(list(set(data_angles))),),
        dtype='float32', force_create=True)
    uniq_angles[:] = list(set(data_angles))

    uniq_data_images, _ = persistent_array(
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
# mkdir_p(images_dir)
# im_max=np.percentile(sinogram, 99.9)
# im_min=np.percentile(sinogram, 10)
# print(im_min, im_max)
# for ia, a in tqdm(list(enumerate(np.argsort(uniq_angles)))):
# #     print('{:34}'.format(ia))
#     plt.imsave(os.path.join(images_dir,'prj_{:03}.png'.format(ia)),
#                np.rot90(sinogram[a],3), vmin=im_min, vmax=im_max,
#                cmap=plt.cm.gray_r)

# !cd {images_dir} && ffmpeg -r 10 -i "prj_%03d.png" -b:v 1000k prj.avi
# !cd {images_dir} && rm prj.mp4

# seraching opposite frames (0 and 180 deg)
def get_angles_at_180_deg(uniq_angles):
    t = np.subtract.outer(uniq_angles, uniq_angles) % 360
    pos = np.argwhere(np.abs(t - 180) % 360 == 0)
    position_0 = []
    position_180 = []
    for tpos in pos:
        p0, p180 = tpos
        if p0 < p180:
            position_0.append(p0)
            position_180.append(p180)
    return position_0, position_180


def test_rec(s1, uniq_angles, vmaxk=1.):
    plt.figure(figsize=(7, 7))
    plt.imshow(s1[np.argsort(uniq_angles)], interpolation='bilinear', cmap=plt.cm.gray_r)
    plt.axis('tight')
    plt.colorbar()
    plt.show()

    bh_corr = 1.0
    t_angles = (uniq_angles - uniq_angles.min()) < 180  # remove angles >180
    rec_slice = recon_2d_parallel(s1[t_angles], uniq_angles[t_angles])

    plt.figure(figsize=(10, 8))
    plt.imshow(safe_median(rec_slice),
               vmin=np.percentile(rec_slice, 2), vmax=np.percentile(rec_slice, 98) * vmaxk, cmap=plt.cm.viridis)
    plt.axis('equal')
    plt.colorbar()
    plt.title('ddddd')
    plt.show()


def reshape_volume(volume, reshape):
    if reshape == 1:
        return volume

    res = np.zeros([s // reshape for s in volume.shape], dtype='float32')
    xs, ys, zs = [s * reshape for s in res.shape]
    for x, y, z in np.ndindex(reshape, reshape, reshape):
        res += volume[x:xs:reshape, y:ys:reshape, z:zs:reshape]
    return res / reshape ** 3


def save_amira(in_array, out_path, name, reshape=3, pixel_size=9.0e-3):
    data_path = str(out_path)
    os.makedirs(data_path, exist_ok=True)
    name = name.replace(' ', '_')
    with open(os.path.join(data_path, name + '.raw'), 'wb') as amira_file:
        reshaped_vol = reshape_volume(in_array, reshape)
        reshaped_vol.tofile(amira_file)
        file_shape = reshaped_vol.shape
        with open(os.path.join(data_path, 'tomo.' + name + '.hx'), 'w') as af:
            af.write('# Amira Script\n')
            # af.write('remove -all\n')
            template_str = '[ load -unit mm -raw ${{SCRIPTDIR}}/{}.raw ' + \
                           'little xfastest float 1 {} {} {}  0 {} 0 {} 0 {} ] setLabel {}\n'
            af.write(template_str.format(
                name,
                file_shape[2], file_shape[1], file_shape[0],
                pixel_size * reshape * (file_shape[2] - 1),
                pixel_size * reshape * (file_shape[1] - 1),
                pixel_size * reshape * (file_shape[0] - 1),
                name)
            )


def show_frames_with_border(data_images, empty_beam, data_angles, image_id, x_min, x_max, y_min, y_max):
    te = np.asarray(empty_beam).T
    te[te < 1] = 1

    angles_sorted_ind = np.argsort(data_angles)
    td = np.asarray(data_images[angles_sorted_ind[image_id]].T)
    td[td < 1] = 1

    d = np.log(te) - np.log(td)

    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.imshow(d, cmap=plt.cm.gray, vmin=np.percentile(d.flat, 1), vmax=np.percentile(d.flat, 99.9))
    plt.axis('image')
    plt.hlines([y_min, y_max], x_min, x_max, 'r')
    plt.vlines([x_min, x_max], y_min, y_max, 'g')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.subplot(122)
    plt.imshow(d[y_min:y_max, x_min:x_max], cmap=plt.cm.gray,
               vmin=np.percentile(d[y_min:y_max, x_min:x_max].flat, 1),
               vmax=np.percentile(d[y_min:y_max, x_min:x_max].flat, 99.9))
    plt.show()
    print("x_min, x_max, y_min, y_max = {}, {}, {}, {}".format(x_min, x_max, y_min, y_max))


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


# import ipyvolume as ipv

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

def find_roi(data_images, empty_beam, data_angles):
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
        q = d > np.percentile(np.asarray(d), 20)
        mask = scipy.ndimage.binary_opening(q, np.ones((9, 9), dtype=int))

        x_mask = np.argwhere(np.sum(mask, axis=1) > 20)  # np.percentile(mask, 99.9, axis=1)
        x_min = np.min(x_mask)
        x_max = np.max(x_mask)

        y_mask = np.argwhere(np.sum(mask, axis=0) > 20)  # np.percentile(mask, 99.9, axis=1)
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

# # For BH with 2 gaussans approximation

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
