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
import cupy as cp
from cupyx.scipy.ndimage import median_filter


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
        last_exception = None

        while remaining_download_tries > 0:
            try:
                urlretrieve(hdf5_url, filename=data_file)
                logging.info('Successfully downloaded: {}'.format(hdf5_url))
                time.sleep(0.1)
            except Exception as e:
                last_exception = e
                logging.warning("error downloading {}  on trial no {}: {}".format(
                    hdf5_url, 6 - remaining_download_tries, e))
                remaining_download_tries = remaining_download_tries - 1
                continue
            else:
                break
        else:
            raise RuntimeError(
                'Failed to download {} after 5 attempts'.format(hdf5_url)
            ) from last_exception
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


def persistent_array(data_file, shape, dtype, force_create=True):
    if force_create:
        logging.info('Force create')
    elif os.path.exists(data_file):
        mm_shape = get_mm_shape(data_file)
        if (shape is None) and (mm_shape is not None):
            res = np.memmap(data_file, dtype=dtype, mode='r+', shape=mm_shape)
            logging.info('Loading existing file: {}'.format(data_file))
            return res, True
        elif (mm_shape is not None) and (shape is not None) and \
                (len(shape) == len(mm_shape)) and \
                all(int(a) == int(b) for a, b in zip(shape, mm_shape)):
            res = np.memmap(data_file, dtype=dtype, mode='r+', shape=shape)
            logging.info('Loading existing file: {}'.format(data_file))
            return res, True
        else:
            logging.info('Shape mismatch: expected {}, found {}'.format(shape, mm_shape))

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


def get_frame_group(data_file, group_name, mmap_file_dir, num_workers=8,
                    hdf5_cache_mb=512):
    """Загружает группу кадров из HDF5-файла.

    Параметры
    ----------
    data_file : str
        Путь к HDF5-файлу.
    group_name : str
        Имя группы ('empty', 'dark', 'data' и т.д.).
    mmap_file_dir : str
        Каталог для временных mmap-файлов (не используется, сохранён для совместимости).
    num_workers : int
        Количество потоков для параллельного чтения (0 — без параллелизма).
    hdf5_cache_mb : int
        Размер chunk-кэша HDF5 в мегабайтах.
    """
    from concurrent.futures import ThreadPoolExecutor

    rdcc_nbytes = hdf5_cache_mb * 1024 * 1024

    # --- Первый проход: метаданные (быстро, один файловый дескриптор) ---
    with h5py.File(data_file, 'r', rdcc_nbytes=rdcc_nbytes) as h5f:
        group = h5f[group_name]
        keys = list(group.keys())          # стабильный порядок, один вызов
        images_count = len(keys)

        first_ds = group[keys[0]]
        frame_h, frame_w = first_ds.shape
        attr_key = list(first_ds.attrs)[0]  # имя атрибута — одинаково для всей группы

        # Pre-allocate: np.empty быстрее np.zeros (не обнуляет память)
        angles = np.empty((images_count,), dtype='float32')

        for i, k in enumerate(keys):
            ds = group[k]
            attributes = json.loads(ds.attrs[attr_key])[0]
            angles[i] = attributes['frame']['object']['angle position']

    images = np.empty((images_count, frame_h, frame_w), dtype='float32')

    # --- Второй проход: параллельное чтение пикселей ---
    def _read_one(args):
        idx, key = args
        # Каждый поток открывает свой дескриптор — безопасно для HDF5 в режиме 'r'
        with h5py.File(data_file, 'r', rdcc_nbytes=rdcc_nbytes) as f:
            images[idx] = f[group_name][key][()]

    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            list(tqdm(
                pool.map(_read_one, enumerate(keys)),
                total=images_count,
                desc=group_name,
            ))
    else:
        # Однопоточный fallback (один открытый файл — меньше overhead)
        with h5py.File(data_file, 'r', rdcc_nbytes=rdcc_nbytes) as h5f:
            group = h5f[group_name]
            for i, k in enumerate(tqdm(keys, desc=group_name)):
                images[i] = group[k][()]

    return images, angles


# def safe_median(data):
#     m_data = cv2.medianBlur(data, 3)
#     mask = np.abs(m_data - data) > 0.1 * np.abs(data)
#     res = data.copy()
#     res[mask] = m_data[mask]
#     return res


def safe_median(data):
    data_gpu = cp.asarray(data)
    m_data = median_filter(data_gpu, size=3)
    mask = cp.abs(m_data - data_gpu) > 0.1 * cp.abs(data_gpu)
    res = data_gpu.copy()
    res[mask] = m_data[mask]
    
    return cp.asnumpy(res)  # Конвертируем обратно в numpy

def recon_2d_parallel(sino, angles, pixel_size=9e-3):
    rec = astra_utils.astra_recon_2d_parallel(sino, angles, 
                                              ['FBP_CUDA', 
                                              ['CGLS_CUDA', 10]])
    return rec / pixel_size


def recon_2d_parallel_nonorm(sino, angles):  # used for axis search
    rec = astra_utils.astra_recon_2d_parallel(sino[angles<180], angles[angles<180], ['FBP_CUDA'])
    return rec

def preview_axis_correction(sinogram_mem, angles, remove_rings=False):
    # from tomopy.prep.stripe import remove_stripe_ti
    from tomo.remove_stripe import remove_stripe_ti, remove_all_stripe
    import cupy as cp
    import cupyx.scipy.ndimage as cndi
    
    if sinogram_mem.ndim >2:
        n_slices = np.min([10, sinogram_mem.shape[0]])
        start_slice = sinogram_mem.shape[0]//n_slices
    else: #for single sliced sinogram
        n_slices = 1
        start_slice = 0
        sinogram_mem = sinogram_mem[None,:,:] 
        
    for slice_idx in tqdm(range(0, n_slices)):
        slice_numb = start_slice*slice_idx
        # sino2d = np.mean(sinogram_mem[:, :, slice_numb:slice_numb+1], axis=-1)
        sino2d = sinogram_mem[slice_numb]
        print(sino2d.shape)
        if remove_rings:
            sino2d = remove_all_stripe(cp.asanyarray(sino2d[:, None, :])).get()
            sino2d = np.squeeze(sino2d)
        recon = recon_2d_parallel_nonorm(sino2d, angles)
        plt.figure(figsize=(10,10))
        plt.imshow(recon, vmin = np.percentile(recon,10), vmax = np.percentile(recon,99.9))
        plt.show()

def show_exp_data(empty_beam, data_images):
    plt.figure()
    plt.imshow(data_images[0]/empty_beam, vmin=0, vmax=1, cmap=plt.cm.gray, interpolation='bilinear')
    cbar = plt.colorbar()
    cbar.set_label('Интенсивность, усл.ед.', rotation=90)
    plt.title('Нормированное изображение объекта')
    plt.xlabel('Номер канала детектора')
    plt.ylabel('Номер канала детектора')
    plt.show()


def load_tomo_data(data_file, tmp_dir):
    empty_images, _ = get_frame_group(data_file, 'empty', tmp_dir)
    dark_images, _ = get_frame_group(data_file, 'dark', tmp_dir)

    empty_image = np.median(empty_images, axis=0)
    dark_image = np.median(dark_images, axis=0)

    empty_image -= dark_image

    # Загружаем кадры с даннымии
    # TODO: добавить поддержку, когда много кадров на одном угле
    data_images, data_angles = get_frame_group(data_file, 'data', tmp_dir)
    data_images -= dark_image

    return empty_image, data_images, data_angles


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
        gt = scipy.ndimage.gaussian_filter1d(t, level / 2.)
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


def reshape_volume(array_3d, binning_factor):
    """
    Более эффективная реализация pixel binning для трёхмерного numpy массива
    по всем трём осям.
    
    Параметры:
    array_3d (numpy.ndarray): Входной трёхмерный массив
    binning_factor (int): Коэффициент сжатия (должен быть целым числом > 0)
    
    Возвращает:
    numpy.ndarray: Сжатый массив
    """
    # Проверка входных данных
    if not isinstance(array_3d, np.ndarray) or array_3d.ndim != 3:
        raise ValueError("Входные данные должны быть трёхмерным numpy массивом")
    
    if not isinstance(binning_factor, int) or binning_factor <= 0:
        raise ValueError("Коэффициент сжатия должен быть положительным целым числом")
    
    # Получаем размеры исходного массива
    height, width, depth = array_3d.shape
    
    # Вычисляем новые размеры
    new_height = height // binning_factor
    new_width = width // binning_factor
    new_depth = depth // binning_factor
    
    # Обрезаем массив до размеров, кратных binning_factor
    trimmed_array = array_3d[:new_height*binning_factor, 
                             :new_width*binning_factor, 
                             :new_depth*binning_factor]
    
    # Изменяем форму массива для группировки вокселей
    reshaped = trimmed_array.reshape(new_height, binning_factor, 
                                     new_width, binning_factor, 
                                     new_depth, binning_factor)
    
    # Вычисляем среднее по группам вокселей
    result = reshaped.mean(axis=(1, 3, 5), dtype='float32')
    return result

def save_amira(in_array, out_path, name, reshape=3, pixel_size=9.0e-3):
    data_path = str(out_path)
    os.makedirs(data_path, exist_ok=True)
    name = name.replace(' ', '_')

    if reshape != 1:
        reshaped_vol = reshape_volume(in_array, reshape)
        file_shape = reshaped_vol.shape
        shape_str = f'{file_shape[0]}_{file_shape[1]}_{file_shape[2]}'
        out_name = f'{name}.{shape_str}.{reshape}.raw'
        with open(os.path.join(data_path, out_name), 'wb') as amira_file:
            reshaped_vol.tofile(amira_file)
    else:
        file_shape = in_array.shape
        shape_str = f'{file_shape[0]}_{file_shape[1]}_{file_shape[2]}'
        out_name = f'{name}.{shape_str}.{reshape}.raw'
        # with open(os.path.join(data_path, out_name), 'wb') as amira_file:
        #     in_array.tofile(amira_file)
            
    with open(os.path.join(data_path, f'tomo.{name}.{reshape}.hx'), 'w') as af:
        af.write('# Amira Script\n')
        # af.write('remove -all\n')
        template_str = '[ load -unit mm -raw ${{SCRIPTDIR}}/{} ' + \
                       'little xfastest float 1 {} {} {}  0 {} 0 {} 0 {} ] setLabel {}\n'
        af.write(template_str.format(
            out_name,
            file_shape[2], file_shape[1], file_shape[0],
            pixel_size * reshape * (file_shape[2] - 1),
            pixel_size * reshape * (file_shape[1] - 1),
            pixel_size * reshape * (file_shape[0] - 1),
            out_name)
        )


def show_frames_with_border(data_images, empty_beam, data_angles, image_id, x_min, x_max, y_min, y_max):
    te = empty_beam

    angles_sorted_ind = np.argsort(data_angles)
    td = np.asarray(data_images[angles_sorted_ind[image_id]])
    td[td < 1] = 1

    d = np.log(te) - np.log(td)

    filter_step = 2
    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.imshow(d, cmap=plt.cm.gray, 
               vmin=np.percentile(d[::filter_step,::filter_step], 1), 
               vmax=np.percentile(d[::filter_step,::filter_step], 99.9))
    plt.axis('image')
    plt.hlines([y_min, y_max], x_min, x_max, 'r')
    plt.vlines([x_min, x_max], y_min, y_max, 'g')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.subplot(122)
    plt.imshow(d[y_min:y_max, x_min:x_max], cmap=plt.cm.gray,
               vmin=np.percentile(d[y_min:y_max:filter_step, x_min:x_max:filter_step], 1),
               vmax=np.percentile(d[y_min:y_max:filter_step, x_min:x_max:filter_step], 99.9))
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
        if isinstance(item, h5py.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans


def find_roi(data_images, empty_beam, data_angles):
    te = np.asarray(empty_beam)
    te[te < 1] = 1
    x_mins = []
    x_maxs = []
    y_mins = []
    y_maxs = []
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

    x_min = np.maximum(0, np.min(x_mins) - 50)
    y_min = np.maximum(0, np.min(y_mins) - 50)
    x_max = np.minimum(te.shape[0] - 1, np.max(x_maxs) + 50)
    y_max = np.minimum(te.shape[1] - 1, np.max(y_maxs) + 50)

    return x_min, x_max, y_min, y_max

