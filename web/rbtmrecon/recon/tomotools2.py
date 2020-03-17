import errno
import json
import logging
import os
import shutil
import time
from urllib.request import urlretrieve

import h5py
import numpy as np
import requests
from tqdm import tqdm_notebook

# STORAGE_SERVER = "http://10.0.7.153:5006/"
STORAGE_SERVER = "http://rbtmstorage_server_1:5006/"


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


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


def load_create_mm(data_file, shape, dtype, force_create=False):
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
        for k, v in tqdm_notebook(h5f[group_name].items()):
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
