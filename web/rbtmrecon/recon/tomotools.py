import matplotlib as mpl
try:
    mpl.use("Agg")
except:
    pass

import logging

import errno
import os
import h5py
import numpy as np
import pylab as plt
import cv2
import json
import requests
import time

from tqdm import tqdm_notebook

# STORAGE_SERVER = "http://10.0.7.153:5006/"
STORAGE_SERVER = "http://rbtmstorage_server_1:5006/"

import sys

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    # Not Python 3 - today, it is most likely to be Python 2
    # But note that this might need an update when Python 4
    # might be around one day
    from urllib import urlretrieve
    

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_experiment_hdf5(experiment_id, output_dir):
    data_file = os.path.join(output_dir, experiment_id + '.h5')
    logging.info('Output experiment HDF5 file: {}'.format(data_file))
    if not os.path.isfile(data_file):
        hdf5_url = STORAGE_SERVER + 'storage/experiments/{}.h5'.format(
            experiment_id)
        logging.info('Downloading file: {}'.format(hdf5_url))
        
        remaining_download_tries = 5

        while remaining_download_tries > 0 :
            try:
                urlretrieve(hdf5_url, filename=data_file)
                logging.info('Successfully downloaded: {}'.format(hdf5_url))
                time.sleep(0.1)
            except Exception as e:
                logging.warn("error downloading {}  on trial no {}: {}".format(hdf5_url, 6 - remaining_download_tries, e))
                remaining_download_tries = remaining_download_tries - 1
                continue
            else:
                break
        # try:
        #     urlretrieve(hdf5_url, filename=data_file)
        # except Exception as e:
        #     logging.warn("error downloading {}: {}".format(hdf5_url, e))
    else:
        logging.info('File exests. Use local copy')

    return data_file

def get_tomoobject_info(experiment_id):
    
    exp_info = json.dumps(({"_id": experiment_id}))
    experiment = requests.post(STORAGE_SERVER + 'storage/experiments/get',
                               exp_info, timeout=1000)
    experiment_info = json.loads(experiment.content)[0]
    return  experiment_info


def show_statistics(data_file):
    """
    Load data frames from hdf5 file and show intesity
    """
    plt.figure(figsize=(7, 5))
    for image_type in ['empty', 'dark', 'data']:
        frame_count = len(h5py.File(data_file, 'r')[image_type])
        logging.info('Frames count: {}'.format(frame_count))
        s = []
        with h5py.File(data_file, 'r') as h5f:
            for k, v in log_progress(h5f[image_type].items()):
                s.append([int(k), np.sum(np.log(v.value + 1))])
                del v

        y = [d[1] for d in s]
        x = [d[0] for d in s]

        plt.plot(x, y, 'o', label=image_type)
        # plt.gca().set_ylim([np.min(y)*0.9,np.max(y)*1.05])
        plt.grid(True)
    plt.ylabel('Frame number')
    plt.ylabel('Total intensity')
    plt.legend(loc=0)
    plt.show()


def get_mm_shape(data_file):
    if os.path.exists(data_file + '.size'):
        res = np.loadtxt(data_file + '.size').astype('uint16')
        if res.ndim > 0:
            return tuple(res)
        else:
            return (res,)
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
            return (res, True)
        elif (np.array(shape) == mm_shape).all():
            res = np.memmap(data_file, dtype=dtype, mode='r+', shape=shape)
            logging.info('Loading existing file: {}'.format(data_file))
            return (res, True)
        else:
            logging.info('Shape missmatch.')

    logging.info('Creating new file: {}'.format(data_file))
    res = np.memmap(data_file, dtype=dtype, mode='w+', shape=shape)
    np.savetxt(data_file + '.size', res.shape, fmt='%5u')
    return(res, False)


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
                    os.path.join(mmap_file_dir, 'group_' + group_name + '.tmp'),
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
            # exposure = attributes['frame']['image_data']['exposure']
            angles[file_number] = attributes['frame']['object']['angle position']
            tmp_image = np.flipud(v.value.astype('float32').swapaxes(0, 1))
#             tmp_image = median_filter(tmp_image,3)
            # tmp_image = cv2.medianBlur(tmp_image, 3)
            # TODO: fix exposition for dark frames
            images[file_number] = tmp_image  # /exposure
            file_number = file_number + 1

    return images, angles
