"""tomotools4.py — расширение tomotools2 с поддержкой AdvancedExperiment.

Новое в v4:
  - is_advanced_experiment()          — детектор формата
  - AdvancedTomoData                  — dataclass с разделёнными данными
  - load_tomo_data_advanced()         — загрузка advanced HDF5
  - load_tomo_data()                  — автодетекция (backward compat)
  - normalize_projections_with_timeline() — нормировка с интерполяцией empty
  - measure_repositioning_shifts()    — кросс-корреляция data vs data_check
  - apply_repositioning_correction()  — sub-pixel сдвиг кадров in-place
  - analyze_source_drift()            — визуализация дрейфа источника
  - analyze_repositioning_accuracy()  — визуализация точности позиционирования

get_frame_group() расширена параметром return_frame_numbers=False
(backward compatible).
"""
import configparser
import dataclasses
import json
import logging
import os
import shutil
import time
from urllib.request import urlretrieve

import h5py
import numpy as np
import pylab as plt
import requests
import scipy.ndimage as ndi
import scipy.optimize as optimize
import scipy.interpolate as interp
from skimage.registration import phase_cross_correlation
import cupy as cp
import cupyx.scipy.ndimage as cndi
from cupyx.scipy.ndimage import median_filter
from tqdm.notebook import tqdm  # noqa

import tomo.recon.astra_utils as astra_utils  # noqa

# STORAGE_SERVER = "http://10.0.7.153:5006/"
STORAGE_SERVER = "http://rbtmstorage_server_1:5006/"


# =============================================================================
# --- I/O ---
# =============================================================================

def mkdir_p(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_experiment_hdf5(experiment_id: str, output_dir: str,
                        experiment_files_dir: str | None = None,
                        storage_server: str = STORAGE_SERVER) -> str:
    """Возвращает путь к локальному HDF5-файлу эксперимента.

    Если файл уже существует и читается — возвращает его.
    Иначе копирует из ``experiment_files_dir`` или скачивает с сервера.
    """
    data_file = os.path.join(output_dir, experiment_id + '.h5')
    logging.info('Output experiment HDF5 file: {}'.format(data_file))

    if os.path.isfile(data_file):
        try:
            with h5py.File(data_file, 'r'):
                pass
        except OSError:
            logging.info('Deleting damaged file: {}'.format(data_file))
            os.remove(data_file)
        except Exception as e:
            raise e
        else:
            logging.info('File exists. Use local copy')
            return data_file

    if experiment_files_dir is None:
        hdf5_url = storage_server + 'storage/experiments/{}.h5'.format(experiment_id)
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
                logging.warning("error downloading {} on trial no {}: {}".format(
                    hdf5_url, 6 - remaining_download_tries, e))
                remaining_download_tries -= 1
                continue
            else:
                break
        else:
            raise RuntimeError(
                'Failed to download {} after 5 attempts'.format(hdf5_url)
            ) from last_exception
    else:
        src_file = os.path.join(experiment_files_dir, experiment_id + '.h5')
        logging.info('Copying local file: {}'.format(src_file))
        shutil.copy(src_file, data_file)

    return data_file


def get_tomoobject_info(experiment_id: str, storage_server: str = STORAGE_SERVER) -> dict:
    """Возвращает метаданные эксперимента из хранилища."""
    exp_info = json.dumps({"_id": experiment_id})
    experiment = requests.post(storage_server + 'storage/experiments/get',
                               exp_info, timeout=1000)
    return json.loads(experiment.content)[0]


def get_mm_shape(data_file: str) -> tuple | None:
    """Читает форму memmap-массива из .size файла рядом с data_file."""
    size_file = data_file + '.size'
    if os.path.exists(size_file):
        res = np.loadtxt(size_file).astype('uint16')
        return tuple(res) if res.ndim > 0 else (res,)
    return None


def persistent_array(data_file: str, shape: tuple | None,
                     dtype: str | np.dtype | type,
                     force_create: bool = True) -> tuple[np.memmap, bool]:
    """Создаёт или открывает memmap-массив.

    Возвращает (array, loaded_from_disk).
    """
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


def get_frame_group(data_file: str, group_name: str, mmap_file_dir: str,
                    num_workers: int = 8, hdf5_cache_mb: int = 512,
                    return_frame_numbers: bool = False,
                    ) -> tuple:
    """Загружает группу кадров из HDF5-файла.

    Параметры
    ----------
    data_file : str
        Путь к HDF5-файлу.
    group_name : str
        Имя группы ('empty', 'dark', 'data', 'data_check' и т.д.).
    mmap_file_dir : str
        Каталог для временных mmap-файлов (не используется, сохранён для совместимости).
    num_workers : int
        Количество потоков для параллельного чтения (0 — без параллелизма).
    hdf5_cache_mb : int
        Размер chunk-кэша HDF5 в мегабайтах.
    return_frame_numbers : bool
        Если True — возвращает третий элемент: массив глобальных номеров кадров
        (int-ключи датасетов внутри группы). Дефолт False (backward compatible).

    Возвращает
    ----------
    images       : np.ndarray, shape (N, H, W)
    angles       : np.ndarray, shape (N,)
    frame_numbers: np.ndarray, shape (N,)  — только если return_frame_numbers=True
    """
    from concurrent.futures import ThreadPoolExecutor

    rdcc_nbytes = hdf5_cache_mb * 1024 * 1024

    with h5py.File(data_file, 'r', rdcc_nbytes=rdcc_nbytes) as h5f:
        group = h5f[group_name]  # type: ignore[index]
        keys = list(group.keys())  # type: ignore[union-attr]
        images_count = len(keys)
        first_ds = group[keys[0]]  # type: ignore[index]
        frame_h, frame_w = first_ds.shape  # type: ignore[union-attr]
        angles = np.empty((images_count,), dtype='float32')
        for i, k in enumerate(keys):
            ds = group[k]  # type: ignore[index]
            attributes = json.loads(str(ds.attrs['frame_info']))[0]  # type: ignore[union-attr]
            angles[i] = attributes['frame']['object']['angle position']

    frame_numbers = np.array([int(k) for k in keys], dtype=np.int64)

    images = np.empty((images_count, frame_h, frame_w), dtype='float32')

    def _read_one(args):
        idx, key = args
        with h5py.File(data_file, 'r', rdcc_nbytes=rdcc_nbytes) as f:
            images[idx] = f[group_name][key][()]  # type: ignore[index]

    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            list(tqdm(pool.map(_read_one, enumerate(keys)),
                      total=images_count, desc=group_name))
    else:
        with h5py.File(data_file, 'r', rdcc_nbytes=rdcc_nbytes) as h5f:
            group = h5f[group_name]  # type: ignore[index]
            for i, k in enumerate(tqdm(keys, desc=group_name)):
                images[i] = group[k][()]  # type: ignore[index]

    if return_frame_numbers:
        return images, angles, frame_numbers
    return images, angles


# =============================================================================
# --- Advanced experiment detection and data structures ---
# =============================================================================

def is_advanced_experiment(data_file: str) -> bool:
    """Возвращает True, если HDF5 содержит непустую группу data_check.

    Признак продвинутого (AdvancedExperiment) формата данных.
    """
    with h5py.File(data_file, 'r') as h5f:
        return 'data_check' in h5f and len(h5f['data_check']) > 0


@dataclasses.dataclass
class AdvancedTomoData:
    """Контейнер данных продвинутого эксперимента.

    Поля
    ----
    dark_image               : медиана dark кадров минус нуль, shape (H, W)
    initial_empty            : медиана начальной empty серии (dark-subtracted), shape (H, W)
    periodic_empties         : list из K np.ndarray shape (H, W) — медианы periodic серий
    periodic_empty_fnumbers  : list из K int — глобальный frame_number первого кадра
                               каждой periodic серии
    data_images              : dark-subtracted проекции, shape (N, H, W)
    data_angles              : углы проекций, shape (N,)
    data_numbers             : глобальные frame_numbers data кадров, shape (N,)
    data_check_images        : dark-subtracted контрольные кадры, shape (M, H, W)
    data_check_angles        : углы data_check, shape (M,)
    data_check_numbers       : глобальные frame_numbers data_check, shape (M,)
    series_length            : длина каждой empty/dark серии
    """
    dark_image:               np.ndarray
    initial_empty:            np.ndarray
    periodic_empties:         list         # list of np.ndarray
    periodic_empty_fnumbers:  list         # list of int
    data_images:              np.ndarray
    data_angles:              np.ndarray
    data_numbers:             np.ndarray
    data_check_images:        np.ndarray
    data_check_angles:        np.ndarray
    data_check_numbers:       np.ndarray
    series_length:            int


def load_tomo_data_advanced(data_file: str, tmp_dir: str) -> AdvancedTomoData:
    """Загружает данные advanced эксперимента из HDF5-файла.

    Алгоритм:
      1. Загружает dark → dark_image = median
      2. Загружает empty с frame_numbers → сортирует по frame_number
      3. Читает series_length из exp_info атрибута HDF5
      4. Первые series_length пустых кадров → initial_empty
      5. Оставшиеся, группами по series_length → periodic_empties
      6. Загружает data с frame_numbers → вычитает dark_image
      7. Загружает data_check с frame_numbers → вычитает dark_image

    Возвращает
    ----------
    AdvancedTomoData
    """
    # --- Dark ---
    dark_images, _ = get_frame_group(data_file, 'dark', tmp_dir)
    dark_image = np.median(dark_images, axis=0).astype('float32')
    del dark_images

    # --- Empty: загружаем с frame_numbers и сортируем ---
    empty_images_raw, empty_angles_raw, empty_fnums_raw = get_frame_group(
        data_file, 'empty', tmp_dir, return_frame_numbers=True)

    sort_idx = np.argsort(empty_fnums_raw)
    empty_images = empty_images_raw[sort_idx]
    empty_fnums = empty_fnums_raw[sort_idx]
    del empty_images_raw, empty_angles_raw, empty_fnums_raw

    # Читаем series_length из метаданных HDF5
    series_length = _read_series_length_from_hdf5(data_file, empty_images.shape[0])

    # Вычитаем dark
    empty_images = empty_images.astype('float32') - dark_image

    # Начальная серия
    initial_empty_frames = empty_images[:series_length]
    initial_empty = np.median(initial_empty_frames, axis=0).astype('float32')

    # Периодические серии
    periodic_empties = []
    periodic_empty_fnumbers = []
    remaining = empty_images[series_length:]
    remaining_fnums = empty_fnums[series_length:]
    n_periodic = len(remaining) // series_length
    for k in range(n_periodic):
        start = k * series_length
        end = start + series_length
        chunk = remaining[start:end]
        periodic_empties.append(np.median(chunk, axis=0).astype('float32'))
        periodic_empty_fnumbers.append(int(remaining_fnums[start]))

    del empty_images, remaining

    # --- Data ---
    data_images_raw, data_angles, data_numbers = get_frame_group(
        data_file, 'data', tmp_dir, return_frame_numbers=True)

    sort_idx = np.argsort(data_numbers)
    data_images = data_images_raw[sort_idx].astype('float32') - dark_image
    data_angles = data_angles[sort_idx]
    data_numbers = data_numbers[sort_idx]
    del data_images_raw

    # --- Data check ---
    data_check_raw, data_check_angles_raw, data_check_fnums_raw = get_frame_group(
        data_file, 'data_check', tmp_dir, return_frame_numbers=True)

    sort_idx = np.argsort(data_check_fnums_raw)
    data_check_images = data_check_raw[sort_idx].astype('float32') - dark_image
    data_check_angles = data_check_angles_raw[sort_idx]
    data_check_numbers = data_check_fnums_raw[sort_idx]
    del data_check_raw, data_check_angles_raw, data_check_fnums_raw

    return AdvancedTomoData(
        dark_image=dark_image,
        initial_empty=initial_empty,
        periodic_empties=periodic_empties,
        periodic_empty_fnumbers=periodic_empty_fnumbers,
        data_images=data_images,
        data_angles=data_angles,
        data_numbers=data_numbers,
        data_check_images=data_check_images,
        data_check_angles=data_check_angles,
        data_check_numbers=data_check_numbers,
        series_length=series_length,
    )


def _read_series_length_from_hdf5(data_file: str, total_empty_count: int) -> int:
    """Читает series_length из атрибута exp_info HDF5 или оценивает по данным.

    Fallback: считает initial empty как все кадры до первого data кадра.
    """
    try:
        with h5py.File(data_file, 'r') as h5f:
            exp_info_raw = h5f.attrs.get('exp_info')
            if exp_info_raw is not None:
                exp_info = json.loads(exp_info_raw)
                if 'series_length' in exp_info:
                    return int(exp_info['series_length'])
    except Exception as e:
        logging.warning('Could not read series_length from exp_info: {}'.format(e))

    # Fallback: вычисляем из количества empty и data кадров
    # (для старых файлов, созданных до добавления series_length в exp_info)
    try:
        with h5py.File(data_file, 'r') as h5f:
            empty_fnums = sorted(int(k) for k in h5f['empty'].keys())
            data_fnums = sorted(int(k) for k in h5f['data'].keys())
            if data_fnums:
                first_data_fnum = data_fnums[0]
                initial_count = sum(1 for fn in empty_fnums if fn < first_data_fnum)
                if initial_count > 0:
                    logging.info('series_length estimated from HDF5 structure: {}'.format(initial_count))
                    return initial_count
    except Exception as e:
        logging.warning('Fallback series_length estimation failed: {}'.format(e))

    # Последний fallback: всё делим поровну
    fallback = max(1, total_empty_count // 2)
    logging.warning('Using fallback series_length = {}'.format(fallback))
    return fallback


def load_tomo_data(data_file: str, tmp_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Загружает данные томографии из HDF5-файла.

    Автоматически определяет формат (standard / advanced).
    Для advanced усредняет все empty серии для backward-compatible empty_image.

    Возвращает
    ----------
    empty_image : np.ndarray  — медиана пустых кадров минус тёмный ток
    data_images : np.ndarray  — проекции минус тёмный ток
    data_angles : np.ndarray  — углы проекций

    Сигнатура и возвращаемые значения идентичны tomotools2.load_tomo_data().
    """
    if is_advanced_experiment(data_file):
        logging.info('Advanced experiment detected, using load_tomo_data_advanced')
        adv = load_tomo_data_advanced(data_file, tmp_dir)
        # Усредняем все empty (initial + periodic) для единого empty_image
        all_empties = [adv.initial_empty] + adv.periodic_empties
        empty_image = np.median(np.stack(all_empties, axis=0), axis=0).astype('float32')
        data_images = adv.data_images
        data_angles = adv.data_angles
    else:
        logging.info('Standard experiment detected')
        empty_images, _ = get_frame_group(data_file, 'empty', tmp_dir)
        dark_images, _ = get_frame_group(data_file, 'dark', tmp_dir)

        empty_image = np.median(empty_images, axis=0)
        dark_image = np.median(dark_images, axis=0)
        empty_image -= dark_image

        data_images, data_angles = get_frame_group(data_file, 'data', tmp_dir)
        data_images -= dark_image

    empty_image[empty_image < 1] = 1
    return empty_image, data_images, data_angles


def load_recon_config(tmp_dir: str, storage_dir: str, experiment_id: str) -> dict:
    """Ищет и загружает rec_config.ini из нескольких стандартных мест.

    Порядок поиска:
      1. ./rec_config.ini (текущий каталог)
      2. {tmp_dir}/rec_config.ini
      3. {storage_dir}/{experiment_id}/reconstruction/rec_config.ini

    Возвращает dict секций конфига (или пустой dict, если файл не найден).
    """
    candidates = [
        "rec_config.ini",
        os.path.join(tmp_dir, 'rec_config.ini'),
        os.path.join(storage_dir, experiment_id, 'reconstruction', 'rec_config.ini'),
    ]
    config_file = next((p for p in candidates if os.path.exists(p)), None)

    if config_file is not None:
        recon_conf = configparser.ConfigParser()
        recon_conf.read(config_file)
        return {s: dict(recon_conf[s]) for s in recon_conf.sections()}
    return {}


def save_recon_config(recon_config: dict, tmp_dir: str) -> None:
    """Сохраняет секции 'roi', 'corr', 'axis_corr' в {tmp_dir}/rec_config.ini."""
    cfg = configparser.ConfigParser()
    for key in ['roi', 'corr', 'axis_corr']:
        if key in recon_config:
            cfg[key] = recon_config[key]
    config_path = os.path.join(tmp_dir, 'rec_config.ini')
    with open(config_path, 'w') as configfile:
        cfg.write(configfile)
    logging.info('Saved recon config: {}'.format(config_path))


# =============================================================================
# --- Preprocessing ---
# =============================================================================

def safe_median(data: np.ndarray) -> np.ndarray:
    """Заменяет выбросы (>10 % отклонение от медианы) медианным значением.

    Выполняется на GPU через CuPy.
    """
    data_gpu = cp.asarray(data)
    m_data = median_filter(data_gpu, size=3)
    mask = cp.abs(m_data - data_gpu) > 0.1 * cp.abs(data_gpu)
    res = data_gpu.copy()
    res[mask] = m_data[mask]  # type: ignore[index]
    return cp.asnumpy(res)


def normalize_projections(data_images_crop: np.ndarray,
                          empty_beam_crop: np.ndarray) -> None:
    """Нормирует проекции in-place: d = log(empty) − log(data), clip ≥ 0.

    Применяет safe_median для подавления одиночных выбросов.
    Модифицирует data_images_crop на месте.
    """
    te = empty_beam_crop.copy()
    te[te < 1] = 1
    log_te = np.log(te)

    for di in tqdm(range(data_images_crop.shape[0])):
        td = data_images_crop[di]
        td[td < 1] = 1
        d = log_te - np.log(td)
        d = safe_median(d)
        d[d < 0] = 0
        data_images_crop[di] = d


def _interpolate_empty(adv_data: AdvancedTomoData,
                       frame_number: int,
                       x_min: int, x_max: int,
                       y_min: int, y_max: int) -> np.ndarray:
    """Возвращает интерполированный empty_crop для кадра с заданным frame_number.

    Линейно интерполирует между ближайшими empty сериями по frame_number.
    Для кадров до первой periodic вставки использует initial_empty.
    Для кадров после последней periodic вставки использует последнюю periodic.
    """
    # Список (frame_number, empty_image) всех серий по возрастанию
    all_fnums = [-1] + adv_data.periodic_empty_fnumbers  # -1 = initial (до всех data)
    all_empties = [adv_data.initial_empty] + adv_data.periodic_empties

    # Находим, между какими двумя сериями находится данный frame_number
    # Ищем i такое, что all_fnums[i] <= frame_number < all_fnums[i+1]
    idx = 0
    for i in range(len(all_fnums) - 1):
        if all_fnums[i + 1] <= frame_number:
            idx = i + 1
        else:
            break

    # Если кадр после последней вставки — используем последний empty
    if idx >= len(all_empties) - 1:
        e = all_empties[-1][y_min:y_max, x_min:x_max].copy()
        e[e < 1] = 1
        return e

    # Если кадр до первой вставки — используем initial
    if idx == 0 and (len(adv_data.periodic_empty_fnumbers) == 0
                     or frame_number < adv_data.periodic_empty_fnumbers[0]):
        e = adv_data.initial_empty[y_min:y_max, x_min:x_max].copy()
        e[e < 1] = 1
        return e

    # Линейная интерполяция
    fn0 = all_fnums[idx]
    fn1 = all_fnums[idx + 1]
    e0 = all_empties[idx][y_min:y_max, x_min:x_max].astype('float32')
    e1 = all_empties[idx + 1][y_min:y_max, x_min:x_max].astype('float32')

    w = float(frame_number - fn0) / float(fn1 - fn0) if fn1 != fn0 else 0.0
    e_interp = ((1.0 - w) * e0 + w * e1).astype('float32')
    e_interp[e_interp < 1] = 1
    return e_interp


def normalize_projections_with_timeline(
        data_images_crop: np.ndarray,
        adv_data: 'AdvancedTomoData',
        x_min: int, x_max: int,
        y_min: int, y_max: int) -> None:
    """Нормирует проекции in-place с учётом временного дрейфа источника.

    Для каждой проекции i интерполирует empty между ближайшими checkpoint-ами
    (по frame_number), затем применяет стандартное log-нормирование.

    Параметры
    ----------
    data_images_crop : dark-subtracted кадры, обрезанные по ROI, shape (N, H_roi, W_roi).
                       Изменяется in-place.
    adv_data         : AdvancedTomoData с initial_empty, periodic_empties, data_numbers
    x_min, x_max, y_min, y_max : границы ROI

    Примечание
    ----------
    data_images_crop должен быть уже обрезан (crop) до [y_min:y_max, x_min:x_max].
    adv_data.data_numbers должны соответствовать порядку кадров в data_images_crop.
    """
    for di in tqdm(range(data_images_crop.shape[0])):
        fn = int(adv_data.data_numbers[di])
        empty_crop = _interpolate_empty(adv_data, fn, x_min, x_max, y_min, y_max)

        td = data_images_crop[di].copy()
        td[td < 1] = 1
        d = np.log(empty_crop) - np.log(td)
        d = safe_median(d)
        d[d < 0] = 0
        data_images_crop[di] = d


# =============================================================================
# --- Repositioning shift measurement and correction ---
# =============================================================================

def _find_matching_data_frame(angle: float,
                               data_angles: np.ndarray,
                               data_numbers: np.ndarray,
                               segment_end_fnumber: int) -> int | None:
    """Находит индекс data-кадра с ближайшим углом, снятого до segment_end_fnumber.

    Возвращает индекс в data_angles/data_numbers или None если не найдено.
    """
    # Ищем среди кадров до периодической вставки
    mask = data_numbers < segment_end_fnumber
    if not mask.any():
        return None
    candidate_indices = np.where(mask)[0]
    angle_diffs = np.abs(data_angles[candidate_indices] - angle) % 360
    angle_diffs = np.minimum(angle_diffs, 360 - angle_diffs)
    best = candidate_indices[np.argmin(angle_diffs)]
    return int(best)


def measure_repositioning_shifts(
        adv_data: 'AdvancedTomoData',
        x_min: int, x_max: int,
        y_min: int, y_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Измеряет сдвиг позиционирования объекта на каждом checkpoint-е.

    Для каждого checkpoint k (соответствует periodic_empties[k]):
      1. Берём data_check кадр из группы data_check_images, привязанный к checkpoint k.
      2. Находим data кадр при том же угле из сегмента ДО checkpoint k.
      3. Нормируем оба кадра ОДНИМ empty (periodic_empties[k]) для устранения фона и дрейфа.
      4. Измеряем сдвиг фазовой кросс-корреляцией (субпиксель, upsample_factor=10).

    Сдвиги характеризуют каждый checkpoint k: насколько объект сместился
    при возврате на стол. Кадры из сегмента k+1 (после checkpoint k до
    следующего checkpoint k+1) нужно скорректировать на shifts[k].

    Параметры
    ----------
    adv_data  : AdvancedTomoData
    x_min, x_max, y_min, y_max : границы ROI

    Возвращает
    ----------
    checkpoint_angles : np.ndarray shape (K,) — углы checkpoint-ов
    shifts_y          : np.ndarray shape (K,) — сдвиг по Y (пикс., субпиксель)
    shifts_x          : np.ndarray shape (K,) — сдвиг по X (пикс., субпиксель)
    """
    K = len(adv_data.periodic_empties)
    if K == 0:
        logging.warning('No periodic empties found — no checkpoints to measure')
        return np.array([]), np.array([]), np.array([])

    checkpoint_angles = np.empty(K, dtype='float32')
    shifts_y = np.empty(K, dtype='float64')
    shifts_x = np.empty(K, dtype='float64')

    for k in tqdm(range(K), desc='measure_repositioning_shifts'):
        # Кадр data_check, привязанный к этому checkpoint
        # Ищем data_check кадры между periodic_empty_fnumbers[k] и следующим checkpoint
        next_fn = (adv_data.periodic_empty_fnumbers[k + 1]
                   if k + 1 < K else int(adv_data.data_numbers[-1]) + 1)
        fn_start = adv_data.periodic_empty_fnumbers[k]

        mask_dc = ((adv_data.data_check_numbers >= fn_start) &
                   (adv_data.data_check_numbers < next_fn))
        dc_indices = np.where(mask_dc)[0]

        if len(dc_indices) == 0:
            logging.warning(f'Checkpoint {k}: no data_check frames found, skipping')
            checkpoint_angles[k] = 0.0
            shifts_y[k] = 0.0
            shifts_x[k] = 0.0
            continue

        # Берём первый data_check кадр для этого checkpoint
        dc_idx = dc_indices[0]
        dc_angle = float(adv_data.data_check_angles[dc_idx])
        checkpoint_angles[k] = dc_angle

        dc_frame = adv_data.data_check_images[dc_idx, y_min:y_max, x_min:x_max].copy()

        # Ищем соответствующий data кадр до checkpoint k
        data_idx = _find_matching_data_frame(
            dc_angle, adv_data.data_angles, adv_data.data_numbers, fn_start)

        if data_idx is None:
            logging.warning(f'Checkpoint {k}: no matching data frame at angle {dc_angle:.2f}, skipping')
            shifts_y[k] = 0.0
            shifts_x[k] = 0.0
            continue

        data_frame = adv_data.data_images[data_idx, y_min:y_max, x_min:x_max].copy()

        # Нормируем ОДНИМ empty (periodic_empties[k]) для вычитания фона и дрейфа
        ref_empty = adv_data.periodic_empties[k][y_min:y_max, x_min:x_max].copy()
        ref_empty[ref_empty < 1] = 1

        # data frame
        data_frame[data_frame < 1] = 1
        data_norm = np.log(ref_empty) - np.log(data_frame)
        data_norm = safe_median(data_norm)

        # data_check frame
        dc_frame[dc_frame < 1] = 1
        dc_norm = np.log(ref_empty) - np.log(dc_frame)
        dc_norm = safe_median(dc_norm)

        # Фазовая кросс-корреляция
        shift, _error, _phasediff = phase_cross_correlation(
            data_norm, dc_norm, upsample_factor=10)
        shifts_y[k] = float(shift[0])
        shifts_x[k] = float(shift[1])

        logging.info(f'Checkpoint {k}, angle={dc_angle:.2f}: '
                     f'shift_y={shifts_y[k]:.3f}, shift_x={shifts_x[k]:.3f}')

    return checkpoint_angles, shifts_y, shifts_x


def apply_repositioning_correction(
        data_images: np.ndarray,
        data_numbers: np.ndarray,
        adv_data: 'AdvancedTomoData',
        shifts_y: np.ndarray,
        shifts_x: np.ndarray,
) -> None:
    """Корректирует сдвиг позиционирования in-place для каждого data кадра.

    Для каждого кадра i определяет к какому сегменту он относится
    (по data_numbers[i] и periodic_empty_fnumbers), затем применяет
    sub-pixel shift через scipy.ndimage.shift с обратным знаком.

    Сегмент 0 (до первого checkpoint) — без коррекции (референсная позиция).
    Сегмент k (k >= 1) — кадры после checkpoint k-1 до checkpoint k → сдвиг shifts[k-1].

    Параметры
    ----------
    data_images   : dark-subtracted кадры shape (N, H, W), изменяются in-place
    data_numbers  : глобальные frame_numbers, shape (N,)
    adv_data      : AdvancedTomoData (нужны periodic_empty_fnumbers)
    shifts_y      : shape (K,) — сдвиги по Y на каждом checkpoint
    shifts_x      : shape (K,) — сдвиги по X на каждом checkpoint
    """
    if len(shifts_y) == 0:
        logging.info('apply_repositioning_correction: no shifts to apply')
        return

    K = len(adv_data.periodic_empty_fnumbers)

    for i in tqdm(range(data_images.shape[0]), desc='apply_repositioning_correction'):
        fn = int(data_numbers[i])

        # Определяем номер сегмента: 0 = до первого checkpoint
        segment = 0
        for k in range(K):
            if fn > adv_data.periodic_empty_fnumbers[k]:
                segment = k + 1

        if segment == 0:
            continue  # референсная позиция — не корректируем

        # Сдвиг для этого сегмента
        sy = -shifts_y[segment - 1]
        sx = -shifts_x[segment - 1]

        if abs(sy) < 1e-6 and abs(sx) < 1e-6:
            continue

        data_images[i] = ndi.shift(data_images[i], [sy, sx], order=3,
                                   mode='nearest').astype('float32')


# =============================================================================
# --- Analysis and visualization ---
# =============================================================================

def analyze_source_drift(adv_data: 'AdvancedTomoData') -> None:
    """Визуализирует дрейф интенсивности рентгеновского источника.

    Сравнивает средние интенсивности начального и периодических empty.
    Монотонный рост/падение = дрейф трубки.

    Параметры
    ----------
    adv_data : AdvancedTomoData
    """
    all_fnums = [-1] + adv_data.periodic_empty_fnumbers
    all_empties = [adv_data.initial_empty] + adv_data.periodic_empties
    labels = ['initial'] + [f'periodic {k+1}' for k in range(len(adv_data.periodic_empties))]

    means = [float(np.mean(e)) for e in all_empties]
    relative = [m / means[0] for m in means]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(range(len(means)), means, 'o-', color='steelblue')
    axes[0].set_xticks(range(len(means)))
    axes[0].set_xticklabels(labels, rotation=30, ha='right')
    axes[0].set_ylabel('Средняя интенсивность (counts)')
    axes[0].set_title('Абсолютная интенсивность empty серий')
    axes[0].grid(True, alpha=0.4)

    axes[1].axhline(1.0, color='gray', linestyle='--', alpha=0.7, label='baseline')
    axes[1].plot(range(len(relative)), relative, 's-', color='tomato')
    axes[1].set_xticks(range(len(relative)))
    axes[1].set_xticklabels(labels, rotation=30, ha='right')
    axes[1].set_ylabel('Относительная интенсивность (к initial)')
    axes[1].set_title('Дрейф источника')
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()

    print('Дрейф от начала к концу: {:.2f}%'.format((relative[-1] - 1.0) * 100))


def analyze_repositioning_accuracy(
        adv_data: 'AdvancedTomoData',
        checkpoint_angles: np.ndarray,
        shifts_y: np.ndarray,
        shifts_x: np.ndarray,
        x_min: int = 0, x_max: int = -1,
        y_min: int = 0, y_max: int = -1,
) -> None:
    """Визуализирует точность возврата объекта после каждой empty-вставки.

    График 1: shifts_x и shifts_y vs номер checkpoint-а.
    График 2: |shift| vs checkpoint_angle.
    График 3: overlay data[θ] vs data_check[θ] для первого checkpoint (до коррекции).

    Параметры
    ----------
    adv_data          : AdvancedTomoData
    checkpoint_angles : shape (K,) — из measure_repositioning_shifts
    shifts_y          : shape (K,)
    shifts_x          : shape (K,)
    x_min, x_max, y_min, y_max : ROI для overlay
    """
    if len(shifts_y) == 0:
        print('Нет checkpoint-ов для анализа')
        return

    K = len(shifts_y)
    shift_magnitude = np.sqrt(shifts_y ** 2 + shifts_x ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # График 1: сдвиги vs checkpoint
    x_axis = np.arange(K)
    axes[0].plot(x_axis, shifts_y, 'o-', label='shift Y', color='steelblue')
    axes[0].plot(x_axis, shifts_x, 's-', label='shift X', color='tomato')
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Номер checkpoint')
    axes[0].set_ylabel('Сдвиг (пикселей)')
    axes[0].set_title('Ошибка позиционирования vs checkpoint')
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    # График 2: |shift| vs angle
    axes[1].scatter(checkpoint_angles, shift_magnitude, c=x_axis, cmap='viridis', s=60)
    axes[1].set_xlabel('Угол checkpoint (°)')
    axes[1].set_ylabel('|shift| (пикселей)')
    axes[1].set_title('Величина сдвига vs угловая позиция')
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()

    print(f'Среднеквадратичный сдвиг: {np.mean(shift_magnitude):.3f} пикс.')
    print(f'Максимальный сдвиг: {np.max(shift_magnitude):.3f} пикс. '
          f'(checkpoint {np.argmax(shift_magnitude)}, угол {checkpoint_angles[np.argmax(shift_magnitude)]:.1f}°)')

    # График 3: overlay первого checkpoint
    K_available = len(adv_data.periodic_empties)
    if K_available == 0:
        return

    k = 0
    fn_start = adv_data.periodic_empty_fnumbers[k]
    fn_end = (adv_data.periodic_empty_fnumbers[k + 1]
              if k + 1 < K_available else int(adv_data.data_numbers[-1]) + 1)

    mask_dc = ((adv_data.data_check_numbers >= fn_start) &
               (adv_data.data_check_numbers < fn_end))
    dc_indices = np.where(mask_dc)[0]
    if len(dc_indices) == 0:
        return

    dc_idx = dc_indices[0]
    dc_angle = float(adv_data.data_check_angles[dc_idx])
    data_idx = _find_matching_data_frame(
        dc_angle, adv_data.data_angles, adv_data.data_numbers, fn_start)

    if data_idx is None:
        return

    ref_empty = adv_data.periodic_empties[k]
    _ymax = y_max if y_max > 0 else adv_data.data_images.shape[1]
    _xmax = x_max if x_max > 0 else adv_data.data_images.shape[2]

    def _norm(frame, empty):
        e = empty[y_min:_ymax, x_min:_xmax].copy()
        e[e < 1] = 1
        f = frame[y_min:_ymax, x_min:_xmax].copy()
        f[f < 1] = 1
        d = np.log(e) - np.log(f)
        d[d < 0] = 0
        return d

    d_norm = _norm(adv_data.data_images[data_idx], ref_empty)
    dc_norm = _norm(adv_data.data_check_images[dc_idx], ref_empty)

    vmin = float(np.percentile(d_norm, 1))
    vmax = float(np.percentile(d_norm, 99))

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    axes2[0].imshow(d_norm, cmap='gray', vmin=vmin, vmax=vmax)
    axes2[0].set_title(f'data[{dc_angle:.1f}°] (до вставки)')
    axes2[1].imshow(dc_norm, cmap='gray', vmin=vmin, vmax=vmax)
    axes2[1].set_title(f'data_check[{dc_angle:.1f}°] (после возврата)')
    diff = dc_norm - d_norm
    axes2[2].imshow(diff, cmap='seismic',
                    vmin=-float(np.percentile(np.abs(diff), 99)),
                    vmax=float(np.percentile(np.abs(diff), 99)))
    axes2[2].set_title('Разность (data_check − data)')
    for ax in axes2:
        ax.axis('off')
    plt.suptitle(f'Checkpoint 0: shift_y={shifts_y[0]:.3f}, shift_x={shifts_x[0]:.3f} пикс.',
                 fontsize=13)
    plt.tight_layout()
    plt.show()


# =============================================================================
# --- Axis correction ---
# =============================================================================

def transform_image(im: np.ndarray, shift_x: float, angle: float) -> np.ndarray:
    """Сдвигает и поворачивает изображение на GPU (CuPy).

    Параметры
    ----------
    im      : входное 2D-изображение (float32)
    shift_x : сдвиг по горизонтали в пикселях
    angle   : угол поворота в градусах

    Возвращает np.ndarray той же формы.
    """
    imcu = cp.asarray(im)
    imcu = cndi.shift(imcu, [0, shift_x], order=3, mode='nearest')
    imcu = cndi.rotate(imcu, angle, order=3, reshape=False, mode='nearest')
    return imcu.get()


def find_axis_correction(data_images_crop: np.ndarray,
                         data_angles: np.ndarray) -> tuple[float, float]:
    """Автоматически определяет поправку оси вращения методом Пауэлла.

    Минимизирует L2-норму разности трансформированных кадров 0° и 180°.

    Параметры
    ----------
    data_images_crop : нормированные кадры, shape (N, H, W)
    data_angles      : углы в градусах, shape (N,)

    Возвращает
    ----------
    shift_x : float — горизонтальный сдвиг оси вращения в пикселях
    alfa    : float — угол наклона оси вращения в градусах
    """
    position_0, position_180 = get_angles_at_180_deg(data_angles)
    data_0_orig = data_images_crop[position_0[0]]
    data_180_orig = np.fliplr(data_images_crop[position_180[0]])

    im0 = data_0_orig / (data_0_orig ** 2).sum() ** 0.5
    im1 = data_180_orig / (data_180_orig ** 2).sum() ** 0.5

    cm0 = ndi.center_of_mass(im0)  # type: ignore[assignment]
    cm1 = ndi.center_of_mass(im1)  # type: ignore[assignment]
    initial_shift = (float(cm0[0]) - float(cm1[0])) / 2  # type: ignore[arg-type]

    def _objective(shift_angle, img0, img1):
        s, a = shift_angle
        diff = transform_image(img0, s, a) - transform_image(img1, -s, -a)
        return (diff ** 2).sum()

    result = optimize.minimize(
        _objective,
        np.array([initial_shift, 0.0]),
        args=(im0, im1),
        method='Powell',
        options={'return_all': True},
    )
    shift_x, alfa = result.x
    return float(shift_x), float(alfa)


def apply_axis_correction(data_images_crop: np.ndarray,
                          shift_x: float,
                          alfa: float) -> np.ndarray:
    """Применяет коррекцию оси вращения и возвращает синограмму.

    Параметры
    ----------
    data_images_crop : нормированные кадры, shape (N, H, W)
    shift_x          : горизонтальный сдвиг в пикселях
    alfa             : угол наклона оси в градусах

    Возвращает
    ----------
    sinogram_fixed : np.ndarray, shape (H, N, W), dtype float32
    """
    n_frames, height, width = data_images_crop.shape
    sinogram_fixed = np.zeros((height, n_frames, width), dtype='float32')

    for i in tqdm(range(n_frames)):
        sinogram_fixed[:, i, :] = transform_image(data_images_crop[i], shift_x, alfa)

    return sinogram_fixed


# =============================================================================
# --- Reconstruction ---
# =============================================================================

def recon_2d_parallel(sino: np.ndarray, angles: np.ndarray,
                      pixel_size: float = 9e-3) -> np.ndarray:
    """FBP + CGLS реконструкция одного 2D среза.

    Результат масштабируется на 1/pixel_size.
    """
    rec = astra_utils.astra_recon_2d_parallel(
        sino, angles, [['FBP_CUDA'], ['CGLS_CUDA', 10]]
    )
    return rec / pixel_size


def recon_2d_parallel_nonorm(sino: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """FBP реконструкция одного 2D среза без нормировки (для предпросмотра)."""
    return astra_utils.astra_recon_2d_parallel(
        sino[angles < 180], angles[angles < 180], [['FBP_CUDA']]
    )


# =============================================================================
# --- Visualization ---
# =============================================================================

def disable_output_scrolling() -> None:
    """Отключает сворачивание вывода ячеек при большом количестве изображений."""
    from IPython.display import display, Javascript
    display(Javascript("""
        document.querySelectorAll('.jp-Cell.jp-mod-outputsScrolled').forEach(function(el) {
            el.classList.remove('jp-mod-outputsScrolled');
        });
        if (window._noScrollObserver) {
            window._noScrollObserver.disconnect();
        }
        window._noScrollObserver = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                    var el = mutation.target;
                    if (el.classList.contains('jp-mod-outputsScrolled')) {
                        el.classList.remove('jp-mod-outputsScrolled');
                    }
                }
            });
        });
        document.querySelectorAll('.jp-Cell').forEach(function(cell) {
            window._noScrollObserver.observe(cell, { attributes: true, attributeFilter: ['class'] });
        });
    """))


def show_exp_data(empty_beam: np.ndarray, data_images: np.ndarray) -> None:
    """Отображает первый нормированный кадр данных."""
    plt.figure()
    plt.imshow(data_images[0] / empty_beam, vmin=0, vmax=1,
               cmap='gray', interpolation='bilinear')
    cbar = plt.colorbar()
    cbar.set_label('Интенсивность, усл.ед.', rotation=90)
    plt.title('Нормированное изображение объекта')
    plt.xlabel('Номер канала детектора')
    plt.ylabel('Номер канала детектора')
    plt.show()


def show_frames_with_border(data_images: np.ndarray, empty_beam: np.ndarray,
                             data_angles: np.ndarray, image_id: int,
                             x_min: int, x_max: int,
                             y_min: int, y_max: int) -> None:
    """Показывает кадр с отмеченной областью ROI."""
    te = empty_beam
    angles_sorted_ind = np.argsort(data_angles)
    td = np.asarray(data_images[angles_sorted_ind[image_id]])
    td[td < 1] = 1
    d = np.log(te) - np.log(td)

    filter_step = 2
    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.imshow(d, cmap='gray',
               vmin=float(np.percentile(d[::filter_step, ::filter_step], 1)),
               vmax=float(np.percentile(d[::filter_step, ::filter_step], 99.9)))
    plt.axis('image')
    plt.hlines([y_min, y_max], x_min, x_max, 'r')
    plt.vlines([x_min, x_max], y_min, y_max, 'g')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.subplot(122)
    plt.imshow(d[y_min:y_max, x_min:x_max], cmap='gray',
               vmin=float(np.percentile(d[y_min:y_max:filter_step, x_min:x_max:filter_step], 1)),
               vmax=float(np.percentile(d[y_min:y_max:filter_step, x_min:x_max:filter_step], 99.9)))
    plt.show()
    print("x_min, x_max, y_min, y_max = {}, {}, {}, {}".format(x_min, x_max, y_min, y_max))


def show_center_of_mass(data_images_crop: np.ndarray) -> None:
    """Отображает траекторию центра масс объекта по кадрам."""
    cxy = np.asarray([ndi.center_of_mass(data_images_crop[i])
                      for i in range(data_images_crop.shape[0])])
    plt.figure(figsize=(6, 6))
    plt.scatter(cxy[:, 1], cxy[:, 0], c=range(cxy.shape[0]), cmap='viridis')
    plt.grid()
    plt.show()


def show_reconstruction_cuts(rec_vol: np.ndarray, n_cuts: int = 20) -> None:
    """Отображает n_cuts срезов вдоль осей 0 и 1 реконструированного объёма."""
    for j in range(2):
        for i in range(n_cuts):
            plt.figure(figsize=(10, 8))
            data = rec_vol.take(i * rec_vol.shape[j] // n_cuts, axis=j)
            plt.imshow(data, cmap='viridis',
                       vmin=float(np.maximum(0, np.percentile(data[:], 10))),
                       vmax=float(np.percentile(data[:], 99.9)))
            plt.axis('image')
            plt.title(str(i * rec_vol.shape[j] // n_cuts))
            plt.colorbar()
            plt.show()


def preview_axis_correction(sinogram_mem: np.ndarray, angles: np.ndarray,
                             remove_rings: bool = False) -> None:
    """Реконструирует и показывает до 10 срезов синограммы для контроля оси."""
    from tomo.remove_stripe import remove_all_stripe

    if sinogram_mem.ndim > 2:
        n_slices = min(10, sinogram_mem.shape[0])
        start_slice = sinogram_mem.shape[0] // n_slices
    else:
        n_slices = 1
        start_slice = 0
        sinogram_mem = sinogram_mem[None, :, :]

    for slice_idx in tqdm(range(n_slices)):
        slice_numb = start_slice * slice_idx
        sino2d = sinogram_mem[slice_numb]
        print(sino2d.shape)
        if remove_rings:
            sino2d = remove_all_stripe(cp.asanyarray(sino2d[:, None, :])).get()
            sino2d = np.squeeze(sino2d)
        recon = recon_2d_parallel_nonorm(sino2d, angles)
        plt.figure(figsize=(10, 10))
        plt.imshow(recon,
                   vmin=float(np.percentile(recon, 10)),
                   vmax=float(np.percentile(recon, 99.9)))
        plt.show()


def create_axis_search_widget(sinogram_fixed: np.ndarray,
                               data_images_crop: np.ndarray,
                               data_angles: np.ndarray,
                               shift_x: float,
                               alfa: float):
    """Создаёт интерактивный виджет ручной коррекции оси вращения.

    Параметры
    ----------
    sinogram_fixed   : синограмма для обновления in-place, shape (H, N, W)
    data_images_crop : нормированные кадры, shape (N, H, W)
    data_angles      : углы в градусах, shape (N,)
    shift_x          : начальное значение сдвига
    alfa             : начальное значение угла

    Возвращает
    ----------
    ui         : ipywidgets.VBox — виджет для отображения через display()
    shift_text : FloatText с текущим сдвигом
    angle_text : FloatText с текущим углом
    """
    import ipywidgets as widgets
    from IPython.display import display  # noqa: F401

    position_0, position_180 = get_angles_at_180_deg(data_angles)
    im_0 = data_images_crop[position_0[0]]
    im_180 = data_images_crop[position_180[0]]

    def _show_alignment(shift, angle):
        t_im_0 = transform_image(im_0, shift, angle)
        t_im_180 = transform_image(im_180, shift, angle)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        im1 = ax1.imshow(t_im_0 - np.fliplr(t_im_180), cmap='seismic')
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        ax1.set_title('Разность (0° − flip(180°))')
        im2 = ax2.imshow(t_im_0, cmap='viridis')
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_title('Кадр 0°')
        plt.tight_layout()
        plt.show()

    def _apply_and_reconstruct(shift, angle):
        for i in tqdm(range(data_images_crop.shape[0])):
            sinogram_fixed[:, i, :] = transform_image(data_images_crop[i], shift, angle)
        preview_axis_correction(sinogram_fixed, data_angles, remove_rings=True)

    shift_slider = widgets.FloatSlider(
        min=-200, max=200, step=0.05, value=shift_x,
        description='Shift:', layout=widgets.Layout(width='400px'))
    shift_text = widgets.FloatText(
        value=shift_x, step=0.05, layout=widgets.Layout(width='100px'))
    angle_slider = widgets.FloatSlider(
        min=-3., max=3., step=0.001, value=alfa,
        description='Angle:', layout=widgets.Layout(width='400px'))
    angle_text = widgets.FloatText(
        value=alfa, step=0.001, layout=widgets.Layout(width='100px'))

    widgets.jslink((shift_slider, 'value'), (shift_text, 'value'))
    widgets.jslink((angle_slider, 'value'), (angle_text, 'value'))

    btn_show = widgets.Button(description='Показать совмещение', button_style='info',
                               layout=widgets.Layout(width='220px'))
    btn_apply = widgets.Button(description='Применить + реконструкция', button_style='primary',
                                layout=widgets.Layout(width='250px'))
    output = widgets.Output(layout=widgets.Layout(border='none'))

    def on_show_click(b):
        with output:
            output.clear_output(wait=True)
            _show_alignment(shift_slider.value, angle_slider.value)

    def on_apply_click(b):
        with output:
            output.clear_output(wait=True)
            _apply_and_reconstruct(shift_slider.value, angle_slider.value)

    btn_show.on_click(on_show_click)
    btn_apply.on_click(on_apply_click)

    ui = widgets.VBox([
        widgets.HBox([shift_slider, shift_text]),
        widgets.HBox([angle_slider, angle_text]),
        widgets.HBox([btn_show, btn_apply]),
        output,
    ])
    return ui, shift_text, angle_text


# =============================================================================
# --- Volume utilities ---
# =============================================================================

def get_angles_at_180_deg(uniq_angles: np.ndarray) -> tuple[list[int], list[int]]:
    """Находит пары кадров под углами 0° и 180°.

    Возвращает (position_0, position_180) — списки индексов.
    """
    t = np.subtract.outer(uniq_angles, uniq_angles) % 360
    pos = np.argwhere(np.abs(t - 180) % 360 == 0)
    position_0, position_180 = [], []
    for tpos in pos:
        p0, p180 = tpos
        if p0 < p180:
            position_0.append(p0)
            position_180.append(p180)
    return position_0, position_180


def reshape_volume(array_3d: np.ndarray, binning_factor: int) -> np.ndarray:
    """Pixel binning для трёхмерного массива по всем трём осям.

    Параметры
    ----------
    array_3d       : входной 3D-массив
    binning_factor : коэффициент сжатия (целое число > 0)

    Возвращает сжатый массив (float32).
    """
    if not isinstance(array_3d, np.ndarray) or array_3d.ndim != 3:
        raise ValueError("Входные данные должны быть трёхмерным numpy массивом")
    if not isinstance(binning_factor, int) or binning_factor <= 0:
        raise ValueError("Коэффициент сжатия должен быть положительным целым числом")

    height, width, depth = array_3d.shape
    nh, nw, nd = height // binning_factor, width // binning_factor, depth // binning_factor

    trimmed = array_3d[:nh * binning_factor, :nw * binning_factor, :nd * binning_factor]
    reshaped = trimmed.reshape(nh, binning_factor, nw, binning_factor, nd, binning_factor)
    return reshaped.mean(axis=(1, 3, 5), dtype='float32')


def save_amira(in_array: np.ndarray, out_path: str, name: str,
               reshape: int = 3, pixel_size: float = 9.0e-3) -> None:
    """Сохраняет объём в формате Amira raw + .hx скрипт.

    Параметры
    ----------
    in_array   : реконструированный объём
    out_path   : каталог для сохранения
    name       : имя образца (пробелы заменяются на _)
    reshape    : коэффициент биннинга (1 — без биннинга)
    pixel_size : размер пикселя в мм
    """
    data_path = str(out_path)
    os.makedirs(data_path, exist_ok=True)
    name = name.replace(' ', '_')

    if reshape != 1:
        vol = reshape_volume(in_array, reshape)
    else:
        vol = in_array
    file_shape = vol.shape
    shape_str = '{}_{}_{}' .format(*file_shape)
    out_name = '{}.{}.{}.raw'.format(name, shape_str, reshape)

    if reshape != 1:
        with open(os.path.join(data_path, out_name), 'wb') as f:
            vol.tofile(f)

    hx_path = os.path.join(data_path, 'tomo.{}.{}.hx'.format(name, reshape))
    with open(hx_path, 'w') as af:
        af.write('# Amira Script\n')
        template = ('[ load -unit mm -raw ${{SCRIPTDIR}}/{} '
                    'little xfastest float 1 {} {} {}  0 {} 0 {} 0 {} ] setLabel {}\n')
        af.write(template.format(
            out_name,
            file_shape[2], file_shape[1], file_shape[0],
            pixel_size * reshape * (file_shape[2] - 1),
            pixel_size * reshape * (file_shape[1] - 1),
            pixel_size * reshape * (file_shape[0] - 1),
            out_name,
        ))


def remove_stripes_sinogram(sinogram_fixed: np.ndarray) -> None:
    """Удаляет кольцевые артефакты из синограммы in-place.

    Обрабатывает синограмму батчами по ~48 срезов на GPU.
    """
    from tomo.remove_stripe import remove_all_stripe

    indexes = range(sinogram_fixed.shape[0])
    num_subarrays = len(indexes) // 48 + 1

    for subarr in tqdm(np.array_split(indexes, num_subarrays)):
        t = sinogram_fixed[subarr]
        t = remove_all_stripe(cp.asanyarray(t.swapaxes(0, 1))).get().swapaxes(0, 1)
        sinogram_fixed[subarr] = t
