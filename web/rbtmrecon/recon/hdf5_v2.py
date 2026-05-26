"""
hdf5_v2.py — Модуль чтения экспериментов формата HDF5 v2 для реконструкции.

Автодетекция версии формата, быстрое чтение через timeline и mapping.
"""
import json
import logging
from typing import Dict, Any, Tuple, Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# Enum для режимов кадров (должен совпадать с hdf5_v2.py в storage)
FRAME_MODES = {
    'dark': 0,
    'empty': 1,
    'data': 2,
    'data_check': 3,
}

MODE_NAMES = {v: k for k, v in FRAME_MODES.items()}


def is_hdf5_v2(filepath: str) -> bool:
    """
    Определяет версию HDF5-файла.
    
    Returns:
        True если файл формата v2 (имеет группу /timeline и format_version='v2'),
        False иначе.
    """
    try:
        with h5py.File(filepath, 'r') as f:
            if 'timeline' not in f:
                return False
            # Дополнительная проверка версии в metadata
            if 'metadata' in f and 'format_version' in f['metadata']:
                version = str(f['metadata']['format_version'][()], 'utf8')
                return version == 'v2'
            return True
    except Exception:
        return False


def get_experiment_info_v2(data_file: str) -> Dict[str, Any]:
    """
    Читает метаданные эксперимента из HDF5 v2.
    
    Args:
        data_file: Путь к HDF5-файлу
        
    Returns:
        Dict с метаданными
    """
    with h5py.File(data_file, 'r') as f:
        if not is_hdf5_v2(data_file):
            raise ValueError(f'File {data_file} is not HDF5 v2 format')
        
        metadata = f['metadata']
        
        info = {
            'format_version': str(metadata['format_version'][()], 'utf8'),
            'experiment_id': str(metadata['experiment_id'][()], 'utf8'),
            'specimen': str(metadata['specimen'][()], 'utf8'),
            'tags': str(metadata['tags'][()], 'utf8'),
            'timestamp': float(metadata['timestamp'][()]),
            'datetime': str(metadata['datetime'][()], 'utf8'),
            'is_advanced': bool(metadata['is_advanced'][()]),
            'detector_model': str(metadata['detector_model'][()], 'utf8'),
            'pixel_size': float(metadata['pixel_size'][()]),
            'source_voltage': float(metadata['source_voltage'][()]),
            'source_current': float(metadata['source_current'][()]),
        }
        
        if info['is_advanced']:
            info['series_length'] = int(metadata['series_length'][()])
            info['empty_period'] = int(metadata['empty_period'][()])
            info['data_total'] = int(metadata['data_total'][()])
            info['data_angle_step'] = float(metadata['data_angle_step'][()])
            info['data_count_per_step'] = int(metadata['data_count_per_step'][()])
        
        # Статистика по кадрам
        timeline = f['timeline']
        modes = timeline['modes'][:]
        
        info['total_frames'] = len(modes)
        info['dark_count'] = int(np.sum(modes == FRAME_MODES['dark']))
        info['empty_count'] = int(np.sum(modes == FRAME_MODES['empty']))
        info['data_count'] = int(np.sum(modes == FRAME_MODES['data']))
        info['data_check_count'] = int(np.sum(modes == FRAME_MODES['data_check']))
        
        return info


def get_frame_group_v2(
    data_file: str,
    group_name: str,
    mmap_file_dir: str,
    num_workers: int = 8,
    hdf5_cache_mb: int = 512,
    return_frame_numbers: bool = False,
) -> Tuple:
    """
    Загружает группу кадров из HDF5 v2 файла.
    
    Аналог get_frame_group() для v1, но работает с timeline + mapping.
    
    Параметры
    ----------
    data_file : str
        Путь к HDF5-файлу.
    group_name : str
        Имя группы ('empty', 'dark', 'data', 'data_check').
    mmap_file_dir : str
        Каталог для временных mmap-файлов (не используется в v2).
    num_workers : int
        Количество потоков для параллельного чтения (игнорируется в v2).
    hdf5_cache_mb : int
        Размер chunk-кэша HDF5 в мегабайтах.
    return_frame_numbers : bool
        Если True — возвращает третий элемент: массив глобальных номеров кадров.

    Возвращает
    ----------
    images       : np.ndarray, shape (N, H, W)
    angles       : np.ndarray, shape (N,)
    frame_numbers: np.ndarray, shape (N,)  — только если return_frame_numbers=True
    """
    import logging
    logger = logging.getLogger(__name__)
    
    rdcc_nbytes = hdf5_cache_mb * 1024 * 1024
    
    with h5py.File(data_file, 'r', rdcc_nbytes=rdcc_nbytes) as f:
        if not is_hdf5_v2(data_file):
            raise ValueError(f'File {data_file} is not HDF5 v2 format')
        
        # Получаем индексы из mapping
        mapping = f['mapping']
        indices = mapping[f'{group_name}_indices'][:]
        
        logger.info(f'Loading {group_name}: {len(indices)} frames from indices {indices[:3]}...{indices[-3:]}')
        
        if len(indices) == 0:
            # Пустая группа
            return (np.array([]), np.array([]), np.array([])) if return_frame_numbers else (np.array([]), np.array([]))
        
        # Читаем кадры batch'ами (последовательно — h5py оптимизирует chunk-чтение)
        images_all = f['images/all']
        logger.info(f'Reading {len(indices)} frames from images/all (shape={images_all.shape}, dtype={images_all.dtype})')
        
        H, W = images_all.shape[1], images_all.shape[2]
        images = np.empty((len(indices), H, W), dtype=images_all.dtype)
        
        from tqdm.notebook import tqdm
        batch_size = 5  # плавный прогресс-бар (больше шагов)
        for start in tqdm(range(0, len(indices), batch_size), desc=f'Reading {group_name}',
                         mininterval=0.1, ncols=80):
            end = min(start + batch_size, len(indices))
            images[start:end] = images_all[indices[start:end]]
        
        logger.info(f'Loaded {group_name} images: shape={images.shape}, dtype={images.dtype}')
        
        # Читаем углы и frame_numbers из timeline
        timeline = f['timeline']
        angles = timeline['angles'][indices]
        
        if return_frame_numbers:
            frame_numbers = timeline['frame_numbers'][indices]
            return images.astype('float32'), angles.astype('float32'), frame_numbers
        
        return images.astype('float32'), angles.astype('float32')


def load_tomo_data_v2(data_file: str, tmp_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Загружает данные томографии из HDF5 v2 файла.
    
    Автоматически определяет advanced/simple режим.
    Для advanced усредняет все empty серии для backward-compatible empty_image.
    
    Возвращает
    ----------
    empty_image : np.ndarray  — медиана пустых кадров минус тёмный ток
    data_images : np.ndarray  — проекции минус тёмный ток
    data_angles : np.ndarray  — углы проекций

    Сигнатура идентична tomotools2.load_tomo_data() и tomotools4.load_tomo_data().
    """
    logger.info(f'load_tomo_data_v2: starting for {data_file}')
    
    with h5py.File(data_file, 'r') as f:
        if not is_hdf5_v2(data_file):
            raise ValueError(f'File {data_file} is not HDF5 v2 format')
        
        is_advanced = bool(f['metadata/is_advanced'][()])
        logger.info(f'Experiment type: {"advanced" if is_advanced else "simple"}')
        
        # Читаем dark
        logger.info('Loading dark frames...')
        dark_images, _ = get_frame_group_v2(data_file, 'dark', tmp_dir)
        if len(dark_images) == 0:
            raise ValueError('No dark frames found')
        dark_image = np.median(dark_images, axis=0).astype('float32')
        logger.info(f'Dark image computed: shape={dark_image.shape}')
        
        # Читаем empty
        logger.info('Loading empty frames...')
        empty_images, _ = get_frame_group_v2(data_file, 'empty', tmp_dir)
        if len(empty_images) == 0:
            raise ValueError('No empty frames found')
        
        if is_advanced:
            # Advanced: усредняем все empty (initial + periodic)
            empty_image = np.median(empty_images, axis=0).astype('float32')
        else:
            # Simple: медиана всех empty
            empty_image = np.median(empty_images, axis=0).astype('float32')
        
        # Вычитаем dark
        empty_image -= dark_image
        empty_image[empty_image < 1] = 1
        logger.info(f'Empty image computed: shape={empty_image.shape}')
        
        # Читаем data
        logger.info('Loading data frames...')
        data_images, data_angles = get_frame_group_v2(data_file, 'data', tmp_dir)
        if len(data_images) == 0:
            raise ValueError('No data frames found')
        
        data_images = data_images.astype('float32') - dark_image
        logger.info(f'Data loaded: shape={data_images.shape}, angles={len(data_angles)}')
        
        return empty_image, data_images, data_angles


def load_tomo_data_advanced_v2(data_file: str, tmp_dir: str) -> 'AdvancedTomoData':
    """
    Загружает данные advanced эксперимента из HDF5 v2 файла.
    
    Возвращает AdvancedTomoData с разделёнными initial/periodic empty,
    data, data_check и готовыми segment_ids.
    
    Возвращает
    ----------
    AdvancedTomoData
    """
    from dataclasses import dataclass
    
    with h5py.File(data_file, 'r') as f:
        if not is_hdf5_v2(data_file):
            raise ValueError(f'File {data_file} is not HDF5 v2 format')
        
        if not bool(f['metadata/is_advanced'][()]):
            raise ValueError(f'File {data_file} is not an advanced experiment')
        
        series_length = int(f['metadata/series_length'][()])
        
        # Читаем dark
        dark_images, _ = get_frame_group_v2(data_file, 'dark', tmp_dir)
        dark_image = np.median(dark_images, axis=0).astype('float32')
        
        # Читаем empty с frame_numbers
        empty_images, empty_angles, empty_fnums = get_frame_group_v2(
            data_file, 'empty', tmp_dir, return_frame_numbers=True)
        
        # Сортируем по frame_numbers
        sort_idx = np.argsort(empty_fnums)
        empty_images = empty_images[sort_idx]
        empty_fnums = empty_fnums[sort_idx]
        
        # Вычитаем dark
        empty_images = empty_images.astype('float32') - dark_image
        
        # Разделяем на initial и periodic
        initial_empty = np.median(empty_images[:series_length], axis=0).astype('float32')
        
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
        
        # Читаем data
        data_images, data_angles, data_numbers = get_frame_group_v2(
            data_file, 'data', tmp_dir, return_frame_numbers=True)
        
        # Сортируем по frame_numbers
        sort_idx = np.argsort(data_numbers)
        data_images = data_images[sort_idx].astype('float32') - dark_image
        data_angles = data_angles[sort_idx]
        data_numbers = data_numbers[sort_idx]
        
        # Читаем data_check
        data_check_images, data_check_angles, data_check_fnums = get_frame_group_v2(
            data_file, 'data_check', tmp_dir, return_frame_numbers=True)
        
        sort_idx = np.argsort(data_check_fnums)
        data_check_images = data_check_images[sort_idx].astype('float32') - dark_image
        data_check_angles = data_check_angles[sort_idx]
        data_check_numbers = data_check_fnums[sort_idx]
        
        # Импортируем AdvancedTomoData из tomotools4 (лежит в том же каталоге)
        from tomotools4 import AdvancedTomoData
        
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


def get_checkpoint_mapping_v2(data_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Получает mapping checkpoint индексов для advanced эксперимента.
    
    Args:
        data_file: Путь к HDF5-файлу
        
    Returns:
        (checkpoint_data_indices, checkpoint_dc_indices) — массивы индексов
        
    Raises:
        ValueError если файл не v2 или не advanced
    """
    with h5py.File(data_file, 'r') as f:
        if not is_hdf5_v2(data_file):
            raise ValueError(f'File {data_file} is not HDF5 v2 format')
        
        if not bool(f['metadata/is_advanced'][()]):
            raise ValueError(f'File {data_file} is not an advanced experiment')
        
        mapping = f['mapping']
        
        if 'checkpoint_data_indices' not in mapping:
            raise ValueError('No checkpoint mapping found — experiment may not be finalized')
        
        data_indices = mapping['checkpoint_data_indices'][:]
        dc_indices = mapping['checkpoint_dc_indices'][:]
        
        return data_indices, dc_indices
