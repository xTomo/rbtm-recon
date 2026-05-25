"""
convert_v1_to_v2.py — Конвертер экспериментов HDF5 v1 → v2.

Использование:
    python convert_v1_to_v2.py <exp_id> [--input-dir <src>] [--output-dir <dst>]

Конвертирует старый формат HDF5 (раздельные группы) в новый формат v2
(timeline + metadata + mapping).
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime

import h5py
import numpy as np

# Добавляем путь к модулям recon
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rbtmrecon', 'recon'))

from hdf5_v2 import FRAME_MODES, compute_total_frames  # noqa

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def read_v1_exp_info(h5f: h5py.File) -> dict:
    """Читает exp_info из HDF5 v1 файла."""
    exp_info_raw = h5f.attrs.get('exp_info')
    if exp_info_raw is None:
        raise ValueError("No 'exp_info' attribute found in HDF5 v1 file")
    return json.loads(exp_info_raw)


def read_v1_group(h5f: h5py.File, group_name: str) -> tuple:
    """
    Читает группу из HDF5 v1 файла.
    
    Returns:
        (images, angles, frame_numbers) — массивы данных
    """
    group = h5f[group_name]
    keys = sorted(group.keys(), key=int)  # Сортируем по числовому значению
    
    images = []
    angles = []
    frame_numbers = []
    
    for key in keys:
        ds = group[key]
        frame_info_raw = ds.attrs.get('frame_info')
        if frame_info_raw:
            frame_info = json.loads(frame_info_raw)
            if isinstance(frame_info, list):
                frame_info = frame_info[0]
            frame_payload = frame_info.get('frame', frame_info)
            angle = frame_payload.get('object', {}).get('angle position', 0.0)
        else:
            angle = 0.0
        
        images.append(ds[()])
        angles.append(angle)
        frame_numbers.append(int(key))
    
    return np.array(images, dtype='uint16'), np.array(angles, dtype='float32'), np.array(frame_numbers, dtype='int64')


def convert_v1_to_v2(v1_path: str, v2_path: str) -> None:
    """
    Конвертирует HDF5 v1 файл в формат v2.
    
    Args:
        v1_path: Путь к исходному файлу v1
        v2_path: Путь к целевому файлу v2
    """
    logger.info(f'Converting {v1_path} → {v2_path}')
    
    with h5py.File(v1_path, 'r') as v1f:
        # Читаем метаданные
        exp_info = read_v1_exp_info(v1f)
        exp_params = exp_info.get('experiment parameters', {})
        is_advanced = exp_params.get('advanced', False)
        
        # Читаем все группы
        dark_images, dark_angles, dark_fnums = read_v1_group(v1f, 'dark')
        empty_images, empty_angles, empty_fnums = read_v1_group(v1f, 'empty')
        data_images, data_angles, data_fnums = read_v1_group(v1f, 'data')
        
        # data_check может отсутствовать в старых файлах
        if 'data_check' in v1f and len(v1f['data_check']) > 0:
            dc_images, dc_angles, dc_fnums = read_v1_group(v1f, 'data_check')
        else:
            dc_images, dc_angles, dc_fnums = np.array([]), np.array([]), np.array([])
        
        logger.info(f'Loaded: dark={len(dark_images)}, empty={len(empty_images)}, '
                    f'data={len(data_images)}, data_check={len(dc_images)}')
        
        # Вычисляем общее число кадров
        total_frames = len(dark_images) + len(empty_images) + len(data_images) + len(dc_images)
        
        # Определяем series_length для advanced
        series_length = 0
        empty_period = 0
        if is_advanced:
            # Пытаемся прочитать из exp_info
            series_length = exp_params.get('series_length', 0)
            empty_period = exp_params.get('empty_period', 0)
            
            if series_length == 0:
                # Оцениваем по структуре: initial empty = empty до первого data
                if len(data_fnums) > 0 and len(empty_fnums) > 0:
                    first_data_fn = data_fnums[0]
                    initial_count = np.sum(empty_fnums < first_data_fn)
                    series_length = int(initial_count)
                    logger.info(f'Estimated series_length={series_length} from structure')
        
        # Создаём v2 файл
        with h5py.File(v2_path, 'w') as v2f:
            # Атрибуты
            v2f.attrs['format_version'] = 'v2'
            v2f.attrs['created_at'] = datetime.now().isoformat()
            v2f.attrs['exp_info_json'] = json.dumps(exp_info)
            v2f.attrs['images_initialized'] = True
            v2f.attrs['total_frames'] = total_frames
            
            # Metadata
            metadata = v2f.create_group('metadata')
            metadata.create_dataset('format_version', data='v2')  # Явная версия в metadata
            metadata.create_dataset('experiment_id', data=str(exp_info.get('_id', '')).encode('utf8'))
            metadata.create_dataset('specimen', data=str(exp_info.get('specimen', '')).encode('utf8'))
            metadata.create_dataset('tags', data=str(exp_info.get('tags', '')).encode('utf8'))
            metadata.create_dataset('timestamp', data=float(exp_info.get('timestamp', 0.0)))
            metadata.create_dataset('datetime', data=str(exp_info.get('datetime', '')).encode('utf8'))
            metadata.create_dataset('is_advanced', data=is_advanced)
            metadata.create_dataset('series_length', data=series_length)
            metadata.create_dataset('empty_period', data=empty_period)
            metadata.create_dataset('data_total', data=exp_params.get('data_total', 0))
            metadata.create_dataset('data_angle_step', data=exp_params.get('data_angle_step', 0.0))
            metadata.create_dataset('data_count_per_step', data=exp_params.get('data_count_per_step', 1))
            
            # Детектор (берём из первого кадра)
            detector_model = ''
            pixel_size = 4.25e-3
            if len(dark_images) > 0:
                ds = v1f['dark'][str(dark_fnums[0])]
                detector_model = str(ds.attrs.get('detector_model', ''))
                pixel_size = float(ds.attrs.get('pixel_size', 4.25e-3))
            
            metadata.create_dataset('detector_model', data=detector_model.encode('utf8'))
            metadata.create_dataset('pixel_size', data=pixel_size)
            metadata.create_dataset('source_voltage', data=0.0)
            metadata.create_dataset('source_current', data=0.0)
            
            # Timeline
            timeline = v2f.create_group('timeline')
            
            # Собираем все данные в единую временную ось
            all_modes = []
            all_angles = []
            all_fnums = []
            all_segment_ids = []
            
            # Dark: segment_id = -1
            all_modes.extend([FRAME_MODES['dark']] * len(dark_images))
            all_angles.extend(dark_angles)
            all_fnums.extend(dark_fnums)
            all_segment_ids.extend([-1] * len(dark_images))
            
            # Empty: segment_id = 0 (initial) или 1+ (periodic)
            # Определяем initial vs periodic по сравнению с data
            if len(data_fnums) > 0:
                first_data_fn = data_fnums[0]
                for i, fn in enumerate(empty_fnums):
                    if fn < first_data_fn:
                        all_segment_ids.append(0)  # initial
                    else:
                        # periodic — определяем номер по порядку
                        periodic_num = np.sum(np.array(all_segment_ids[-len(empty_images)+i:]) == 0) // series_length if series_length > 0 else 1
                        all_segment_ids.append(1 + periodic_num)
            else:
                all_segment_ids.extend([0] * len(empty_images))
            
            all_modes.extend([FRAME_MODES['empty']] * len(empty_images))
            all_angles.extend(empty_angles)
            all_fnums.extend(empty_fnums)
            
            # Data: segment_id = 0 или 1+ (после periodic empty)
            # Упрощённо: все data = 0, если не advanced
            if is_advanced and series_length > 0:
                # Считаем сегменты по номерам кадров
                current_segment = 0
                last_empty_fn = empty_fnums[-1] if len(empty_fnums) > 0 else 0
                
                for fn in data_fnums:
                    if fn > last_empty_fn:
                        current_segment += 1
                        last_empty_fn = fn  # обновляем для следующей итерации
                    all_segment_ids.append(current_segment)
            else:
                all_segment_ids.extend([0] * len(data_images))
            
            all_modes.extend([FRAME_MODES['data']] * len(data_images))
            all_angles.extend(data_angles)
            all_fnums.extend(data_fnums)
            
            # Data check
            if len(dc_images) > 0:
                all_modes.extend([FRAME_MODES['data_check']] * len(dc_images))
                all_angles.extend(dc_angles)
                all_fnums.extend(dc_fnums)
                # segment_id для data_check = сегмент + 1
                all_segment_ids.extend([s + 1 for s in all_segment_ids[-len(dc_images)-1:-1][-len(dc_images):]])
            
            # Создаём датасеты timeline
            timeline.create_dataset('frame_numbers', data=np.array(all_fnums, dtype='int64'))
            timeline.create_dataset('modes', data=np.array(all_modes, dtype='uint8'))
            timeline.create_dataset('angles', data=np.array(all_angles, dtype='float32'))
            timeline.create_dataset('exposures', data=np.zeros(total_frames, dtype='float32'))
            timeline.create_dataset('timestamps', data=np.zeros(total_frames, dtype='float64'))
            timeline.create_dataset('object_present', data=np.ones(total_frames, dtype='bool'))
            timeline.create_dataset('shutter_open', data=np.ones(total_frames, dtype='bool'))
            timeline.create_dataset('chip_temp', data=np.zeros(total_frames, dtype='float32'))
            timeline.create_dataset('hous_temp', data=np.zeros(total_frames, dtype='float32'))
            timeline.create_dataset('horizontal_pos', data=np.zeros(total_frames, dtype='int32'))
            timeline.create_dataset('vertical_pos', data=np.zeros(total_frames, dtype='int32'))
            timeline.create_dataset('segment_ids', data=np.array(all_segment_ids, dtype='int32'))
            
            # Images
            H, W = dark_images.shape[1] if len(dark_images) > 0 else (1024, 1024)
            images_group = v2f.create_group('images')
            
            # Определяем chunk size
            chunk_size = max(series_length, 10) if is_advanced else min(100, max(total_frames // 10, 10))
            
            images_all = images_group.create_dataset(
                'all',
                shape=(total_frames, H, W),
                dtype='uint16',
                chunks=(chunk_size, H, W),
                compression='gzip',
                compression_opts=4
            )
            
            # Записываем кадры в порядке timeline
            idx = 0
            for img in dark_images:
                images_all[idx] = img
                idx += 1
            for img in empty_images:
                images_all[idx] = img
                idx += 1
            for img in data_images:
                images_all[idx] = img
                idx += 1
            for img in dc_images:
                images_all[idx] = img
                idx += 1
            
            # Mapping
            mapping = v2f.create_group('mapping')
            
            modes_arr = np.array(all_modes, dtype='uint8')
            fnums_arr = np.array(all_fnums, dtype='int64')
            
            for mode_name, mode_code in FRAME_MODES.items():
                indices = np.where(modes_arr == mode_code)[0].astype('int32')
                mapping.create_dataset(f'{mode_name}_indices', data=indices)
            
            # checkpoint mapping для advanced
            if is_advanced and len(dc_images) > 0:
                # Упрощённо: сопоставляем data_check с предыдущими data
                dc_indices = mapping['data_check_indices'][:]
                data_indices = mapping['data_indices'][:]
                
                checkpoint_data = []
                checkpoint_dc = []
                
                for dc_idx in dc_indices:
                    dc_fn = fnums_arr[dc_idx]
                    # Находим data перед этим data_check
                    data_before = data_indices[data_indices < dc_idx]
                    if len(data_before) > 0:
                        checkpoint_data.append(int(data_before[-1]))
                        checkpoint_dc.append(int(dc_idx))
                
                if len(checkpoint_data) > 0:
                    mapping.create_dataset('checkpoint_data_indices', data=np.array(checkpoint_data, dtype='int32'))
                    mapping.create_dataset('checkpoint_dc_indices', data=np.array(checkpoint_dc, dtype='int32'))
        
        logger.info(f'Conversion complete: {v2_path}')


def main():
    parser = argparse.ArgumentParser(description='Конвертер HDF5 v1 → v2')
    parser.add_argument('exp_id', help='ID эксперимента')
    parser.add_argument('--input-dir', default='.', help='Директория с исходными файлами v1')
    parser.add_argument('--output-dir', default='.', help='Директория для файлов v2')
    
    args = parser.parse_args()
    
    v1_path = os.path.join(args.input_dir, f'{args.exp_id}.h5')
    v2_path = os.path.join(args.output_dir, f'{args.exp_id}_v2.h5')
    
    if not os.path.exists(v1_path):
        logger.error(f'File not found: {v1_path}')
        sys.exit(1)
    
    convert_v1_to_v2(v1_path, v2_path)
    logger.info('Done!')


if __name__ == '__main__':
    main()
