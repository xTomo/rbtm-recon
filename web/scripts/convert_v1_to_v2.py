"""
convert_v1_to_v2.py — Конвертер экспериментов HDF5 v1 → v2.

Использование:
    python convert_v1_to_v2.py <exp_id> [--input-dir <src>] [--output-dir <dst>]

Конвертирует старый формат HDF5 (раздельные группы) в новый формат v2
(timeline + metadata + mapping).

Кадры укладываются в timeline/images в порядке возрастания глобального
frame_number — то есть в порядке съёмки, как их пишет rbtm-storage.
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

from hdf5_v2 import FRAME_MODES  # noqa

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Допуск совпадения углов data и data_check при построении checkpoint-пар
ANGLE_MATCH_TOL = 0.5


def read_v1_exp_info(h5f: h5py.File) -> dict:
    """Читает exp_info из HDF5 v1 файла."""
    exp_info_raw = h5f.attrs.get('exp_info')
    if exp_info_raw is None:
        raise ValueError("No 'exp_info' attribute found in HDF5 v1 file")
    return json.loads(exp_info_raw)


def read_v1_group(h5f: h5py.File, group_name: str) -> list:
    """
    Читает группу из HDF5 v1 файла.

    Returns:
        Список dict с ключами frame_number / angle / image / key.
        Ключ ``key`` — реальное имя датасета в файле (оно может быть
        дополнено нулями: '00042'), по нему читаются атрибуты кадра.
    """
    if group_name not in h5f:
        return []

    group = h5f[group_name]
    keys = sorted(group.keys(), key=int)  # Сортируем по числовому значению

    frames = []
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

        frames.append({
            'frame_number': int(key),
            'angle': float(angle),
            'image': ds[()],
            'key': key,
            'mode': group_name,
        })

    return frames


def build_timeline(frames_by_group: dict) -> list:
    """Сливает группы в единую временную ось и проставляет segment_ids.

    Семантика segment_id (как в rbtm-storage):
      -1  — dark
       0  — начальная empty-серия и data до первой periodic-вставки
       k≥1 — k-я periodic-вставка: её empty-серия, её data_check и data после неё
    """
    all_frames = []
    for group_frames in frames_by_group.values():
        all_frames.extend(group_frames)
    all_frames.sort(key=lambda fr: fr['frame_number'])

    current_segment = 0
    seen_data = False
    prev_mode = None

    for fr in all_frames:
        mode = fr['mode']
        if mode == 'dark':
            fr['segment'] = -1
            # dark не влияет на нумерацию сегментов и не прерывает empty-серию
            continue

        if mode == 'empty':
            # Новая periodic-серия начинается, когда empty идёт после data/data_check
            if seen_data and prev_mode not in ('empty', None):
                current_segment += 1
            fr['segment'] = current_segment
        else:
            if mode == 'data':
                seen_data = True
            fr['segment'] = current_segment

        prev_mode = mode

    return all_frames


def build_checkpoint_pairs(timeline: list) -> tuple:
    """Строит checkpoint-пары: одна на periodic-серию.

    Первый data_check серии k сопоставляется с ПОСЛЕДНИМ предыдущим data-кадром,
    снятым при том же угле (драйвер снимает data_check при угле последнего data).
    """
    checkpoint_data, checkpoint_dc = [], []
    used_segments = set()

    for i, fr in enumerate(timeline):
        if fr['mode'] != 'data_check':
            continue
        segment = fr['segment']
        if segment in used_segments:
            continue

        match = None
        for j in range(i - 1, -1, -1):
            prev = timeline[j]
            if prev['mode'] != 'data':
                continue
            diff = abs(prev['angle'] - fr['angle']) % 360
            diff = min(diff, 360 - diff)
            if diff <= ANGLE_MATCH_TOL:
                match = j
                break

        if match is None:
            logger.warning('data_check #%d (segment %s, angle %.2f): '
                           'не найден data-кадр с тем же углом', i, segment, fr['angle'])
            continue

        used_segments.add(segment)
        checkpoint_data.append(match)
        checkpoint_dc.append(i)

    return checkpoint_data, checkpoint_dc


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
        frames_by_group = {name: read_v1_group(v1f, name)
                           for name in ('dark', 'empty', 'data', 'data_check')}

        counts = {name: len(fr) for name, fr in frames_by_group.items()}
        logger.info('Loaded: dark={dark}, empty={empty}, data={data}, '
                    'data_check={data_check}'.format(**counts))

        timeline = build_timeline(frames_by_group)
        # Общее число кадров считаем по реально прочитанным группам
        total_frames = len(timeline)
        if total_frames == 0:
            raise ValueError(f'No frames found in {v1_path}')

        # Определяем series_length для advanced
        series_length = 0
        empty_period = 0
        if is_advanced:
            series_length = exp_params.get('series_length', 0)
            empty_period = exp_params.get('empty_period', 0)

            if series_length == 0:
                # Оцениваем по структуре: initial empty = empty до первого data
                initial = [fr for fr in timeline
                           if fr['mode'] == 'empty' and fr['segment'] == 0]
                series_length = len(initial)
                logger.info(f'Estimated series_length={series_length} from structure')

        # Детектор (берём из первого кадра, ключ — реальное имя датасета)
        detector_model = ''
        pixel_size = 4.25e-3
        first_frames = (frames_by_group['dark'] or frames_by_group['empty']
                        or frames_by_group['data'])
        if first_frames:
            ds = v1f[first_frames[0]['mode']][first_frames[0]['key']]
            detector_model = str(ds.attrs.get('detector_model', ''))
            pixel_size = float(ds.attrs.get('pixel_size', 4.25e-3))

        H, W = timeline[0]['image'].shape[:2]

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
            metadata.create_dataset('format_version', data=b'v2')  # Явная версия в metadata
            metadata.create_dataset('experiment_id', data=str(exp_info.get('_id', '')).encode('utf8'))
            metadata.create_dataset('specimen', data=str(exp_info.get('specimen', '')).encode('utf8'))
            metadata.create_dataset('tags', data=str(exp_info.get('tags', '')).encode('utf8'))
            metadata.create_dataset('timestamp', data=float(exp_info.get('timestamp', 0.0)))
            metadata.create_dataset('datetime', data=str(exp_info.get('datetime', '')).encode('utf8'))
            metadata.create_dataset('is_advanced', data=bool(is_advanced))
            metadata.create_dataset('series_length', data=series_length)
            metadata.create_dataset('empty_period', data=empty_period)
            metadata.create_dataset('data_total', data=exp_params.get('data_total', counts['data']))
            metadata.create_dataset('data_angle_step', data=float(exp_params.get('data_angle_step', 0.0)))
            metadata.create_dataset('data_count_per_step', data=exp_params.get('data_count_per_step', 1))
            metadata.create_dataset('detector_model', data=detector_model.encode('utf8'))
            metadata.create_dataset('pixel_size', data=pixel_size)
            metadata.create_dataset('source_voltage', data=0.0)
            metadata.create_dataset('source_current', data=0.0)

            # Timeline
            modes_arr = np.array([FRAME_MODES[fr['mode']] for fr in timeline], dtype='uint8')
            fnums_arr = np.array([fr['frame_number'] for fr in timeline], dtype='int64')
            angles_arr = np.array([fr['angle'] for fr in timeline], dtype='float32')
            segments_arr = np.array([fr['segment'] for fr in timeline], dtype='int32')

            tl = v2f.create_group('timeline')
            tl.create_dataset('frame_numbers', data=fnums_arr)
            tl.create_dataset('modes', data=modes_arr)
            tl.create_dataset('angles', data=angles_arr)
            tl.create_dataset('exposures', data=np.zeros(total_frames, dtype='float32'))
            tl.create_dataset('timestamps', data=np.zeros(total_frames, dtype='float64'))
            tl.create_dataset('object_present',
                              data=(modes_arr != FRAME_MODES['empty']))
            tl.create_dataset('shutter_open',
                              data=(modes_arr != FRAME_MODES['dark']))
            tl.create_dataset('chip_temp', data=np.zeros(total_frames, dtype='float32'))
            tl.create_dataset('hous_temp', data=np.zeros(total_frames, dtype='float32'))
            tl.create_dataset('horizontal_pos', data=np.zeros(total_frames, dtype='int32'))
            tl.create_dataset('vertical_pos', data=np.zeros(total_frames, dtype='int32'))
            tl.create_dataset('segment_ids', data=segments_arr)

            # Images — в том же порядке, что и timeline
            images_group = v2f.create_group('images')
            chunk_size = (max(series_length, 10) if is_advanced
                          else min(100, max(total_frames // 10, 10)))
            chunk_size = max(1, min(chunk_size, total_frames))

            images_all = images_group.create_dataset(
                'all',
                shape=(total_frames, H, W),
                dtype='uint16',
                chunks=(chunk_size, H, W),
                compression='gzip',
                compression_opts=4
            )
            for idx, fr in enumerate(timeline):
                images_all[idx] = fr['image']

            # Mapping
            mapping = v2f.create_group('mapping')
            for mode_name, mode_code in FRAME_MODES.items():
                indices = np.where(modes_arr == mode_code)[0].astype('int32')
                mapping.create_dataset(f'{mode_name}_indices', data=indices)

            # checkpoint mapping для advanced: одна пара на periodic-серию
            if is_advanced and counts['data_check'] > 0:
                checkpoint_data, checkpoint_dc = build_checkpoint_pairs(timeline)
                if checkpoint_data:
                    mapping.create_dataset('checkpoint_data_indices',
                                           data=np.array(checkpoint_data, dtype='int32'))
                    mapping.create_dataset('checkpoint_dc_indices',
                                           data=np.array(checkpoint_dc, dtype='int32'))

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
