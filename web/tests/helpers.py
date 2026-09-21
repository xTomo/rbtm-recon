"""Генераторы синтетических HDF5-файлов для тестов.

``make_v2_file`` повторяет layout, который пишет rbtm-storage (hdf5_v2):
metadata/*, timeline/*, images/all, mapping/* (mapping создаётся только при
finalize, поэтому его можно отключить).

``make_v1_file`` повторяет старый формат с раздельными группами
dark/empty/data/data_check и attrs ``frame_info``.
"""
import json

import h5py
import numpy as np

FRAME_MODES = {'dark': 0, 'empty': 1, 'data': 2, 'data_check': 3}


def build_advanced_timeline(n_dark=4, series_length=3, n_periodic=2,
                            data_per_segment=4, data_check_per_step=1,
                            angle_step=10.0):
    """Строит план кадров advanced-эксперимента.

    Порядок съёмки (как в драйвере):
      dark × n_dark
      empty × series_length                     (начальная серия, segment 0)
      data × data_per_segment                   (segment 0)
      [ empty × series_length                   (k-я вставка, segment k)
        data_check × data_check_per_step        (угол последнего data)
        data × data_per_segment ] × n_periodic

    Возвращает список словарей с ключами mode/angle/segment.
    """
    frames = []
    for _ in range(n_dark):
        frames.append({'mode': 'dark', 'angle': 0.0, 'segment': -1})
    for _ in range(series_length):
        frames.append({'mode': 'empty', 'angle': 0.0, 'segment': 0})

    pos = 0
    for _ in range(data_per_segment):
        frames.append({'mode': 'data', 'angle': round(pos * angle_step, 2) % 360,
                       'segment': 0})
        pos += 1

    for k in range(1, n_periodic + 1):
        last_angle = frames[-1]['angle']
        for _ in range(series_length):
            frames.append({'mode': 'empty', 'angle': 0.0, 'segment': k})
        for _ in range(data_check_per_step):
            frames.append({'mode': 'data_check', 'angle': last_angle, 'segment': k})
        for _ in range(data_per_segment):
            frames.append({'mode': 'data', 'angle': round(pos * angle_step, 2) % 360,
                           'segment': k})
            pos += 1
    return frames


def make_v2_file(path, frames, images=None, H=8, W=10, series_length=3,
                 is_advanced=True, with_mapping=True, empty_period=4,
                 data_angle_step=10.0, data_count_per_step=1, seed=0):
    """Создаёт синтетический HDF5 v2 файл.

    frames        : список dict с ключами mode/angle/segment (см. build_advanced_timeline)
    images        : (N, H, W) uint16 или None (тогда генерируется псевдослучайно)
    with_mapping  : False — эмуляция прерванного эксперимента без finalize
    Возвращает (path, images).
    """
    n = len(frames)
    if images is None:
        rng = np.random.default_rng(seed)
        images = rng.integers(100, 4000, size=(n, H, W)).astype('uint16')
    else:
        images = np.asarray(images, dtype='uint16')
        H, W = images.shape[1], images.shape[2]

    modes = np.array([FRAME_MODES[f['mode']] for f in frames], dtype='uint8')
    angles = np.array([f['angle'] for f in frames], dtype='float32')
    segment_ids = np.array([f['segment'] for f in frames], dtype='int32')
    frame_numbers = np.arange(n, dtype='int64')

    with h5py.File(path, 'w') as f:
        f.attrs['format_version'] = 'v2'
        f.attrs['total_frames'] = n

        md = f.create_group('metadata')
        md.create_dataset('format_version', data=b'v2')
        md.create_dataset('experiment_id', data=b'synthetic')
        md.create_dataset('specimen', data='образец 50% Ni'.encode('utf8'))
        md.create_dataset('tags', data=b'test')
        md.create_dataset('timestamp', data=1700000000.0)
        md.create_dataset('datetime', data=b'2026-01-01T00:00:00')
        md.create_dataset('is_advanced', data=bool(is_advanced))
        md.create_dataset('detector_model', data=b'synthetic-detector')
        md.create_dataset('pixel_size', data=4.25e-3)
        md.create_dataset('source_voltage', data=100.0)
        md.create_dataset('source_current', data=0.1)
        if is_advanced:
            md.create_dataset('series_length', data=series_length)
            md.create_dataset('empty_period', data=empty_period)
            md.create_dataset('data_total',
                              data=int((modes == FRAME_MODES['data']).sum()))
            md.create_dataset('data_angle_step', data=data_angle_step)
            md.create_dataset('data_count_per_step', data=data_count_per_step)

        tl = f.create_group('timeline')
        tl.create_dataset('frame_numbers', data=frame_numbers)
        tl.create_dataset('modes', data=modes)
        tl.create_dataset('angles', data=angles)
        tl.create_dataset('exposures', data=np.full(n, 0.5, dtype='float32'))
        tl.create_dataset('timestamps',
                          data=np.arange(n, dtype='float64') + 1700000000.0)
        tl.create_dataset('object_present',
                          data=(modes != FRAME_MODES['empty']))
        tl.create_dataset('shutter_open', data=(modes != FRAME_MODES['dark']))
        tl.create_dataset('chip_temp', data=np.full(n, 25.0, dtype='float32'))
        tl.create_dataset('hous_temp', data=np.full(n, 30.0, dtype='float32'))
        tl.create_dataset('horizontal_pos', data=np.zeros(n, dtype='int32'))
        tl.create_dataset('vertical_pos', data=np.zeros(n, dtype='int32'))
        tl.create_dataset('segment_ids', data=segment_ids)

        images_group = f.create_group('images')
        images_group.create_dataset('all', data=images, dtype='uint16',
                                    chunks=(min(4, n), H, W))

        if with_mapping:
            mapping = f.create_group('mapping')
            for name, code in FRAME_MODES.items():
                mapping.create_dataset(
                    f'{name}_indices',
                    data=np.where(modes == code)[0].astype('int32'))

            # checkpoint-пары: одна на periodic-серию — первый data_check
            # серии ↔ последний предыдущий data с тем же углом
            dc_idx = np.where(modes == FRAME_MODES['data_check'])[0]
            data_idx = np.where(modes == FRAME_MODES['data'])[0]
            ck_data, ck_dc = [], []
            seen_segments = set()
            for i in dc_idx:
                seg = int(segment_ids[i])
                if seg in seen_segments:
                    continue
                seen_segments.add(seg)
                before = data_idx[data_idx < i]
                if len(before) == 0:
                    continue
                ck_data.append(int(before[-1]))
                ck_dc.append(int(i))
            if ck_data:
                mapping.create_dataset('checkpoint_data_indices',
                                       data=np.array(ck_data, dtype='int32'))
                mapping.create_dataset('checkpoint_dc_indices',
                                       data=np.array(ck_dc, dtype='int32'))

    return str(path), images


def make_v1_file(path, frames, images=None, H=8, W=10, series_length=3,
                 empty_period=4, seed=0):
    """Создаёт синтетический HDF5 v1 файл (раздельные группы + frame_info)."""
    n = len(frames)
    if images is None:
        rng = np.random.default_rng(seed)
        images = rng.integers(100, 4000, size=(n, H, W)).astype('uint16')
    else:
        images = np.asarray(images, dtype='uint16')
        H, W = images.shape[1], images.shape[2]

    exp_info = {
        '_id': 'synthetic',
        'specimen': 'образец 50% Ni',
        'tags': 'test',
        'timestamp': 1700000000.0,
        'datetime': '2026-01-01T00:00:00',
        'experiment parameters': {
            'advanced': True,
            'series_length': series_length,
            'empty_period': empty_period,
            'data_total': sum(1 for fr in frames if fr['mode'] == 'data'),
            'data_angle_step': 10.0,
            'data_count_per_step': 1,
        },
    }

    with h5py.File(path, 'w') as f:
        f.attrs['exp_info'] = json.dumps(exp_info)
        groups = {}
        for name in ('dark', 'empty', 'data', 'data_check'):
            groups[name] = f.create_group(name)
        for i, fr in enumerate(frames):
            # Ключи датасетов — глобальный номер кадра с нулевым паддингом
            key = '{:05d}'.format(i)
            ds = groups[fr['mode']].create_dataset(key, data=images[i])
            ds.attrs['frame_info'] = json.dumps([
                {'frame': {'object': {'angle position': float(fr['angle'])},
                           'number': i}}
            ])
            ds.attrs['detector_model'] = 'synthetic-detector'
            ds.attrs['pixel_size'] = 4.25e-3

    return str(path), images
