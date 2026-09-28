"""Синтетические сканы HDF5 v2 для тестов recon-engine.

``write_scan`` повторяет раскладку rbtm-storage (``storage/hdf5_v2.py``): ``images/all`` (N, H, W) uint16
с чанком (C, H, W), gzip уровня 4 без shuffle (или с shuffle — для проверки фолбэка), ``timeline/*``,
``mapping/*_indices`` (только если эксперимент «финализирован»), ``metadata/*`` и attrs файла.

Порядок кадров как у драйвера (rbtm-drivers ``experiment.py``):
- advanced: dark × n_dark, empty × S, затем data по позициям; после каждой empty_period-й позиции
  (кроме последней) — вставка empty × S + data_check × n_check при угле последнего data;
- simple: dark × n_dark, empty × (n_empty_series · S), затем все data.

Содержимое кадров детерминированное: фантом, зависящий от угла, + номер кадра + шум от seed,
поэтому побайтные сравнения с h5py осмысленны.
"""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime

import h5py
import numpy as np

MODES = {'dark': 0, 'empty': 1, 'data': 2, 'data_check': 3}


@dataclasses.dataclass
class SynthScan:
    path: str
    images: np.ndarray        # (N, H, W) uint16 — ровно то, что записано в images/all
    modes: np.ndarray         # (N,) uint8
    angles: np.ndarray        # (N,) float32 (как в timeline)
    frame_numbers: np.ndarray
    segment_ids: np.ndarray
    series_length: int
    empty_period: int
    chunk_frames: int

    @property
    def n_frames(self) -> int:
        return len(self.modes)

    def idx(self, mode: str) -> np.ndarray:
        return np.flatnonzero(self.modes == MODES[mode])


def frame_plan(n_dark, n_empty_series, series_length, n_data, n_check, angle_step, advanced):
    """План кадров: (modes, angles, segment_ids, empty_period)."""
    modes, angles, segments = [], [], []

    def add(mode, angle, seg, count=1):
        for _ in range(count):
            modes.append(MODES[mode])
            angles.append(angle)
            segments.append(seg)

    add('dark', 0.0, -1, n_dark)
    if not advanced:
        add('empty', 0.0, 0, n_empty_series * series_length)
        for pos in range(n_data):
            add('data', round(pos * angle_step, 4) % 360, 0)
        return modes, angles, segments, 0

    add('empty', 0.0, 0, series_length)
    n_inserts = max(0, n_empty_series - 1)
    empty_period = max(1, n_data // (n_inserts + 1)) if n_inserts else 0
    seg, inserted = 0, 0
    for pos in range(n_data):
        angle = round(pos * angle_step, 4) % 360
        add('data', angle, seg)
        is_last = pos == n_data - 1
        if n_inserts and not is_last and inserted < n_inserts and (pos + 1) % empty_period == 0:
            seg += 1
            inserted += 1
            add('empty', angle, seg, series_length)
            add('data_check', angle, seg, n_check)
    return modes, angles, segments, empty_period


def make_images(modes, angles, H, W, seed=0):
    """Кадры (N, H, W) uint16: dark ~ 100, empty ~ 3000 с градиентом, data — empty с поглощающим
    вращающимся объектом; ко всем добавлены номер кадра и шум."""
    rng = np.random.default_rng(seed)
    n = len(modes)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    flat = 3000.0 + 400.0 * xx / max(W - 1, 1) - 200.0 * yy / max(H - 1, 1)
    images = np.empty((n, H, W), dtype=np.uint16)
    for i in range(n):
        mode = int(modes[i])
        if mode == MODES['dark']:
            img = np.full((H, W), 100.0)
        elif mode == MODES['empty']:
            img = flat.copy()
        else:
            a = np.radians(angles[i])
            cx = W / 2 + 0.25 * W * np.cos(a)
            r2 = ((xx - cx) / (0.15 * W)) ** 2 + ((yy - H / 2) / (0.35 * H)) ** 2
            mu = np.where(r2 < 1, 1.2 * np.sqrt(np.clip(1 - r2, 0, 1)), 0.0)
            mu += np.where(np.abs(xx - W / 2) < 0.1 * W, 0.3, 0.0) * (np.abs(yy - H / 2) < 0.4 * H)
            img = 100.0 + (flat - 100.0) * np.exp(-mu)
        img = img + i + rng.integers(0, 32, size=(H, W))
        images[i] = np.clip(np.round(img), 0, 65535).astype(np.uint16)
    return images


def write_scan(path, n_dark=3, n_empty_series=3, series_length=3, n_data=24, n_check=1, H=48, W=64,
               chunk_frames=7, angle_step=7.5, advanced=True, with_mapping=True, shuffle=False, seed=0,
               experiment_id='synthetic-scan'):
    """Записать синтетический скан HDF5 v2 и вернуть SynthScan (с записанными кадрами для сравнения).

    n_empty_series — число empty-серий по series_length кадров: для advanced это начальная + (n−1) вставок,
    для simple — все empty подряд после dark. n_check — data_check-кадров в каждой вставке (advanced).
    """
    path = str(path)
    modes, angles, segments, empty_period = frame_plan(
        n_dark, n_empty_series, series_length, n_data, n_check, angle_step, advanced)
    modes = np.asarray(modes, dtype=np.uint8)
    angles = np.asarray(angles, dtype=np.float32)
    segments = np.asarray(segments, dtype=np.int32)
    n = len(modes)
    frame_numbers = np.arange(n, dtype=np.int64) + 1000
    images = make_images(modes, angles, H, W, seed)
    chunk_frames = max(1, min(int(chunk_frames), n))

    with h5py.File(path, 'w') as f:
        f.attrs['format_version'] = 'v2'
        f.attrs['created_at'] = datetime(2026, 9, 1, 12, 0, 0).isoformat()
        f.attrs['exp_info_json'] = json.dumps({'_id': experiment_id, 'experiment parameters': {'advanced': advanced}})
        f.attrs['images_initialized'] = True
        f.attrs['total_frames'] = n
        f.attrs['current_frame_index'] = n
        f.attrs['current_segment'] = int(segments.max(initial=0))
        f.attrs['last_mode'] = 'data'

        md = f.create_group('metadata')
        md.create_dataset('format_version', data='v2')
        md.create_dataset('experiment_id', data=experiment_id.encode('utf8'))
        md.create_dataset('specimen', data='образец 50% Ni'.encode('utf8'))
        md.create_dataset('tags', data=b'synthetic')
        md.create_dataset('timestamp', data=1788000000.0)
        md.create_dataset('datetime', data=b'01.09.2026 12:00:00')
        md.create_dataset('is_advanced', data=bool(advanced))
        md.create_dataset('detector_model', data=b'synthetic-detector')
        md.create_dataset('pixel_size', data=4.25e-3)
        md.create_dataset('source_voltage', data=40.0)
        md.create_dataset('source_current', data=80.0)
        if advanced:
            md.create_dataset('series_length', data=int(series_length))
            md.create_dataset('empty_period', data=int(empty_period))
            md.create_dataset('data_total', data=int(n_data))
            md.create_dataset('data_angle_step', data=float(angle_step))
            md.create_dataset('data_count_per_step', data=1)
        else:
            md.create_dataset('series_length', data=0)
            md.create_dataset('empty_period', data=0)
            md.create_dataset('data_total', data=0)
            md.create_dataset('data_angle_step', data=0.0)
            md.create_dataset('data_count_per_step', data=0)

        tl = f.create_group('timeline')
        columns = {
            'frame_numbers': frame_numbers,
            'modes': modes,
            'angles': angles,
            'exposures': np.full(n, 500.0, dtype=np.float32),
            'timestamps': 1788000000.0 + np.arange(n, dtype=np.float64),
            'object_present': modes != MODES['empty'],
            'shutter_open': modes != MODES['dark'],
            'chip_temp': np.full(n, 25.0, dtype=np.float32),
            'hous_temp': np.full(n, 30.0, dtype=np.float32),
            'horizontal_pos': np.zeros(n, dtype=np.int32),
            'vertical_pos': np.zeros(n, dtype=np.int32),
            'segment_ids': segments,
        }
        for name, values in columns.items():
            tl.create_dataset(name, data=values, maxshape=(n,), chunks=True)

        img = f.create_group('images')
        ds = img.create_dataset('all', shape=(n, H, W), dtype='uint16', chunks=(chunk_frames, H, W),
                                compression='gzip', compression_opts=4, shuffle=bool(shuffle))
        # По кадру, как пишет storage (add_frame_v2)
        for i in range(n):
            ds[i] = images[i]

        if with_mapping:
            mp = f.create_group('mapping')
            for name, code in MODES.items():
                mp.create_dataset('{}_indices'.format(name), data=np.flatnonzero(modes == code).astype(np.int32))
            if advanced:
                ck_data, ck_dc = [], []
                dc_idx = np.flatnonzero(modes == MODES['data_check'])
                data_idx = np.flatnonzero(modes == MODES['data'])
                seen = set()
                for i in dc_idx:
                    if segments[i] in seen:
                        continue
                    seen.add(int(segments[i]))
                    before = data_idx[(data_idx < i) & (np.abs(angles[data_idx] - angles[i]) < 0.01)]
                    ck_data.append(int(before[-1]) if len(before) else -1)
                    ck_dc.append(int(i))
                mp.create_dataset('checkpoint_data_indices', data=np.array(ck_data, dtype=np.int32))
                mp.create_dataset('checkpoint_dc_indices', data=np.array(ck_dc, dtype=np.int32))

    return SynthScan(path=path, images=images, modes=modes, angles=angles, frame_numbers=frame_numbers,
                     segment_ids=segments, series_length=int(series_length) if advanced else 0,
                     empty_period=int(empty_period), chunk_frames=chunk_frames)
