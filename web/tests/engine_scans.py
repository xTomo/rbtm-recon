"""Синтетические сканы с аналитическими проекциями, записанные в HDF5 v2 (для тестов конвейера и сервиса)."""
import h5py
import numpy as np

import engine_phantom as ph

ANGLES = np.arange(0.0, 360.0, 3.0)          # 120 углов, полный оборот


def write_h5(ss: ph.SyntheticScan, path, chunk_frames: int = 5, exp_id: str = 'synthetic') -> str:
    """Синтетический скан → HDF5 v2 в раскладке rbtm-storage (gzip-чанки из целых кадров)."""
    sc = ss.scan
    n, h, w = ss.frames.shape
    with h5py.File(str(path), 'w') as f:
        md = f.create_group('metadata')
        md.create_dataset('experiment_id', data=exp_id.encode('utf8'))
        md.create_dataset('is_advanced', data=bool(sc.is_advanced))
        md.create_dataset('series_length', data=int(sc.series_length))
        md.create_dataset('pixel_size', data=0.009)
        md.create_dataset('detector_model', data=b'synthetic-detector')
        tl = f.create_group('timeline')
        tl.create_dataset('modes', data=sc.modes.astype('uint8'))
        tl.create_dataset('angles', data=sc.angles.astype('float32'))
        tl.create_dataset('frame_numbers', data=sc.frame_numbers.astype('int64'))
        ds = f.create_group('images').create_dataset(
            'all', shape=(n, h, w), dtype='uint16', chunks=(chunk_frames, h, w), compression='gzip',
            compression_opts=4, shuffle=False)
        for i in range(n):
            ds[i] = ss.frames[i]
    return str(path)


def simple_scan(**kw):
    kw.setdefault('height', 40)
    kw.setdefault('width', 72)
    kw.setdefault('center_x', 35.3)
    kw.setdefault('y_ref', 19.5)
    kw.setdefault('tilt_deg', 0.6)
    return ph.make_synthetic_scan(ANGLES, **kw)


def advanced_scan(**kw):
    # кадр крупнее: на 40×72 фантом обрезается краями и фазовая корреляция пары ошибается
    kw.setdefault('height', 96)
    kw.setdefault('width', 128)
    kw.setdefault('center_x', 63.3)
    kw.setdefault('y_ref', 47.5)
    kw.setdefault('tilt_deg', 0.6)
    return ph.make_synthetic_scan(ANGLES, advanced=True, n_segments=3,
                                  segment_offsets=[(0.0, 0.0), (1.3, -0.8), (2.1, 0.6)], **kw)
