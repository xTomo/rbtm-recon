"""Чтение HDF5 v2 (формат rbtm-storage) без распаковки лишнего.

Раскладка ``images/all``: uint16 (N, H, W), чанк (C, H, W) — C целых кадров, gzip без shuffle.
h5py при любом чтении распаковывает чанк целиком (C·H·W·2 байт, у реальных данных 300 МБ+),
поэтому здесь чанки читаются напрямую:

- кадр/строки кадра: ``get_chunk_info_by_coord`` → ``os.pread`` сжатых байт кусками →
  ``zlib.decompressobj().decompress(buf, max_length)`` до нужного байта; кадр j в чанке стоит (j+1) кадров
  распаковки, начало чанка — один кадр (замер: 0,28 с против 1,36 с у h5py на кадре 2968×5056);
- кроп по всем кадрам: ``read_direct_chunk`` + ``zlib.decompress`` в пуле потоков (zlib отпускает GIL),
  чанки по порядку, результат — memmap uint16 в кэше.

Если у датасета не «чистый gzip» (shuffle, иные фильтры, нет сжатия и т.п.) — ``ScanInfo.fast_path = False``
и все чтения идут через h5py (медленнее, но корректно). ``filter_mask`` чанка ≠ 0 — тоже h5py.

Реализация:
- смещения чанков берутся один раз при создании ``ChunkSampler`` (``get_chunk_info_by_coord``); сжатые байты
  читаются по этим смещениям мимо h5py — ``os.pread`` с общим дескриптором (Linux) или свой ``open`` на каждое
  чтение (Windows, где ``os.pread`` нет); это ровно то, что делает ``read_direct_chunk``, но без глобальной
  блокировки h5py, поэтому чтение и распаковка идут параллельно;
- кадры, запрошенные из одного чанка, распаковываются одним проходом до последнего нужного байта;
- h5py (фолбэк) — один файл на сэмплер под блокировкой: h5py всё равно сериализует вызовы HDF5.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import logging
import math
import os
import threading
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from .model import (MODE_DARK, MODE_DATA, MODE_DATA_CHECK, MODE_EMPTY, Cancelled, CropData, Overview,
                    ProgressFn, ROI, ScanInfo, check_cancel, no_progress)

logger = logging.getLogger(__name__)

#: потоков по умолчанию (на сервере 6 ядер)
DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 1))

_IMAGES = 'images/all'
_MODE_NAMES = {MODE_DARK: 'dark', MODE_EMPTY: 'empty', MODE_DATA: 'data', MODE_DATA_CHECK: 'data_check'}
_READ_BLOCK = 4 << 20      # сжатые байты читаются кусками по 4 МБ
_MAX_OUT = 8 << 20         # не больше 8 МБ распакованных байт за один вызов decompress
_HAS_PREAD = hasattr(os, 'pread')
_CROP_FORMAT = 'rbtm-recon-crop/1'


# --------------------------------------------------------------------------- open_scan

def _to_py(value):
    """Значение metadata → питоновский тип; байтовые строки → str."""
    if isinstance(value, bytes):
        return value.decode('utf8', errors='replace')
    if isinstance(value, np.ndarray):
        if value.dtype.kind in 'SOU':
            return [_to_py(v) for v in value.tolist()]
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_metadata(f: h5py.File) -> Dict[str, object]:
    out: Dict[str, object] = {}
    group = f.get('metadata')
    if isinstance(group, h5py.Group):
        for key in group:
            obj = group[key]
            if isinstance(obj, h5py.Dataset):
                out[key] = _to_py(obj[()])
    return out


def _filter_codes(ds: h5py.Dataset) -> List[int]:
    plist = ds.id.get_create_plist()
    return [int(plist.get_filter(i)[0]) for i in range(plist.get_nfilters())]


def _mode_indices(f: h5py.File, modes: np.ndarray, n: int) -> Dict[int, np.ndarray]:
    """Индексы режимов: из mapping (если есть все четыре), иначе из timeline/modes."""
    mapping = f.get('mapping')
    names = {code: '{}_indices'.format(name) for code, name in _MODE_NAMES.items()}
    if isinstance(mapping, h5py.Group) and all(name in mapping for name in names.values()):
        out = {}
        for code, name in names.items():
            idx = np.unique(np.asarray(mapping[name][()], dtype=np.int64))
            out[code] = idx[(idx >= 0) & (idx < n)]
        return out
    if mapping is not None:
        logger.warning('mapping неполный — индексы режимов восстанавливаются по timeline/modes')
    return {code: np.flatnonzero(modes == code).astype(np.int64) for code in names}


def _first_run(idx: np.ndarray) -> np.ndarray:
    """Первый непрерывный (по индексам timeline) блок."""
    if len(idx) == 0:
        return idx
    breaks = np.flatnonzero(np.diff(idx) != 1)
    return idx[:breaks[0] + 1] if len(breaks) else idx


def _fingerprint(f: h5py.File, ds: h5py.Dataset, size: int, filters: List[int]) -> str:
    h = hashlib.sha256()
    opts = ds.compression_opts
    head = {
        'shape': [int(s) for s in ds.shape],
        'chunks': [int(c) for c in ds.chunks] if ds.chunks else None,
        'dtype': ds.dtype.str,
        'compression': ds.compression,
        'compression_opts': list(opts) if isinstance(opts, tuple) else opts,
        'shuffle': bool(ds.shuffle),
        'filters': filters,
        'size': int(size),
    }
    h.update(json.dumps(head, sort_keys=True, default=str).encode('utf8'))
    for gname in ('timeline', 'mapping'):
        group = f.get(gname)
        if not isinstance(group, h5py.Group):
            continue
        for name in sorted(group):
            obj = group[name]
            if not isinstance(obj, h5py.Dataset):
                continue
            arr = np.ascontiguousarray(obj[()])
            h.update('{}/{}:{}:{}'.format(gname, name, arr.dtype.str, arr.shape).encode('utf8'))
            h.update(arr.tobytes())
    return h.hexdigest()


def open_scan(path: str, exp_id: Optional[str] = None) -> ScanInfo:
    """Прочитать структуру скана: форма, чанки, фильтры, timeline, индексы режимов, metadata, fingerprint.

    Индексы режимов берутся из ``mapping/<mode>_indices``, при отсутствии — из ``timeline/modes``.
    Строки metadata декодируются в str. ``exp_id`` по умолчанию — ``metadata/experiment_id`` или имя файла.
    Fingerprint: sha256 от (форма, чанки, фильтры, размер файла, timeline/*, mapping/*), без чтения изображений.

    ``n_frames`` — длина timeline (не больше первой оси images/all): у прерванного эксперимента
    images/all создан на весь план, а записаны только кадры timeline.
    """
    path = os.path.abspath(os.fspath(path))
    size = os.path.getsize(path)
    with h5py.File(path, 'r') as f:
        ds = f.get(_IMAGES)
        if not isinstance(ds, h5py.Dataset) or ds.ndim != 3:
            raise ValueError('{}: нет images/all (N, H, W)'.format(path))
        timeline = f.get('timeline')
        if not isinstance(timeline, h5py.Group) or 'modes' not in timeline:
            raise ValueError('{}: не HDF5 v2 (нет timeline/modes)'.format(path))

        shape = tuple(int(s) for s in ds.shape)
        chunks = tuple(int(c) for c in ds.chunks) if ds.chunks else None
        dtype = np.dtype(ds.dtype)
        filters = _filter_codes(ds)
        compression = ds.compression
        shuffle = bool(ds.shuffle)
        userblock = f.id.get_create_plist().get_userblock()
        fast_path = bool(
            compression == 'gzip' and not shuffle
            and filters == [h5py.h5z.FILTER_DEFLATE]
            and chunks is not None and chunks[1:] == shape[1:]   # чанк — целые кадры
            and dtype.isnative and userblock == 0
            and hasattr(ds.id, 'get_chunk_info_by_coord'))

        modes = np.asarray(timeline['modes'][()], dtype=np.uint8)
        n = min(len(modes), shape[0])
        if 'angles' in timeline:
            angles = np.asarray(timeline['angles'][()], dtype=np.float64)
            n = min(n, len(angles))
        else:
            angles = np.zeros(n, dtype=np.float64)
        if 'frame_numbers' in timeline:
            frame_numbers = np.asarray(timeline['frame_numbers'][()], dtype=np.int64)
            n = min(n, len(frame_numbers))
        else:
            frame_numbers = np.arange(n, dtype=np.int64)
        modes, angles, frame_numbers = modes[:n], angles[:n], frame_numbers[:n]

        idx = _mode_indices(f, modes, n)
        metadata = _read_metadata(f)
        fingerprint = _fingerprint(f, ds, size, filters)

    if 'is_advanced' in metadata:
        is_advanced = bool(metadata['is_advanced'])
    else:
        is_advanced = len(idx[MODE_DATA_CHECK]) > 0
    series_length = int(metadata.get('series_length') or 0)
    if is_advanced and series_length <= 0:
        series_length = len(_first_run(idx[MODE_EMPTY]))
    empty_period = int(metadata.get('empty_period') or 0)
    if not exp_id:
        exp_id = str(metadata.get('experiment_id') or '') or os.path.splitext(os.path.basename(path))[0]

    return ScanInfo(
        path=path, exp_id=str(exp_id), n_frames=n, height=shape[1], width=shape[2], dtype=str(dtype),
        chunk_frames=chunks[0] if chunks else 1, compression=compression, shuffle=shuffle, fast_path=fast_path,
        angles=angles, modes=modes, frame_numbers=frame_numbers,
        dark_idx=idx[MODE_DARK], empty_idx=idx[MODE_EMPTY], data_idx=idx[MODE_DATA], check_idx=idx[MODE_DATA_CHECK],
        is_advanced=is_advanced, series_length=series_length, empty_period=empty_period,
        metadata=metadata, fingerprint=fingerprint)


def initial_empty_indices(scan: ScanInfo) -> np.ndarray:
    """Индексы начальной empty-серии: первые series_length empty-кадров (advanced) или первый непрерывный
    блок empty-кадров (simple)."""
    empty = np.sort(np.asarray(scan.empty_idx, dtype=np.int64))
    if scan.is_advanced and scan.series_length > 0:
        return empty[:scan.series_length]
    return _first_run(empty)


# --------------------------------------------------------------------------- ChunkSampler

class _ShortChunk(Exception):
    """Поток чанка кончился раньше нужного байта или распакованный размер не C·H·W·itemsize."""


class _RawReader:
    """Чтение сжатых байт по смещению: os.pread по общему дескриптору или свой файл на время чтения."""

    def __init__(self, path: str, fd: Optional[int]):
        self._path = path
        self._fd = fd
        self._fh = None

    def __enter__(self):
        if self._fd is None:
            self._fh = open(self._path, 'rb', buffering=0)  # noqa: SIM115 — закрывается в __exit__
        return self

    def __exit__(self, *exc):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def read(self, offset: int, n: int) -> bytes:
        parts = []
        while n > 0:
            if self._fd is not None:
                part = os.pread(self._fd, n, offset)
            else:
                self._fh.seek(offset)
                part = self._fh.read(n)
            if not part:
                break
            parts.append(part)
            offset += len(part)
            n -= len(part)
        return parts[0] if len(parts) == 1 else b''.join(parts)


def _bin2d(a: np.ndarray, b: int) -> np.ndarray:
    """Среднее по блокам b×b двух последних осей, float32; края, не кратные b, отбрасываются."""
    if b == 1:
        return a.astype(np.float32)
    h = a.shape[-2] // b * b
    w = a.shape[-1] // b * b
    v = a[..., :h, :w].reshape(a.shape[:-2] + (h // b, b, w // b, b))
    acc = np.uint64 if a.dtype.kind in 'ub' else (np.int64 if a.dtype.kind == 'i' else np.float64)
    s = v.sum(axis=(-3, -1), dtype=acc).astype(np.float32)
    s /= np.float32(b * b)
    return s


class ChunkSampler:
    """Чтение отдельных кадров или их строк с распаковкой только нужной части чанка. Потокобезопасен.

    Держит дескриптор файла (Linux) и, при фолбэке, открытый h5py.File — закрывать через ``close()``
    или ``with ChunkSampler(scan) as s``.
    """

    def __init__(self, scan: ScanInfo):
        self.scan = scan
        self.C = max(1, int(scan.chunk_frames))
        self._dtype = np.dtype(scan.dtype)
        self._row_bytes = scan.width * self._dtype.itemsize
        self._frame_bytes = scan.height * self._row_bytes
        self._lock = threading.RLock()
        self._h5 = None
        self._ds = None
        self._fd = None
        self._chunks: Dict[int, Tuple[Optional[int], int, int]] = {}
        self.fast_path = bool(scan.fast_path)
        if self.fast_path:
            n_chunks = -(-scan.n_frames // self.C)
            try:
                with h5py.File(scan.path, 'r') as f:
                    dsid = f[_IMAGES].id
                    for c in range(n_chunks):
                        info = dsid.get_chunk_info_by_coord((c * self.C, 0, 0))
                        offset = None if info.byte_offset is None else int(info.byte_offset)
                        self._chunks[c] = (offset, int(info.size or 0), int(info.filter_mask or 0))
            except Exception as exc:  # noqa: BLE001 — старый h5py/HDF5: работаем через h5py
                logger.warning('смещения чанков недоступны (%s) — чтение через h5py', exc)
                self.fast_path = False
                self._chunks = {}
        if self.fast_path and _HAS_PREAD:
            self._fd = os.open(scan.path, os.O_RDONLY | getattr(os, 'O_BINARY', 0))

    # -- жизненный цикл -------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            if self._h5 is not None:
                try:
                    self._h5.close()
                finally:
                    self._h5 = None
                    self._ds = None
            if self._fd is not None:
                try:
                    os.close(self._fd)
                finally:
                    self._fd = None

    def __enter__(self) -> 'ChunkSampler':
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 — финализатор
            pass

    # -- низкий уровень ---------------------------------------------------------------

    def _fast_info(self, c: int) -> Optional[Tuple[int, int]]:
        """(смещение, размер) чанка c, если его можно читать напрямую, иначе None."""
        if not self.fast_path:
            return None
        info = self._chunks.get(c)
        if info is None:
            return None
        offset, size, mask = info
        if offset is None or mask != 0 or size <= 0:
            return None
        return offset, size

    def _h5_read(self, sel):
        with self._lock:
            if self._h5 is None:
                self._h5 = h5py.File(self.scan.path, 'r')
                self._ds = self._h5[_IMAGES]
            return self._ds[sel]

    def _read_raw(self, offset: int, size: int) -> bytes:
        with _RawReader(self.scan.path, self._fd) as rd:
            raw = rd.read(offset, size)
        if len(raw) != size:
            raise _ShortChunk('прочитано {} байт из {}'.format(len(raw), size))
        return raw

    def _stream_rows(self, offset: int, size: int, rel_frames: Sequence[int], y0: int, y1: int) -> List[np.ndarray]:
        """Строки [y0, y1) кадров rel_frames (номера внутри чанка, по возрастанию) одним проходом распаковки,
        который останавливается на последнем нужном байте."""
        fb, rb = self._frame_bytes, self._row_bytes
        targets = [(j * fb + y0 * rb, j * fb + y1 * rb) for j in rel_frames]
        bufs = [bytearray(e - s) for s, e in targets]
        end_needed = targets[-1][1]
        d = zlib.decompressobj()
        pos = 0          # распаковано байт
        t = 0            # текущая цель
        read_off, remaining = offset, size
        data = b''
        with _RawReader(self.scan.path, self._fd) as rd:
            while pos < end_needed and not d.eof:
                if not data and remaining > 0:
                    data = rd.read(read_off, min(_READ_BLOCK, remaining))
                    if not data:
                        remaining = 0
                    read_off += len(data)
                    remaining -= len(data)
                before = len(data)
                out = d.decompress(data, min(_MAX_OUT, end_needed - pos))
                data = d.unconsumed_tail
                if not out:
                    if not data and remaining <= 0:
                        break
                    if data and len(data) == before:
                        raise _ShortChunk('распаковка не продвигается')
                    continue
                p1 = pos + len(out)
                mv = memoryview(out)
                while t < len(targets):
                    s, e = targets[t]
                    if s >= p1:
                        break
                    a, b = max(s, pos), min(e, p1)
                    if a < b:
                        bufs[t][a - s:b - s] = mv[a - pos:b - pos]
                    if e <= p1:
                        t += 1
                    else:
                        break
                pos = p1
        if pos < end_needed:
            raise _ShortChunk('чанк распаковался в {} байт, нужно {}'.format(pos, end_needed))
        shape = (y1 - y0, self.scan.width)
        return [np.frombuffer(b, dtype=self._dtype).reshape(shape) for b in bufs]

    def _read_chunk_rows(self, c: int, frames: Sequence[int], y0: int, y1: int) -> List[np.ndarray]:
        """Строки [y0, y1) кадров frames (индексы timeline чанка c, по возрастанию, без повторов)."""
        info = self._fast_info(c)
        if info is not None:
            try:
                return self._stream_rows(info[0], info[1], [j - c * self.C for j in frames], y0, y1)
            except (zlib.error, _ShortChunk) as exc:
                logger.warning('чанк %d: быстрое чтение не удалось (%s) — через h5py', c, exc)
        if len(frames) == 1:
            return [np.asarray(self._h5_read((int(frames[0]), slice(y0, y1))))]
        arr = np.asarray(self._h5_read((list(map(int, frames)), slice(y0, y1))))
        return list(arr)

    def _read_chunk_crop(self, c: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """Кроп [c0:c1, y0:y1, x0:x1] кадров чанка c (c1 ≤ n_frames), C-непрерывный массив."""
        c0 = c * self.C
        c1 = min(c0 + self.C, self.scan.n_frames)
        info = self._fast_info(c)
        if info is not None:
            try:
                raw = self._read_raw(*info)
                full = self.C * self._frame_bytes
                dec = zlib.decompress(raw, bufsize=full)
                del raw
                if len(dec) != full:   # HDF5 хранит полный чанк, в т.ч. последний неполный
                    raise _ShortChunk('распаковано {} байт, ожидалось {}'.format(len(dec), full))
                arr = np.frombuffer(dec, dtype=self._dtype, count=(c1 - c0) * self.scan.height * self.scan.width)
                arr = arr.reshape(c1 - c0, self.scan.height, self.scan.width)
                return np.ascontiguousarray(arr[:, y0:y1, x0:x1])
            except (zlib.error, _ShortChunk) as exc:
                logger.warning('чанк %d: быстрое чтение не удалось (%s) — через h5py', c, exc)
        return np.ascontiguousarray(self._h5_read((slice(c0, c1), slice(y0, y1), slice(x0, x1))))

    def _rows(self, rows: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        if rows is None:
            return 0, self.scan.height
        y0, y1 = int(rows[0]), int(rows[1])
        if not (0 <= y0 < y1 <= self.scan.height):
            raise ValueError('строки [{}, {}) вне кадра высотой {}'.format(y0, y1, self.scan.height))
        return y0, y1

    def _read_many(self, indices: Sequence[int], rows: Optional[Tuple[int, int]], bin: int, workers: int,
                   progress=None) -> np.ndarray:
        """Общая часть read_frame/read_frames: кадры группируются по чанкам, каждый чанк — одна задача пула.
        progress(frac) вызывается из вызывающего потока по мере готовности чанков."""
        idx = np.asarray(indices, dtype=np.int64).ravel()
        y0, y1 = self._rows(rows)
        bin = int(bin)
        if bin < 1:
            raise ValueError('bin должен быть ≥ 1')
        if bin == 1:
            out_shape, out_dtype = (y1 - y0, self.scan.width), self._dtype
        else:
            out_shape, out_dtype = ((y1 - y0) // bin, self.scan.width // bin), np.dtype(np.float32)
            if 0 in out_shape:
                raise ValueError('bin {} больше области {}×{}'.format(bin, y1 - y0, self.scan.width))
        out = np.empty((len(idx),) + out_shape, dtype=out_dtype)
        if len(idx) == 0:
            return out
        bad = (idx < 0) | (idx >= self.scan.n_frames)
        if bad.any():
            raise IndexError('кадр {} вне [0, {})'.format(int(idx[bad][0]), self.scan.n_frames))

        slots: Dict[int, List[int]] = {}
        for pos, j in enumerate(idx.tolist()):
            slots.setdefault(j, []).append(pos)
        groups: Dict[int, List[int]] = {}
        for j in sorted(slots):
            groups.setdefault(j // self.C, []).append(j)
        tasks = sorted(groups.items())

        def run(task):
            c, frames = task
            n_done = 0
            for j, a in zip(frames, self._read_chunk_rows(c, frames, y0, y1)):
                v = _bin2d(a, bin) if bin > 1 else a
                for p in slots[j]:
                    out[p] = v
                    n_done += 1
            return n_done

        total, done = len(idx), 0
        workers = max(1, min(int(workers), len(tasks)))
        if workers == 1:
            for task in tasks:
                done += run(task)
                if progress is not None:
                    progress(done / total)
            return out
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(run, task) for task in tasks]
            try:
                for fut in cf.as_completed(futures):
                    done += fut.result()
                    if progress is not None:
                        progress(done / total)
            except BaseException:
                for fut in futures:
                    fut.cancel()
                raise
        return out

    # -- контракт -----------------------------------------------------------------------

    def frame_cost(self, idx: int) -> int:
        """Сколько кадров придётся распаковать ради кадра idx (быстрый путь: idx % C + 1, иначе C)."""
        idx = int(idx)
        if self._fast_info(idx // self.C) is not None:
            return idx % self.C + 1
        return self.C

    def read_frame(self, idx: int, rows: Optional[Tuple[int, int]] = None, bin: int = 1) -> np.ndarray:
        """Кадр idx (или строки [y0, y1)) полного разрешения.

        bin=1 → uint16 (h, W); bin>1 → float32, среднее bin×bin (края, не кратные bin, отбрасываются).
        Распаковка останавливается на последнем нужном байте.
        """
        return self._read_many([idx], rows, bin, 1)[0]

    def read_frames(self, indices: Sequence[int], rows: Optional[Tuple[int, int]] = None,
                    bin: int = 1, workers: int = DEFAULT_WORKERS) -> np.ndarray:
        """Несколько кадров параллельно; форма (k, h, w) в порядке indices.
        Кадры одного чанка распаковываются одним проходом (до последнего нужного из них)."""
        return self._read_many(indices, rows, bin, workers)


def _nominal_cost(scan: ScanInfo, idx: np.ndarray) -> np.ndarray:
    """Стоимость распаковки кадров по одной структуре скана (без проверки filter_mask чанков)."""
    C = max(1, int(scan.chunk_frames))
    idx = np.asarray(idx, dtype=np.int64)
    if scan.fast_path:
        return idx % C + 1
    return np.full(idx.shape, C, dtype=np.int64)


def pick_sample_indices(scan: ScanInfo, n: int) -> np.ndarray:
    """n data-кадров для обзора: равномерно по диапазону углов, среди кандидатов около каждого целевого
    угла выбирается кадр с наименьшей стоимостью распаковки. Результат — индексы timeline по возрастанию угла.

    Целевые углы — ``linspace(min, max, n)`` по data-углам; кандидаты — в окне ±(шаг целей)/2; при равной
    стоимости — ближайший по углу; кадры не повторяются (если окно исчерпано — ближайший свободный).
    """
    data = np.asarray(scan.data_idx, dtype=np.int64)
    n = int(n)
    if n <= 0 or len(data) == 0:
        return np.empty(0, dtype=np.int64)
    ang = np.asarray(scan.angles, dtype=np.float64)[data]
    if n >= len(data):
        return data[np.lexsort((data, ang))]
    amin, amax = float(ang.min()), float(ang.max())
    if n == 1:
        targets, half = np.array([(amin + amax) / 2]), (amax - amin) / 2
    else:
        targets, half = np.linspace(amin, amax, n), (amax - amin) / (n - 1) / 2
    eps = 1e-9 * max(1.0, abs(amin), abs(amax))
    cost = _nominal_cost(scan, data)
    used = np.zeros(len(data), dtype=bool)
    chosen = []
    for t in targets:
        dist = np.abs(ang - t)
        cand = np.flatnonzero((dist <= half + eps) & ~used)
        if len(cand) == 0:
            cand = np.flatnonzero(~used)
            cand = cand[dist[cand] == dist[cand].min()]
        best = int(cand[np.lexsort((data[cand], dist[cand], cost[cand]))[0]])
        used[best] = True
        chosen.append(best)
    chosen = np.asarray(chosen, dtype=np.int64)
    return data[chosen[np.lexsort((data[chosen], ang[chosen]))]]


def _cheapest(sampler: ChunkSampler, idx: np.ndarray, k: int) -> np.ndarray:
    """k кадров из idx с наименьшей стоимостью распаковки (при равенстве — меньший индекс), по возрастанию."""
    idx = np.asarray(idx, dtype=np.int64)
    if k <= 0 or len(idx) == 0:
        return np.empty(0, dtype=np.int64)
    cost = np.array([sampler.frame_cost(i) for i in idx], dtype=np.int64)
    return np.sort(idx[np.lexsort((idx, cost))[:k]])


def sample_overview(scan: ScanInfo, n: int = 16, bin: int = 4, n_dark: int = 3, n_empty: int = 3,
                    workers: int = DEFAULT_WORKERS, progress: ProgressFn = no_progress) -> Overview:
    """Обзор для шага «Поле зрения»: медианы dark и начальной empty (по n_dark / n_empty самым дешёвым кадрам)
    и n data-кадров выборки, всё с биннингом bin.

    Нет dark-кадров — dark нулевой; нет empty-кадров — ValueError. Стадия прогресса — ``'overview'``.
    """
    progress(0.0, 'overview')
    with ChunkSampler(scan) as sampler:
        dark_sel = _cheapest(sampler, scan.dark_idx, n_dark)
        initial = initial_empty_indices(scan)
        if len(initial) == 0:
            raise ValueError('{}: нет empty-кадров'.format(scan.exp_id))
        empty_sel = _cheapest(sampler, initial, n_empty)
        if len(empty_sel) == 0:
            raise ValueError('n_empty должно быть ≥ 1')
        sample_idx = pick_sample_indices(scan, n)
        all_idx = np.concatenate([dark_sel, empty_sel, sample_idx])
        frames = sampler._read_many(all_idx, None, bin, workers,
                                    progress=lambda frac: progress(0.98 * frac, 'overview'))
    frames = frames.astype(np.float32, copy=False)
    nd, ne = len(dark_sel), len(empty_sel)
    h, w = frames.shape[1:]
    dark = np.median(frames[:nd], axis=0).astype(np.float32) if nd else np.zeros((h, w), np.float32)
    empty = np.median(frames[nd:nd + ne], axis=0).astype(np.float32)
    samples = np.ascontiguousarray(frames[nd + ne:])
    progress(1.0, 'overview')
    return Overview(bin=int(bin), dark=dark, empty=empty, samples=samples, sample_idx=sample_idx,
                    sample_angles=np.asarray(scan.angles, dtype=np.float64)[sample_idx],
                    full_height=scan.height, full_width=scan.width)


def read_row_sinogram(scan: ScanInfo, indices: Sequence[int], row: int,
                      workers: int = DEFAULT_WORKERS) -> np.ndarray:
    """Одна строка детектора row на кадрах indices: float32 (k, W), распаковка каждого кадра только до строки."""
    row = int(row)
    with ChunkSampler(scan) as sampler:
        a = sampler.read_frames(indices, rows=(row, row + 1), bin=1, workers=workers)
    return a[:, 0, :].astype(np.float32)


# --------------------------------------------------------------------------- CropLoader

def _remove_quiet(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning('не удалось удалить %s: %s', path, exc)


def _roi_dict(roi: ROI) -> Dict[str, int]:
    return {'x0': roi.x0, 'x1': roi.x1, 'y0': roi.y0, 'y1': roi.y1}


class CropLoader:
    """Загрузка кропа всех кадров скана в memmap uint16 (N, h, w) в каталоге кэша.

    Кэш: ``<cache_dir>/crop-<hash>.u16`` + ``.json`` (roi, fingerprint, shape, complete). Готовый кэш с тем же
    fingerprint и ROI переиспользуется без чтения HDF5; незавершённый — перезаписывается.
    Ключ кэша — fingerprint и прямоугольник ROI (строка превью на кроп не влияет).
    """

    def __init__(self, scan: ScanInfo, cache_dir: str):
        self.scan = scan
        self.cache_dir = os.fspath(cache_dir)

    def cache_path(self, roi: ROI) -> str:
        key = '{}|{},{},{},{}'.format(self.scan.fingerprint, roi.x0, roi.x1, roi.y0, roi.y1)
        digest = hashlib.sha1(key.encode('utf8')).hexdigest()[:16]
        return os.path.join(self.cache_dir, 'crop-{}.u16'.format(digest))

    @staticmethod
    def _meta_path(path: str) -> str:
        return os.path.splitext(path)[0] + '.json'

    def _shape(self, roi: ROI) -> Tuple[int, int, int]:
        return self.scan.n_frames, roi.height, roi.width

    def _open_cached(self, roi: ROI, path: str) -> Optional[np.memmap]:
        try:
            with open(self._meta_path(path), encoding='utf8') as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            return None
        shape = self._shape(roi)
        dtype = np.dtype(self.scan.dtype)
        nbytes = int(np.prod(shape)) * dtype.itemsize
        ok = (isinstance(meta, dict) and meta.get('complete') is True
              and meta.get('fingerprint') == self.scan.fingerprint
              and meta.get('roi') == _roi_dict(roi)
              and list(meta.get('shape') or []) == list(shape)
              and os.path.isfile(path) and os.path.getsize(path) == nbytes)
        if not ok:
            return None
        return np.memmap(path, dtype=dtype, mode='r', shape=shape)

    def _write_meta(self, path: str, roi: ROI, complete: bool) -> None:
        meta = {
            'format': _CROP_FORMAT, 'complete': bool(complete), 'fingerprint': self.scan.fingerprint,
            'roi': _roi_dict(roi), 'shape': list(self._shape(roi)), 'dtype': str(np.dtype(self.scan.dtype)),
            'exp_id': self.scan.exp_id, 'scan_path': self.scan.path, 'data': os.path.basename(path),
        }
        meta_path = self._meta_path(path)
        tmp = meta_path + '.tmp'
        with open(tmp, 'w', encoding='utf8') as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, meta_path)

    def load(self, roi: ROI, progress: ProgressFn = no_progress, cancel=None,
             workers: int = DEFAULT_WORKERS) -> CropData:
        """Прочитать кроп [y0, y1) × [x0, x1) всех N кадров. Прогресс — доля обработанных чанков (стадия 'crop').
        Отмена (cancel.is_set()) проверяется между чанками → model.Cancelled, незавершённый файл удаляется."""
        scan = self.scan
        roi.validate(scan.height, scan.width)
        if scan.n_frames <= 0:
            raise ValueError('{}: в скане нет кадров'.format(scan.exp_id))
        path = self.cache_path(roi)
        cached = self._open_cached(roi, path)
        if cached is not None:
            try:
                os.utime(self._meta_path(path))     # время использования — для вытеснения старых кропов (LRU)
            except OSError:
                pass
            progress(1.0, 'crop')
            return CropData(roi=roi, frames=cached, path=path, fingerprint=scan.fingerprint)

        check_cancel(cancel)
        os.makedirs(self.cache_dir, exist_ok=True)
        meta_path = self._meta_path(path)
        _remove_quiet(meta_path)
        progress(0.0, 'crop')
        shape = self._shape(roi)
        dtype = np.dtype(scan.dtype)
        nbytes = int(np.prod(shape)) * dtype.itemsize
        try:
            # Кроп чанка — непрерывный кусок (n, h, w) файла, поэтому пишем обычным файлом, чанки строго по порядку
            # (готовые раньше очереди ждут в буфере). Итог тот же, что у memmap w+, но без открытого отображения
            # (на Windows оно мешает удалить файл при отмене) и без предварительного расширения файла
            # (truncate на Windows заполняет нулями: 4,7 с на 3,4 ГБ).
            with open(path, 'wb') as fh:
                self._write_meta(path, roi, complete=False)
                with ChunkSampler(scan) as sampler:
                    self._fill(sampler, fh, roi, progress, cancel, workers)
                fh.flush()
                os.fsync(fh.fileno())
            if os.path.getsize(path) != nbytes:
                raise IOError('{}: записано {} байт, ожидалось {}'.format(path, os.path.getsize(path), nbytes))
            self._write_meta(path, roi, complete=True)
        except BaseException:
            _remove_quiet(path)
            _remove_quiet(meta_path)
            _remove_quiet(meta_path + '.tmp')
            raise
        frames = np.memmap(path, dtype=dtype, mode='r', shape=shape)
        return CropData(roi=roi, frames=frames, path=path, fingerprint=scan.fingerprint)

    def _fill(self, sampler: ChunkSampler, fh, roi: ROI, progress: ProgressFn, cancel, workers: int) -> None:
        C = sampler.C
        n_chunks = -(-self.scan.n_frames // C)
        frame_out = roi.height * roi.width * np.dtype(self.scan.dtype).itemsize
        box = (roi.y0, roi.y1, roi.x0, roi.x1)

        def write(c: int, arr: np.ndarray) -> None:
            fh.seek(c * C * frame_out)
            fh.write(memoryview(np.ascontiguousarray(arr)).cast('B'))
            progress((c + 1) / n_chunks, 'crop')

        workers = max(1, min(int(workers), n_chunks))
        if workers == 1:
            for c in range(n_chunks):
                check_cancel(cancel)
                write(c, sampler._read_chunk_crop(c, *box))
            return

        def task(c: int):
            if cancel is not None and cancel.is_set():
                return None
            return sampler._read_chunk_crop(c, *box)

        # Окно — 2·workers чанков от первого незаписанного: workers распаковываются, остальные ждут в очереди
        # (без памяти) или готовы и ждут записи по порядку (кроп чанка — десятки МБ).
        ex = cf.ThreadPoolExecutor(max_workers=workers)
        try:
            pending: Dict[cf.Future, int] = {}
            ready: Dict[int, np.ndarray] = {}
            next_c = written = 0
            while written < n_chunks:
                while next_c < n_chunks and next_c - written < 2 * workers:
                    check_cancel(cancel)
                    pending[ex.submit(task, next_c)] = next_c
                    next_c += 1
                finished, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
                for fut in finished:
                    c = pending.pop(fut)
                    arr = fut.result()
                    if arr is None:
                        raise Cancelled()
                    ready[c] = arr
                while written in ready:
                    write(written, ready.pop(written))
                    written += 1
                check_cancel(cancel)
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
