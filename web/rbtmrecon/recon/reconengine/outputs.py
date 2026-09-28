"""Запись объёма (Amira raw + .hx), биннинг «по ходу», статистика и result.json.

Имена файлов и формат ``.hx`` повторяют ``tomotools4.sanitize_amira_name`` / ``amira_raw_name`` /
``save_amira`` (эти функции здесь не импортируются — переносятся как самостоятельные, чтобы модуль не
тянул зависимости ноутбучного кода). Копия с биннингом ``b`` строится потоково, теми же блоками
``b×b×b`` (среднее, float32), что и ``tomotools4.reshape_volume`` — хвост срезов, не кратный ``b``,
отбрасывается так же, как при обрезке в ``reshape_volume``.
"""
from __future__ import annotations

import datetime
import json
import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

RESULT_SCHEMA = 'rbtm-recon-result/1'

_RAW_DTYPE = '<f4'  # float32, little-endian


def sanitize_amira_name(name: str) -> str:
    """Имя образца, пригодное для имени файла (пробелы → подчёркивания)."""
    return str(name).replace(' ', '_')


def amira_raw_name(name: str, shape, reshape: int = 1) -> str:
    """Имя raw-файла объёма: ``<name>.<d0>_<d1>_<d2>.<reshape>.raw`` (как в ``tomotools4``)."""
    return '{}.{}_{}_{}.{}.raw'.format(sanitize_amira_name(name), *shape, reshape)


def hx_text(raw_name: str, shape_zyx: Sequence[int], voxel_mm: float) -> str:
    """Текст ``.hx``-скрипта Amira, ровно как пишет ``tomotools4.save_amira``.

    ``shape_zyx`` — форма объёма (nz, ny, nx); ``voxel_mm`` — итоговый размер вокселя (для биннинга
    ``b`` это уже ``pixel_size * b``).
    """
    nz, ny, nx = shape_zyx
    template = ('# Amira Script\n'
               '[ load -unit mm -raw ${{SCRIPTDIR}}/{} '
               'little xfastest float 1 {} {} {}  0 {} 0 {} 0 {} ] setLabel {}\n')
    return template.format(
        raw_name, nx, ny, nz,
        voxel_mm * (nx - 1), voxel_mm * (ny - 1), voxel_mm * (nz - 1),
        raw_name,
    )


def _write_text_atomic(path: str, text: str) -> None:
    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix='.tmp-hx-', suffix='.hx', dir=directory)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            fh.write(text)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _write_size_sidecar(raw_path: str, shape: Sequence[int]) -> None:
    np.savetxt(raw_path + '.size', np.asarray(shape), fmt='%5u')


def _close_memmap(mm: np.memmap) -> None:
    """Закрыть memmap немедленно (не дожидаясь сборщика мусора) — на Windows файл нельзя удалить,
    пока он замаплен."""
    try:
        mm.flush()
    except Exception:  # noqa: BLE001 — лучшая попытка, не должна ронять abort()
        pass
    underlying = getattr(mm, '_mmap', None)
    if underlying is not None:
        try:
            underlying.close()
        except Exception:  # noqa: BLE001
            pass


class _Binner:
    """Копит срезы полного разрешения и пишет усреднённый блок ``b×b×b`` по мере накопления."""

    def __init__(self, out_dir: str, name: str, full_shape, voxel_mm: float, b: int):
        nz, ny, nx = full_shape
        self.b = b
        self.nyb = ny // b
        self.nxb = nx // b
        self.ny_trim = self.nyb * b
        self.nx_trim = self.nxb * b
        self.nzb = nz // b
        self.bshape = (self.nzb, self.nyb, self.nxb)

        self.mm: Optional[np.memmap] = None
        self.raw_name = ''
        self.raw_path = ''
        self.hx_path = ''
        self.created_files: List[str] = []

        if self.nzb > 0 and self.nyb > 0 and self.nxb > 0:
            self.raw_name = amira_raw_name(name, self.bshape, b)
            self.raw_path = os.path.join(out_dir, self.raw_name)
            self.mm = np.memmap(self.raw_path, dtype=_RAW_DTYPE, mode='w+', shape=self.bshape)
            _write_size_sidecar(self.raw_path, self.bshape)
            self.hx_path = os.path.join(out_dir, 'tomo.{}.{}.hx'.format(name, b))
            _write_text_atomic(self.hx_path, hx_text(self.raw_name, self.bshape, voxel_mm * b))
            self.created_files = [self.raw_path, self.raw_path + '.size', self.hx_path]

        self._pending: List[np.ndarray] = []
        self._pending_rows = 0
        self._next_bz = 0

    def feed(self, slab: np.ndarray) -> None:
        if self.mm is None or self._next_bz >= self.nzb:
            return
        self._pending.append(slab)
        self._pending_rows += slab.shape[0]
        self._flush_groups()

    def _flush_groups(self) -> None:
        while self._pending_rows >= self.b and self._next_bz < self.nzb:
            buf = self._pending[0] if len(self._pending) == 1 else np.concatenate(self._pending, axis=0)
            group, rest = buf[:self.b], buf[self.b:]
            self._pending = [rest] if rest.shape[0] else []
            self._pending_rows = rest.shape[0]

            trimmed = group[:, :self.ny_trim, :self.nx_trim]
            binned = trimmed.reshape(self.b, self.nyb, self.b, self.nxb, self.b) \
                            .mean(axis=(0, 2, 4), dtype='float32')
            self.mm[self._next_bz] = binned.astype('float32', copy=False)
            self._next_bz += 1

    def close(self) -> None:
        if self.mm is not None:
            _close_memmap(self.mm)

    def discard(self) -> None:
        """Закрыть memmap (не дожидаясь сборщика мусора), чтобы файл можно было удалить сразу."""
        if self.mm is not None:
            _close_memmap(self.mm)
            self.mm = None

    def result(self) -> Optional[Dict[str, Any]]:
        if self.mm is None:
            return None
        return {'factor': self.b, 'raw': self.raw_name,
               'hx': os.path.basename(self.hx_path), 'shape': list(self.bshape)}


class VolumeWriter:
    """Пишет полный объём (memmap float32) и потоковые копии с биннингом, в формате Amira raw + .hx.

    ``shape`` — (nz, ny, nx); срезы пишутся строго по порядку через :meth:`write` (``z0`` должен
    совпадать с концом уже записанного); :meth:`close` возвращает описание созданных файлов,
    :meth:`abort` удаляет всё, что успело быть создано.
    """

    def __init__(self, out_dir: Any, name: str, shape, voxel_mm: float, binning=(4,)):
        self.out_dir = str(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        self.name = sanitize_amira_name(name)
        self.shape = tuple(int(s) for s in shape)
        self.voxel_mm = float(voxel_mm)
        self.binning = tuple(int(b) for b in binning)

        self._next_z = 0
        self._closed = False

        self._full_raw_name = amira_raw_name(self.name, self.shape, 1)
        self._full_path = os.path.join(self.out_dir, self._full_raw_name)
        self._full_mm = np.memmap(self._full_path, dtype=_RAW_DTYPE, mode='w+', shape=self.shape)
        _write_size_sidecar(self._full_path, self.shape)
        self._full_hx_path = os.path.join(self.out_dir, 'tomo.{}.1.hx'.format(self.name))
        _write_text_atomic(self._full_hx_path, hx_text(self._full_raw_name, self.shape, self.voxel_mm))

        self._created_files = [self._full_path, self._full_path + '.size', self._full_hx_path]

        self._binners: Dict[int, _Binner] = {}
        for b in self.binning:
            binner = _Binner(self.out_dir, self.name, self.shape, self.voxel_mm, b)
            self._binners[b] = binner
            self._created_files.extend(binner.created_files)

    def write(self, z0: int, slab: np.ndarray) -> None:
        """Записать слой срезов [z0, z0 + slab.shape[0]). ``z0`` должен продолжать уже записанное."""
        if self._closed:
            # memmap уже отпущен — запись в него обратилась бы к освобождённой памяти
            raise ValueError('VolumeWriter уже закрыт')
        slab = np.asarray(slab, dtype='float32')
        if slab.ndim != 3:
            raise ValueError('slab должен быть 3D массивом (s, ny, nx), получено ndim={}'.format(slab.ndim))
        if slab.shape[1:] != self.shape[1:]:
            raise ValueError('slab имеет форму {}, ожидалось (*, {}, {})'.format(
                slab.shape, self.shape[1], self.shape[2]))
        z0 = int(z0)
        if z0 != self._next_z:
            raise ValueError('запись не по порядку: ожидался z0={}, получено {}'.format(self._next_z, z0))
        s = slab.shape[0]
        if z0 + s > self.shape[0]:
            raise ValueError('запись выходит за пределы объёма: {} + {} > {}'.format(z0, s, self.shape[0]))

        self._full_mm[z0:z0 + s] = slab
        self._next_z += s
        for binner in self._binners.values():
            binner.feed(slab)

    def close(self) -> Dict[str, Any]:
        """Дописать всё на диск, освободить memmap (файлы можно сразу переносить) и вернуть описание файлов."""
        _close_memmap(self._full_mm)
        for binner in self._binners.values():
            binner.close()
        self._closed = True
        return {
            'full': {
                'raw': self._full_raw_name,
                'hx': os.path.basename(self._full_hx_path),
                'shape': list(self.shape),
            },
            'binned': [r for r in (binner.result() for binner in self._binners.values()) if r is not None],
        }

    def abort(self) -> None:
        """Удалить все файлы, созданные этим writer'ом (используется при ошибке/отмене)."""
        _close_memmap(self._full_mm)
        self._full_mm = None
        for binner in self._binners.values():
            binner.discard()
        for path in self._created_files:
            if os.path.exists(path):
                os.remove(path)
        self._closed = True


def volume_stats(sample: np.ndarray, bins: int = 256) -> Dict[str, Any]:
    """Статистика по выборке срезов: min/max, персентили 0.1/99.9, гистограмма."""
    arr = np.asarray(sample, dtype='float64').ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {'min': 0.0, 'max': 0.0, 'p0_1': 0.0, 'p99_9': 0.0,
               'hist': {'edges': [], 'counts': []}}
    vmin = float(arr.min())
    vmax = float(arr.max())
    p0_1 = float(np.percentile(arr, 0.1))
    p99_9 = float(np.percentile(arr, 99.9))
    hist_range = (vmin, vmax) if vmax > vmin else (vmin - 0.5, vmin + 0.5)
    counts, edges = np.histogram(arr, bins=bins, range=hist_range)
    return {
        'min': vmin, 'max': vmax, 'p0_1': p0_1, 'p99_9': p99_9,
        'hist': {'edges': edges.tolist(), 'counts': counts.tolist()},
    }


def write_json(path: Any, obj: Any) -> None:
    """Записать JSON атомарно (временный файл + ``os.replace``), UTF-8, ``indent=2``."""
    path = str(path)
    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix='.tmp-json-', suffix='.json', dir=directory)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def result_document(recipe_dict: Dict[str, Any], recipe_sha: str, files: Optional[Dict[str, Any]],
                    shape, voxel_mm: float, stats: Dict[str, Any], timings: Dict[str, Any],
                    warnings: Sequence[str], engine_version: str, gpu_name: Optional[str]) -> Dict[str, Any]:
    """Собрать документ результата по схеме ``rbtm-recon-result/1``.

    ``files`` — словарь, который возвращает :meth:`VolumeWriter.close` (``{'full': {...}, 'binned': [...]}``).
    """
    files = files or {}
    full = files.get('full') or {}
    binned = files.get('binned') or []
    return {
        'schema': RESULT_SCHEMA,
        'run_id': uuid.uuid4().hex,
        'created': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'recipe_sha256': recipe_sha,
        'recipe': recipe_dict,
        'volume': {
            'file': full.get('raw'),
            'shape': list(shape),
            'dtype': 'float32',
            'byteorder': 'little',
            'voxel_mm': voxel_mm,
            'units': '1/mm',
        },
        'binned': binned,
        'stats': stats,
        'timings': timings,
        'warnings': list(warnings),
        'engine': {'version': engine_version},
        'gpu': gpu_name,
    }
