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
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .model import CropData, Overview, ProgressFn, ROI, ScanInfo, no_progress


def open_scan(path: str, exp_id: Optional[str] = None) -> ScanInfo:
    """Прочитать структуру скана: форма, чанки, фильтры, timeline, индексы режимов, metadata, fingerprint.

    Индексы режимов берутся из ``mapping/<mode>_indices``, при отсутствии — из ``timeline/modes``.
    Строки metadata декодируются в str. ``exp_id`` по умолчанию — ``metadata/experiment_id`` или имя файла.
    Fingerprint: sha256 от (форма, чанки, фильтры, размер файла, timeline/*, mapping/*), без чтения изображений.
    """
    raise NotImplementedError


class ChunkSampler:
    """Чтение отдельных кадров или их строк с распаковкой только нужной части чанка. Потокобезопасен."""

    def __init__(self, scan: ScanInfo):
        raise NotImplementedError

    def frame_cost(self, idx: int) -> int:
        """Сколько кадров придётся распаковать ради кадра idx (быстрый путь: idx % C + 1, иначе C)."""
        raise NotImplementedError

    def read_frame(self, idx: int, rows: Optional[Tuple[int, int]] = None, bin: int = 1) -> np.ndarray:
        """Кадр idx (или строки [y0, y1)) полного разрешения.

        bin=1 → uint16 (h, W); bin>1 → float32, среднее bin×bin (края, не кратные bin, отбрасываются).
        Распаковка останавливается на последнем нужном байте.
        """
        raise NotImplementedError

    def read_frames(self, indices: Sequence[int], rows: Optional[Tuple[int, int]] = None,
                    bin: int = 1, workers: int = 8) -> np.ndarray:
        """Несколько кадров параллельно; форма (k, h, w) в порядке indices."""
        raise NotImplementedError


def pick_sample_indices(scan: ScanInfo, n: int) -> np.ndarray:
    """n data-кадров для обзора: равномерно по диапазону углов, среди кандидатов около каждого целевого
    угла выбирается кадр с наименьшей стоимостью распаковки. Результат — индексы timeline по возрастанию угла."""
    raise NotImplementedError


def sample_overview(scan: ScanInfo, n: int = 16, bin: int = 4, n_dark: int = 3, n_empty: int = 3,
                    workers: int = 8, progress: ProgressFn = no_progress) -> Overview:
    """Обзор для шага «Поле зрения»: медианы dark и начальной empty (по n_dark / n_empty самым дешёвым кадрам)
    и n data-кадров выборки, всё с биннингом bin."""
    raise NotImplementedError


def read_row_sinogram(scan: ScanInfo, indices: Sequence[int], row: int, workers: int = 8) -> np.ndarray:
    """Одна строка детектора row на кадрах indices: float32 (k, W), распаковка каждого кадра только до строки."""
    raise NotImplementedError


class CropLoader:
    """Загрузка кропа всех кадров скана в memmap uint16 (N, h, w) в каталоге кэша.

    Кэш: ``<cache_dir>/crop-<hash>.u16`` + ``.json`` (roi, fingerprint, shape, complete). Готовый кэш с тем же
    fingerprint и ROI переиспользуется без чтения HDF5; незавершённый — перезаписывается.
    """

    def __init__(self, scan: ScanInfo, cache_dir: str):
        raise NotImplementedError

    def cache_path(self, roi: ROI) -> str:
        raise NotImplementedError

    def load(self, roi: ROI, progress: ProgressFn = no_progress, cancel=None, workers: int = 8) -> CropData:
        """Прочитать кроп [y0, y1) × [x0, x1) всех N кадров. Прогресс — доля обработанных чанков.
        Отмена (cancel.is_set()) проверяется между чанками → model.Cancelled, незавершённый файл удаляется."""
        raise NotImplementedError
