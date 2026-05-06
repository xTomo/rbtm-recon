import configparser
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
                    num_workers: int = 8, hdf5_cache_mb: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Загружает группу кадров из HDF5-файла.

    Параметры
    ----------
    data_file : str
        Путь к HDF5-файлу.
    group_name : str
        Имя группы ('empty', 'dark', 'data' и т.д.).
    mmap_file_dir : str
        Каталог для временных mmap-файлов (не используется, сохранён для совместимости).
    num_workers : int
        Количество потоков для параллельного чтения (0 — без параллелизма).
    hdf5_cache_mb : int
        Размер chunk-кэша HDF5 в мегабайтах.

    Возвращает
    ----------
    images : np.ndarray, shape (N, H, W)
    angles : np.ndarray, shape (N,)
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

    return images, angles


def load_tomo_data(data_file: str, tmp_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Загружает данные томографии из HDF5-файла.

    Возвращает
    ----------
    empty_image : np.ndarray  — медиана пустых кадров минус тёмный ток
    data_images : np.ndarray  — проекции минус тёмный ток
    data_angles : np.ndarray  — углы проекций
    """
    empty_images, _ = get_frame_group(data_file, 'empty', tmp_dir)
    dark_images, _ = get_frame_group(data_file, 'dark', tmp_dir)

    empty_image = np.median(empty_images, axis=0)
    dark_image = np.median(dark_images, axis=0)
    empty_image -= dark_image

    # TODO: добавить поддержку, когда много кадров на одном угле
    data_images, data_angles = get_frame_group(data_file, 'data', tmp_dir)
    data_images -= dark_image

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
        # Копируем секции, исключая служебный DEFAULT
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
                      pixel_size: float = 9e-3,
                      gpu_id: int = 0) -> np.ndarray:
    """FBP + CGLS реконструкция одного 2D среза.

    Результат масштабируется на 1/pixel_size.

    Параметры
    ----------
    sino      : синограмма, shape (N_angles, W)
    angles    : углы в градусах, shape (N_angles,)
    pixel_size: размер пикселя в мм
    gpu_id    : индекс GPU для ASTRA
    """
    rec = astra_utils.astra_recon_2d_parallel(
        sino, angles, [['FBP_CUDA'], ['CGLS_CUDA', 10]], gpu_id=gpu_id
    )
    return rec / pixel_size


def recon_2d_parallel_nonorm(sino: np.ndarray, angles: np.ndarray,
                              gpu_id: int = 0) -> np.ndarray:
    """FBP реконструкция одного 2D среза без нормировки (для предпросмотра)."""
    return astra_utils.astra_recon_2d_parallel(
        sino[angles < 180], angles[angles < 180], ['FBP_CUDA'], gpu_id=gpu_id
    )



def recon_volume_astra3d(sinogram_fixed: np.ndarray,
                          data_angles: np.ndarray,
                          pixel_size: float,
                          rec_vol: np.ndarray,
                          gpu_indices: list | tuple = (0,),
                          chunk_size: int = 64) -> None:
    """Реконструкция всего объёма с нативным ASTRA 3D multi-GPU по чанкам (FBP).

    Использует ``FBP3D_CUDA`` — единственный 3D-алгоритм ASTRA, пригодный
    для чанковой обработки. В отличие от итерационных алгоритмов (CGLS3D),
    FBP3D обрабатывает каждый срез независимо, поэтому разбивка на чанки
    не нарушает корректность реконструкции.

    ``astra.set_gpu_index(gpu_indices)`` вызывается один раз до цикла —
    ASTRA распределяет вычисления каждого чанка по указанным GPU.

    Параметры
    ----------
    sinogram_fixed : np.ndarray, shape (H, N_angles, W)
        Синограмма после коррекции оси и удаления колец.
    data_angles    : np.ndarray, shape (N_angles,)
        Углы проекций в градусах.
    pixel_size     : float
        Размер пикселя в мм.
    rec_vol        : np.ndarray, shape (H, W, W)
        Выходной массив для записи результата (memmap или обычный ndarray).
    gpu_indices    : list or tuple of int
        Индексы GPU для нативного multi-GPU в ASTRA, например [0, 1].
    chunk_size     : int
        Количество срезов в одном ASTRA 3D вызове (default: 64).
        VRAM на чанк ≈ chunk_size × W × W × 4 байт.
        При W=1000, chunk_size=64: ~256 MB.
    """
    import astra  # noqa

    H = sinogram_fixed.shape[0]
    angles_f32 = data_angles.astype('float32', copy=False)

    # Устанавливаем multi-GPU один раз до цикла
    astra.set_gpu_index(list(gpu_indices))

    for start in range(0, H, chunk_size):
        end = min(start + chunk_size, H)
        sino_chunk = sinogram_fixed[start:end]   # (chunk, N_angles, W)
        # BP3D_CUDA: обратная проекция без фильтра.
        # FBP3D_CUDA не существует в ASTRA для параллельного пучка.
        # Для получения качества FBP нужно предварительно отфильтровать
        # синограмму ramp-фильтром и использовать BP3D_CUDA.
        # CGLS3D_CUDA/SIRT3D_CUDA нельзя разбивать на чанки.
        rec_chunk = astra_utils.astra_recon_3d_parallel(
            sino_chunk,
            angles_f32,
            [['BP3D_CUDA']],
        )
        rec_vol[start:end] = (rec_chunk / pixel_size).astype(rec_vol.dtype, copy=False)


# ---------------------------------------------------------------------------
# Multi-GPU reconstruction via shared memory
# ---------------------------------------------------------------------------

def _recon_worker_shmem(args: dict) -> None:
    """Воркер реконструкции для отдельного процесса.

    Читает синограмму из SharedMemory, пишет результат в SharedMemory выхода.
    Оба буфера размещены в /dev/shm (RAM) — нет дискового I/O.

    Параметры в словаре args
    ------------------------
    shm_sino_name     : str   — имя SharedMemory синограммы
    shm_out_name      : str   — имя SharedMemory выходного объёма
    sino_shape        : tuple — (H, N_angles, W)
    out_shape         : tuple — (H, W, W)
    dtype_sino        : str   — dtype синограммы ('float32')
    dtype_out         : str   — dtype выходного объёма ('float32')
    slice_indices     : list  — список индексов срезов для обработки
    angles            : bytes — np.ndarray углов, сериализованный через tobytes()
    angles_shape      : tuple — форма массива углов
    pixel_size        : float
    gpu_id            : int
    use_cgls          : bool  — если True, применяется FBP_CUDA + CGLS_CUDA×10
    norm_thresh_factor: float — срез считается пустым если L2-норма < max*factor
                                (0 — не проверять, всегда использовать use_cgls)
    """
    import multiprocessing.shared_memory as shm_mod
    import astra  # noqa: импорт внутри процесса (spawn)
    import cupy as cp  # noqa
    import tomo.recon.astra_utils as astra_utils  # noqa

    # Распаковываем аргументы
    shm_sino_name      = args['shm_sino_name']
    shm_out_name       = args['shm_out_name']
    sino_shape         = args['sino_shape']
    out_shape          = args['out_shape']
    dtype_sino         = args['dtype_sino']
    dtype_out          = args['dtype_out']
    slice_indices      = args['slice_indices']
    angles_bytes       = args['angles']
    angles_shape       = args['angles_shape']
    pixel_size         = args['pixel_size']
    gpu_id             = args['gpu_id']
    use_cgls           = args.get('use_cgls', True)
    norm_thresh_factor = args.get('norm_thresh_factor', 0.0)

    # Восстанавливаем массив углов
    angles = np.frombuffer(angles_bytes, dtype='float32').reshape(angles_shape)

    # Присоединяемся к SharedMemory (чтение sino, запись out)
    shm_sino = shm_mod.SharedMemory(name=shm_sino_name)
    shm_out  = shm_mod.SharedMemory(name=shm_out_name)
    sino_arr = np.ndarray(sino_shape, dtype=dtype_sino, buffer=shm_sino.buf)
    out_arr  = np.ndarray(out_shape,  dtype=dtype_out,  buffer=shm_out.buf)

    # Предвычисляем порог для пустых срезов один раз на воркер
    if norm_thresh_factor > 0:
        local_sino  = sino_arr[slice_indices]
        norms_local = np.linalg.norm(local_sino.reshape(len(slice_indices), -1), axis=1)
        norm_thresh = float(norms_local.max()) * norm_thresh_factor
    else:
        norm_thresh = -1.0  # отключить проверку

    method_full = [['FBP_CUDA'], ['CGLS_CUDA', 10]] if use_cgls else [['FBP_CUDA']]
    method_fbp  = [['FBP_CUDA']]

    # Устанавливаем GPU для этого процесса
    with cp.cuda.Device(gpu_id):
        for i in slice_indices:
            sino_i = sino_arr[i]
            if norm_thresh >= 0:
                norm_i = float(np.linalg.norm(sino_i))
                method = method_fbp if norm_i <= norm_thresh else method_full
            else:
                method = method_full
            rec = astra_utils.astra_recon_2d_parallel(
                sino_i, angles, method, gpu_id=gpu_id
            )
            out_arr[i] = rec / pixel_size

    shm_sino.close()
    shm_out.close()


def recon_volume_multi_gpu(sinogram_fixed: np.ndarray,
                            data_angles: np.ndarray,
                            pixel_size: float,
                            rec_vol: np.ndarray,
                            num_gpus: int = 2,
                            use_cgls: bool = True,
                            norm_thresh_factor: float = 1e-6,
                            pool=None) -> None:
    """Реконструкция всего объёма с разбивкой срезов по нескольким GPU.

    Синограмма передаётся через ``multiprocessing.shared_memory`` (без копирования).
    Результат пишется напрямую в ``/dev/shm`` (tmpfs) через ``np.memmap``,
    что избегает медленной записи на диск.
    Дочерние процессы запускаются методом ``spawn`` для гарантии чистого
    CUDA-контекста (обязательно на Windows).

    Параметры
    ----------
    sinogram_fixed      : np.ndarray, shape (H, N_angles, W)
        Синограмма после коррекции оси и удаления колец.
    data_angles         : np.ndarray, shape (N_angles,)
        Углы проекций в градусах.
    pixel_size          : float
        Размер пикселя в мм.
    rec_vol             : np.ndarray, shape (H, W, W)
        Выходной массив (memmap или обычный ndarray) для записи результата.
    num_gpus            : int
        Количество GPU для параллельной реконструкции.
    use_cgls            : bool
        Если True — FBP_CUDA + CGLS_CUDA×10, иначе только FBP_CUDA.
    norm_thresh_factor  : float
        Срез считается пустым (только FBP, без CGLS), если L2-норма его
        синограммы < max_norm * norm_thresh_factor. Предотвращает NaN/Inf
        в CGLS на нулевых/почти нулевых срезах. 0 — отключить проверку.
    pool                : multiprocessing.pool.Pool или None
        Готовый пул процессов для переиспользования. Если None — создаётся
        и уничтожается внутри функции (overhead spawn каждый раз ~2-3 с).
        Передайте предварительно созданный пул для устранения этого overhead:
            ctx = multiprocessing.get_context('spawn')
            pool = ctx.Pool(processes=num_gpus)
            recon_volume_multi_gpu(..., pool=pool)
            pool.close(); pool.join()
    """
    import multiprocessing
    import multiprocessing.shared_memory as shm_mod

    n_slices   = sinogram_fixed.shape[0]
    sino_shape = sinogram_fixed.shape          # (H, N_angles, W)
    out_shape  = rec_vol.shape                 # (H, W, W)
    dtype_sino = sinogram_fixed.dtype.str      # '<f4'
    dtype_out  = rec_vol.dtype.str

    # --- SharedMemory для синограммы (чтение воркерами) ---
    nbytes_sino = int(np.prod(sino_shape)) * np.dtype(dtype_sino).itemsize
    # --- SharedMemory для выходного объёма (воркеры пишут напрямую в RAM) ---
    # Оба буфера размещаются в /dev/shm (tmpfs) — нет дискового I/O.
    # /dev/shm в контейнере настроен на 6 GB через shm_size в docker-compose.yml.
    nbytes_out  = int(np.prod(out_shape))  * np.dtype(dtype_out).itemsize

    shm_sino = shm_mod.SharedMemory(create=True, size=nbytes_sino)
    shm_out  = shm_mod.SharedMemory(create=True, size=nbytes_out)

    try:
        # Копируем синограмму в SharedMemory
        sino_shared = np.ndarray(sino_shape, dtype=dtype_sino, buffer=shm_sino.buf)
        np.copyto(sino_shared, sinogram_fixed.astype(dtype_sino, copy=False))

        # Инициализируем выходной буфер нулями
        out_shared = np.ndarray(out_shape, dtype=dtype_out, buffer=shm_out.buf)
        out_shared[:] = 0

        # Разбиваем срезы по GPU
        all_indices = list(range(n_slices))
        chunks = [list(c) for c in np.array_split(all_indices, num_gpus)]

        angles_f32 = data_angles.astype('float32', copy=False)
        worker_args = [
            {
                'shm_sino_name':      shm_sino.name,
                'shm_out_name':       shm_out.name,
                'sino_shape':         sino_shape,
                'out_shape':          out_shape,
                'dtype_sino':         dtype_sino,
                'dtype_out':          dtype_out,
                'slice_indices':      chunk,
                'angles':             angles_f32.tobytes(),
                'angles_shape':       angles_f32.shape,
                'pixel_size':         pixel_size,
                'gpu_id':             gpu_id,
                'use_cgls':           use_cgls,
                'norm_thresh_factor': norm_thresh_factor,
            }
            for gpu_id, chunk in enumerate(chunks)
        ]

        # Запускаем воркеры методом spawn (обязателен на Windows)
        if pool is not None:
            pool.map(_recon_worker_shmem, worker_args)
        else:
            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(processes=num_gpus) as _pool:
                _pool.map(_recon_worker_shmem, worker_args)

        # Копируем результат из SharedMemory в rec_vol (RAM→RAM, быстро)
        np.copyto(rec_vol, out_shared.astype(rec_vol.dtype, copy=False))

    finally:
        shm_sino.unlink()
        shm_sino.close()
        shm_out.unlink()
        shm_out.close()


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
