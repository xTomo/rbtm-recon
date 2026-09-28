"""reconengine.preprocess: нормировка 1-в-1 со старым путём tomotools4, огибающая и авто-ROI, репозиционирование."""
import logging

import numpy as np
import pytest

import engine_phantom as ph
import hdf5_v2
import tomotools4 as t4
from helpers import make_v2_file
from reconengine import gpu, preprocess as pp
from reconengine.model import ROI

ROI_CROP = ROI(10, 86, 6, 58)


@pytest.fixture(autouse=True)
def _cpu_backend(monkeypatch):
    monkeypatch.setenv('RECON_ENGINE_CPU', '1')
    gpu.reset_backend()
    yield
    gpu.reset_backend()


def _write_h5(tmp_path, ss, series_length=3):
    tl = ph.Timeline(ss.scan.modes, ss.scan.angles.astype('float32'),
                     np.zeros(ss.scan.n_frames, dtype='int64'), ss.scan.frame_numbers)
    path, _ = make_v2_file(tmp_path / 'scan.h5', tl.as_frame_dicts(), images=ss.frames,
                           series_length=series_length, is_advanced=ss.scan.is_advanced)
    return path


def _data_in_fn_order(scan):
    idx = scan.data_idx[np.argsort(scan.frame_numbers[scan.data_idx], kind='stable')]
    return idx, scan.frame_numbers[idx]


# --- нормировка: совпадение со старым путём ----------------------------------------------------------------

def test_normalize_slab_matches_normalize_projections(tmp_path):
    """Простой скан: dark/empty по кропу + normalize_slab == load_tomo_data_v2 + normalize_projections."""
    ss = ph.make_synthetic_scan(np.arange(0, 180, 6.0), noise=1.0, seed=5)
    # горячие и «мёртвые» пиксели — чтобы работали safe_median и обрезка снизу
    ss.frames[10, 20, 30] = 60000
    ss.frames[12, 25, 40] = 0
    path = _write_h5(tmp_path, ss)

    empty_image, data_images, _ = hdf5_v2.load_tomo_data_v2(path, str(tmp_path))
    r = ROI_CROP
    old = data_images[:, r.y0:r.y1, r.x0:r.x1].copy()
    t4.normalize_projections(old, empty_image[r.y0:r.y1, r.x0:r.x1])

    crop = ph.make_crop(ss.frames, r)
    de = pp.dark_empty_from_crop(ss.scan, crop)
    assert de.periodic_empties == []
    idx, fns = _data_in_fn_order(ss.scan)
    new = pp.normalize_slab(crop.frames[idx], fns, de, xp=np)

    assert new.dtype == np.float32 and new.shape == old.shape
    np.testing.assert_array_equal(new, old)


def test_normalize_slab_matches_timeline_normalization(tmp_path):
    """Advanced со дрейфом источника: == load_tomo_data_advanced_v2 + normalize_projections_with_timeline."""
    ss = ph.make_synthetic_scan(np.arange(0, 180, 4.0), advanced=True, n_segments=3, series_length=3,
                                drift=0.3, noise=1.0, seed=7)
    path = _write_h5(tmp_path, ss, series_length=3)
    adv = hdf5_v2.load_tomo_data_advanced_v2(path, str(tmp_path))
    r = ROI_CROP
    old = adv.data_images[:, r.y0:r.y1, r.x0:r.x1].copy()
    t4.normalize_projections_with_timeline(old, adv, r.x0, r.x1, r.y0, r.y1)

    crop = ph.make_crop(ss.frames, r)
    de = pp.dark_empty_from_crop(ss.scan, crop)
    # опорные кадры совпадают с загрузчиком старого пути
    np.testing.assert_array_equal(de.dark, adv.dark_image[r.y0:r.y1, r.x0:r.x1])
    np.testing.assert_array_equal(de.initial_empty, adv.initial_empty[r.y0:r.y1, r.x0:r.x1])
    assert de.initial_empty_fnumber == adv.initial_empty_fnumber
    assert de.periodic_empty_fnumbers == adv.periodic_empty_fnumbers
    assert len(de.periodic_empties) == 2
    for a, b in zip(de.periodic_empties, adv.periodic_empties):
        np.testing.assert_array_equal(a, b[r.y0:r.y1, r.x0:r.x1])

    idx, fns = _data_in_fn_order(ss.scan)
    np.testing.assert_array_equal(fns, adv.data_numbers)
    new = pp.normalize_slab(crop.frames[idx], fns, de, xp=np)
    np.testing.assert_array_equal(new, old)
    # дрейф действительно учтён: одна начальная empty дала бы другой результат
    de_flat = pp.DarkEmpty(de.dark, de.initial_empty, de.initial_empty_fnumber, [], [])
    flat = pp.normalize_slab(crop.frames[idx], fns, de_flat, xp=np)
    assert np.abs(flat[-1] - new[-1]).max() > 0.1


def test_normalize_slab_rows_subset_differs_only_at_edges():
    """Слой строк с dark/empty по тем же строкам совпадает с полным кропом, кроме крайних строк слоя (медиана 3×3)."""
    ss = ph.make_synthetic_scan(np.arange(0, 180, 10.0), noise=1.0, seed=3)
    crop = ph.make_crop(ss.frames, ROI_CROP)
    idx, fns = _data_in_fn_order(ss.scan)
    full = pp.normalize_slab(crop.frames[idx], fns, pp.dark_empty_from_crop(ss.scan, crop), xp=np)
    r0, r1 = 20, 36
    de = pp.dark_empty_from_crop(ss.scan, crop, rows=(r0, r1))
    part = pp.normalize_slab(crop.frames[idx, r0:r1], fns, de, xp=np)
    np.testing.assert_array_equal(part[:, 1:-1], full[:, r0 + 1:r1 - 1])
    assert not np.array_equal(part[:, 0], full[:, r0])
    # без медианы совпадают все строки
    part_nm = pp.normalize_slab(crop.frames[idx, r0:r1], fns, de, xp=np, median3=False)
    full_nm = pp.normalize_slab(crop.frames[idx], fns, pp.dark_empty_from_crop(ss.scan, crop), xp=np,
                                median3=False)
    np.testing.assert_array_equal(part_nm, full_nm[:, r0:r1])


def test_normalize_slab_uses_default_backend_and_validates_shapes():
    ss = ph.make_synthetic_scan(np.arange(0, 180, 30.0))
    crop = ph.make_crop(ss.frames, ROI_CROP)
    de = pp.dark_empty_from_crop(ss.scan, crop, rows=(0, 10))
    idx, fns = _data_in_fn_order(ss.scan)
    res = pp.normalize_slab(crop.frames[idx, 0:10], fns, de)
    assert isinstance(res, np.ndarray) and res.shape == (len(idx), 10, ROI_CROP.width)
    with pytest.raises(ValueError):
        pp.normalize_slab(crop.frames[idx, 0:12], fns, de)
    with pytest.raises(ValueError):
        pp.normalize_slab(crop.frames[idx, 0:10], fns[:-1], de)


@pytest.mark.parametrize('frame_number, expected', [
    (2, 100.0), (0, 100.0), (50, 149.0), (100, 200.0), (150, 300.0), (200, 400.0), (250, 400.0),
])
def test_empty_for_frame_matches_interpolate_empty(frame_number, expected):
    def const(v):
        return np.full((4, 5), float(v), dtype='float32')

    de = pp.DarkEmpty(const(0), const(100), 2, [const(200), const(400)], [100, 200])
    adv = t4.AdvancedTomoData(
        dark_image=const(0), initial_empty=const(100), initial_empty_fnumber=2,
        periodic_empties=[const(200), const(400)], periodic_empty_fnumbers=[100, 200],
        data_images=np.empty((0, 4, 5), 'float32'), data_angles=np.empty(0, 'float32'),
        data_numbers=np.empty(0, 'int64'), data_check_images=np.empty((0, 4, 5), 'float32'),
        data_check_angles=np.empty(0, 'float32'), data_check_numbers=np.empty(0, 'int64'), series_length=2)
    res = pp.empty_for_frame(de, frame_number)
    assert res.dtype == np.float32 and res.shape == (4, 5)
    np.testing.assert_array_equal(res, t4._interpolate_empty(adv, frame_number, 0, 5, 0, 4))
    assert res == pytest.approx(np.full((4, 5), expected), abs=0.51)


def test_empty_for_frame_clips_and_copies():
    e = np.array([[0.5, 5.0]], dtype='float32')
    de = pp.DarkEmpty(np.zeros((1, 2), 'float32'), e, 0, [], [])
    res = pp.empty_for_frame(de, 10)
    np.testing.assert_array_equal(res, [[1.0, 5.0]])
    res[0, 1] = 7
    assert e[0, 1] == 5.0


def test_dark_empty_series_split():
    """Серии empty делятся по series_length, frame_number серии — первый кадр; хвост не кратный — отбрасывается."""
    ss = ph.make_synthetic_scan(np.arange(0, 90, 10.0), advanced=True, n_segments=3, series_length=2)
    crop = ph.make_crop(ss.frames, ROI_CROP)
    de = pp.dark_empty_from_crop(ss.scan, crop, rows=(5, 9))
    sc = ss.scan
    first = sc.frame_numbers[sc.empty_idx]
    assert de.initial_empty_fnumber == first[0]
    assert de.periodic_empty_fnumbers == [int(first[2]), int(first[4])]
    assert de.dark.shape == de.initial_empty.shape == (4, ROI_CROP.width)
    expected = np.median(crop.frames[sc.empty_idx[2:4], 5:9].astype('float32') - de.dark, axis=0)
    np.testing.assert_array_equal(de.periodic_empties[0], expected)

    sc.series_length = 0
    with pytest.raises(ValueError, match='series_length'):
        pp.dark_empty_from_crop(sc, crop)
    with pytest.raises(ValueError):
        pp.dark_empty_from_crop(sc, crop, rows=(10, 5))


# --- огибающая, авто-ROI, углы за пределами ROI ------------------------------------------------------------

def _scan_for_roi(noise=1.0):
    # центральное тело, пятно на радиусе 15 (уходит в стороны около 0° и 180°) и пятно у оси
    blobs = [ph.Blob(0.0, 0.0, 0.0, 4.0, 0.05), ph.Blob(15.0, 0.0, 4.0, 2.0, 0.08),
             ph.Blob(-2.0, 2.0, -8.0, 2.5, 0.05)]
    angles = np.arange(0, 360, 22.5)
    return ph.make_synthetic_scan(angles, height=64, width=128, center_x=70.2, y_ref=33.0, blobs=blobs,
                                  noise=noise, seed=11)


def test_envelope_and_suggest_roi_find_object():
    ss = _scan_for_roi()
    sc = ss.scan
    ov = ph.make_overview(ss, sc.data_idx, b=2)
    env = pp.envelope(ov)
    assert env.dtype == np.float32 and env.shape == (32, 64)
    roi = pp.suggest_roi(env, ov.bin, sc.height, sc.width)
    # объект: там, где проекции заметны (≫ шума) хотя бы на одном угле
    pmax = ss.projections[sc.data_idx].max(axis=0)
    ys, xs = np.where(pmax > 0.15)
    assert roi.x0 <= xs.min() and roi.x1 > xs.max()
    assert roi.y0 <= ys.min() and roi.y1 > ys.max()
    # и ROI не раздут: объект ~ 70 ± 20 столбцов из 128
    assert roi.width < 64 and roi.x0 > 30 and roi.x1 < 110
    assert roi.preview_row == (roi.y0 + roi.y1) // 2
    roi.validate(sc.height, sc.width)


def test_suggest_roi_whole_frame_without_object():
    env = np.zeros((20, 30), dtype='float32')
    assert pp.suggest_roi(env, 4, 80, 120) == ROI(0, 120, 0, 80)


def test_suggest_roi_margins_and_clipping():
    env = np.zeros((50, 100), dtype='float32')
    env[10:40, 5:60] = 1.0
    roi = pp.suggest_roi(env, 2, 100, 200, margin_frac=0.05)
    assert (roi.x0, roi.x1, roi.y0, roi.y1) == (0, 130, 15, 85)


def test_angles_outside_flags_narrow_roi():
    ss = _scan_for_roi()
    sc = ss.scan
    ov = ph.make_overview(ss, sc.data_idx, b=2)
    roi_ok = pp.suggest_roi(pp.envelope(ov), ov.bin, sc.height, sc.width)
    assert pp.angles_outside(ov, roi_ok) == []
    # узкий ROI ±11 столбцов вокруг оси: пятно на радиусе 15 выходит за него около 0° и 180°, но не около 90°
    narrow = ROI(59, 82, 0, 64)
    out = pp.angles_outside(ov, narrow)
    assert 0.0 in out and 180.0 in out
    assert 90.0 not in out and 270.0 not in out


# --- репозиционирование ------------------------------------------------------------------------------------

ROI_REPOS = ROI(4, 92, 4, 84)


def _advanced_with_shifts():
    """Advanced-скан с объектом с резкими краями целиком внутри ROI_REPOS — на нём работает и старая фазовая
    корреляция tomotools4 (ей нужны высокие частоты), так что результаты можно сравнить."""
    offsets = [(0.0, 0.0), (1.5, -2.0), (0.5, 1.0)]   # абсолютные (dy, dx) сегментов
    ss = ph.make_synthetic_scan(np.arange(0, 180, 4.0), height=88, width=96, y_ref=43.5,
                                blobs=ph.sharp_objects(), advanced=True, n_segments=3, series_length=3,
                                segment_offsets=offsets, drift=0.1, noise=0.5, seed=21)
    return ss, offsets


def test_repositioning_shifts_recover_known_shifts(tmp_path):
    ss, offsets = _advanced_with_shifts()
    crop = ph.make_crop(ss.frames, ROI_REPOS)
    de = pp.dark_empty_from_crop(ss.scan, crop)
    angles, sy, sx = pp.repositioning_shifts(ss.scan, crop, de)
    assert len(sy) == len(sx) == len(angles) == 2
    # соглашение phase_cross_correlation(reference=data, moving=data_check): сдвиг возвращает кадр назад
    steps = np.diff(np.array(offsets), axis=0)
    np.testing.assert_allclose(sy, -steps[:, 0], atol=0.11)
    np.testing.assert_allclose(sx, -steps[:, 1], atol=0.11)
    cum_y, cum_x = pp.cumulative_shifts(sy, sx)
    np.testing.assert_allclose(cum_y, [0.0, -1.5, -0.5], atol=0.11)
    np.testing.assert_allclose(cum_x, [0.0, 2.0, -1.0], atol=0.11)

    # пары кадров те же, что у старого measure_repositioning_shifts; на резком объекте и его фазовая
    # корреляция близка к истине
    path = _write_h5(tmp_path, ss, series_length=3)
    adv = hdf5_v2.load_tomo_data_advanced_v2(path, str(tmp_path))
    r = ROI_REPOS
    a_old, sy_old, sx_old = t4.measure_repositioning_shifts(adv, r.x0, r.x1, r.y0, r.y1, debug=False)
    np.testing.assert_array_equal(angles, a_old)
    np.testing.assert_allclose(sy, sy_old, atol=0.15)
    np.testing.assert_allclose(sx, sx_old, atol=0.15)


@pytest.mark.parametrize('noise', [0.0, 0.5])
@pytest.mark.parametrize('offsets', [
    [(0.0, 0.0), (1.3, -0.8), (2.1, 0.6)],
    [(0.0, 0.0), (-0.7, 1.35), (0.45, 0.25)],
])
def test_repositioning_shifts_fractional_on_smooth_object(offsets, noise):
    """Дробные сдвиги гладкого объекта (кадр 96×128, наклонная ось, полный оборот): накопленные сдвиги ±0,11 px.
    Фазовая корреляция tomotools4 здесь ошибается на 0,5–1,9 px — поэтому в движке normalization=None."""
    ss = ph.make_synthetic_scan(np.arange(0, 360, 3.0), height=96, width=128, center_x=63.3, y_ref=47.5,
                                tilt_deg=0.6, advanced=True, n_segments=3, series_length=3,
                                segment_offsets=offsets, noise=noise, seed=3)
    crop = ph.make_crop(ss.frames, ROI(4, 124, 4, 92))
    de = pp.dark_empty_from_crop(ss.scan, crop)
    _, sy, sx = pp.repositioning_shifts(ss.scan, crop, de)
    cum_y, cum_x = pp.cumulative_shifts(sy, sx)
    # накопленный сдвиг сегмента возвращает его к сегменту 0: cum = −offset
    np.testing.assert_allclose(cum_y, [-o[0] for o in offsets], atol=0.11)
    np.testing.assert_allclose(cum_x, [-o[1] for o in offsets], atol=0.11)


def test_repositioning_shifts_skip_unmatched_checkpoint(caplog):
    ss, _ = _advanced_with_shifts()
    sc = ss.scan
    sc.angles = sc.angles.copy()
    sc.angles[sc.check_idx[0]] = 7.0   # такого угла среди data нет
    crop = ph.make_crop(ss.frames, ROI_REPOS)
    de = pp.dark_empty_from_crop(sc, crop)
    with caplog.at_level(logging.WARNING):
        angles, sy, sx = pp.repositioning_shifts(sc, crop, de)
    assert sy[0] == 0.0 and sx[0] == 0.0 and angles[0] == pytest.approx(7.0)
    assert sy[1] != 0.0 or sx[1] != 0.0
    assert any('допуск' in rec.getMessage() for rec in caplog.records)


def test_repositioning_shifts_not_advanced_or_wrong_rows():
    ss = ph.make_synthetic_scan(np.arange(0, 180, 30.0))
    crop = ph.make_crop(ss.frames, ROI_CROP)
    a, sy, sx = pp.repositioning_shifts(ss.scan, crop, pp.dark_empty_from_crop(ss.scan, crop))
    assert a.size == sy.size == sx.size == 0

    ss, _ = _advanced_with_shifts()
    crop = ph.make_crop(ss.frames, ROI_REPOS)
    with pytest.raises(ValueError):
        pp.repositioning_shifts(ss.scan, crop, pp.dark_empty_from_crop(ss.scan, crop, rows=(0, 10)))


def test_segment_index():
    fns = [0, 5, 10, 11, 20, 21, 99]
    np.testing.assert_array_equal(pp.segment_index(fns, [10, 20]), [0, 0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(pp.segment_index(fns, []), np.zeros(7))


def test_cumulative_shifts_nan_is_zero(caplog):
    with caplog.at_level(logging.ERROR):
        cy, cx = pp.cumulative_shifts(np.array([np.nan, 3.0]), np.array([np.nan, 0.5]))
    np.testing.assert_array_equal(cy, [0.0, 0.0, 3.0])
    np.testing.assert_array_equal(cx, [0.0, 0.0, 0.5])
    assert any('ненадёжны' in rec.getMessage() for rec in caplog.records)
    cy, cx = pp.cumulative_shifts(np.array([]), np.array([]))
    np.testing.assert_array_equal(cy, [0.0])
