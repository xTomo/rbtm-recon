"""Тесты вытеснения старых кропов из кэша на /fast (reconservice.cache)."""
import os
import time

from reconservice import cache
from service_helpers import make_config


def _crop(cfg, exp_id, name, size, used):
    d = cfg.cache_dir(exp_id)
    os.makedirs(d, exist_ok=True)
    data_path = os.path.join(d, 'crop-{}.u16'.format(name))
    with open(data_path, 'wb') as fh:
        fh.truncate(size)
    meta_path = os.path.splitext(data_path)[0] + '.json'
    with open(meta_path, 'w') as fh:
        fh.write('{}')
    os.utime(meta_path, (used, used))
    return data_path


def test_cleanup_removes_least_recently_used_over_limit(tmp_path):
    now = time.time()
    mb = 1024 ** 2
    cfg = make_config(tmp_path, fast_limit_gb=2.5 * mb / 1024 ** 3, session_ttl_s=600)
    old = _crop(cfg, 'e1', 'old', mb, now - 7200)
    mid = _crop(cfg, 'e2', 'mid', mb, now - 3600)
    fresh = _crop(cfg, 'e1', 'fresh', mb, now - 10)
    removed = cache.cleanup(cfg, now=now)
    assert removed == [old]                                   # 3 МБ > 2,5 МБ: хватает удалить самый старый
    assert not os.path.exists(old) and not os.path.exists(os.path.splitext(old)[0] + '.json')
    assert os.path.exists(mid) and os.path.exists(fresh)


def test_cleanup_keeps_recently_used_even_over_limit(tmp_path):
    now = time.time()
    mb = 1024 ** 2
    cfg = make_config(tmp_path, fast_limit_gb=0.5 * mb / 1024 ** 3, session_ttl_s=600)
    a = _crop(cfg, 'e1', 'a', mb, now - 100)                  # в пределах TTL сессии — защищены
    b = _crop(cfg, 'e1', 'b', mb, now - 5000)
    assert cache.cleanup(cfg, now=now) == [b]
    assert os.path.exists(a)


def test_crop_loader_touches_meta_on_cache_hit(tmp_path):
    import engine_scans as es
    from reconengine import data
    from reconengine.model import ROI
    ss = es.simple_scan()
    path = es.write_h5(ss, tmp_path / 'scan.h5')
    scan = data.open_scan(path)
    loader = data.CropLoader(scan, str(tmp_path / 'cache'))
    roi = ROI(2, 70, 4, 36)
    crop = loader.load(roi)
    meta = os.path.splitext(crop.path)[0] + '.json'
    os.utime(meta, (1_000_000, 1_000_000))
    del crop
    loader.load(roi)
    assert os.path.getmtime(meta) > time.time() - 60
