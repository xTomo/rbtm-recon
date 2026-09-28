"""Тесты очереди заданий (mongomock вместо реальной MongoDB).

Проверяют FIFO-порядок get_rec_queue_next_obj и то, что recon- и web-копии
tomo_queue.py идентичны.
"""
import filecmp
import os

import mongomock
import pytest

import tomo_queue


@pytest.fixture
def queue(monkeypatch):
    """Подменяет коллекцию MongoDB на mongomock."""
    coll = mongomock.MongoClient()['autotom']['tomoobjects']
    monkeypatch.setattr(tomo_queue, 'to', coll)
    return coll


def test_next_obj_is_fifo_by_id(queue):
    tomo_queue.put_object_rec_queue('a')
    tomo_queue.put_object_rec_queue('b')
    tomo_queue.put_object_rec_queue('c')

    obj = tomo_queue.get_rec_queue_next_obj()
    assert obj['obj_id'] == 'a'
    assert obj['action'] == 'reconstruct'


def test_next_obj_skips_superseded_waiting(queue):
    tomo_queue.put_object_rec_queue('a')
    tomo_queue.put_object_rec_queue('b')
    # у 'a' появился более свежий статус → задание больше не актуально
    tomo_queue.set_object_status('a', 'canceled')

    obj = tomo_queue.get_rec_queue_next_obj()
    assert obj['obj_id'] == 'b'


def test_next_obj_skips_records_without_action(queue):
    queue.insert_one({'obj_id': 'a', 'status': 'waiting'})
    tomo_queue.put_object_rec_queue('b')

    obj = tomo_queue.get_rec_queue_next_obj()
    assert obj['obj_id'] == 'b'


def test_next_obj_empty_queue(queue):
    assert tomo_queue.get_rec_queue_next_obj() is None


def test_put_object_rec_queue_dedup(queue):
    assert tomo_queue.put_object_rec_queue('a') is True
    assert tomo_queue.put_object_rec_queue('a') is False
    assert queue.count_documents({'obj_id': 'a'}) == 1

    tomo_queue.set_object_status('a', 'done')
    assert tomo_queue.put_object_rec_queue('a') is True


def test_get_object_status_for_unknown(queue):
    assert tomo_queue.get_object_status('nope').startswith('hm...')


def test_get_logs_returns_all_records(queue):
    tomo_queue.put_object_rec_queue('a')
    tomo_queue.set_object_status('a', 'reconstructing')
    tomo_queue.set_object_status('a', 'done')

    logs = tomo_queue.get_logs('a')
    assert len(logs) == 3
    assert 'done' in logs[0]


def test_cancel_all_waiting(queue):
    tomo_queue.put_object_rec_queue('a')
    tomo_queue.put_object_rec_queue('b')

    assert tomo_queue.cancel_all_waiting() == 2
    assert tomo_queue.get_rec_queue_next_obj() is None


def test_recon_and_web_queue_modules_are_identical():
    here = os.path.dirname(os.path.abspath(__file__))
    web_dir = os.path.dirname(here)
    recon_copy = os.path.join(web_dir, 'rbtmrecon', 'recon', 'tomo_queue.py')
    web_copy = os.path.join(web_dir, 'rbtmwebrecon', 'webrecon', 'tomo_queue.py')
    assert filecmp.cmp(recon_copy, web_copy, shallow=False), (
        'rbtmrecon/recon/tomo_queue.py и rbtmwebrecon/webrecon/tomo_queue.py '
        'должны быть идентичны'
    )
