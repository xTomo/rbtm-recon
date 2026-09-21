"""Тесты tomo_worker: разбор ошибок выполненного ноутбука и устойчивость цикла."""
import json

import pytest

import tomo_worker


SYNTHETIC_NB = json.loads("""
{
  "cells": [
    {"cell_type": "markdown", "source": ["# title"]},
    {"cell_type": "code", "source": ["print(1)"],
     "outputs": [{"output_type": "stream", "name": "stdout", "text": ["1\\n"]}]},
    {"cell_type": "code", "source": ["1/0"],
     "outputs": [{"output_type": "error", "ename": "ZeroDivisionError",
                  "evalue": "division by zero", "traceback": ["..."]}]},
    {"cell_type": "code", "source": ["pass"], "outputs": []}
  ],
  "nbformat": 4
}
""")

CLEAN_NB = json.loads("""
{
  "cells": [
    {"cell_type": "code", "source": ["print(1)"],
     "outputs": [{"output_type": "stream", "name": "stdout", "text": ["1\\n"]}]},
    {"cell_type": "markdown", "source": ["ok"]}
  ],
  "nbformat": 4
}
""")


def test_find_notebook_errors_detects_error_output():
    errors = tomo_worker.find_notebook_errors(SYNTHETIC_NB)
    assert len(errors) == 1
    assert errors[0]['ename'] == 'ZeroDivisionError'


def test_find_notebook_errors_on_clean_notebook():
    assert tomo_worker.find_notebook_errors(CLEAN_NB) == []


def test_find_notebook_errors_tolerates_missing_fields():
    assert tomo_worker.find_notebook_errors({}) == []
    assert tomo_worker.find_notebook_errors({'cells': [{'cell_type': 'code'}]}) == []
    assert tomo_worker.find_notebook_errors(
        {'cells': [{'cell_type': 'code', 'outputs': None}]}) == []


def test_format_notebook_error_truncates_evalue():
    err = {'ename': 'ValueError', 'evalue': 'x' * 400}
    status = tomo_worker.format_notebook_error(err)
    assert status.startswith('error: ValueError: ')
    assert status == 'error: ValueError: ' + 'x' * 150


def test_notebook_name_is_reconstructor4():
    assert tomo_worker.NOTEBOOK_NAME == 'reconstructor4.py'


def test_reconstruct_sets_error_status_when_notebook_cell_failed(monkeypatch):
    statuses = []
    monkeypatch.setattr(tomo_worker, 'set_object_status',
                        lambda obj_id, status: statuses.append(status))
    monkeypatch.setattr(tomo_worker, 'copy_python_files',
                        lambda obj_id, storage_dir: '/tmp/out')
    monkeypatch.setattr(tomo_worker, '_notebook_auto_run',
                        lambda nb: (SYNTHETIC_NB,
                                    tomo_worker.find_notebook_errors(SYNTHETIC_NB)))

    tomo_worker.reconstruct({'obj_id': 'exp1'})

    assert statuses == ['reconstructing',
                        'error: ZeroDivisionError: division by zero']


def test_reconstruct_sets_done_when_no_errors(monkeypatch):
    statuses = []
    monkeypatch.setattr(tomo_worker, 'set_object_status',
                        lambda obj_id, status: statuses.append(status))
    monkeypatch.setattr(tomo_worker, 'copy_python_files',
                        lambda obj_id, storage_dir: '/tmp/out')
    monkeypatch.setattr(tomo_worker, '_notebook_auto_run',
                        lambda nb: (CLEAN_NB, []))

    tomo_worker.reconstruct({'obj_id': 'exp1'})

    assert statuses == ['reconstructing', 'done']


def test_process_once_ignores_unknown_action(monkeypatch):
    calls = []
    monkeypatch.setattr(tomo_worker, 'get_rec_queue_next_obj',
                        lambda: {'obj_id': 'x', 'action': 'fly_to_the_moon'})
    monkeypatch.setattr(tomo_worker, 'reconstruct', lambda o: calls.append('rec'))
    monkeypatch.setattr(tomo_worker, 'copyfiles', lambda o: calls.append('copy'))

    assert tomo_worker.process_once() is False
    assert calls == []


def test_process_once_ignores_missing_action(monkeypatch):
    monkeypatch.setattr(tomo_worker, 'get_rec_queue_next_obj',
                        lambda: {'obj_id': 'x'})
    assert tomo_worker.process_once() is False


def test_worker_loop_survives_queue_exception(monkeypatch):
    """Недоступная MongoDB на старте не должна убивать воркер."""
    calls = {'n': 0}

    def _boom():
        calls['n'] += 1
        raise RuntimeError('ServerSelectionTimeoutError')

    monkeypatch.setattr(tomo_worker, 'get_rec_queue_next_obj', _boom)
    monkeypatch.setattr(tomo_worker.time, 'sleep', lambda s: None)

    tomo_worker.worker_loop(iterations=3, sleep_seconds=0)

    assert calls['n'] == 3


def test_worker_loop_sleeps_on_empty_queue(monkeypatch):
    sleeps = []
    monkeypatch.setattr(tomo_worker, 'get_rec_queue_next_obj', lambda: None)
    monkeypatch.setattr(tomo_worker.time, 'sleep', lambda s: sleeps.append(s))

    tomo_worker.worker_loop(iterations=2, sleep_seconds=7)

    assert sleeps == [7, 7]


def test_copy_python_files_uses_script_dir(monkeypatch, tmp_path):
    """Скрипты копируются относительно каталога воркера, а не cwd."""
    import os

    monkeypatch.setattr(tomo_worker.tomotools, 'get_tomoobject_info',
                        lambda to: {'_id': 'exp42', 'specimen': '100% Fe',
                                    'tags': 'a,b'})
    monkeypatch.chdir(tmp_path)

    out_dir = tomo_worker.copy_python_files('exp42', str(tmp_path / 'storage'))

    assert os.path.exists(os.path.join(out_dir, 'tomotools4.py'))
    assert os.path.exists(os.path.join(out_dir, 'hdf5_v2.py'))
    assert os.path.exists(os.path.join(out_dir, 'reconstructor4.py'))


def test_tomo_ini_written_with_percent_in_specimen(monkeypatch, tmp_path):
    """'%' в specimen не должен ломать запись tomo.ini (interpolation=None)."""
    import configparser
    import os

    monkeypatch.setattr(tomo_worker.tomotools, 'get_tomoobject_info',
                        lambda to: {'_id': 'exp43',
                                    'specimen': 'сплав 30% Ni / 70% Cu'})

    out_dir = tomo_worker.copy_python_files('exp43', str(tmp_path / 'storage'))

    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read(os.path.join(out_dir, 'tomo.ini'), encoding='utf8')
    assert cfg['SAMPLE']['specimen'] == 'сплав 30% Ni / 70% Cu'


def test_tomo_ini_fails_with_default_interpolation():
    """Контроль: дефолтный ConfigParser действительно падает на '%'."""
    import configparser

    cfg = configparser.ConfigParser()
    with pytest.raises(ValueError):
        cfg['SAMPLE'] = {'specimen': '30% Ni'}
