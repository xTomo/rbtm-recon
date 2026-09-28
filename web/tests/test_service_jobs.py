"""Тесты очереди задач recon-service: API /jobs, исполнитель с настоящим процессом движка, отмена, остановка,
восстановление после перезапуска, старая очередь ноутбуков, статусы в tomoobjects."""
import json
import os
import threading
import time
import types

import mongomock
import pytest

from engine_scans import simple_scan
from jobs_helpers import (EXP, StubScans, engine_run, jobs_service, scan_and_recipe, slow_engine, tomo_statuses,
                          wait_for)
from reconengine import gpu
from reconservice import jobs, publish
from service_helpers import HEADERS, headers, make_config, write_scan


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv('RECON_ENGINE_CPU', '1')
    gpu.reset_backend()
    yield
    gpu.reset_backend()


def post_job(client, recipe, name=None, user='alice'):
    body = {'recipe': recipe}
    if name is not None:
        body['name'] = name
    return client.post('/jobs', json=body, headers=headers(user))


def job_doc(svc, job_id):
    return svc.coll.find_one({'_id': job_id})


# --- полный цикл ---------------------------------------------------------------------------------------------

def test_full_cycle_and_rerun(tmp_path, monkeypatch):
    app, client, cfg, svc = jobs_service(tmp_path, monkeypatch)
    _, recipe = scan_and_recipe(cfg)
    dest = cfg.reconstruction_dir(EXP)
    os.makedirs(dest)
    foreign = os.path.join(dest, 'отчёт ноутбука.html')          # не от движка: не трогается никогда
    with open(foreign, 'w', encoding='utf-8') as fh:
        fh.write('<html/>')

    r = post_job(client, recipe, name='образец 1')
    assert r.status_code == 201, r.get_json()
    job = r.get_json()
    assert job['status'] == 'queued' and job['id'] == job['_id'] and job['user'] == 'alice'
    assert job['name'] == 'образец 1' and job['recipe']['author'] == 'alice'
    assert job['created'].endswith('+00:00')
    assert client.get('/health').get_json()['jobs']['queued'] == 1

    assert svc.runner.run_once() is True
    job = client.get('/jobs/' + job['id'], headers=HEADERS).get_json()
    assert job['status'] == 'done', job
    assert job['progress'] == 1.0 and job['stage'] == 'done' and job['finished']
    assert job['result']['volume']['file'] == 'образец_1.32_68_68.1.raw'
    assert job['result']['run_id'] == job['run_id']

    # опубликовано туда же, куда копирует ноутбук; result.json — того же запуска
    names = set(os.listdir(dest))
    assert {'образец_1.32_68_68.1.raw', 'образец_1.32_68_68.1.raw.size', 'tomo.образец_1.1.hx',
            'образец_1.8_17_17.4.raw', 'образец_1.8_17_17.4.raw.size', 'tomo.образец_1.4.hx',
            'recipe.json', 'result.json'} <= names
    first = publish.read_json(os.path.join(dest, 'result.json'))
    assert first['run_id'] == job['run_id']
    assert tomo_statuses(svc) == ['reconstructing', 'done']
    assert os.path.isfile(os.path.join(cfg.storage_dir, EXP + '.h5'))           # архивная копия
    # в каталоге запуска остались лог и входной рецепт
    run_dir = os.path.join(cfg.runs_dir(EXP), job['run_id'])
    assert sorted(os.listdir(run_dir)) == ['engine.log', 'recipe.in.json']
    log = client.get('/jobs/{}/log'.format(job['id']), headers=HEADERS)
    assert log.status_code == 200 and log.mimetype == 'text/plain'
    text = log.get_data(as_text=True)
    assert '"stage": "recon"' in text and 'run_recipe' in text
    assert len(client.get('/jobs/{}/log?lines=2'.format(job['id']), headers=HEADERS).get_data(as_text=True)
               .splitlines()) == 2

    listed = client.get('/jobs?exp_id=' + EXP, headers=HEADERS).get_json()
    assert [j['id'] for j in listed] == [job['id']] and 'recipe' not in listed[0]
    health = client.get('/health').get_json()['jobs']
    assert health['queued'] == 0 and health['current'] is None
    assert svc.runner.run_once() is False                                   # очередь пуста, старая очередь выкл.

    # пересчёт с другими срезами: другие имена файлов → прошлый объём удалён, рецепт и метаданные — в history/
    recipe2 = dict(recipe, recon=dict(recipe['recon'], slices=[8, 32]))
    archive_mtime = os.path.getmtime(os.path.join(cfg.storage_dir, EXP + '.h5'))
    job2 = post_job(client, recipe2, name='образец 1').get_json()
    assert svc.runner.run_once() is True
    job2 = client.get('/jobs/' + job2['id'], headers=HEADERS).get_json()
    assert job2['status'] == 'done', job2
    names = set(os.listdir(dest))
    assert 'образец_1.24_68_68.1.raw' in names and 'образец_1.6_17_17.4.raw' in names
    assert not any(n.startswith('образец_1.32_68_68') or n.startswith('образец_1.8_17_17') for n in names)
    assert os.path.isfile(foreign)
    hist = os.path.join(dest, 'history', job['run_id'])
    assert sorted(os.listdir(hist)) == ['recipe.json', 'result.json']
    assert publish.read_json(os.path.join(hist, 'result.json'))['run_id'] == job['run_id']
    assert publish.read_json(os.path.join(dest, 'result.json'))['run_id'] == job2['run_id']
    assert os.path.getmtime(os.path.join(cfg.storage_dir, EXP + '.h5')) == archive_mtime   # копия уже была
    assert tomo_statuses(svc) == ['reconstructing', 'done', 'reconstructing', 'done']
    listed = client.get('/jobs', headers=HEADERS).get_json()
    assert [j['id'] for j in listed] == [job2['id'], job['id']]
    assert len(client.get('/jobs?limit=1', headers=HEADERS).get_json()) == 1


# --- создание -------------------------------------------------------------------------------------------------

def test_create_validation_and_conflict(tmp_path, monkeypatch):
    _, client, cfg, svc = jobs_service(tmp_path, monkeypatch)
    _, recipe = scan_and_recipe(cfg)

    assert client.post('/jobs', json={'recipe': recipe}).status_code == 403                  # без токена
    assert client.post('/jobs', data='not json', headers=HEADERS).status_code == 400
    assert client.post('/jobs', json={'recipe': 'x'}, headers=HEADERS).status_code == 400
    assert post_job(client, dict(recipe, schema='other/1')).status_code == 400
    assert post_job(client, dict(recipe, fov={'x0': 0})).status_code == 400                  # нет полей fov
    bad_fov = dict(recipe, fov=dict(recipe['fov'], x1=1000))                                   # вне кадра 72
    r = post_job(client, bad_fov)
    assert r.status_code == 400 and 'ROI' in r.get_json()['error']
    bad_slices = dict(recipe, recon=dict(recipe['recon'], slices=[0, 39]))                     # вне fov по y
    assert post_job(client, bad_slices).status_code == 400
    assert post_job(client, dict(recipe, input=dict(recipe['input'], exp_id='../x'))).status_code == 400
    assert post_job(client, dict(recipe, input=dict(recipe['input'], exp_id='nope'))).status_code == 404
    for name in ('../x', 'a/b', '.hidden', 'x' * 101, 'a:b'):
        assert post_job(client, recipe, name=name).status_code == 400, name
    assert svc.coll.count_documents({}) == 0

    changed = dict(recipe, input=dict(recipe['input'], fingerprint='другой'))
    r = post_job(client, changed, name='  ')
    assert r.status_code == 201
    first = r.get_json()
    assert first['name'] == EXP and first['warnings'] and 'отпечаток' in first['warnings'][0]

    r = post_job(client, recipe)                                                               # активная задача
    assert r.status_code == 409 and r.get_json()['job_id'] == first['id']
    svc.set_fields(first['id'], {'status': 'running'})
    assert post_job(client, recipe).status_code == 409
    svc.set_fields(first['id'], {'status': 'error'})
    assert post_job(client, recipe).status_code == 201

    assert client.get('/jobs/nope', headers=HEADERS).status_code == 404
    assert client.get('/jobs?exp_id=..', headers=HEADERS).status_code == 400
    assert client.get('/jobs?limit=x', headers=HEADERS).status_code == 400


# --- отмена ---------------------------------------------------------------------------------------------------

def test_cancel_queued(tmp_path, monkeypatch):
    _, client, cfg, svc = jobs_service(tmp_path, monkeypatch)
    _, recipe = scan_and_recipe(cfg)
    job = post_job(client, recipe).get_json()
    r = client.post('/jobs/{}/cancel'.format(job['id']), headers=headers('bob'))
    assert r.status_code == 200
    job = r.get_json()
    assert job['status'] == 'canceled' and job['cancel_requested'] is True and job['canceled_by'] == 'bob'
    assert job['finished']
    assert svc.runner.run_once() is False                            # отменённая не запускается
    assert tomo_statuses(svc) == []                                  # до запуска старую страницу не трогаем
    again = client.post('/jobs/{}/cancel'.format(job['id']), headers=HEADERS).get_json()
    assert again['status'] == 'canceled' and again['canceled_by'] == 'bob'
    assert client.post('/jobs/nope/cancel', headers=HEADERS).status_code == 404
    assert client.get('/jobs/{}/log'.format(job['id']), headers=HEADERS).get_data(as_text=True) == ''


def test_cancel_running(tmp_path, monkeypatch):
    """Отмена из другого потока во время run_once: процесс останавливается, run_dir удаляется, статус canceled."""
    slow_engine(tmp_path, monkeypatch, 2.0)
    _, client, cfg, svc = jobs_service(tmp_path, monkeypatch)
    _, recipe = scan_and_recipe(cfg)
    job_id = post_job(client, recipe).get_json()['id']

    writes = []
    set_fields = svc.set_fields

    def recording(jid, fields):
        if 'progress' in fields and fields.get('status') is None:
            writes.append(time.monotonic())
        set_fields(jid, fields)

    monkeypatch.setattr(svc, 'set_fields', recording)
    answers = []

    def canceller():
        wait_for(lambda: (job_doc(svc, job_id) or {}).get('stage') in ('prepare', 'recon'), what='движок считает')
        answers.append(client.post('/jobs/{}/cancel'.format(job_id), headers=headers('bob')).get_json())

    t = threading.Thread(target=canceller)
    t.start()
    assert svc.runner.run_once() is True
    t.join(5)

    assert answers and answers[0]['status'] == 'running' and answers[0]['cancel_requested'] is True
    doc = job_doc(svc, job_id)
    assert doc['status'] == 'canceled' and doc['finished'] is not None
    run_dir = os.path.join(cfg.runs_dir(EXP), doc['run_id'])
    assert not os.path.exists(run_dir)
    assert not os.path.exists(os.path.join(cfg.reconstruction_dir(EXP), 'result.json'))
    assert tomo_statuses(svc) == ['reconstructing', 'canceled']
    # прогресс в Mongo — не чаще раза в секунду
    assert all(b - a >= 0.95 for a, b in zip(writes, writes[1:])), writes


# --- остановка сервиса ----------------------------------------------------------------------------------------

def test_stop_interrupts_running_job(tmp_path, monkeypatch):
    """shutdown_all (хук gunicorn worker_exit) → JobService.stop(): процесс задачи прерывается, задача interrupted,
    её run_dir удалён, поток остановлен; новые задачи не берутся; повторный stop() безопасен."""
    from reconservice import app as app_mod
    slow_engine(tmp_path, monkeypatch, 2.0)
    monkeypatch.setattr(app_mod, '_STATES', [])
    _, client, cfg, svc = jobs_service(tmp_path, monkeypatch, start_threads=True)
    assert svc.runner.alive()
    _, recipe = scan_and_recipe(cfg)
    job_id = post_job(client, recipe).get_json()['id']
    wait_for(lambda: (job_doc(svc, job_id) or {}).get('stage') in ('prepare', 'recon'), what='движок считает')
    run_dir = os.path.join(cfg.runs_dir(EXP), job_doc(svc, job_id)['run_id'])
    assert os.path.isdir(run_dir)

    t0 = time.monotonic()
    app_mod.shutdown_all()
    assert time.monotonic() - t0 < 8.0
    assert not svc.runner.alive()
    doc = job_doc(svc, job_id)
    assert doc['status'] == 'interrupted' and doc['error'] and doc['finished'] is not None
    assert not os.path.exists(run_dir)
    assert tomo_statuses(svc) == ['reconstructing', 'error: interrupted']

    svc.stop()                                                        # повторно — без ошибок и ожидания
    job2 = post_job(client, recipe).get_json()
    assert svc.runner.run_once() is False
    assert job_doc(svc, job2['id'])['status'] == 'queued'


def test_stop_without_thread_is_safe(tmp_path, monkeypatch):
    _, _, _, svc = jobs_service(tmp_path, monkeypatch)
    t0 = time.monotonic()
    svc.stop()
    svc.stop()
    assert time.monotonic() - t0 < 1.0


# --- ошибки ---------------------------------------------------------------------------------------------------

def test_engine_error(tmp_path, monkeypatch):
    """Движок завершился с ошибкой: статус error с хвостом лога, кратко — в tomoobjects, run_dir с логом остаётся."""
    _, client, cfg, svc = jobs_service(tmp_path, monkeypatch)
    _, recipe = scan_and_recipe(cfg)
    job_id = post_job(client, recipe).get_json()['id']
    write_scan(cfg, EXP, simple_scan(height=20, width=40))          # скан подменили после постановки в очередь
    assert svc.runner.run_once() is True
    doc = job_doc(svc, job_id)
    assert doc['status'] == 'error'
    assert 'кодом 1' in doc['error'] and 'ValueError' in doc['error'] and 'ROI' in doc['error']
    status = tomo_statuses(svc)[-1]
    assert status.startswith('error: ValueError') and len(status) <= len('error: ') + 150
    run_dir = os.path.join(cfg.runs_dir(EXP), doc['run_id'])
    assert os.path.isfile(os.path.join(run_dir, 'engine.log'))
    assert 'Traceback' in client.get('/jobs/{}/log'.format(job_id), headers=HEADERS).get_data(as_text=True)


def test_service_error_before_engine(tmp_path, monkeypatch):
    _, client, cfg, svc = jobs_service(tmp_path, monkeypatch)
    _, recipe = scan_and_recipe(cfg)
    job_id = post_job(client, recipe).get_json()['id']
    os.remove(cfg.scan_path(EXP))
    assert svc.runner.run_once() is True
    doc = job_doc(svc, job_id)
    assert doc['status'] == 'error' and doc['error'].startswith('FileNotFoundError')
    assert tomo_statuses(svc) == ['reconstructing', 'error: FileNotFoundError: нет скана exp1']


def test_archive_failure_is_warning(tmp_path, monkeypatch):
    _, client, cfg, svc = jobs_service(tmp_path, monkeypatch)
    _, recipe = scan_and_recipe(cfg)
    job_id = post_job(client, recipe).get_json()['id']

    def broken(cfg_, exp_id):
        raise OSError('диск заполнен')

    monkeypatch.setattr(jobs.publish, 'archive_h5', broken)
    assert svc.runner.run_once() is True
    doc = job_doc(svc, job_id)
    assert doc['status'] == 'done'
    assert any('архивная копия' in w and 'диск заполнен' in w for w in doc['warnings'])


# --- процесс движка -------------------------------------------------------------------------------------------

def test_engine_command_and_environment(tmp_path, monkeypatch):
    monkeypatch.setenv('PYTHONPATH', 'other')
    _, _, cfg, svc = jobs_service(tmp_path, monkeypatch, job_gpu='0', gpu_lock=str(tmp_path / 'gpu.lock'),
                                  job_python='py')
    job = {'_id': 'j', 'exp_id': EXP, 'run_id': 'r1', 'name': 'обр 1'}
    cmd = svc.runner._engine_cmd(job, 'scan.h5', 'rin.json', 'rd')
    assert cmd[:5] == ['py', '-m', 'reconengine', 'run', 'scan.h5']
    assert cmd[cmd.index('--name') + 1] == 'обр 1' and cmd[cmd.index('--run-id') + 1] == 'r1'
    assert cmd[cmd.index('--cache') + 1] == cfg.cache_dir(EXP)
    assert '--progress-json' in cmd and cmd[cmd.index('--gpu-lock') + 1] == str(tmp_path / 'gpu.lock')
    env = svc.runner._engine_env()
    assert env['CUDA_VISIBLE_DEVICES'] == '0'
    assert env['PYTHONPATH'].split(os.pathsep) == [jobs.ENGINE_ROOT, 'other']
    assert os.path.isfile(os.path.join(jobs.ENGINE_ROOT, 'reconengine', '__main__.py'))

    cfg.gpu_lock = None
    cfg.job_gpu = None
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    assert '--gpu-lock' not in svc.runner._engine_cmd(job, 'scan.h5', 'rin.json', 'rd')
    assert svc.runner._engine_env()['CUDA_VISIBLE_DEVICES'] == '1'


# --- перезапуск сервиса ---------------------------------------------------------------------------------------

def test_recovery_after_restart(tmp_path, monkeypatch):
    """Задачи running/publishing прошлого процесса: running → interrupted (run_dir удалён); publishing с результатом
    в run_dir — публикуется заново; publishing, уже опубликованная, — done."""
    cfg = make_config(tmp_path)
    mongo = mongomock.MongoClient()
    _, recipe = scan_and_recipe(cfg)
    old = jobs.JobService(cfg, mongo['autotom'], StubScans(cfg))

    def insert(job_id, status, run_id):
        old.coll.insert_one({'_id': job_id, 'exp_id': EXP, 'user': 'u', 'name': EXP, 'recipe': recipe,
                             'status': status, 'run_id': run_id, 'warnings': [], 'created': jobs._utcnow()})

    insert('run', 'running', 'r-run')
    junk = os.path.join(cfg.runs_dir(EXP), 'r-run')
    os.makedirs(junk)
    open(os.path.join(junk, 'partial.raw'), 'wb').close()
    insert('pub', 'publishing', 'r-pub')
    engine_run(cfg, EXP, recipe, 'r-pub')
    insert('pub-empty', 'publishing', 'r-gone')                     # run_dir уже нет — прервана
    insert('queued', 'queued', None)

    svc = jobs.JobService(cfg, mongo['autotom'], StubScans(cfg))
    assert svc.runner.recover() == 3
    st = {d['_id']: d['status'] for d in svc.coll.find()}
    assert st == {'run': 'interrupted', 'pub': 'done', 'pub-empty': 'interrupted', 'queued': 'queued'}
    assert not os.path.exists(junk)
    assert publish.read_json(os.path.join(cfg.reconstruction_dir(EXP), 'result.json'))['run_id'] == 'r-pub'
    statuses = tomo_statuses(svc)
    assert statuses.count('error: interrupted') == 2 and 'done' in statuses

    # опубликованная до перезапуска (result.json уже r-pub), но статус не успел смениться
    svc.set_fields('pub', {'status': 'publishing'})
    assert svc.runner.recover() == 1
    assert svc.coll.find_one({'_id': 'pub'})['status'] == 'done'

    # run_once при первом вызове восстанавливает сам
    svc.set_fields('run', {'status': 'running'})
    svc2 = jobs.JobService(cfg, mongo['autotom'], StubScans(cfg))
    monkeypatch.setattr(svc2.runner, '_execute', lambda job: None)
    assert svc2.runner.run_once() is True                            # взяла queued
    assert svc2.coll.find_one({'_id': 'run'})['status'] == 'interrupted'


# --- старая очередь ноутбуков ---------------------------------------------------------------------------------

def test_legacy_queue_step(tmp_path, monkeypatch):
    calls = []
    fake = types.SimpleNamespace(queue=[])

    def next_obj():
        return fake.queue[0] if fake.queue else None

    def process_once():
        calls.append(('process', dict(locked)))
        fake.queue.pop(0)
        return True

    fake.get_rec_queue_next_obj = next_obj
    fake.process_once = process_once
    locked = {'path': None}

    import contextlib

    @contextlib.contextmanager
    def fake_lock(path):
        locked['path'] = path
        yield
        locked['path'] = None

    monkeypatch.setattr(jobs, '_legacy_worker', lambda: fake)
    monkeypatch.setattr(jobs.gpu, 'gpu_lock', fake_lock)
    monkeypatch.setenv('RECON_JOB_GPU', '')                     # не задана; восстановится после теста
    _, client, cfg, svc = jobs_service(tmp_path, monkeypatch, legacy_queue=True, gpu_lock='/fast/.gpu0.lock',
                                       job_gpu='0')

    assert svc.runner.run_once() is False                        # пусто: блокировку GPU не берём
    assert calls == []
    assert os.environ['RECON_JOB_GPU'] == '0'                    # ноутбук — на GPU задач (tomo_worker)
    fake.queue.append({'obj_id': 'old1', 'action': 'reconstruct'})
    assert svc.runner.run_once() is False                        # пустую очередь перепроверяем не сразу
    svc.runner.legacy_poll_s = 0.0
    svc.runner._legacy_next = 0.0
    assert svc.runner.run_once() is True
    assert calls == [('process', {'path': '/fast/.gpu0.lock'})]

    # новая очередь — первой
    _, recipe = scan_and_recipe(cfg)
    post_job(client, recipe)
    fake.queue.append({'obj_id': 'old2', 'action': 'reconstruct'})
    executed = []
    monkeypatch.setattr(svc.runner, '_execute', lambda job: executed.append(job['_id']))
    assert svc.runner.run_once() is True
    assert len(executed) == 1 and len(calls) == 1
    assert svc.runner.run_once() is True and len(calls) == 2

    # ошибка шага старой очереди не роняет исполнитель
    def boom():
        raise RuntimeError('ServerSelectionTimeoutError')

    fake.get_rec_queue_next_obj = boom
    assert svc.runner.run_once() is False


def test_legacy_queue_unavailable(tmp_path, monkeypatch):
    def broken():
        raise ImportError('No module named nbformat')

    monkeypatch.setattr(jobs, '_legacy_worker', broken)
    _, _, _, svc = jobs_service(tmp_path, monkeypatch, legacy_queue=True)
    assert svc.runner.run_once() is False
    assert svc.runner.run_once() is False


def test_summary_without_mongo(tmp_path):
    class Down:
        def __getitem__(self, name):
            raise RuntimeError('mongo down')

    svc = jobs.JobService(make_config(tmp_path), Down(), None)
    s = svc.summary()
    assert 'mongo down' in s['error'] and s['runner'] is False


def test_progress_parsing_and_short_error():
    assert jobs._parse_progress('{"progress": 0.5, "stage": "recon"}') == (0.5, 'recon')
    assert jobs._parse_progress('{"progress": "x"}') is None
    assert jobs._parse_progress('INFO {"progress": 1}') is None
    tail = ['Traceback (most recent call last):', '  File "x.py", line 1', 'ValueError: ROI x [2, 70) вне кадра']
    assert jobs._short_error(tail, 1) == 'ValueError: ROI x [2, 70) вне кадра'
    assert jobs._short_error(['что-то пошло не так'], 1) == 'что-то пошло не так'
    assert jobs._short_error([], -9) == 'код выхода -9'
    assert json.loads(json.dumps(jobs.public({'_id': 'a', 'created': jobs._utcnow()})))['id'] == 'a'
