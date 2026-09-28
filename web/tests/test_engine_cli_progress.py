"""Тесты ``python -m reconengine run`` для recon-service: прогресс строками JSON, --run-id, отмена по сигналам
(SIGTERM/SIGINT, на Windows — Ctrl+Break): начатый объём удаляется, код выхода 130."""
import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import threading

import pytest

from engine_scans import simple_scan, write_h5
from jobs_helpers import slow_engine
from reconengine import cli, data, gpu
from reconengine import recipe as recipe_mod
from reconengine.model import Axis, ROI
from reconservice import jobs


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv('RECON_ENGINE_CPU', '1')
    gpu.reset_backend()
    yield
    gpu.reset_backend()


def make_run(tmp_path):
    ss = simple_scan()
    path = write_h5(ss, tmp_path / 'scan.h5', exp_id='exp-1')
    scan = data.open_scan(path)
    r = recipe_mod.default_recipe(scan.exp_id, scan.fingerprint, ROI(2, 70, 4, 36), 0.009, 'user', False)
    r.axis = Axis(ss.center_x, ss.y_ref, ss.tilt_deg)
    r.rings = {'preset': 'off', 'params': None}
    rpath = tmp_path / 'recipe.json'
    recipe_mod.save(r, rpath)
    return path, str(rpath), tmp_path / 'out'


def json_lines(text):
    return [json.loads(line) for line in text.splitlines() if line.startswith('{"progress"')]


def volume_files(out):
    if not out.exists():
        return []
    return [p.name for p in out.iterdir() if p.suffix in ('.raw', '.hx', '.size') or p.name == 'result.json']


def test_json_progress_printer_throttles(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(cli.time, 'monotonic', lambda: clock[0])
    stream = io.StringIO()
    p = cli._json_progress_printer(stream)

    def step(dt, frac, stage):
        clock[0] += dt
        p(frac, stage)

    step(0, 0.0, 'crop')          # первая
    step(0.1, 0.1, 'crop')        # чаще двух раз в секунду — пропуск
    step(0.3, 0.2, 'crop')
    step(0.2, 0.25, 'crop')       # 0,5 с с прошлой
    step(0, 0.3, 'prepare')       # смена стадии — сразу
    step(0, 0.35, 'recon')
    step(0.1, 0.5, 'recon')
    step(0.1, 1.0, 'recon')       # конец — всегда
    step(0, 1.0, 'done')
    lines = [json.loads(s) for s in stream.getvalue().splitlines()]
    assert lines == [{'progress': 0.0, 'stage': 'crop'}, {'progress': 0.25, 'stage': 'crop'},
                     {'progress': 0.3, 'stage': 'prepare'}, {'progress': 0.35, 'stage': 'recon'},
                     {'progress': 1.0, 'stage': 'recon'}, {'progress': 1.0, 'stage': 'done'}]


def test_run_progress_json_and_run_id(tmp_path, capsys):
    path, rpath, out = make_run(tmp_path)
    assert cli.main(['run', path, '--recipe', rpath, '--out', str(out), '--backend', 'cpu', '--progress-json',
                     '--run-id', 'run-42', '--slab-rows', '8']) == 0
    captured = capsys.readouterr()
    lines = json_lines(captured.err)
    stages = [x['stage'] for x in lines]
    assert stages[0] == 'crop' and 'prepare' in stages and 'recon' in stages and lines[-1] == {
        'progress': 1.0, 'stage': 'done'}
    progress = [x['progress'] for x in lines]
    assert progress == sorted(progress)
    assert '%' not in captured.err                                  # человекочитаемых строк прогресса нет
    assert json.loads(captured.out)['run_id'] == 'run-42'
    assert json.loads((out / 'result.json').read_text(encoding='utf-8'))['run_id'] == 'run-42'

    with pytest.raises(SystemExit):
        cli.main(['run', path, '--recipe', rpath, '--out', str(out), '--run-id', '../x'])


def _cancel_signals():
    names = ['SIGTERM', 'SIGINT'] + (['SIGBREAK'] if hasattr(signal, 'SIGBREAK') else [])
    return [getattr(signal, n) for n in names]


@pytest.mark.parametrize('sig', _cancel_signals(), ids=lambda s: signal.Signals(s).name)
def test_signal_cancels_run_and_removes_volume(tmp_path, monkeypatch, capsys, sig):
    """Сигнал посреди расчёта (обработчик вызывается напрямую через raise_signal — на Windows SIGTERM снаружи
    не доставить): cancel → Cancelled между слоями → объём удалён, код 130, обработчики восстановлены."""
    path, rpath, out = make_run(tmp_path)
    real = cli.pipeline.run_recipe
    fired = []

    def run_recipe(*args, progress, **kwargs):
        def hooked(frac, stage):
            progress(frac, stage)
            if stage == 'recon' and not fired:
                fired.append(stage)
                signal.raise_signal(sig)
        return real(*args, progress=hooked, **kwargs)

    monkeypatch.setattr(cli.pipeline, 'run_recipe', run_recipe)
    before = {s: signal.getsignal(s) for s in _cancel_signals()}
    rc = cli.main(['run', path, '--recipe', rpath, '--out', str(out), '--backend', 'cpu', '--progress-json',
                   '--slab-rows', '4'])
    assert rc == 130 and fired
    assert volume_files(out) == []
    assert 'отменено' in capsys.readouterr().err
    assert {s: signal.getsignal(s) for s in _cancel_signals()} == before


def test_signal_while_waiting_for_gpu_lock(tmp_path, monkeypatch, capsys):
    """Пока ждём блокировку GPU (flock после сигнала перезапускается), обработчик прерывает ожидание сам."""
    path, rpath, out = make_run(tmp_path)
    called = []

    @contextlib.contextmanager
    def busy_lock(lock_path):
        signal.raise_signal(signal.SIGTERM)                          # «пришёл SIGTERM, пока ждём flock»
        called.append('after-signal')
        yield

    monkeypatch.setattr(cli.gpu, 'lock_busy', lambda p: True)
    monkeypatch.setattr(cli.gpu, 'gpu_lock', busy_lock)
    monkeypatch.setattr(cli.pipeline, 'run_recipe', lambda *a, **k: called.append('run'))
    rc = cli.main(['run', path, '--recipe', rpath, '--out', str(out), '--progress-json',
                   '--gpu-lock', str(tmp_path / 'gpu.lock')])
    assert rc == 130 and called == []
    assert json_lines(capsys.readouterr().err)[0] == {'progress': 0.0, 'stage': 'wait_gpu'}


def test_handlers_not_installed_outside_main_thread(tmp_path):
    cancel = threading.Event()
    before = signal.getsignal(signal.SIGTERM)
    seen = []

    def worker():
        with cli._cancel_on_signals(cancel, {'on': False}):
            seen.append(signal.getsignal(signal.SIGTERM))

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen == [before]


def _engine_process(tmp_path, monkeypatch, extra=()):
    path, rpath, out = make_run(tmp_path)
    slow_engine(tmp_path, monkeypatch, 1.0)
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([env['PYTHONPATH'], jobs.ENGINE_ROOT])
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUNBUFFERED'] = '1'
    cmd = [sys.executable, '-m', 'reconengine', 'run', path, '--recipe', rpath, '--out', str(out),
           '--cache', str(tmp_path / 'cache'), '--progress-json'] + list(extra)
    proc = subprocess.Popen(cmd, cwd=jobs.ENGINE_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            **jobs._popen_kwargs())
    return proc, out


def _wait_stage(proc, stage):
    seen = []
    for raw in iter(proc.stderr.readline, b''):
        line = raw.decode('utf-8', 'replace')
        seen.append(line)
        if '"stage": "{}"'.format(stage) in line:
            return seen
    raise AssertionError('процесс завершился раньше стадии {}: {}'.format(stage, ''.join(seen)))


def test_engine_process_cancelled_by_signal(tmp_path, monkeypatch):
    """Настоящий процесс ``python -m reconengine run``: сигнал отмены посреди расчёта (Linux — SIGTERM, как шлёт
    recon-service; Windows — Ctrl+Break группе процесса) → код 130, начатый объём удалён."""
    proc, out = _engine_process(tmp_path, monkeypatch, ['--slab-rows', '8'])     # 4 слоя по ≥1 с
    try:
        _wait_stage(proc, 'recon')                                  # первый слой записан
        if os.name == 'nt':
            try:
                os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
            except OSError as exc:
                pytest.skip('Ctrl+Break не доставить без консоли: {}'.format(exc))
        else:
            proc.send_signal(signal.SIGTERM)
        rest = proc.stderr.read().decode('utf-8', 'replace')
        assert proc.wait(30) == 130, rest
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()
        proc.stderr.close()
    assert 'отменено' in rest
    assert volume_files(out) == []


@pytest.mark.skipif(os.name == 'nt', reason='flock — только Linux')
def test_engine_process_cancelled_while_waiting_for_gpu_lock(tmp_path, monkeypatch):
    import fcntl
    lock = tmp_path / 'gpu.lock'
    with open(lock, 'a+') as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)                     # «Jupyter держит GPU»
        proc, out = _engine_process(tmp_path, monkeypatch, ['--gpu-lock', str(lock)])
        try:
            _wait_stage(proc, 'wait_gpu')
            proc.send_signal(signal.SIGTERM)
            assert proc.wait(15) == 130
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stdout.close()
            proc.stderr.close()
    assert volume_files(out) == []
