"""Помощники тестов очереди задач, публикации и результатов recon-service.

Реестр сканов и интерактивная сессия здесь — заглушки (очереди от реестра нужны только ``path()`` и ``info()``),
чтобы тесты задач не зависели от модулей scans/sessions.

Замедление движка в дочернем процессе (для отмены и остановки посреди расчёта) — через ``sitecustomize.py`` в
PYTHONPATH процесса: при ``RECON_TEST_SLOW_FBP=<секунды>`` каждый вызов ``fbp.recon_rows`` сначала спит.
"""
import os
import textwrap
import time

from engine_scans import simple_scan
from reconengine import data, outputs, pipeline
from reconengine import recipe as recipe_mod
from reconengine.model import Axis, ROI
from service_helpers import make_service, write_scan

EXP = 'exp1'
ROI_BOX = ROI(2, 70, 4, 36)          # объём 32×68×68, копия ×4 — 8×17×17


class StubScans:
    """ScanRegistry в объёме, нужном очереди задач."""

    def __init__(self, cfg):
        self.cfg = cfg

    def path(self, exp_id):
        p = self.cfg.scan_path(exp_id)
        if not os.path.isfile(p):
            raise FileNotFoundError('нет скана {}'.format(exp_id))
        return p

    def info(self, exp_id):
        return data.open_scan(self.path(exp_id), exp_id)


class StubSessions:
    def __init__(self, cfg, scans):
        pass

    def summary(self):
        return None

    def start_reaper(self):
        pass

    def stop_reaper(self):
        pass


def jobs_service(tmp_path, monkeypatch, **overrides):
    """make_service с заглушками реестра сканов и сессии: (app, client, cfg, JobService)."""
    from reconservice import scans, sessions
    monkeypatch.setattr(scans, 'ScanRegistry', StubScans)
    monkeypatch.setattr(sessions, 'SessionManager', StubSessions)
    app, client, cfg = make_service(tmp_path, **overrides)
    return app, client, cfg, app.extensions['recon'].jobs


def scan_and_recipe(cfg, exp_id=EXP, ss=None, roi=ROI_BOX, slices=None):
    """Записать синтетический скан и вернуть (скан, рецепт-dict): ось известна, кольца выключены (быстро и без cupy)."""
    ss = ss or simple_scan()
    path = write_scan(cfg, exp_id, ss)
    scan = data.open_scan(path, exp_id)
    r = recipe_mod.default_recipe(exp_id, scan.fingerprint, roi, 0.009, 'user', scan.is_advanced)
    r.axis = Axis(ss.center_x, ss.y_ref, ss.tilt_deg)
    r.rings = {'preset': 'off', 'params': None}
    if slices:
        r.recon['slices'] = list(slices)
    return ss, recipe_mod.to_dict(r)


def engine_run(cfg, exp_id, recipe, run_id, name=None):
    """То, что сделал бы процесс задачи, — в этом процессе: run_dir = runs_dir/<run_id> с result.json этого run_id."""
    r = recipe_mod.from_dict(recipe)
    run_dir = os.path.join(cfg.runs_dir(exp_id), run_id)
    res = pipeline.run_recipe(r, cfg.scan_path(exp_id), run_dir, cfg.cache_dir(exp_id), name=name, backend='cpu')
    res.result['run_id'] = run_id
    outputs.write_json(os.path.join(run_dir, 'result.json'), res.result)
    return run_dir, res.result


def slow_engine(tmp_path, monkeypatch, seconds):
    """Замедлить FBP в дочерних процессах движка (sitecustomize в PYTHONPATH)."""
    hook = tmp_path / 'slowhook'
    hook.mkdir(exist_ok=True)
    (hook / 'sitecustomize.py').write_text(textwrap.dedent('''\
        import os
        import time

        _delay = float(os.environ.get('RECON_TEST_SLOW_FBP') or 0)
        if _delay > 0:
            from reconengine import fbp

            _orig = fbp.recon_rows

            def _slow(*args, **kwargs):
                time.sleep(_delay)
                return _orig(*args, **kwargs)

            fbp.recon_rows = _slow
        '''), encoding='utf-8')
    monkeypatch.setenv('PYTHONPATH', str(hook))
    monkeypatch.setenv('RECON_TEST_SLOW_FBP', str(seconds))


def wait_for(pred, timeout=30.0, step=0.05, what='условие'):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = pred()
        if value:
            return value
        time.sleep(step)
    raise AssertionError('не дождались: {}'.format(what))


def tomo_statuses(jobs_svc, exp_id=EXP):
    return [d['status'] for d in jobs_svc.tomo.find({'obj_id': exp_id}).sort('_id', 1)]
