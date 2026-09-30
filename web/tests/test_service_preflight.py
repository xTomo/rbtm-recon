"""reconservice.preflight: проверка окружения после выкатки (без GPU, Mongo и сети — подменены)."""
import os

from reconservice import preflight


def test_preflight_reports_and_exit_code(tmp_path, monkeypatch, capsys):
    for d in ('exp_src', 'fast', 'storage'):
        (tmp_path / d).mkdir()
    env = {'RECON_TOKEN': 't' * 43, 'RECON_EXP_SRC': str(tmp_path / 'exp_src'), 'RECON_FAST': str(tmp_path / 'fast'),
           'RECON_STORAGE': str(tmp_path / 'storage'), 'RECON_GPU_LOCK': str(tmp_path / 'fast' / '.gpu0.lock'),
           'CUPY_CACHE_DIR': str(tmp_path / 'fast' / '.cupy')}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(preflight, 'check_versions', lambda: preflight._report('OK', 'библиотеки'))
    monkeypatch.setattr(preflight, 'check_mongo', lambda cfg: preflight._report('OK', 'Mongo'))
    monkeypatch.setattr(preflight, 'check_storage', lambda cfg: preflight._report('WARN', 'storage'))
    monkeypatch.setattr(preflight, 'check_service', lambda cfg: None)
    assert preflight.main(['--no-gpu']) == 0
    out = capsys.readouterr().out
    assert 'OK    RECON_TOKEN — задан (43 символов)' in out and 'итог: 0 FAIL, 1 WARN' in out
    assert not [p for p in os.listdir(tmp_path / 'storage') if p.startswith('.preflight')]   # проба записи убрана

    monkeypatch.setenv('RECON_TOKEN', '')
    assert preflight.main(['--no-gpu', '--exp', 'nope']) == 1                  # пустой токен и нет скана
    out = capsys.readouterr().out
    assert 'FAIL  RECON_TOKEN' in out and 'FAIL  скан nope' in out


def test_long_details_are_truncated():
    preflight._results.clear()
    preflight._report('FAIL', 'x', 'a\n' * 400)
    assert len(preflight._results[-1][2]) <= 240
