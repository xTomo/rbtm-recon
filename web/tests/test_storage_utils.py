"""Тесты вспомогательных функций веб-интерфейса."""
import os

import storage_utils


def _make_exp_dir(tmp_path, monkeypatch, experiment_id, files):
    """Создаёт static/tomo_data/<id> с указанными файлами."""
    app_root = tmp_path / 'webrecon'
    exp_dir = app_root / 'static' / 'tomo_data' / experiment_id
    exp_dir.mkdir(parents=True)
    for name in files:
        (exp_dir / name).write_text('x', encoding='utf8')
    monkeypatch.setattr(storage_utils.os.path, 'abspath',
                        lambda p: str(app_root / 'storage_utils.py'))
    return app_root


def test_reconstructed_files_list_finds_reconstructor4_report(tmp_path, monkeypatch):
    _make_exp_dir(tmp_path, monkeypatch, 'exp1',
                  ['reconstructor4.html', 'tomo_rec.h5'])
    res = storage_utils.get_reconstructed_files_list('exp1', True)
    assert 'tomo_reports' in res
    assert any('reconstructor4.html' in r for r in res['tomo_reports'])
    assert 'tomo_rec' in res


def test_reconstructed_files_list_finds_legacy_report(tmp_path, monkeypatch):
    _make_exp_dir(tmp_path, monkeypatch, 'exp2', ['reconstructor-v3.html'])
    res = storage_utils.get_reconstructed_files_list('exp2', True)
    assert any('reconstructor-v3.html' in r for r in res['tomo_reports'])


def test_reconstructed_files_list_missing_dir(tmp_path, monkeypatch):
    app_root = tmp_path / 'webrecon'
    app_root.mkdir()
    monkeypatch.setattr(storage_utils.os.path, 'abspath',
                        lambda p: str(app_root / 'storage_utils.py'))
    assert storage_utils.get_reconstructed_files_list('nope', True) == {}


def test_get_tomoobjects_list_survives_storage_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError('storage down')

    monkeypatch.setattr(storage_utils.requests, 'post', boom)
    assert storage_utils.get_tomoobjects_list() == []


def test_web_recon_sorts_with_get_timestamp():
    """web_recon сортирует по x.get('timestamp', 0), а не по x['timestamp']."""
    web_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(web_dir, 'rbtmwebrecon', 'webrecon', 'web_recon.py'),
               encoding='utf8').read()
    assert "x.get('timestamp', 0)" in src
    assert "key=lambda x: x['timestamp']" not in src
