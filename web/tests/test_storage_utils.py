"""Тесты вспомогательных функций веб-интерфейса."""
import ast
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


def _extract_sort_key_lambda(src):
    """Находит лямбду ``key=`` в первом вызове ``.sort(...)`` исходника и
    компилирует её в вызываемый объект — вместо хрупкого поиска подстроки
    в тексте, который ломается при любом косметическом изменении форматирования."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'sort'):
            for kw in node.keywords:
                if kw.arg == 'key' and isinstance(kw.value, ast.Lambda):
                    expr = ast.Expression(body=kw.value)
                    ast.fix_missing_locations(expr)
                    return eval(compile(expr, '<web_recon.sort key>', 'eval'))
    raise AssertionError('ключ .sort(key=lambda ...) не найден в web_recon.py')


def test_web_recon_timestamp_sort_key_handles_missing_and_string_values():
    """Ключ сортировки списка объектов должен переживать объекты без
    timestamp, с timestamp=None и с timestamp-строкой (так отдаёт storage) —
    иначе вся страница со списком падает с KeyError/TypeError при сравнении
    разнотипных ключей."""
    web_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(web_dir, 'rbtmwebrecon', 'webrecon', 'web_recon.py'),
               encoding='utf8').read()
    key_fn = _extract_sort_key_lambda(src)

    objs = [{'timestamp': 100}, {'timestamp': '50'}, {'timestamp': None}, {}]
    objs.sort(key=key_fn, reverse=True)

    keys = [key_fn(o) for o in objs]
    assert keys == sorted(keys, reverse=True)
    assert keys[0] == 100.0
    assert keys[-1] == 0.0
