"""Статические проверки ноутбука reconstructor4.py (без выполнения)."""
import ast
import io
import os

import tomotools4 as t4

NOTEBOOK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'rbtmrecon', 'recon', 'reconstructor4.py')


def _source():
    return io.open(NOTEBOOK, encoding='utf8').read()


def test_notebook_parses():
    ast.parse(_source())


def test_notebook_imports_only_existing_tomotools4_names():
    tree = ast.parse(_source())
    missing = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == 'tomotools4':
            for alias in node.names:
                if not hasattr(t4, alias.name):
                    missing.append(alias.name)
    assert missing == [], f'нет в tomotools4: {missing}'


def test_measure_repositioning_shifts_runs_without_debug():
    assert 'debug=False' in _source()
    assert 'debug=True)' not in _source()


def test_manual_axis_search_disabled_by_default():
    tree = ast.parse(_source())
    values = [node.value.value
              for node in ast.walk(tree)
              if isinstance(node, ast.Assign)
              and isinstance(node.value, ast.Constant)
              and any(isinstance(t, ast.Name) and t.id == 'manual_axis_search'
                      for t in node.targets)]
    assert values == [False], values
