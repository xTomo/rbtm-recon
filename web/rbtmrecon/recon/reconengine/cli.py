"""Командная строка движка: предложить рецепт и выполнить его.

    python -m reconengine suggest <scan> [--out recipe.json] [--pixel-size-mm X]
    python -m reconengine migrate <scan> --ini rec_config.ini [--out recipe.json]
    python -m reconengine run <scan> --recipe recipe.json --out DIR [--cache DIR] [--name ИМЯ] [--slices Z0 Z1]
    python -m reconengine compare <ноутбук .1.raw> <каталог результата> [--step 16]

``<scan>`` — путь к файлу HDF5 v2 или id эксперимента (тогда файл ищется как ``<src-dir>/<id>.h5``,
по умолчанию ``$RECON_EXP_SRC`` или ``/exp_src``). Прогресс пишется в stderr, итог — JSON в stdout.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from typing import List, Optional

from . import data, gpu, pipeline, pixelsize, preprocess
from . import recipe as recipe_mod
from .model import Cancelled

logger = logging.getLogger(__name__)


def resolve_scan_path(scan: str, src_dir: Optional[str]) -> str:
    if os.path.isfile(scan):
        return scan
    base = src_dir or os.environ.get('RECON_EXP_SRC', '/exp_src')
    path = os.path.join(base, scan + '.h5')
    if not os.path.isfile(path):
        raise FileNotFoundError('нет файла скана: {} (и {})'.format(scan, path))
    return path


def _progress_printer(stream=sys.stderr, min_interval: float = 1.0):
    state = {'t': 0.0, 'stage': None}

    def progress(frac: float, stage: str) -> None:
        now = time.time()
        if stage != state['stage'] or now - state['t'] >= min_interval or frac >= 1.0:
            state['t'], state['stage'] = now, stage
            stream.write('{:5.1f}% {}\n'.format(100.0 * frac, stage))
            stream.flush()
    return progress


def suggest(scan_path: str, exp_id: Optional[str] = None, pixel_size_mm: Optional[float] = None,
            n: int = 16, bin: int = 4, mongo_doc: Optional[dict] = None) -> recipe_mod.Recipe:
    """Рецепт по умолчанию: ROI по огибающей обзора, размер пикселя с источником, ось — авто при запуске."""
    scan = data.open_scan(scan_path, exp_id)
    ov = data.sample_overview(scan, n=n, bin=bin)
    roi = preprocess.suggest_roi(preprocess.envelope(ov), bin, scan.height, scan.width)
    ps = pixelsize.resolve(mongo_doc, scan.metadata, pixel_size_mm)
    for w in ps.warnings:
        logger.warning('%s', w)
    return recipe_mod.default_recipe(scan.exp_id, scan.fingerprint, roi, ps.value_mm, ps.source, scan.is_advanced)


def _cmd_suggest(args) -> int:
    path = resolve_scan_path(args.scan, args.src_dir)
    r = suggest(path, args.exp, args.pixel_size_mm, n=args.n, bin=args.bin)
    if args.out:
        recipe_mod.save(r, args.out)
    json.dump(recipe_mod.to_dict(r), sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write('\n')
    return 0


def migrate(scan_path: str, ini_path: str, exp_id: Optional[str] = None,
            pixel_size_mm: Optional[float] = None) -> recipe_mod.Recipe:
    """Рецепт из ``rec_config.ini`` ноутбука (ROI и ось ноутбука) с размером пикселя по метаданным скана."""
    scan = data.open_scan(scan_path, exp_id)
    r = recipe_mod.from_rec_config_ini(ini_path, scan.exp_id, scan.fingerprint, scan.height, scan.width)
    ps = pixelsize.resolve(None, scan.metadata, pixel_size_mm)
    for w in ps.warnings:
        logger.warning('%s', w)
    r.pixel_size = {'value_mm': ps.value_mm, 'source': ps.source, 'user_edited': ps.source == 'user'}
    r.repositioning['enabled'] = bool(scan.is_advanced)
    return r


def _cmd_migrate(args) -> int:
    path = resolve_scan_path(args.scan, args.src_dir)
    r = migrate(path, args.ini, args.exp, args.pixel_size_mm)
    if args.out:
        recipe_mod.save(r, args.out)
    json.dump(recipe_mod.to_dict(r), sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write('\n')
    return 0


def _cmd_run(args) -> int:
    path = resolve_scan_path(args.scan, args.src_dir)
    r = recipe_mod.load(args.recipe)
    if args.slices:
        r.recon['slices'] = [int(args.slices[0]), int(args.slices[1])]
    cache = args.cache or os.path.join(args.out, '.cache')
    cancel = threading.Event()
    with gpu.gpu_lock(args.gpu_lock):
        try:
            res = pipeline.run_recipe(r, path, args.out, cache, name=args.name, progress=_progress_printer(),
                                      cancel=cancel, backend=args.backend, slab_rows=args.slab_rows,
                                      workers=args.workers)
        except Cancelled:
            sys.stderr.write('отменено\n')
            return 130
    doc = res.result
    summary = {k: doc[k] for k in ('run_id', 'volume', 'binned', 'timings', 'warnings')}
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write('\n')
    return 0


def _cmd_compare(args) -> int:
    from . import compare  # noqa: WPS433
    rows = compare.compare(args.old_raw, args.result, args.step)
    if not rows:
        sys.stderr.write('нет общих срезов\n')
        return 1
    for z, c, ratio in rows:
        sys.stdout.write('строка {:5d}: корреляция {:.5f}, среднее движок/ноутбук {:.4f}\n'.format(z, c, ratio))
    sys.stdout.write('минимум корреляции {:.5f} по {} срезам\n'.format(min(c for _, c, _ in rows), len(rows)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='python -m reconengine', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-v', '--verbose', action='store_true', help='подробный лог')
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(sp):
        sp.add_argument('scan', help='файл HDF5 v2 или id эксперимента')
        sp.add_argument('--src-dir', help='каталог файлов экспериментов (по умолчанию $RECON_EXP_SRC или /exp_src)')
        sp.add_argument('--exp', help='id эксперимента (по умолчанию из metadata или имени файла)')

    sp = sub.add_parser('suggest', help='предложить рецепт по обзору скана')
    common(sp)
    sp.add_argument('--out', help='куда сохранить recipe.json')
    sp.add_argument('--pixel-size-mm', type=float, help='размер пикселя, мм (иначе из метаданных)')
    sp.add_argument('--n', type=int, default=16, help='число углов обзора')
    sp.add_argument('--bin', type=int, default=4, help='биннинг обзора')
    sp.set_defaults(func=_cmd_suggest)

    sp = sub.add_parser('migrate', help='рецепт из rec_config.ini ноутбука (ROI и ось)')
    common(sp)
    sp.add_argument('--ini', required=True, help='rec_config.ini')
    sp.add_argument('--out', help='куда сохранить recipe.json')
    sp.add_argument('--pixel-size-mm', type=float, help='размер пикселя, мм (иначе из метаданных)')
    sp.set_defaults(func=_cmd_migrate)

    sp = sub.add_parser('run', help='реконструкция по рецепту')
    common(sp)
    sp.add_argument('--recipe', required=True, help='recipe.json')
    sp.add_argument('--out', required=True, help='каталог результата')
    sp.add_argument('--cache', help='каталог кэша кропа (по умолчанию <out>/.cache)')
    sp.add_argument('--name', help='имя образца для файлов Amira (по умолчанию id эксперимента)')
    sp.add_argument('--backend', choices=('auto', 'astra', 'cpu'), default='auto', help='FBP')
    sp.add_argument('--slices', type=int, nargs=2, metavar=('Z0', 'Z1'),
                    help='строки детектора [Z0, Z1) вместо recon.slices рецепта (для пробного запуска)')
    sp.add_argument('--slab-rows', type=int, help='строк в слое (по умолчанию по памяти GPU)')
    sp.add_argument('--workers', type=int, default=data.DEFAULT_WORKERS, help='потоков распаковки')
    sp.add_argument('--gpu-lock', help='файл блокировки GPU (flock), общий с Jupyter')
    sp.set_defaults(func=_cmd_run)

    sp = sub.add_parser('compare', help='сравнить объём движка с объёмом ноутбука по общим срезам')
    sp.add_argument('old_raw', help='raw ноутбука полного разрешения (<имя>.<nz>_<ny>_<nx>.1.raw)')
    sp.add_argument('result', help='каталог результата движка или его result.json')
    sp.add_argument('--step', type=int, default=16, help='шаг по срезам')
    sp.set_defaults(func=_cmd_compare)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, stream=sys.stderr,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    return int(args.func(args) or 0)
