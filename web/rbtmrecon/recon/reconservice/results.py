"""Результат реконструкции для шага «Результат» (``/results/<exp_id>/...``).

| Метод и путь                          | Ответ |
|---------------------------------------|-------|
| GET /results/<id>                     | JSON: опубликованный ``result.json``, история (``history/*/result.json``: run_id, created, recipe_sha256), список доступных файлов; 404 — результата движка нет |
| GET /results/<id>/slice?axis=z|y|x&i=&max_px= | binary uint16: срез копии с наибольшим биннингом из ``result.json['binned']`` (memmap, без чтения объёма целиком); окно квантования — ``stats.p0_1``/``p99_9`` результата; X-Meta: axis, i, n, binning, voxel_mm |
| GET /results/<id>/file/<name>         | файл потоком (as_attachment): только ``recipe.json``, ``result.json`` и файлы копий с биннингом (raw, .size, .hx) из ``result.json``; полный объём через сервис не отдаётся (он доступен как раньше, через ``/reconstruct/static``) |
"""
from __future__ import annotations

from flask import Blueprint

bp = Blueprint('results', __name__, url_prefix='/results')
