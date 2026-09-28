"""Перенос результата запуска в хранилище.

``publish(cfg, exp_id, run_dir)`` — ``run_dir`` на SSD (``cfg.runs_dir(exp)/<run_id>``) содержит то, что написал
``pipeline.run_recipe``: raw/.size/.hx полного объёма и копий с биннингом, ``recipe.json``, ``result.json``.
Назначение — ``cfg.reconstruction_dir(exp)`` (``/storage/<id>/reconstruction``, туда же копирует ноутбук):

1. если там уже есть ``result.json`` прошлого запуска движка — его ``recipe.json``/``result.json`` переносятся в
   ``history/<run_id прошлого>/``, а файлы объёма, перечисленные в прошлом ``result.json``, удаляются
   (пересчёт заменяет объём; история хранит только рецепты и метаданные);
2. файлы, которые движок не создавал (результаты ноутбука, отчёты), не удаляются никогда; совпадающие по имени
   перезаписываются новым объёмом;
3. файлы объёма переносятся (``shutil.move``: между дисками — копия и удаление), ``recipe.json`` — затем,
   ``result.json`` — последним и атомарно: читатель видит либо прошлый результат целиком, либо новый;
4. пустой ``run_dir`` удаляется.

``archive_h5(cfg, exp_id)`` — архивная копия ``<storage_dir>/<id>.h5`` из ``<exp_src>/<id>.h5``, если её ещё нет
(как ``mv /fast/<id>.h5 /storage/`` в ноутбуке): копия во временное имя в том же каталоге и ``os.replace``.
"""
from __future__ import annotations

from typing import Any, Dict

from .config import Config


def publish(cfg: Config, exp_id: str, run_dir: str) -> Dict[str, Any]:
    """Перенести результат; вернуть опубликованный result.json (dict)."""
    raise NotImplementedError


def archive_h5(cfg: Config, exp_id: str) -> bool:
    """Скопировать исходный HDF5 в архив, если копии нет. True — скопирован сейчас."""
    raise NotImplementedError
