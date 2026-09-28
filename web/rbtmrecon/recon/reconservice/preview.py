"""Вычисления превью по кропу сессии: срез, авто-ось, перебор центра, вид 0°−180°, кольца, сдвиги образца.

Строится на тех же функциях движка, что и ``pipeline.run_recipe`` (``preprocess.normalize_slab``,
``pipeline._apply_shifts``, ``axis.align_rows``, ``rings.apply``, ``fbp.recon_rows``), поэтому превью совпадает со
срезом итоговой реконструкции при тех же параметрах (проверяется тестом). Кэш полосы строк — см. ``sessions``.
"""
from __future__ import annotations
