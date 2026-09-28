# Ревью rbtm-recon (сентябрь 2026)

Ветка: `review/recon-cleanup`. База: `develop` @ `0004b56`. Код в `web/`.
Общий документ по стыкам модулей: `xtomo/plans/REVIEW-2026-09-cross-module.md` (C1, C2, C7, C9, C11, C12).
Заменяет `xtomo/plans/rbtm-recon-web-review.md` (май 2026): из 25 его пунктов закрыт только #14, #8 — наполовину (§3.5).

Что смотрели: `rbtmrecon/recon` (`tomo_worker.py`, `tomo_queue.py`, `hdf5_v2.py` reader, `tomotools4.py`, `tomotools2.py`,
`reconstructor4.py`, `reconstructor-axis_search3b.py`, `hdf2vtk.py`), `rbtmwebrecon/webrecon` (`web_recon.py`, `tomo_queue.py`,
`storage_utils.py`, шаблоны), `scripts/convert_v1_to_v2.py`, Docker, README. Ссылки `file:line` — по состоянию базы.

---

## 1. Главное

| # | Проблема | Где | Риск |
|---|---|---|---|
| 1 | **Автореконструкция не работает для новых экспериментов**: воркер запускает legacy-ноутбук (`tomotools2`, только HDF5 v1), storage пишет только v2; ошибки ячеек глушатся `--allow-errors`, статус всегда `done`. `reconstructor4`/`tomotools4`/`hdf5_v2` копируются в каталог задания, но не запускаются; README описывает их как активные. | `tomo_worker.py:16,30,58` | blocker |
| 2 | **Коррекция репозиционирования**: k-му сегменту применяется `shifts[k-1]` без накопления (сдвиг измерен относительно предыдущего сегмента → нужен `cumsum`); знак сдвига `phase_cross_correlation(ref=data, moving=dc)` применён наоборот; README закрепляет ненакопительную схему и «коррекцию до нормировки», код — после. | `tomotools4.py:799-802, 846-860`, `README.md:99, 344-357` | high |
| 3 | **Интерполяция empty**: ветка `idx==0` всегда возвращает `initial_empty` без интерполяции — дрейф между начальной и первой periodic-серией не компенсируется, на первом checkpoint скачок нормировки. Сегменты восстанавливаются нарезкой empty по `series_length`, а не по `segment_ids` — потеря кадра молча сдвигает все checkpoint-ы. | `tomotools4.py:537-582, 330-344`, `hdf5_v2.py:277-291` | high |
| 4 | **Оси и углы**: пары 0°/180° ищутся точным `==` по float32 → при развёртке < 180° или неточных углах `IndexError`; `initial_shift` — Y-компонента центра масс вместо X (и в `tomotools2`); `_find_matching_data_frame` без допуска по углу берёт первое вхождение при повторных оборотах. | `tomotools4.py:1358-1371, 1075-1077, 623-639` | high |
| 5 | **Reader требует `mapping/*`** (есть только после успешного finish) → прерванный эксперимент не читается; пустые группы дают `shape (0,)` и ломают медиану; `is_hdf5_v2` открывает файл повторно внутри открытого. | `hdf5_v2.py:141-142, 148, 261-262, 304-310` | high |
| 6 | **Воркер/очередь**: главный цикл без `try/except` — недоступная Mongo на старте убивает процесс; `ValueError` на неизвестном `action`; `get_rec_queue_next_obj` без сортировки и атомарного захвата; статусы `copying/reconstructing` без TTL; два разошедшихся `tomo_queue.py`; `copy_python_files` зависит от cwd; `tomo.ini` через ConfigParser с интерполяцией (`%` в specimen). Незавершённые эксперименты ставятся в очередь (фильтр `finished` закомментирован). | `tomo_worker.py:79-120`, `tomo_queue.py` ×2, `storage_utils.py:95` | high |
| 7 | **Docker**: все `restart:` закомментированы → после ребута стек не поднимается; `./data/db` относительный; `depends_on` без healthcheck; нет `.dockerignore` (в образ попадает `.git` сабмодуля `tomo/`); Mongo наружу на 27017; SHA1-хэш пароля Jupyter и IP в Dockerfile; ubuntu18.04. | `docker-compose.yml`, `rbtmrecon/Dockerfile:1,57`, `rbtmwebrecon/Dockerfile:32` | high (ребут), security |
| 8 | `get_experiment_hdf5`: ветка HTTP-скачивания мёртвая (всегда `/exp_src`), целостность не проверяется, недокачанный файл принимается; `save_amira` теряет пробелы в имени raw; `show_frames_with_border` мутирует входной массив; `safe_median` на лог-кадре сглаживает весь фон; отчёт `reconstructor*.html` не находится (`glob 'reconstructor-v*.html'`). | `tomotools4.py:54-105, 1397-1423, 1187-1196, 504-514`, `storage_utils.py:33` | med |
| 9 | `convert_v1_to_v2.py` не запускается (импорт несуществующего `compute_total_frames`); `H, W = shape[1]`; ключи v1 без zero-pad; своя третья семантика `segment_ids`/checkpoint. | `convert_v1_to_v2.py:23, 147-149, 180, 214, 231, 272-290` | med |
| 10 | Web-часть: мутирующие GET без auth (`/reconstruct|copyfiles|reset|cancel_all`), path traversal в `storage_utils.py:14,47`, `timeout=1000` с (5 мест), `is_local_ip` только `10.*`, `sort(key=x['timestamp'])` KeyError, `get_tomoobjects_list` без try/except, `hdf2vtk` на удалённом PyVista API. | `web_recon.py`, `storage_utils.py`, `hdf2vtk.py:9,13` | med/security |

---

## 2. Архитектура

`web` (Flask + gunicorn -w 8, :5550) и `reconstructor` (воркер `tomo_worker.py`) общаются через Mongo `autotom.tomoobjects`
(insert-only лог статусов: `waiting → copying/reconstructing → done | error: … | canceled`, «текущий» = последний по `_id`).
Воркер копирует скрипты в `/storage/<id>/`, jupytext → nbconvert `--execute` → HTML. HDF5 берётся из `/exp_src` (bind-mount storage, ro).
GPU только в контейнере (cupy/astra) — локально тесты идут через заглушки `cupy`/`astra` в `web/tests/conftest.py`.

---

## 3. По компонентам

### 3.1 `hdf5_v2.py` (reader) — C1/C2
Читает `metadata/*`, `timeline/{modes,angles,frame_numbers}`, `images/all` батчами по 5 через `mapping/*_indices`; `segment_ids`, `checkpoint_*`
не использует (`get_checkpoint_mapping_v2` мёртв). Две ветки `if is_advanced` идентичны. Round-trip с writer storage (`xtomo/scripts/check_hdf5_contract.py`):
имена/dtype совпадают, `periodic_empty_fnumbers`, `data_numbers`, `series_length` — корректны; падает без `mapping`.

### 3.2 `tomotools4.py`
Нормировка `log(empty)−log(data)` с клипом ≥1 до логарифма — NaN невозможен; `clip(≥0)` даёт положительное смещение шума.
`_interpolate_empty` по `frame_number` первого кадра серии; после последней periodic — константа (финальной empty-серии драйвер не снимает).
`measure_repositioning_shifts` матчит по углу, берёт первый data_check окна (при `data_count_per_step>1` остальные игнорируются).
`recon_2d_parallel`: `CGLS_CUDA` выключен (в `tomotools2` включён) — расхождение не задокументировано. v1-advanced даёт медиану медиан серий, v2 — медиану всех empty.

### 3.3 `reconstructor4.py` под nbconvert
`interact_manual` безопасен (тело не выполняется, следующая ячейка читает начальные значения виджета — как в legacy);
`create_axis_search_widget` + ручной re-apply повторяют авто-результат лишним проходом; `measure_repositioning_shifts(debug=True)` — 3 фигуры на checkpoint.
Пути `/fast`, `/storage`, `/exp_src` и `STORAGE_SERVER` совпадают с compose. HTML-имя `reconstructor4.html`.

### 3.4 Воркер и очередь
См. §1 п.6. `copyfiles` копирует скрипты + `tomo.ini`; проверка `'action' not in rec_obj` недостижима.

### 3.5 Статус пунктов прежнего ревью (`plans/rbtm-recon-web-review.md`)

| # | Пункт | Статус до ветки | В ветке |
|---|---|---|---|
| 1 | хэш Jupyter в Dockerfile | open | документируем |
| 2 | мутации через GET без auth | open | документируем |
| 3 | Mongo 27017 наружу | open | документируем |
| 4 | IP Jupyter в Dockerfile/conf.py | open | документируем |
| 5 | path traversal | open | документируем |
| 6 | bare `except:` | open | fix |
| 7 | `ValueError` в цикле воркера | open | fix |
| 8 | сортировка `date` vs `_id` | partial (recon fixed) | fix (web) |
| 9 | мёртвая `objs is None` | open | fix |
| 10 | `__import__('datetime')` | open | fix |
| 11 | `hdf2vtk` PyVista API | open | fix |
| 12 | `NotebookApp` флаг | open | fix |
| 13 | `get_tomoobjects_list` без try | open | fix |
| 14 | `get_frame_group` первый атрибут | **fixed** | — |
| 15 | `timeout=1000` (теперь 5 мест) | open | документируем |
| 16 | `is_local_ip` | open | документируем |
| 17 | `app.run(debug=True, host=IP)` | open | документируем |
| 18 | `MongoClient` на уровне модуля | open | документируем |
| 19 | `/storage` захардкожен | open | документируем |
| 20 | дубли compose-сервисов | open | документируем |
| 21 | ubuntu18.04 | open | документируем |
| 22 | `restart` закомментирован | open | fix |
| 23 | нет `.dockerignore` | open | fix |
| 24 | bootstrap в репо | open | документируем |
| 25 | `./data/db` относительный | open | документируем (решение) |

---

## 4. План работ (ветка `review/recon-cleanup`)

| Коммит | Содержание | Проверка |
|---|---|---|
| воркер | `reconstructor4`, статус `error:` по ошибкам ячеек ipynb, `ServerApp`, unknown action → лог, устойчивый цикл (Mongo), пути от `__file__`, `interpolation=None` | тест сканера ошибок, `process_once` |
| очередь | единый `tomo_queue.py` (сортировка `_id`, дедуп), `except Exception`, мёртвые проверки | mongomock |
| reader | fallback индексов из `timeline/modes`, пустые группы, один `is_hdf5_v2`, мёртвые ветки | синтетический v2 с/без mapping |
| алгоритмы | `cumsum` сдвигов, знак (по round-trip тесту), `_interpolate_empty` с `initial_empty_fnumber`, допуск углов + `ValueError`, X-компонента, последний кадр в допуске, `save_amira`, копия в `show_frames_with_border`, `series_length` fallback → ошибка, `get_experiment_hdf5` (HTTP-ветка, размер) | pytest на синтетике |
| ноутбук | `debug=False`, `manual_axis_search=False` | статически |
| web/scripts | glob отчётов, try/except, `.get('timestamp')`, `convert_v1_to_v2` | тест v1 → v2 → reader |
| `hdf2vtk` | `ImageData`, `point_data` | — |
| compose | `restart`, healthcheck Mongo, `depends_on`, `.dockerignore`, `.gitignore` | — |
| README | активный ноутбук, статусы, прерванные эксперименты, накопительные сдвиги, порядок коррекции, старт после ребута | — |

Не делаем без отдельного решения: auth/POST для мутирующих роутов, path traversal, порт Mongo, хэш Jupyter, ubuntu18.04, `timeout=1000`,
удаление legacy `reconstructor-axis_search3b.py`/`tomotools2.py`, перенос тома Mongo, `safe_median` на фоне.

---

## 5. Статус на 22.09.2026

Ветка `review/recon-cleanup`: 27 коммитов поверх `develop` (`f7feeeb` … `101921d`), рабочее дерево чистое. Сделано всё из §4
(коммиты подписаны `Co-Authored-By: Claude Opus 5 (1M context)`). Сверх плана — исправлен знак начального приближения сдвига оси в
`find_axis_correction` (`tomotools4`, `tomotools2`): бралась Y-компонента центра масс (для пары 0°/180° всегда ≈0) и с неверным знаком;
ревью диффа подтвердило вывод `s = (cm1.x − cm0.x)/2` по целевой функции `transform_image(im0, s) − transform_image(im1, −s)`.

Ревью диффа (Opus) — математика подтверждена (cumsum, знак `phase_cross_correlation`, границы сегментов, fallback индексов из `timeline/modes`),
12 замечаний; в follow-up коммитах закрыты: неизвестный `action` → статус `error:` (иначе запись блокировала очередь навсегда),
`ValueError` от одного checkpoint не роняет прогон (пропуск с предупреждением), явный `logging.error` при пропущенном checkpoint (cumsum делает
хвост ненадёжным), no-op флаг `iopub_data_rate_limit`, семантика `segment_ids`/checkpoint в конвертере 1:1 со storage, `display` импорт,
`timestamp` разных типов при сортировке, `.pytest_cache`, файл зависимостей тестов.

Проверено локально:

| Проверка | Результат |
|---|---|
| `python -m pytest web/tests -q` (заглушки cupy/astra в `conftest.py`) | 95 passed |
| Round-trip: файл writer'а storage → `load_tomo_data_v2`/`load_tomo_data_advanced_v2` с `mapping` и без | 27/27 (`xtomo/scripts/check_hdf5_contract.py`) |
| Знак сдвига репозиционирования (синтетика, `dc = shift(data, [1.5, −2])`) | RMS без коррекции 0.063, старый знак 0.102, новый 0.014 |
| Накопление сдвигов на 2 checkpoint | сегмент 2 получает `shifts[0]+shifts[1]` |
| Конвертер v1 → v2 → reader | результат ≡ прямому чтению v2 |
| `docker-compose.yml` | валидный YAML, `restart` у всех кроме `test`, healthcheck `mongo`, `depends_on: service_healthy` |

NOT VERIFIED — требует стенда (GPU, Mongo, /exp_src): автозапуск `reconstructor4` из очереди (статус `done`, `reconstructor4.html`;
эксперимент с ошибкой ячейки → `error: …`); коррекция репозиционирования на реальных данных (`analyze_repositioning_accuracy` до/после);
старт стека после ребута; `hdf2vtk` на актуальном PyVista.

Не сделано (решение пользователя): auth/POST для мутирующих роутов, path traversal `storage_utils`, порт Mongo 27017 наружу, хэш Jupyter и IP
в Dockerfile, ubuntu18.04, `timeout=1000`, перенос тома `./data/db`, удаление legacy `reconstructor-axis_search3b.py`/`tomotools2.py`
(`tomotools2.get_experiment_hdf5` остался на `urlretrieve`), `safe_median` на лог-кадре, опорный `frame_number` серии empty (первый кадр vs центр).

---

## 6. Межмодульные контракты (сторона recon)

- **← storage**: HDF5 v2 по `/exp_src` (bind-mount) или nginx-alias `/storage/experiments/<id>.h5`; документ эксперимента через `POST /storage/experiments/get`
  (`finished` — признак завершённости, в очередь стоит ставить только `finished=True` или явно понимать, что `mapping` может отсутствовать).
- **Семантика v2**, на которую reader опирается: `timeline/frame_numbers` глобально монотонны; empty-серии по `series_length`; data_check снят при угле последнего data
  предыдущего сегмента; сдвиг k-го сегмента накопительный. `segment_ids`/`checkpoint_*` из storage — информационные.
- **← web (rbtm-web)**: только ссылка на `/view/tomo_object/<id>` через Apache-прокси `/reconstruct`.
