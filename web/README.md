# rbtm-recon/web — Система томографической реконструкции

Веб-сервис для управления очередью томографических реконструкций на базе Flask + MongoDB + CUDA.

---

## Содержание

- [Архитектура](#архитектура)
- [Компоненты](#компоненты)
- [Быстрый старт](#быстрый-старт)
- [Настройка](#настройка)
- [Запуск после перезагрузки](#запуск-после-перезагрузки)
- [Использование](#использование)
- [Разработка](#разработка)
- [tomotools4 — поддержка AdvancedExperiment](#tomotools4--поддержка-advancedexperiment)

---

## Архитектура

```
Браузер
  │
  ├─► :5550  rbtmwebrecon   (Flask + Gunicorn)
  │           │
  │           ├─► :27017  MongoDB            (очередь заданий)
  │           └─► :5006   rbtmstorage        (метаданные экспериментов)
  │
  └─► :5551  reconstructor-jupyter  (JupyterLab — ручная реконструкция)

rbtm-web (Django) ──► :5560 recon-service (reconstructor, порт не публикуется, токен RECON_TOKEN)
  ├─► :27017  MongoDB                (очередь задач autotom.jobs, статусы в tomoobjects, старая очередь)
  ├─► :5006   rbtmstorage            (метаданные: размер пикселя)
  ├─► /exp_src                       (исходные HDF5, чтение без копирования)
  ├─► GPU 1                          (интерактивная сессия студии)
  └─► GPU 0 под /fast/.gpu0.lock     (задачи: python -m reconengine run; старая очередь ноутбуков)
```

До этапа 7 студии recon-service обслуживает и старую очередь `tomoobjects` (кнопки на странице
`:5550`) — тем же ноутбуком `reconstructor4.py` через nbconvert, что и раньше `tomo_worker.py`.

Контейнеры используют **две сети Docker**:
- `default` — внутренняя сеть стека `web`
- `rbtmstorage_default` — внешняя сеть стека `rbtmstorage` (должна существовать до запуска)

---

## Компоненты

### `rbtmwebrecon` — веб-интерфейс

| Файл | Описание |
|---|---|
| `webrecon/web_recon.py` | Flask-приложение: маршруты, API |
| `webrecon/tomo_queue.py` | Работа с MongoDB: чтение/запись очереди |
| `webrecon/storage_utils.py` | Запросы к rbtmstorage, обход файловой системы |
| `webrecon/conf.py` | Конфигурация (генерируется при сборке Docker) |
| `webrecon/templates/` | Jinja2-шаблоны (Bootstrap 4) |

**Маршруты:**

| URL | Метод | Описание |
|---|---|---|
| `/` или `/view/tomo_objects` | GET | Список всех томографических объектов |
| `/view/tomo_object/<id>` | GET | Карточка объекта: статус, файлы, метаданные |
| `/reconstruct/<id>` | GET | Поставить в очередь на реконструкцию |
| `/copyfiles/<id>` | GET | Скопировать файлы для ручной реконструкции |
| `/reset/<id>` | GET | Сбросить статус объекта (→ `canceled`) |
| `/queue` | GET | Текущая очередь заданий |
| `/queue/cancel_all` | GET | Отменить все задания в очереди |
| `/status/<n>` | GET | Последние N записей в базе |
| `/tomo_objects` | GET | JSON-список объектов (REST API) |
| `/tomo_object/<id>` | GET | JSON-карточка объекта (REST API) |

### `rbtmrecon` — recon-service и воркер реконструкции

Контейнер `reconstructor` запускает **recon-service** (`recon/reconservice/`, gunicorn на порту 5560): API
студии реконструкции для rbtm-web и очередь задач. Вычисления — пакет **reconengine** (`recon/reconengine/`,
`python -m reconengine suggest|migrate|run|compare`): чтение HDF5 без распаковки лишнего, конвейер по слоям
строк, рецепт `recipe.json` и `result.json`. Подробности — в докстрингах `reconservice/__init__.py` и
`reconengine/__init__.py`.

| Файл | Описание |
|---|---|
| `recon/reconservice/` | recon-service: сканы и обзор, интерактивная сессия, задачи, результаты; `/health` без токена |
| `recon/reconengine/` | движок реконструкции (используется сервисом, CLI и — с этапа 6 — ноутбуком) |
| `recon/tomo_worker.py` | Шаг старой очереди (`process_once`): ноутбук через nbconvert; вызывается recon-service |
| `recon/tomo_queue.py` | Работа с MongoDB |
| `recon/tomotools2.py` | Инструменты для стандартного эксперимента (legacy, нужен `tomo_worker.py` и старому ноутбуку) |
| `recon/tomotools4.py` | Расширенные инструменты: поддержка AdvancedExperiment, коррекция позиционирования, нормировка с дрейфом |
| `recon/hdf5_v2.py` | Читатель HDF5 v2 (timeline + mapping), автодетекция формата |
| `recon/reconstructor4.py` | **Активный** ноутбук-шаблон реконструкции v4 (AdvancedExperiment + Standard, jupytext-формат) |
| `recon/reconstructor-axis_search3b.py` | Устаревший ноутбук-шаблон (только Standard), оставлен для ручных запусков |
| `recon/hdf2vtk.py` | Конвертация HDF5 → VTK (ручной запуск) |
| `recon/tomo.ini` | Пример конфигурации одного эксперимента |
| `environment.yml` | conda-окружение `xrecon` (Python 3.11 + CUDA 12) |

**Алгоритм реконструкции (один объект).** Воркер запускает ноутбук
`NOTEBOOK_NAME` из [`tomo_worker.py`](rbtmrecon/recon/tomo_worker.py) — сейчас
это `reconstructor4.py` (`reconstructor-axis_search3b.py` оставлен только для
ручных запусков в JupyterLab).

```
1. Получить метаданные эксперимента из rbtmstorage
2. Скопировать скрипты в /storage/<experiment_id>/ (относительно каталога воркера)
3. Получить HDF5-файл в /fast/<experiment_id>/
   ├── локальная копия из /exp_src/<id>/before_processing/<id>.h5
   └── если её нет — скачать с rbtmstorage по HTTP
   (после копирования/скачивания размер сверяется с источником)
4. jupytext: .py → .ipynb
5. nbconvert --execute --allow-errors: выполнить ноутбук (CUDA)
   ├── Автодетекция формата: is_advanced_experiment() → advanced / standard
   ├── [Advanced] load_tomo_data_advanced() — раздельная загрузка empty/data/data_check
   ├── [Advanced] analyze_source_drift() — визуализация дрейфа трубки
   ├── [Advanced] measure_repositioning_shifts() — кросс-корреляция data vs data_check
   ├── Нормировать проекции
   │   ├── [Advanced] normalize_projections_with_timeline() — с интерполяцией empty
   │   └── [Standard] normalize_projections() — единый empty
   ├── [Advanced] apply_repositioning_correction() — sub-pixel коррекция
   │                ПОСЛЕ нормировки, накопительным сдвигом cumsum(shifts[:m])
   ├── Найти коррекцию оси вращения (метод Пауэлла)
   ├── Применить коррекцию, построить синограмму
   ├── Удалить кольцевые артефакты
   └── FBP реконструкция (astra-toolbox)
6. nbconvert: .ipynb → HTML-отчёт
7. Установить статус в MongoDB:
   ├── ни одной упавшей ячейки            → 'done'
   └── есть output с output_type=='error' → 'error: <ename>: <evalue>'
```

> Ноутбук выполняется с `--allow-errors`, поэтому HTML-отчёт создаётся даже
> при падении ячейки — по нему видно, на каком шаге всё сломалось. Статус при
> этом всё равно `error: …`, а не `done`.

**Прерванный эксперимент.** `rbtm-storage` пишет группу `mapping` только при
`finalize`. Если эксперимент остановлен или упал, `mapping` в HDF5 нет —
читатель [`hdf5_v2.py`](rbtmrecon/recon/hdf5_v2.py) в этом случае
восстанавливает индексы кадров по `timeline/modes`. Пустые группы
(нет `dark`, нет `data_check`) тоже допустимы: для `dark` используется нулевой
тёмный ток с предупреждением, `data_check` даёт пустые массивы и коррекция
позиционирования просто не выполняется.

### `database` — MongoDB 4.0

Хранит **единую коллекцию** `autotom.tomoobjects` — лог событий очереди.

**Схема документа:**
```json
{
  "_id": ObjectId,
  "obj_id": "uuid-эксперимента",
  "action": "reconstruct" | "copyfiles",
  "status": "waiting" | "copying" | "reconstructing" | "done" | "error: ..." | "canceled",
  "date": ISODate
}
```

Статус объекта — это **статус самого последнего документа** с данным `obj_id`.

---

## Быстрый старт

### Требования

- Docker ≥ 20, docker-compose ≥ 1.25
- NVIDIA GPU с CUDA 12.1, nvidia-docker2
- Запущенный стек `rbtmstorage` (сеть `rbtmstorage_default` должна существовать)

### Монтируемые пути на хосте

| Путь хоста | Контейнер | Назначение |
|---|---|---|
| `/diskmnt/fast/robotom` | `/fast` | Быстрый диск для временных файлов (SSD) |
| `/diskmnt/a/makov/robotom/` | `/storage` | Долгосрочное хранилище результатов |
| `/home/robotom/rbtm_data/rbtm_storage/data/experiments` | `/exp_src` | Исходные данные экспериментов (только чтение) |

### Сборка и запуск

Перед первым запуском создайте `web/.env` с общим токеном recon-service и rbtm-web (образец —
`web/.env.example`; тот же `RECON_TOKEN` задаётся в окружении rbtm-web). Без токена сервис отвечает 503.

```bash
cd rbtm-recon/web
docker-compose build
docker-compose up -d
```

Для перезапуска с очисткой логов (скрипт `restart.sh`):
```bash
bash restart.sh
```

### Проверка работоспособности

```bash
# Статус контейнеров
docker-compose ps

# Логи recon-service и очереди
docker-compose logs -f reconstructor

# recon-service жив: GPU, сессия, очередь, занятость GPU 0
docker exec web_reconstructor_1 wget -qO- http://localhost:5560/health

# Логи веб-сервиса
docker-compose logs -f web

# Тест GPU
docker-compose run --rm test
```

Веб-интерфейс: http://localhost:5550  
JupyterLab: http://localhost:5551

---

## Настройка

### Конфигурация генерируется при сборке

Оба Dockerfile **записывают** `conf.py` при сборке:

**`rbtmrecon/Dockerfile`** — создаёт `/rbtm/recon/conf.py`:
```python
MONGODB_URI = 'mongodb://web_database_1:27017'
```

**`rbtmwebrecon/Dockerfile`** — создаёт `/webrbtm/webrecon/conf.py`:
```python
MONGODB_URI = 'mongodb://web_database_1:27017'
JUPYTER_SERVER = 'http://10.0.7.153:5551'
```

> **⚠️ IP-адрес Jupyter захардкожен в Dockerfile** (строка 32).  
> При переезде на другой хост — поменять вручную.

### Смена пароля JupyterLab

Сгенерировать новый хэш:
```python
from jupyter_server.auth import passwd
print(passwd('ваш_пароль'))
```
Вставить хэш в [`rbtmrecon/Dockerfile`](rbtmrecon/Dockerfile) строку с `c.ServerApp.password`.

### Настройка пользователя в `rbtmrecon`

По умолчанию внутри контейнера создаётся пользователь `makov` (UID=1000, GID=1000).  
Для изменения передайте `--build-arg`:
```bash
docker-compose build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g)
```

### Очередь MongoDB

MongoDB хранит данные в `./data/db` — путь **относительный**, он раскрывается от
каталога, из которого запущен `docker-compose` (обычно `rbtm-recon/web`).
Директория создаётся автоматически, но при запуске из другого каталога Mongo
поднимется с пустой базой и вся история очереди «исчезнет». Запускайте стек
только из `rbtm-recon/web` (или переведите том на абсолютный путь).

---

## Запуск после перезагрузки

Все сервисы стека (кроме одноразового `test`) объявлены с
`restart: unless-stopped`, поэтому после перезагрузки хоста Docker поднимает их
сам. Порядок внутри стека: `database` → healthcheck (`mongo --eval
db.adminCommand('ping')`) → `web` (`depends_on: condition: service_healthy`).
`reconstructor` от Mongo формально не зависит, но и не падает, если она ещё не
готова: главный цикл воркера ловит любое исключение итерации (в том числе
`ServerSelectionTimeoutError`), пишет его в лог и повторяет попытку через 10 с.

**Внешняя сеть `rbtmstorage_default` — главная причина, по которой стек не
поднимается.** Она создаётся стеком `rbtm-storage` и объявлена здесь как
`external: true`. Если на storage выполнили `docker compose down`, сеть
удаляется, и контейнеры `web`/`reconstructor` при старте падают с
`network rbtmstorage_default not found` — restart-политика будет их
перезапускать, но безуспешно, пока сеть не появится.

Правильный порядок запуска:

```bash
# 1. Сначала storage — он создаёт сеть rbtmstorage_default
cd rbtm-storage && docker-compose up -d

# 2. Затем этот стек
cd ../rbtm-recon/web && docker-compose up -d

# Проверка
docker network ls | grep rbtmstorage_default
docker-compose ps            # database должна быть (healthy)
docker-compose logs -f reconstructor
```

На storage вместо `docker compose down` лучше использовать
`docker compose stop` / `restart` — тогда сеть не удаляется.

---

## Использование

### Запуск реконструкции

1. Открыть http://localhost:5550
2. Найти нужный объект в списке (фильтр по названию — поле вверху справа)
3. Нажать **📋 Copy files to reconstruction** — файлы скопируются в `/storage/<id>/`
4. Для ручной реконструкции нажать **⚙ Manual reconstruction** — откроется JupyterLab
5. Для автоматической реконструкции воркер запускается автоматически (кнопка реконструкции появится после копирования)

### Статусы объектов

| Статус | Цвет | Описание |
|---|---|---|
| `waiting` | 🟡 жёлтый | Ждёт в очереди |
| `copying` | 🟡 жёлтый | Копирование файлов |
| `reconstructing` | 🟡 жёлтый | Идёт реконструкция |
| `done` | 🟢 зелёный | Завершено успешно, ни одна ячейка ноутбука не упала |
| `error: <ename>: <evalue>` | 🔴 красный | Упала ячейка ноутбука; HTML-отчёт всё равно создан |
| `error: ...` | 🔴 красный | Ошибка самого воркера (копирование, jupytext, nbconvert) |
| `canceled` | 🔴 красный | Отменено |

Задания берутся из очереди в порядке FIFO (сортировка по `_id`). Запись
считается актуальной, только если это самый свежий документ для своего
`obj_id`.

### Файлы результатов

Результаты сохраняются в `/storage/<experiment_id>/` (монтируется в веб как `static/tomo_data/`):

| Файл | Описание |
|---|---|
| `reconstructor*.html` | HTML-отчёт выполненного ноутбука |
| `tomo_rec.h5` | Реконструированный объём (HDF5) |
| `amira.raw` + `tomo.hx` | Объём в формате Amira |
| `tomo.html` | Предпросмотр (если есть) |
| `rec_config.ini` | Сохранённые параметры реконструкции (ROI, ось) |

---

## Разработка

### Структура conda-окружений

| Окружение | Контейнер | Пакеты |
|---|---|---|
| `xrecon` | `rbtmrecon` | Python 3.11, astra-toolbox, CuPy, CIL, tomopy, Jupyter |
| `xweb` | `rbtmwebrecon` | Python 3.11, Flask, gunicorn, pymongo |

### Тесты

Тесты лежат в `web/tests` и не требуют ни GPU, ни MongoDB: `conftest.py`
подставляет заглушки `cupy`, `cupyx.scipy.ndimage`, `tomo.recon.astra_utils`,
`tqdm.notebook` и `pylab` поверх numpy/scipy/matplotlib, а очередь тестируется
на `mongomock`.

```bash
pip install -r web/tests/requirements-test.txt
python -m pytest web/tests -q
```

Зависимости зафиксированы в [`web/tests/requirements-test.txt`](tests/requirements-test.txt):
`pytest numpy scipy scikit-image h5py mongomock tqdm matplotlib requests nbformat`.

### Отладка ноутбука реконструкции

```bash
# Запустить JupyterLab вместо воркера
docker-compose up reconstructor-jupyter
# → http://localhost:5551
```

### Добавление нового типа задания воркеру

В [`tomo_worker.py`](rbtmrecon/recon/tomo_worker.py) в главном цикле:
```python
elif rec_obj['action'] == 'my_new_action':
    my_new_action_handler(rec_obj)
```

В [`tomo_queue.py`](rbtmwebrecon/webrecon/tomo_queue.py):
```python
tomo_queue.put_object_rec_queue(obj_id, action='my_new_action')
```

### Запуск без Docker (для разработки)

```bash
# MongoDB
docker run -d -p 27017:27017 --name mongo mongo:4.0

# Веб
cd rbtmwebrecon/webrecon
conda activate xweb
gunicorn -w 1 -b 0.0.0.0:5550 web_recon:app

# Воркер
cd rbtmrecon/recon
conda activate xrecon
python tomo_worker.py
```

### Зависимости

- Требует запущенного стека `rbtmstorage` и сети `rbtmstorage_default`
- IP Jupyter-сервера (`10.0.7.153`) захардкожен в Dockerfile веб-сервиса

---

## tomotools4 — поддержка AdvancedExperiment

`tomotools4.py` полностью заменяет `tomotools2.py` для продвинутого режима и совместим с ним для стандартного.

### Ключевые функции

| Функция | Описание |
|---|---|
| `is_advanced_experiment(data_file)` | Детектор формата по наличию непустой группы `data_check` |
| `load_tomo_data_advanced(data_file, tmp_dir)` | Загружает `AdvancedTomoData` (dark, initial/periodic empty, data, data_check) |
| `analyze_source_drift(adv_data)` | График дрейфа интенсивности рентгеновской трубки по сериям empty |
| `measure_repositioning_shifts(adv_data, ...)` | Кросс-корреляция data vs data_check для измерения сдвига позиционирования |
| `apply_repositioning_correction(data_images, ...)` | Sub-pixel коррекция кадров in-place (ПОСЛЕ нормировки, накопительным сдвигом) |
| `analyze_repositioning_accuracy(adv_data, ...)` | Графики ошибки позиционирования по checkpoints |
| `normalize_projections_with_timeline(data_images_crop, adv_data, ...)` | Нормировка с линейной интерполяцией empty между checkpoint-ами |

### Структура данных AdvancedTomoData

```python
AdvancedTomoData(
    dark_image,               # медиана dark кадров, shape (H, W)
    initial_empty,            # медиана начальной empty серии, shape (H, W)
    initial_empty_fnumber,    # frame_number первого кадра начальной серии
    periodic_empties,         # list[np.ndarray] — медианы periodic серий
    periodic_empty_fnumbers,  # list[int] — frame_number первого кадра каждой periodic серии
    data_images,              # dark-subtracted проекции, shape (N, H, W)
    data_angles,              # углы, shape (N,)
    data_numbers,             # глобальные frame_numbers, shape (N,)
    data_check_images,        # dark-subtracted контрольные кадры, shape (M, H, W)
    data_check_angles,        # shape (M,)
    data_check_numbers,       # глобальные frame_numbers, shape (M,)
    series_length,            # длина dark/empty серии
)
```

### Нумерация сегментов (для коррекции позиционирования)

```
frame_numbers:  0..19 (dark)  20..29 (empty_init)  30..69 (data seg0)
                70..79 (periodic_empty[0])  80 (data_check[0])  81..130 (data seg1)
                131..140 (periodic_empty[1])  141 (data_check[1])  ...

periodic_empty_fnumbers = [70, 131, ...]
data_check_numbers      = [80, 141, ...]

Сегмент 0 (data seg0): data_number < 70  → референс, не корректируется
Сегмент 1 (data seg1): data_number > 70  → смещён на shifts[0]
Сегмент 2 (data seg2): data_number > 131 → смещён на shifts[0] + shifts[1]
```

**Сдвиги накопительные.** `shifts[k]` измеряется сравнением `data_check[k]`
(снят сразу после k-й вставки) с ПОСЛЕДНИМ data-кадром сегмента k при том же
угле, то есть характеризует смещение сегмента k+1 *относительно сегмента k*,
а не относительно референса. Поэтому к сегменту m применяется
`cumsum(shifts[:m])`.

**Знак.** `measure_repositioning_shifts` возвращает результат
`phase_cross_correlation(reference=data, moving=data_check)` — такой сдвиг,
что `ndi.shift(data_check, shift) ≈ data`. Кадры сегмента сняты в той же
позиции, что и `data_check`, поэтому `apply_repositioning_correction`
применяет тот же сдвиг без смены знака.

**Порядок.** Коррекция позиционирования применяется **ПОСЛЕ** нормировки,
к нормированным кадрам `data_images_crop` (иначе сдвигались бы «сырые»
отсчёты вместе с полем засветки).

### Отладка measure_repositioning_shifts

Функция принимает параметр `debug=True` для вывода диагностики:

```python
checkpoint_angles, shifts_y, shifts_x = measure_repositioning_shifts(
    adv_data, x_min, x_max, y_min, y_max, debug=True)
```

Вывод включает:
- `series_length`, `periodic_empty_fnumbers`, `data_check_numbers` — для проверки разбиения
- `fn_start` / `next_fn` / `dc_indices` — для каждого checkpoint-а
- `data_norm`/`dc_norm` статистики — для контроля нормировки
- `direct_cc_shift` — сдвиг через прямую кросс-корреляцию (для сравнения с `phase_cross_correlation`)

### Известная ловушка: `series_length` из HDF5

`series_length` читается из атрибута `exp_info` HDF5-файла. Структура атрибута — это полный MongoDB-документ:

```json
{
  "experiment parameters": {
    "series_length": 10,
    "empty_period": 50
  }
}
```

Правильный путь: `exp_info['experiment parameters']['series_length']` (не `exp_info['series_length']`).
Если `series_length` прочитан неправильно — `periodic_empty_fnumbers` смещаются и `measure_repositioning_shifts` возвращает нулевые сдвиги для всех checkpoints. Исправлено в `tomotools4.py`.
