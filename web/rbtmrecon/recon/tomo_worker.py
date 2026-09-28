import configparser
import glob
import logging
import os
import subprocess
import time
from shutil import copy, copytree

import nbformat

import tomotools2 as tomotools
from tomo_queue import get_rec_queue_next_obj, set_object_status

logging.basicConfig(level=logging.INFO)

NOTEBOOK_NAME = 'reconstructor4.py'

# Каталог с исходниками воркера (скрипты копируются относительно него, а не cwd)
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def find_notebook_errors(nb):
    """Возвращает список error-выводов выполненного ноутбука.

    Работает как с объектом nbformat.NotebookNode, так и с обычным dict,
    прочитанным из .ipynb через json.load.
    """
    errors = []
    for cell in nb.get('cells', []) or []:
        for output in cell.get('outputs', []) or []:
            if output.get('output_type') == 'error':
                errors.append(output)
    return errors


def format_notebook_error(error):
    """Формирует строку статуса по первому error-выводу ноутбука."""
    ename = error.get('ename', 'Error')
    evalue = str(error.get('evalue', ''))[:150]
    return 'error: {}: {}'.format(ename, evalue)


def _notebook_auto_run(notebook):
    """Execute a notebook via nbconvert and collect output.
       Сначала конвертирует .py (jupytext) -> .ipynb, затем выполняет через nbconvert.
       Выполнение идёт с --allow-errors, чтобы HTML-отчёт создавался всегда;
       ошибки ячеек возвращаются вызывающему коду отдельным списком.
       :returns (parsed nb object, execution errors)
    """
    # Шаг 1: конвертируем .py (jupytext-формат) в .ipynb
    notebook_ipynb = notebook.replace('.py', '.ipynb')
    args_jupytext = ["jupytext", "--to", "notebook", notebook, "--output", notebook_ipynb]
    subprocess.check_call(args_jupytext)

    # Шаг 2: выполняем .ipynb через nbconvert
    # --ServerApp.iopub_data_rate_limit относится к Jupyter Server (лимит скорости
    # вывода по websocket) и не применим к nbconvert: тот читает iopub напрямую
    # через jupyter_client, а не через сервер. nbconvert просто игнорирует этот
    # флаг с предупреждением "Unrecognized config" — убран как no-op.
    args = ["jupyter", "nbconvert", "--execute", "--allow-errors",
            "--ExecutePreprocessor.timeout=-1",
            "--to", "notebook", '--output', notebook_ipynb, notebook_ipynb]
    subprocess.check_call(args)

    # Шаг 3: конвертируем выполненный ноутбук в HTML
    args = ["jupyter", "nbconvert", "--to", "html", notebook_ipynb]
    subprocess.check_call(args)

    nb = nbformat.read(notebook_ipynb, nbformat.current_nbformat)
    return nb, find_notebook_errors(nb)


def reconstruct(obj):
    storage_dir = '/storage'
    obj_id = obj['obj_id']
    set_object_status(obj_id, 'reconstructing')
    logging.info('Start reconstructing: {}'.format(obj_id))

    try:
        out_dir = copy_python_files(obj_id, storage_dir)
        nb, errors = _notebook_auto_run(os.path.join(out_dir, NOTEBOOK_NAME))
        for e in errors:
            logging.error(e)
        logging.info('Finish reconstructing: {}'.format(obj_id))
        if errors:
            # HTML-отчёт создан (--allow-errors), но реконструкция не завершилась
            set_object_status(obj_id, format_notebook_error(errors[0]))
        else:
            set_object_status(obj_id, 'done')
    except Exception as e:
        logging.error('Error reconstructing {}: {}'.format(obj_id, e), exc_info=True)
        set_object_status(obj_id, 'error: {}'.format(str(e)[:200]))


def copyfiles(obj):
    storage_dir = '/storage'
    obj_id = obj['obj_id']
    set_object_status(obj_id, 'copying')
    logging.info('Start copying files: {}'.format(obj_id))

    try:
        copy_python_files(obj_id, storage_dir)
        logging.info('Finish copying: {}'.format(obj_id))
        set_object_status(obj_id, 'done')
    except Exception as e:
        logging.error('Error copying files for {}: {}'.format(obj_id, e), exc_info=True)
        set_object_status(obj_id, 'error: {}'.format(str(e)[:200]))


def copy_python_files(obj_id, storage_dir):
    to = obj_id
    tomo_info = tomotools.get_tomoobject_info(to)
    experiment_id = tomo_info['_id']

    out_dir = os.path.join(storage_dir, experiment_id, '')
    tomotools.mkdir_p(out_dir)

    logging.info(tomo_info['specimen'])
    # interpolation=None: в specimen встречается '%', ConfigParser по умолчанию
    # трактует его как начало подстановки и падает с InterpolationSyntaxError
    config = configparser.ConfigParser(interpolation=None)
    config["SAMPLE"] = tomo_info
    with open(os.path.join(out_dir, 'tomo.ini'), 'w') as cf:
        config.write(cf)

    # Копируем скрипты из каталога воркера, а не из текущего рабочего каталога
    for pattern in ('reconstructor*.py', 'tomotools*.py', 'hdf5_*.py'):
        for f in glob.glob(os.path.join(SCRIPTS_DIR, pattern)):
            copy(f, out_dir)
    copytree(os.path.join(SCRIPTS_DIR, 'tomo'),
             os.path.join(out_dir, 'tomo'), dirs_exist_ok=True)
    return out_dir


def process_once():
    """Одна итерация главного цикла: берёт задание из очереди и выполняет его.

    Возвращает True, если задание было взято, иначе False.
    """
    rec_obj = get_rec_queue_next_obj()
    if rec_obj is None:
        return False

    action = rec_obj.get('action')
    if action == 'reconstruct':
        reconstruct(rec_obj)
    elif action == 'copyfiles':
        copyfiles(rec_obj)
    else:
        # Неизвестный/отсутствующий action не должен ронять воркер, но и не должен
        # молча оставлять запись в статусе 'waiting' — иначе get_rec_queue_next_obj
        # будет раз за разом возвращать это же задание и очередь встанет намертво
        # (head-of-line blocking для всех задач, поставленных после него).
        logging.error('Skipping task %s: unknown action %r', rec_obj.get('_id'), action)
        set_object_status(rec_obj.get('obj_id'), 'error: unknown action {!r}'.format(action))
        return False
    return True


def worker_loop(iterations=None, sleep_seconds=10):
    """Главный цикл воркера.

    Любая ошибка итерации (в т.ч. недоступная при старте MongoDB —
    ``pymongo.errors.ServerSelectionTimeoutError``) не должна убивать воркер:
    restart-политика контейнера не гарантирована, процесс обязан выжить сам.

    ``iterations`` ограничивает число итераций (используется в тестах);
    None — бесконечный цикл.
    """
    n = 0
    while iterations is None or n < iterations:
        n += 1
        try:
            if not process_once():
                time.sleep(sleep_seconds)
        except Exception:
            logging.exception('Unhandled error in worker loop, retrying in %ss',
                              sleep_seconds)
            time.sleep(sleep_seconds)


if __name__ == "__main__":
    worker_loop()
