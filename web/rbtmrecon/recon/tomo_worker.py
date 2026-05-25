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

NOTEBOOK_NAME = 'reconstructor-axis_search3b.py'


def _notebook_auto_run(notebook):
    """Execute a notebook via nbconvert and collect output.
       Сначала конвертирует .py (jupytext) -> .ipynb, затем выполняет через nbconvert.
       :returns (parsed nb object, execution errors)
    """
    # Шаг 1: конвертируем .py (jupytext-формат) в .ipynb
    notebook_ipynb = notebook.replace('.py', '.ipynb')
    args_jupytext = ["jupytext", "--to", "notebook", notebook, "--output", notebook_ipynb]
    subprocess.check_call(args_jupytext)

    # Шаг 2: выполняем .ipynb через nbconvert
    args = ["jupyter", "nbconvert", "--execute", "--allow-errors",
            "--ExecutePreprocessor.timeout=-1", "--NotebookApp.iopub_data_rate_limit=1.0e10",
            "--to", "notebook", '--output', notebook_ipynb, notebook_ipynb]
    subprocess.check_call(args)

    # Шаг 3: конвертируем выполненный ноутбук в HTML
    args = ["jupyter", "nbconvert", "--to", "html", notebook_ipynb]
    subprocess.check_call(args)

    nb = nbformat.read(notebook_ipynb, nbformat.current_nbformat)
    errors = [output for cell in nb.cells if "outputs" in cell
              for output in cell["outputs"]
              if output.output_type == "error"]
    return nb, errors


def reconstruct(obj):
    storage_dir = '/storage'
    obj_id = obj['obj_id']
    set_object_status(obj_id, 'reconstructing')
    logging.info('Start reconstructing: {}'.format(obj_id))

    try:
        out_dir = copy_python_files(obj_id, storage_dir)
        nb, errors = _notebook_auto_run(os.path.join(out_dir, NOTEBOOK_NAME))
        for e in errors:
            logging.info(e)
        logging.info('Finish reconstructing: {}'.format(obj_id))
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
        out_dir = copy_python_files(obj_id, storage_dir)
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
    config = configparser.ConfigParser()
    config["SAMPLE"] = tomo_info
    with open(os.path.join(out_dir, 'tomo.ini'), 'w') as cf:
        config.write(cf)

    # copy(NOTEBOOK_NAME, out_dir)
    # copy(NOTEBOOK_NAME[:-5] + 'py', out_dir)
    # copy('reconstructor-axis_search2.py', out_dir)
    # copy('tomotools.py', out_dir)
    for f in glob.glob('reconstructor*.py'):
        copy(f, out_dir)
    for f in glob.glob('tomotools*.py'):
        copy(f, out_dir)
    for f in glob.glob('hdf5_*.py'):
        copy(f, out_dir)
    copytree('tomo', os.path.join(out_dir, 'tomo'), dirs_exist_ok=True)
    return out_dir


if __name__ == "__main__":
    while True:
        rec_obj = get_rec_queue_next_obj()
        if rec_obj is not None:
            if 'action' not in rec_obj:
                raise ValueError('No "action" field in reconstruction object')
            if rec_obj['action'] == 'reconstruct':
                reconstruct(rec_obj)
            elif rec_obj['action'] == 'copyfiles':
                copyfiles(rec_obj)
            else:
                raise ValueError('Unknown action {}'.format(rec_obj['action']))
        else:
            time.sleep(10)
