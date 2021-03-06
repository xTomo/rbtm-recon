from tomo_queue import get_rec_queue_next_obj, set_object_status, put_object_rec_queue, get_logs
from pprint import pprint

import time

import json
import requests
import tomotools2 as tomotools
import configparser
import subprocess
import nbformat
import os

from shutil import copy

NOTEBOOK_NAME = 'reconstructor-v-2.0d.ipynb'

def _notebook_auto_run(notebook):
    """Execute a notebook via nbconvert and collect output.
       :returns (parsed nb object, execution errors)
    """

    path = notebook
    args = ["jupyter", "nbconvert", "--execute", "--allow-errors",
            "--ExecutePreprocessor.timeout=-1", "--NotebookApp.iopub_data_rate_limit=1.0e10",
            "--to", "notebook", '--output', notebook, path]
    subprocess.check_call(args)

    args = ["jupyter", "nbconvert", "--to", "html",
            notebook]
    subprocess.check_call(args)

    nb = nbformat.read(path, nbformat.current_nbformat)
    errors = [output for cell in nb.cells if "outputs" in cell
              for output in cell["outputs"]
              if output.output_type == "error"]
    return nb, errors


def reconstruct(obj):
    storage_dir = '/storage'
    obj_id = obj['obj_id']
    set_object_status(obj_id, 'reconstructing')
    print('Start reconstructing: {}'.format(obj_id))

    out_dir = copy_python_files(obj_id, storage_dir)

    nb, errors = _notebook_auto_run(os.path.join(out_dir, NOTEBOOK_NAME))
    for e in errors:
        pprint(e)

    print('Finish reconstructing: {}'.format(obj_id))
    set_object_status(obj_id, 'done')

def copy_python_files(obj_id, storage_dir):
    to = obj_id
    tomo_info = tomotools.get_tomoobject_info(to)
    experiment_id = tomo_info['_id']

    out_dir = os.path.join(storage_dir, experiment_id, '')
    tomotools.mkdir_p(out_dir)

    print(tomo_info['specimen'])
    config = configparser.ConfigParser()
    config["SAMPLE"] = tomo_info
    with open(os.path.join(out_dir, 'tomo.ini'), 'w') as cf:
        config.write(cf)

    copy(NOTEBOOK_NAME, out_dir)
    copy('tomotools2.py', out_dir)
    return out_dir


if __name__ == "__main__":
    while True:
        # print('waiting objects')
        rec_obj = get_rec_queue_next_obj()
        if rec_obj is not None:
            reconstruct(rec_obj)
        else:
            time.sleep(10)
