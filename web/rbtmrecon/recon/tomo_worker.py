import configparser
import os
import subprocess
import time
from pprint import pprint
from shutil import copy

import nbformat

import tomotools2 as tomotools
from tomo_queue import get_rec_queue_next_obj, set_object_status

NOTEBOOK_NAME = 'reconstructor-v-2.1a.ipynb'


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

def copyfiles(obj):
    storage_dir = '/storage'
    obj_id = obj['obj_id']
    set_object_status(obj_id, 'reconstructing')
    print('Coping files: {}'.format(obj_id))

    out_dir = copy_python_files(obj_id, storage_dir)

    print('Finish coping: {}'.format(obj_id))
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
            if rec_obj['action'] == 'reconsruct':
                reconstruct(rec_obj)
            elif rec_obj['action'] == 'copyfiles':
                copyfiles(rec_obj)
            else:
                raise ValueError('Unknown action {}'.format(rec_obj['action']))
        else:
            time.sleep(10)
