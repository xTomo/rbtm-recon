import json
import requests
import tomo_queue
import glob
import os


STORAGE_SERVER = "http://rbtmstorage_server_1:5006/"

def get_reconstructed_files_list(experiment_id):
    app_root = os.path.dirname(os.path.abspath(__file__))   # refers to application_top
    app_tomo_data= os.path.join(app_root, 'static', 'tomo_data', experiment_id)
    res = {}
    if not os.path.isdir(app_tomo_data):
        return res
    
    if os.path.exists(os.path.join(app_tomo_data,'tomo.html')):
        res['tomo_preview'] =  './static/tomo_data/' + experiment_id +'/tomo.html'
    
    if os.path.exists(os.path.join(app_tomo_data,'tomo.hx')):
        res['amira_hx'] =  './static/tomo_data/' + experiment_id +'/tomo.hx'

    if os.path.exists(os.path.join(app_tomo_data,'amira.raw')):
        res['amira_raw'] =  './static/tomo_data/' + experiment_id +'/amira.raw'
    
    if os.path.exists(os.path.join(app_tomo_data,'tomo_rec.h5')):
        res['tomo_rec'] =  './static/tomo_data/' + experiment_id +'/tomo_rec.h5'

    tomo_reports = glob.glob(os.path.join(app_tomo_data,'reconstructor-v*.html'))

    if len(tomo_reports)>0:
       res['tomo_reports'] = [tr[len(app_root):] for tr in tomo_reports]

    return res


def get_tomoobject_info(experiment_id):
    exp_info = json.dumps(({"_id": experiment_id}))
    experiment = requests.post(STORAGE_SERVER + 'storage/experiments/get',
                               exp_info, timeout=1000)
    experiment_info = json.loads(experiment.content)[0]
    tomo_status = tomo_queue.get_object_status(experiment_id)
    experiment_info['tomo_status'] = tomo_status
    experiment_info['files'] = get_reconstructed_files_list(experiment_id)
    return  experiment_info

def get_tomoobjects_list():
    exp_info = json.dumps({'finished': True})
 
    experiment = requests.post(STORAGE_SERVER + 'storage/experiments/get',
                               exp_info, timeout=1000)
    experiment_info = json.loads(experiment.content)
    ids = [x['_id'] for x in experiment_info]
    return  ids
