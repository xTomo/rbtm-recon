import json
import logging
import time
import requests
import tomo_queue
import glob
import os


STORAGE_SERVER = "http://rbtmstorage_server_1:5006/"

def get_reconstructed_files_list(experiment_id, is_local_ip):
    app_root = os.path.dirname(os.path.abspath(__file__))   # refers to application_top
    app_tomo_data= os.path.join(app_root, 'static', 'tomo_data', experiment_id)
    res = {}
    if not os.path.isdir(app_tomo_data):
        return res
    
    url_prefx = '' if is_local_ip else '.'
    logging.debug(f'get_reconstructed_files_list: is_local_ip={is_local_ip}')
    if os.path.exists(os.path.join(app_tomo_data,'tomo.html')):
        res['tomo_preview'] =  url_prefx+'/static/tomo_data/' + experiment_id +'/tomo.html'
    
    if os.path.exists(os.path.join(app_tomo_data,'tomo.hx')):
        res['amira_hx'] =  url_prefx+'/static/tomo_data/' + experiment_id +'/tomo.hx'

    if os.path.exists(os.path.join(app_tomo_data,'amira.raw')):
        res['amira_raw'] =  url_prefx+'/static/tomo_data/' + experiment_id +'/amira.raw'
    
    if os.path.exists(os.path.join(app_tomo_data,'tomo_rec.h5')):
        res['tomo_rec'] =  url_prefx+'/static/tomo_data/' + experiment_id +'/tomo_rec.h5'

    tomo_reports = glob.glob(os.path.join(app_tomo_data,'reconstructor-v*.html'))

    if len(tomo_reports)>0:
       res['tomo_reports'] = [url_prefx+tr[len(app_root):] for tr in tomo_reports]

    return res


def get_files_tree(experiment_id, is_local_ip):
    """Рекурсивно обходит директорию эксперимента и возвращает дерево файлов."""
    app_root = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(app_root, 'static', 'tomo_data', experiment_id)
    url_prefix = '' if is_local_ip else '.'

    if not os.path.isdir(base_dir):
        return None

    def build_tree(path, rel_path=''):
        entries = []
        try:
            items = sorted(os.listdir(path))
        except PermissionError:
            return entries
        for item in items:
            item_path = os.path.join(path, item)
            item_rel = (rel_path + '/' + item) if rel_path else item
            if os.path.isdir(item_path):
                children = build_tree(item_path, item_rel)
                entries.append({'name': item, 'type': 'dir', 'children': children})
            else:
                url = url_prefix + '/static/tomo_data/' + experiment_id + '/' + item_rel
                entries.append({'name': item, 'type': 'file', 'url': url})
        return entries

    return build_tree(base_dir)


def get_tomoobject_info(experiment_id, is_local_ip):
    exp_info = json.dumps({"_id": experiment_id})
    try:
        experiment = requests.post(STORAGE_SERVER + 'storage/experiments/get',
                                   exp_info, timeout=1000)
        experiment_info = json.loads(experiment.content)[0]
    except Exception as e:
        logging.error(f'Ошибка при обращении к storage серверу для {experiment_id}: {e}')
        return {'_id': experiment_id, 'specimen': '???', 'timestamp': 0,
                'tomo_status': 'storage error', 'files': {}}
    tomo_status = tomo_queue.get_object_status(experiment_id)
    experiment_info['tomo_status'] = tomo_status
    experiment_info['files'] = get_reconstructed_files_list(experiment_id, is_local_ip)
    return experiment_info

def get_tomoobjects_list():
    # exp_info = json.dumps({'finished': True})
    exp_info = json.dumps({})
 
    experiment = requests.post(STORAGE_SERVER + 'storage/experiments/get',
                               exp_info, timeout=1000)
    experiment_info = json.loads(experiment.content)
    ids = [x['_id'] for x in experiment_info]
    return ids


# [ДОБАВЛЕНО] Новая функция, заменяющая цепочку get_tomoobjects_list() + N×get_tomoobject_info().
# Вместо 1 + N HTTP-запросов к storage серверу делает всего 1 запрос,
# а статусы из MongoDB получает одним агрегирующим запросом через get_all_object_statuses().
# Это сократило время загрузки страницы с ~77 секунд до ~1-2 секунд.
def get_tomoobjects_full_info():
    t0 = time.time()

    exp_info = json.dumps({})
    # [ДОБАВЛЕНО] Один HTTP-запрос вместо N — получаем все эксперименты сразу
    try:
        experiment = requests.post(STORAGE_SERVER + 'storage/experiments/get',
                                   exp_info, timeout=1000)
        experiments = json.loads(experiment.content)
    except Exception as e:
        logging.error(f'Ошибка при обращении к storage серверу: {e}')
        return []
    t1 = time.time()
    logging.info(f'[PROFILE] HTTP запрос к storage: {t1 - t0:.3f}s')

    t2 = time.time()
    logging.info(f'[PROFILE] JSON парсинг ({len(experiments)} объектов): {t2 - t1:.3f}s')

    ts = time.time()
    # [ДОБАВЛЕНО] Один агрегирующий запрос к MongoDB вместо N отдельных запросов
    all_statuses = tomo_queue.get_all_object_statuses()
    te = time.time()
    logging.info(f'[PROFILE] get_all_object_statuses (1 запрос): {te - ts:.3f}s')

    for exp in experiments:
        exp['tomo_status'] = all_statuses.get(exp['_id'], 'hm... reconstruction not found...')

    t3 = time.time()
    logging.info(f'[PROFILE] Все статусы MongoDB: {t3 - t2:.3f}s')
    logging.info(f'[PROFILE] ИТОГО get_tomoobjects_full_info: {t3 - t0:.3f}s')

    return experiments
