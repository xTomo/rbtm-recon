import logging
from datetime import datetime

from pymongo import MongoClient, DESCENDING

from conf import MONGODB_URI

client = MongoClient(MONGODB_URI)

db = client['autotom']
to = db['tomoobjects']


def put_object_rec_queue(obj_id, action='reconstruct'):
    to.insert_one({'obj_id': obj_id,
                   'action': action,
                   'status': 'waiting',
                   'date': datetime.now()}
                  )


def get_object(obj_id):
    try:
        obj = to.find({'obj_id': obj_id}).sort('date', DESCENDING).limit(1)[0]
        return obj
    except:
        return None


def set_object_status(obj_id, status):
    logging.info('[QUEUE] set_object_status: {} -> {}'.format(obj_id, status))
    to.insert_one({'obj_id': obj_id,
               'status': status,
               'date': datetime.now()}
              )


def get_object_status(obj_id):
    obj = get_object(obj_id)
    if obj is None:
        return 'hm... reconstruction not found...'
    else:
        return obj['status']


def get_rec_queue_next_obj():
    waiting_all = list(to.find({'status': 'waiting'}))
    logging.info('[QUEUE] get_rec_queue_next_obj: всего waiting записей в БД: {}'.format(len(waiting_all)))
    for obj in waiting_all:
        if 'action' not in obj:
            logging.info('[QUEUE] пропускаем запись без action: {}'.format(obj.get('_id')))
            continue
        # Получаем последнюю запись для этого obj_id
        latest_obj = get_object(obj['obj_id'])
        latest_status = latest_obj['status'] if latest_obj else 'None'
        latest_id = latest_obj['_id'] if latest_obj else 'None'
        logging.info('[QUEUE] obj_id={} waiting_id={} latest_id={} latest_status={} match={}'.format(
            obj['obj_id'], obj['_id'], latest_id, latest_status,
            latest_obj and latest_obj['_id'] == obj['_id']
        ))
        # Проверяем, что текущая запись является последней И имеет статус 'waiting'
        if latest_obj and latest_obj['_id'] == obj['_id']:
            logging.info('[QUEUE] -> взяли задание: obj_id={} action={}'.format(obj['obj_id'], obj.get('action')))
            return obj
    logging.info('[QUEUE] -> очередь пуста')
    return None


def get_logs(obj_id):
    objs = to.find({'obj_id': obj_id}).sort('date', DESCENDING)
    if objs is None:
        raise ValueError('Object not found: {}'.format(obj_id))

    res = ["{}: {}".format(str(obj['date']), obj['status']) for obj in objs]
    return res


def get_last_n(n):
    objs = to.find().sort('date', DESCENDING).limit(n)
    return list(objs)
