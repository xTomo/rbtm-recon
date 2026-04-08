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
    for obj in to.find({'status': 'waiting'}):
        if 'action' not in obj:
            continue
        # Получаем последнюю запись для этого obj_id
        latest_obj = get_object(obj['obj_id'])
        # Проверяем, что текущая запись является последней И имеет статус 'waiting'
        if latest_obj and latest_obj['_id'] == obj['_id']:
            return obj
    return None


def get_logs(obj_id):
    objs = to.find({'obj_id': obj_id}).sort('date', DESCENDING)
    if objs is None:
        raise ValueError('Object not found: {}'.format(obj_id))

    res = ["{}: {}".format(str(obj['date']), obj['status']) for obj in objs]
    return res


def get_all_object_statuses():
    """Возвращает словарь {obj_id: status} одним запросом к MongoDB."""
    pipeline = [
        {"$sort": {"date": DESCENDING}},
        {"$group": {"_id": "$obj_id", "status": {"$first": "$status"}}}
    ]
    result = to.aggregate(pipeline)
    return {doc['_id']: doc['status'] for doc in result}


def get_last_n(n):
    objs = to.find().sort('date', DESCENDING).limit(n)
    return list(objs)
