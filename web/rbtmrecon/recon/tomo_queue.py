from pymongo import MongoClient, DESCENDING
from datetime import datetime

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
        obj = to.find({'obj_id': obj_id}).sort('date', DESCENDING)[0]
        return obj
    except:
        return None


def set_object_status(obj_id, status):
    to.insert({'obj_id': obj_id,
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
    for obj in to.find({'obj_id': 'waiting'}):
        if 'action' in obj and get_object_status(obj['obj_id']) == 'waiting':
            return obj
    return None


def get_logs(obj_id):
    objs = to.find({'obj_id': obj_id}).sort('date', DESCENDING)
    if objs is None:
        raise ValueError('Object not found: {}'.format(obj_id))

    res = ["{}: {}".format(str(obj['date']), obj['status']) for obj in objs]
    return res