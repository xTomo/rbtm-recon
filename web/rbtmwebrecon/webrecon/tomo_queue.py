from datetime import datetime

from pymongo import MongoClient, DESCENDING

from conf import MONGODB_URI

client = MongoClient(MONGODB_URI)

db = client['autotom']
to = db['tomoobjects']


def put_object_rec_queue(obj_id, action='reconstruct'):
    # Защита от дублей: не добавляем задание, если объект уже в активном статусе
    current_status = get_object_status(obj_id)
    active_statuses = ('waiting', 'copying', 'reconstructing')
    if any(s in current_status for s in active_statuses):
        return False
    to.insert_one({'obj_id': obj_id,
                   'action': action,
                   'status': 'waiting',
                   'date': datetime.now()}
                  )
    return True


def get_object(obj_id):
    try:
        # Сортируем по _id (ObjectId, генерируется MongoDB) вместо date (datetime.now() клиента),
        # чтобы избежать проблем с расхождением часов между контейнерами
        obj = to.find({'obj_id': obj_id}).sort('_id', DESCENDING).limit(1)[0]
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


# [ДОБАВЛЕНО] Агрегирующий запрос, который за один проход по коллекции MongoDB
# возвращает последний статус для каждого объекта.
# Заменяет N вызовов get_object_status() — ускорение с ~77с до ~1с для 760 объектов.
def get_all_object_statuses():
    """Возвращает словарь {obj_id: status} одним запросом к MongoDB."""
    pipeline = [
        # Сортируем по _id (ObjectId), а не по date — защита от расхождения часов контейнеров
        {"$sort": {"_id": DESCENDING}},
        {"$group": {"_id": "$obj_id", "status": {"$first": "$status"}}}
    ]
    result = to.aggregate(pipeline)
    return {doc['_id']: doc['status'] for doc in result}


def get_waiting_queue():
    """Возвращает все задания со статусом 'waiting', отсортированные по дате (старые первые)."""
    result = []
    for obj in to.find({'status': 'waiting'}).sort('_id', 1):
        if 'action' not in obj:
            continue
        latest = get_object(obj['obj_id'])
        if latest and latest['_id'] == obj['_id']:
            result.append(obj)
    return result


def cancel_all_waiting():
    """Отменяет все задания в очереди с статусом 'waiting'."""
    waiting = get_waiting_queue()
    for obj in waiting:
        to.insert_one({
            'obj_id': obj['obj_id'],
            'status': 'canceled',
            'date': __import__('datetime').datetime.now()
        })
    return len(waiting)


def get_last_n(n):
    objs = to.find().sort('date', DESCENDING).limit(n)
    return list(objs)
