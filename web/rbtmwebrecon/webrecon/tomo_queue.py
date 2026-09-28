"""Работа с очередью заданий реконструкции в MongoDB.

ВАЖНО: этот файл должен быть ПОБАЙТОВО ИДЕНТИЧЕН
``rbtmrecon/recon/tomo_queue.py``. Два экземпляра существуют только потому,
что web и reconstructor собираются в разные Docker-образы и каждый копирует
в образ лишь свой подкаталог. Любую правку нужно вносить в оба файла.
"""
import logging
from datetime import datetime

from pymongo import MongoClient, ASCENDING, DESCENDING

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
    except Exception:
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
    # Сортировка по _id ASC — FIFO: старейшее задание берётся первым.
    # (Без сортировки порядок выдачи курсора не определён.)
    waiting_all = list(to.find({'status': 'waiting'}).sort('_id', ASCENDING))
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
            bool(latest_obj) and latest_obj['_id'] == obj['_id']
        ))
        # Проверяем, что текущая запись является последней И имеет статус 'waiting'
        if latest_obj and latest_obj['_id'] == obj['_id']:
            logging.info('[QUEUE] -> взяли задание: obj_id={} action={}'.format(obj['obj_id'], obj.get('action')))
            return obj
    logging.info('[QUEUE] -> очередь пуста')
    return None


def get_logs(obj_id):
    objs = to.find({'obj_id': obj_id}).sort('_id', DESCENDING)
    return ["{}: {}".format(str(obj['date']), obj['status']) for obj in objs]


# Агрегирующий запрос, который за один проход по коллекции MongoDB
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
    """Возвращает все задания со статусом 'waiting', отсортированные по _id (старые первые)."""
    result = []
    for obj in to.find({'status': 'waiting'}).sort('_id', ASCENDING):
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
            'date': datetime.now()
        })
    return len(waiting)


def get_last_n(n):
    objs = to.find().sort('_id', DESCENDING).limit(n)
    return list(objs)
