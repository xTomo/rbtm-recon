import logging
import pprint
import time

from flask import Flask, render_template, request, redirect
from flask_restful import Resource, Api

import storage_utils
import tomo_queue

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
api = Api(app)


def is_local_ip(ip):
    if ip.startswith('10.'):
        return True
    else:
        return False


class TomoObjects(Resource):
    def get(self):
        return storage_utils.get_tomoobjects_list()


api.add_resource(TomoObjects, '/tomo_objects')


class TomoObject(Resource):
    def get(self, to_id):
        return storage_utils.get_tomoobject_info(to_id, is_local_ip(request.remote_addr))


api.add_resource(TomoObject, '/tomo_object/<to_id>')


# @app.route('/ip')
# def get_ip():
#     return 'IP: ' + request.remote_addr + 'is local: ' + str(is_local_ip(request.remote_addr))

@app.route('/')
@app.route('/view/tomo_objects')
def view_tomo_objects():
    t0 = time.time()
    # [ДОБАВЛЕНО] Используем оптимизированную функцию: 1 HTTP-запрос + 1 запрос MongoDB
    # вместо прежних 1 + N HTTP-запросов
    tomo_objects = storage_utils.get_tomoobjects_full_info()
    t1 = time.time()
    logging.info(f'[PROFILE] get_tomoobjects_full_info в view: {t1 - t0:.3f}s')
    tomo_objects.sort(key=lambda x: x['timestamp'], reverse=True)

    total = len(tomo_objects)
    # [ДОБАВЛЕНО] Пагинация: показываем по 20 объектов на странице
    per_page = 20
    page = request.args.get('page', 1, type=int)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))

    start = (page - 1) * per_page
    end = start + per_page
    page_objects = tomo_objects[start:end]

    # [ДОБАВЛЕНО] Нумерация: старейший объект = №1, новейший = №total
    for i, obj in enumerate(page_objects):
        obj['number'] = total - (start + i)

    t2 = time.time()
    logging.info(f'[PROFILE] ИТОГО view_tomo_objects: {t2 - t0:.3f}s')
    return render_template('tomo_objects.html',
                           tomo_objects=page_objects,
                           page=page,
                           total_pages=total_pages,
                           total=total)


@app.route('/view/tomo_object/<to_id>')
def view_tomo_object(to_id):
    to = TomoObject()
    tomo_object = to.get(to_id)
    return render_template('tomo_object.html',
                           tomo_object_str=pprint.pformat(tomo_object),
                           tomo_object=tomo_object)


@app.route('/reconstruct/<to_id>')
def reconstruct(to_id):
    tomo_queue.put_object_rec_queue(to_id, 'reconstruct')
    return redirect('/view/tomo_object/' + to_id)


@app.route('/copyfiles/<to_id>')
def copyfiles(to_id):
    tomo_queue.put_object_rec_queue(to_id, 'copyfiles')
    return redirect('/view/tomo_object/' + to_id)


@app.route('/reset/<to_id>')
def reset(to_id):
    tomo_queue.set_object_status(to_id, 'canceled')
    return redirect('/view/tomo_object/' + to_id)


@app.route('/status/<int:n>')
def status(n):
    return render_template('status.html',
                           status_str='\n'.join([pprint.pformat(ti) for ti in tomo_queue.get_last_n(n)]))


if __name__ == '__main__':
    app.run(debug=True, host='10.0.7.153', port=5550)
