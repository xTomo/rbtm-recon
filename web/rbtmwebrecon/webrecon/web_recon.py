import logging
import pprint
import time

from flask import Flask, render_template, request, redirect
from flask_restful import Resource, Api

import conf
import storage_utils
import tomo_queue

from datetime import datetime

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
api = Api(app)


@app.context_processor
def inject_config():
    """Передаёт переменные конфигурации во все шаблоны."""
    jupyter_server = getattr(conf, 'JUPYTER_SERVER', 'http://10.0.7.153:5551')
    return {'jupyter_server': jupyter_server}


# [ДОБАВЛЕНО] Фильтр для перевода Unix timestamp в читаемую дату
@app.template_filter('datetimeformat')
def datetimeformat(value):
    try:
        return datetime.fromtimestamp(float(value)).strftime('%Y-%m-%d %H:%M')
    except Exception:
        return value


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

    # [ДОБАВЛЕНО] Нумерация: старейший объект = №1, новейший = №total
    for i, obj in enumerate(tomo_objects):
        obj['number'] = total - i

    t2 = time.time()
    logging.info(f'[PROFILE] ИТОГО view_tomo_objects: {t2 - t0:.3f}s')
    return render_template('tomo_objects.html',
                           tomo_objects=tomo_objects,
                           total=total)


@app.route('/view/tomo_object/<to_id>')
def view_tomo_object(to_id):
    tomo_object = storage_utils.get_tomoobject_info(to_id, is_local_ip(request.remote_addr))
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
