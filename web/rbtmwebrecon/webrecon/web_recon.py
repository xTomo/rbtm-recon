from flask import Flask, render_template, url_for, jsonify, request, redirect
from flask_restful import Resource, Api
import pprint
import tomo_queue

import storage_utils

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
    def get(self,to_id):
        return storage_utils.get_tomoobject_info(to_id, is_local_ip(request.remote_addr))

api.add_resource(TomoObject,'/tomo_object/<to_id>')

# @app.route('/ip')
# def get_ip():
#     return 'IP: ' + request.remote_addr + 'is local: ' + str(is_local_ip(request.remote_addr))

@app.route('/')
@app.route('/view/tomo_objects')
def view_tomo_objects():
    to_ids=TomoObjects()
    to = TomoObject()
    tomo_objects = [to.get(to_id) for to_id in to_ids.get()]
    tomo_objects.sort(key=lambda x:x['timestamp'], reverse=True)
    return render_template('tomo_objects.html',
                           tomo_objects=tomo_objects)


@app.route('/view/tomo_object/<to_id>')
def view_tomo_object(to_id):
    to=TomoObject()
    tomo_object = to.get(to_id)
    return render_template('tomo_object.html',
                       tomo_object_str=pprint.pformat(tomo_object),
                       tomo_object=tomo_object)

@app.route('/reconstruct/<to_id>')
def reconstruct(to_id):
    tomo_queue.put_object_rec_queue(to_id)
    return redirect('/view/tomo_object/'+to_id)

@app.route('/reset/<to_id>')
def reset(to_id):
    tomo_queue.set_object_status(to_id, 'canceled')
    return redirect('/view/tomo_object/'+to_id)

if __name__ == '__main__':
    app.run(debug=True, host='10.0.7.153', port=5550)
