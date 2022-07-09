
import json
from base64 import b64decode

import cv2
import numpy as np
from flask import Flask, request, jsonify

from Archives.data.provider.image import ImageFeeder
from Archives.face.detection import Detector
from Archives.face.verification import Verifier

dataset = ImageFeeder("nutrica.pickle")
detector = Detector()
verifier = Verifier(dataset)

app = Flask(__name__)


@app.route('/insert_image', methods=['PUT'])
def insert_image():
    body = json.loads(request.data.decode())
    image = b64decode(body.get('image').encode())
    dataset.add_profile(image, body.get("metadata", {}))
    return jsonify({
        'status': 'ok',
    })


@app.route('/similar_faces', methods=['GET', 'POST'])
def get_similar_faces():
    body = json.loads(request.data.decode())
    image = b64decode(body.get('image').encode())
    np_arr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    similar_faces = list()

    for detected_face in detector.detect_faces(img):
        starting_point = detected_face['box']['start_point']
        ending_point = detected_face['box']['end_point']
        face = img[starting_point[0]:ending_point[0], starting_point[1]:ending_point[1]]
        similar_faces.append({
            'face_position': detected_face['box'],
            'similar_faces': verifier.get_similar_faces(face)
        })

    return jsonify(similar_faces)


@app.route('/detect_faces', methods=['POST'])
def detect_faces():
    body = json.loads(request.data.decode())
    image = b64decode(body.get('image').encode())
    np_arr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

    identities = list()

    for detected_face in detector.detect_faces(img):
        identities.append({
            'face_position': detected_face['box'],
        })

    return jsonify({identities})


if __name__ == '__main__':
    app.run('0.0.0.0', 9000, debug=True, threaded=True, use_reloader=False)


# import json
# from base64 import b64decode
#
# import cv2
# import numpy as np
# from flask import Flask, jsonify, request
#
# from face.detection import Detector
# from face.verification import Verifier
#
# detector = Detector()
# verifier = Verifier('instagram.json')
#
# app = Flask(__name__)
#
#
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     body = json.loads(request.data.decode())
#     image = b64decode(body.get('image').encode())
#
#     np_arr = np.frombuffer(image, np.uint8)
#
#     img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
#
#     identities = list()
#
#     for detected_face in detector.detect_faces(img):
#         identities.append({
#             'face_position': detected_face['box'],
#             'similar_identities': [verifier.who_is_it(detected_face)],
#         })
#
#     return jsonify({
#         'image_size': list(img.shape),
#         'identities': identities,
#     })
