import json
from base64 import b64encode
from unittest import TestCase

from Archives.app.interfaces.rest import app


class TestMainFunctionality(TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_insert_image_single_face(self):
        image = open('tests/resources/single_face.jpeg', 'rb').read()
        response = self.app.put('/insert_image', data=json.dumps({
            'image': b64encode(image).decode(),
            'metadata': {"name": "Guy"}
        }))

        self.assertEqual(response.status_code, 200)

    def test_insert_image_multiple_faces(self):
        image = open('tests/resources/multiple_faces.jpeg', 'rb').read()
        response = self.app.put('/insert_image', data= json.dumps({
            'image': b64encode(image).decode(),
            'metadata': {"name": "Multiple Guy"}
        }))

        self.assertEqual(response.status_code, 200)

    def test_similar_faces(self):
        image = open('tests/resources/multiple_faces.jpeg', 'rb').read()

        response = self.app.post('/similar_faces', data=json.dumps({
            'image': b64encode(image).decode(),
        }))

        self.assertEqual(response.status_code, 200)
