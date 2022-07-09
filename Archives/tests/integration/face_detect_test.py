from unittest import TestCase
import requests
from base64 import b64encode, b64decode
import json


class TestFaceDetect(TestCase):
    def test_single_face(self):
        with open('../resources/single_face.jpeg', 'rb') as image_file:
            image = image_file.read()
            
        data = {
            'image': b64encode(image).decode()
        }
        print(json.dumps(data))

        response = requests.post('http://localhost:9000/', data=json.dumps(data))

        print(json.dumps(json.loads(response.content.decode())))


    # def test_multiple_faces(self):
    #     with open('../resources/images.jpeg', 'rb') as image_file:

