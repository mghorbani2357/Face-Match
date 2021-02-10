from FaceDetection.FaceDetector import Detector
import os
import json
import base64
import zlib


class DatasetBuilder:
    def __init__(self, database_path):
        self.database_path = database_path
        print('here')

        if not os.path.exists(database_path):
            with open(database_path, 'w') as database:
                json.dump({}, database)
                database.write('\n')

        self.database = open(self.database_path, 'wb')

        self.metadata = json.loads(self.database.readline().decode('utf-8'))

    def add_face(self, path):
        if os.path.exists(path):
            self.database.seek(-1, -1)
            begging_position = self.database.tell()
            with open(path, 'rb') as file:
                data = file.read()

                ending_position = begging_position + data.__len__()

                self.database.write(data)

            self.metadata.append({
                'id': self.metadata.__len__(),
                'begging_position': begging_position,
                'ending_position': ending_position
            })
