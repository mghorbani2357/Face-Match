from FaceDetection.FaceDetector import Detector
import os
import json
import base64
import zlib


class DatasetBuilder:
    def __init__(self, database_path):
        self.database_path = database_path

        self.database = open(self.database_path, 'wb')

        self.database.read()

