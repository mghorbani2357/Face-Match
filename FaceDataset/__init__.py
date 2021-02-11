import json


class FaceCountError(Exception):
    pass


class DatasetBuilder:
    def __init__(self, database_path):
        self.database_path = database_path

        with open(database_path, 'r') as database:
            self.dataset = json.load(database)

        # self.database = open(self.database_path, 'w')
        # faces = self.face_detector.detect_faces(image)
        #
        # if faces.__len__() is not 1:
        #     raise FaceCountError
        #
        # self.face_detector = Detector()

    def add_profile(self, profile):
        self.dataset[profile['id']] = {
            'username': profile['username'],
            'full_name': profile['full_name'],
            'profile_picture': profile['profile_pic'],
            'is_verified ': profile['is_verified'],
            'followed_by_viewer': profile['followed_by_viewer'],
            'requested_by_viewer': profile['requested_by_viewer'],
        }

    def save(self):
        with open(self.database_path, 'w') as database:
            json.dump(database, self.dataset)

    def reload(self):
        with open(self.database_path, 'r') as database:
            self.dataset = json.load(database)

    def load(self, database_path):
        self.database_path = database_path
        with open(database_path, 'r') as database:
            self.dataset = json.load(database)

    def get_profile_by_id(self, profile_id):
        return self.dataset[profile_id]
