import json
import os


class FaceCountError(Exception):
    pass


class Dataset:
    def __init__(self, database_path):
        self.database_path = database_path

        if os.path.exists(database_path):
            with open(database_path, 'r') as database:
                self.dataset = json.load(database)
        else:
            self.dataset = {}
            self.save()

    def add_profile(self, profile):
        self.dataset[profile['id']] = {
            'username': profile['username'],
            'full_name': profile['full_name'],
            'profile_picture': profile['profile_pic'],
            'detected_faces': profile['detected_faces'],
            # 'is_verified ': profile['is_verified'],
            # 'followed_by_viewer': profile['followed_by_viewer'],
            # 'requested_by_viewer': profile['requested_by_viewer'],
        }

    def save(self):
        with open(self.database_path, 'w') as database:
            json.dump(self.dataset, database)

    def reload(self):
        with open(self.database_path, 'r') as database:
            self.dataset = json.load(database)

    def load(self, database_path):
        self.database_path = database_path
        with open(database_path, 'r') as database:
            self.dataset = json.load(database)

    def get_profile_by_id(self, profile_id):
        return self.dataset[profile_id]
