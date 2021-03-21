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

    def add_profiles(self, profiles):
        for profile in profiles:
            self.dataset[profile['id']] = {
                'image': profile['profile_picture'],
                'detected_faces': profile['detected_faces'],
                'metadata': {
                    'username': profile['username'],
                    'full_name': profile['full_name'],
                }
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
