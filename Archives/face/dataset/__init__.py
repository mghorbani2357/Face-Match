import os
import pickle
import uuid

import cv2
import numpy as np


class FaceCountError(Exception):
    pass


class Dataset:
    def __init__(self, database_path):
        self.database_path = database_path

        if os.path.exists(database_path):
            self.data = pickle.load(open(database_path, 'rb'))
        else:
            self.data = {}
            self.save()

    def add_profiles(self, profiles):
        for profile in profiles:
            if profile is None:
                continue

            self.data[uuid.uuid4().__str__()] = profile

    def save(self):
        pickle.dump(self.data, open(self.database_path, 'wb'))

    def reload(self):
        self.data = pickle.load(open(self.database_path, 'rb'))

    def load(self, database_path):
        self.database_path = database_path
        self.reload()

    def get_profile_by_id(self, profile_id):
        return self.data[profile_id]

    def view_profile(self, profile_id):
        np_arr = np.frombuffer(self.data[profile_id]['image'], np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        cv2.imshow(f'Camera ({profile_id})', img)
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
