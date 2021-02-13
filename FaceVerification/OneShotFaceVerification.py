import os
from base64 import b64encode, b64decode
import cv2
import numpy as np
from FaceDataset import Dataset

import FaceVerification.FaceToolKit as ftk


class Verifier:
    verification_threshold = 1.188
    image_size = 160

    def __init__(self, dataset_path):
        self.verifier = ftk.Verification()
        self.verifier.load_model("./models/20180204-160909/")
        self.verifier.initial_input_output_tensors()
        self.dataset = Dataset(dataset_path)

    def img_to_encoding(self, aligned_image):
        return self.verifier.img_to_encoding(aligned_image, self.image_size)

    @staticmethod
    def distance(emb1, emb2):
        diff = np.subtract(emb1, emb2)
        return np.sum(np.square(diff))

    def verify(self, pendent_identity, defined_identity):

        dist = self.distance(pendent_identity, defined_identity)

        if dist < self.verification_threshold:
            return True, dist
        else:
            return False, dist

    def who_is_it(self, face):
        identity = '404'

        encoded_face = self.img_to_encoding(face)

        min_dist = 1000
        for profile in self.dataset.dataset.values():

            nparr = np.frombuffer(b64decode(profile['profile_picture'].encode()), np.uint8)

            dataset_face = cv2.imencode(nparr, cv2.IMREAD_UNCHANGED)

            encoded_dataset_face = self.img_to_encoding(dataset_face)

            verified, dist = self.verify(encoded_face, encoded_dataset_face)

            if dist < min_dist:
                min_dist = dist
                identity = profile['full_name']
                # identity = os.path.splitext(os.path.basename(image_path))[0]

        if min_dist < self.verification_threshold:
            return identity
        else:
            return '404'
