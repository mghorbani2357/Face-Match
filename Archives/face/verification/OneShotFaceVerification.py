from base64 import b64decode

import cv2
import numpy as np

import Archives.face.verification.FaceToolKit as ftk


class Verifier:
    verification_threshold = 1.188
    image_size = 160

    def __init__(self, dataset=None):
        """
            Verifier constructor

            Args:
                dataset(Dataset): Face dataset

        """
        self.verifier = ftk.Verification()
        self.verifier.load_model("face/verification/models/20180204-160909/")
        self.verifier.initial_input_output_tensors()

        self.dataset = dataset

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

    def more_alike(self, checking_face, faces):
        """
            Args:
                 faces(list)
                 checking_face(np.array):
        """

        dist = min_dist = 1000
        similar_face_index = None

        for face in faces:
            verified, dist = self.verify(checking_face, face)

            if dist < min_dist:
                min_dist = dist
                if min_dist < self.verification_threshold:
                    similar_face_index = faces.index(face)

        return similar_face_index, dist

    def get_similar_faces(self, face):
        """

            Args:
                face(np.array):

            Return:
                list

        """
        similar_faces = list()
        encoded_face = self.img_to_encoding(face)

        for uid, profile in self.dataset.data.items():
            for detected_face in profile['detected_faces']:
                nparr = np.frombuffer(b64decode(detected_face.get('face')), np.uint8)
                dataset_face = cv2.imencode(nparr, cv2.IMREAD_UNCHANGED)
                encoded_dataset_face = self.img_to_encoding(dataset_face)
                verified, dist = self.verify(face, encoded_dataset_face)

                if dist < self.verification_threshold:
                    similar_faces.append({
                        "profile": profile.get('metadata'),
                        "dist": dist,
                    })

                break

        return similar_faces
