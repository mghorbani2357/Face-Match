import time

from face_verification.OneShotFaceVerification import Verifier


class Tracker:
    recognized_faces = list()
    unrecognized_faces = list()

    def __init__(self, dataset_path, cache_duration=60):
        self.cache_duration = cache_duration
        self.verifier = Verifier(dataset_path)

    def __garbage_collector(self, face_set):
        for item in face_set:
            if time.time() - item['timestamp'] > self.cache_duration:
                face_set.remove(item)

        return face_set

    def __search_in(self, to_verify_face, verified_faces):
        identity = None

        faces = [verified_face['face'] for verified_face in verified_faces]
        index, dist = self.verifier.more_alike(to_verify_face, faces)

        if index is not None:
            verified_faces[index]['timestamp'] = time.time()
            identity = verified_faces[index]['id']

        return identity, self.__garbage_collector(verified_faces)

    def track_faces(self, faces):
        """

            Args:
                 faces(list):

            Returns:
                list:
        """
        tracked_faces = list()

        for face in faces:
            identity, self.recognized_faces = self.__search_in(face, self.recognized_faces)
            if identity is not None:
                tracked_faces.append(identity)
                continue

            identity, self.unrecognized_faces = self.__search_in(face, self.unrecognized_faces)

            if identity is not None:
                tracked_faces.append(identity)
                continue

            identity = self.verifier.who_is_it(face)

            tracked_faces.append(identity)
            if identity != '404':
                identity = len(self.unrecognized_faces)

            self.recognized_faces.append({
                'face': face,
                'id': identity,
                'timestamp': time.time()
            })
        return tracked_faces
