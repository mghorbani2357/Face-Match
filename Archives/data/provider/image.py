import uuid

import cv2
import numpy as np

from Archives.face.dataset import Dataset
from Archives.face.detection import Detector


class ImageFeeder(Dataset):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.detector = Detector()

    def add_profile(self, image, metadata=None):
        """
        Add a profile to the dataset.

        Args:
            image (bytes): The image to be added.
            metadata (dict): The metadata associated with the image.
        """

        profile = {
            'image': image,
            'detected_faces': list(),
            'source': 'Image',
            'metadata': metadata if metadata is not None else dict(),
        }
        np_arr = np.frombuffer(profile['image'], np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        detected_faces = self.detector.detect_faces(img)

        for detected_face in detected_faces:
            starting_point = detected_face['box']['start_point']
            ending_point = detected_face['box']['end_point']
            face = img[starting_point[0]:ending_point[0], starting_point[1]:ending_point[1]]
            is_success, im_buf_arr = cv2.imencode('.jpg', face)
            profile['detected_faces'].append({
                'face': im_buf_arr.tobytes(),
                'identity': uuid.uuid4().__str__(),
                'confidence': detected_face.get('confidence', 0.0),
                'box': detected_face['box']
            })

        self.add_profiles([profile])
        self.save()
