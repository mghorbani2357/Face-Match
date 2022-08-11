import numpy as np
from cv2 import cv2
from mtcnn.mtcnn import MTCNN
from skimage import transform


class MTCNNFaceDetector:
    __detector = MTCNN()

    def detect_faces(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for detected_face in self.__detector.detect_faces(image_rgb):
            x, y, w, h = detected_face["box"]
            yield self.align_face_with_landmarks(image[int(y):int(y + h), int(x):int(x + w)], detected_face["keypoints"])

    @staticmethod
    def align_face_with_landmarks(image, landmarks):
        src = np.array([
            [54.70657349, 73.85186005],
            [105.04542542, 73.57342529],
            [80.03600311, 102.48085785],
            [59.35614395, 131.95071411],
            [101.04272461, 131.72013855]], dtype=np.float32)
        dst = landmarks.astype(np.float32)
        transformer = transform.SimilarityTransform()
        transformer.estimate(dst, src)
        wrapped = cv2.warpAffine(image, transformer.params[0:2, :], (160, 160), borderValue=0.0)
        return wrapped
