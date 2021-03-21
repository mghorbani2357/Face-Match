import cv2
import numpy as np
from PIL import Image


class Detector:
    image_height = 160
    image_width = 160

    def __init__(self):
        #  Load Face Detector
        deploy = "deploy.json"
        model_path = "res10_300x300_ssd_iter_140000.caffemodel"

        self.detector = cv2.dnn.readNetFromCaffe(deploy, model_path)

    def detect_faces(self, frame):
        (h, w) = frame.shape[:2]

        # construct a blob from the image
        image_blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize faces in the input image
        self.detector.setInput(image_blob)
        detections = self.detector.forward()

        detected = dict()

        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            if confidence > 0.5:
                detected[detected.__len__()] = {
                    'confidence': confidence,
                    'box': {
                        'start_point': (startX, startY), 'end_point': (endX, endY)
                    },
                    'face': frame[startY:endY, startX:endX]

                }

        return list(detected.values())

    def align(self, face):
        return np.array(
            Image.fromarray(face).resize((self.image_height, self.image_width), Image.BILINEAR)).astype(
            np.double)

    def align_all_face(self, frame):
        faces = []
        # boxes = [face['box'] for face in extracted_face]
        for face in self.detect_faces(frame):
            # cropped = frame[box['start_point'][0]:box['start_point'][1], box['end_point'][0]:box['end_point'][1], :]
            scaled = np.array(
                Image.fromarray(face).resize((self.image_height, self.image_width), Image.BILINEAR)).astype(
                np.double)
            faces.append(scaled)
        return faces

    def crop_detected_face(self, frame, box):
        frame = np.array(frame)
        cropped = frame[box['end_point'][0]:box['end_point'][1], box['start_point'][0]:box['start_point'][1], :]

        # cropped = Image.crop((box['start_point'][1], box['end_point'][1], box['start_point'][0], box['end_point'][0]))
        scaled = np.array(
            Image.fromarray(cropped).resize((self.image_height, self.image_width), Image.BILINEAR)).astype(
            np.double)
        return scaled
