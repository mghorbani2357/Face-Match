from face_detection.FaceDetector import Detector
from face_verification.OneShotFaceVerification import Verifier
from ip_camera.CCTV import CCTV
import cv2
from utils import poi
import os
from dotenv import load_dotenv

load_dotenv()

ip_camera_url = 'http://192.168.1.5:8080/video'

cctv1 = CCTV(ip_camera_url)

detector = Detector()

verifier = Verifier('instagram.json')
for frame in cctv1.start_streaming():
    for detected_face in detector.detect_faces(frame):
        # if real_face.verify(detected_face['face']):
        # if True:
        identity = verifier.who_is_it(detector.align(detected_face['face']))

        frame = poi(frame, detected_face['box']['start_point'], detected_face['box']['end_point'], text=identity)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        cctv1.stop_streaming()

cv2.destroyAllWindows()
