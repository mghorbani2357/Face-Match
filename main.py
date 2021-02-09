from FaceDetection.FaceDetector import Detector
# from FaceVerification.OneShotFaceVerification import Verifier
from IPCamera.CCTV import CCTV
import os
import cv2

ip_camera_url = 'http://192.168.1.3:8080/video'

cctv1 = CCTV(ip_camera_url)

detector = Detector()

for frame in cctv1.start_streaming():
    for detected_face in detector.detect_faces(frame):
        # if real_face.verify(detected_face['face']):
        # if True:
        # identity = verifier.who_is_it(detector.align(detected_face['face']))
        # print(identity)

        cv2.rectangle(frame,
                      detected_face['box']['start_point'],
                      detected_face['box']['end_point'],
                      (0, 155, 255),
                      1)

        # cv2.putText(frame, identity, (detected_face['box']['start_point'][0], detected_face['box']['start_point'][1]),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        cctv1.stop_streaming()

cv2.destroyAllWindows()
