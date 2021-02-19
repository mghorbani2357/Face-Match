import instaloader.instaloader
from instaloader import Profile
from FaceDataset import Dataset
from FaceDetection.FaceDetector import Detector
import requests
from base64 import b64encode, b64decode
import cv2
import numpy as np


class InstaFeeder:
    L = instaloader.Instaloader()
    L.load_session_from_file('feeder347',
                             '/home/bluesp/WorkSpaces/PyCharmProjects/AI/Single-Shot-Face-Recognition/feeder347')

    def __init__(self):
        self.dataset = Dataset('instagram.json')
        self.face_detector = Detector()

    def get_profiles_from_user(self, user):
        profile = Profile.from_username(self.L.context, user)

        print('getting profiles')
        i = 0
        followings = profile.get_followees()
        for following in followings:
            i += 1
            print(i)
            # print(f'{i}/{Profile.followees()}')
            response = requests.get(following.profile_pic_url,allow_redirects=True)
            np_arr = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            detected_faces = self.face_detector.detect_faces(img)

            if detected_faces.__len__() != 1:
                print('No face found.')
                continue

                # raise FaceCountError

            starting_point = detected_faces[0]['box']['start_point']
            ending_point = detected_faces[0]['box']['end_point']
            face = img[starting_point[0]:ending_point[0], starting_point[1]:ending_point[1]]
            is_success, im_buf_arr = cv2.imencode('.jpg', face)

            json_profile = {
                'id': following.userid,
                'username': following.username,
                'full_name': following.full_name,
                'profile_pic': b64encode(im_buf_arr.tobytes()).decode(),
                # 'is_verified ': following['is_verified'],
                # 'followed_by_viewer': following['followed_by_viewer'],
                # 'requested_by_viewer': following['requested_by_viewer']

            }

            self.dataset.add_profile(json_profile)

        self.dataset.save()


if __name__ == "__main__":
    feeder = InstaFeeder()
    feeder.get_profiles_from_user('m.ghorbani2357')
