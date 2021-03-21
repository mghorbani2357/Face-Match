import instaloader.instaloader
from instaloader import Profile
from face_dataset import Dataset
from face_detection import Detector
import requests
from base64 import b64encode
import cv2
import numpy as np
import yaspin
import asyncio
from yaspin import yaspin
import time


class InstaFeeder:
    L = instaloader.Instaloader()

    def __init__(self, dataset_path, auth=None, session_path=None):
        """

        Args:
              dataset_path(str)
              auth(tuple)
              session_path(str)
        """
        self.dataset = Dataset(dataset_path)
        self.face_detector = Detector()
        if session_path is not None:
            self.load_session(session_path)

        elif auth is not None:
            self.new_session(auth[0], auth[1])

    def new_session(self, username, password):
        self.L.login(username, password)

    def save_session(self, path):
        self.L.save_session_to_file(path)

    def load_session(self, path):
        self.L.load_session_from_file('', 'path')

    async def get_profile(self, profile):
        response = requests.get(profile.profile_pic_url, allow_redirects=True)
        np_arr = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        detected_faces = self.face_detector.detect_faces(img)

        json_profile = {
            'image': b64encode(cv2.imencode('.jpg', img)).decode(),
            'detected_faces': [],
            'meta': {
                'id': profile.userid,
                'username': profile.username,
                'full_name': profile.full_name,
            }

        }

        for face in detected_faces:
            starting_point = face[0]['box']['start_point']
            ending_point = face[0]['box']['end_point']
            face = img[starting_point[0]:ending_point[0], starting_point[1]:ending_point[1]]
            is_success, im_buf_arr = cv2.imencode('.jpg', face)
            json_profile['detected_faces'].append(b64encode(im_buf_arr).decode())

        return json_profile

    @staticmethod
    async def __print_progress():
        with yaspin(text="Downloading images", color="cyan") as sp:
            while True:
                done_task_count = len([task for task in asyncio.Task.all_tasks() if not task.done()])
                all_tasks_count = len(asyncio.Task.all_tasks())
                sp.write(f"{done_task_count}/{all_tasks_count}")

                if all_tasks_count == done_task_count:
                    break
                else:
                    time.sleep(0.1)

            sp.ok("Profiles downloaded.")

    def get_followers_profile_from_user(self, user):
        profile = Profile.from_username(self.L.context, user)
        followings = profile.get_followees()

        tasks = list()
        tasks.append(self.__print_progress())
        for following in followings:
            tasks.append(self.get_profile(following))

        profiles = list(asyncio.gather(*tasks))

        profiles.remove(None)

        self.dataset.add_profiles(profiles)

        self.dataset.save()


if __name__ == "__main__":
    feeder = InstaFeeder('insta1.json', session_path='feeder347')
    feeder.get_followers_profile_from_user('m.ghorbani2357')
