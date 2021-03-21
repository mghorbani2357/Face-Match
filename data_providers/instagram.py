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

from utils import poi


class InstaFeeder:
    L = instaloader.Instaloader()
    sp = yaspin(text="Downloading images", color="cyan")

    def __init__(self, dataset_path, username, password=None, session_path=None):
        """

        Args:
              dataset_path(str)
              auth(tuple)
              session_path(str)
        """
        self.dataset = Dataset(dataset_path)
        self.face_detector = Detector()
        if session_path is not None:
            self.load_session(username, session_path)

        elif password is not None:
            self.new_session(username, password)

    def new_session(self, username, password):
        self.L.login(username, password)

    def save_session(self, path):
        self.L.save_session_to_file(path)

    def load_session(self, username, path):
        self.L.load_session_from_file(username, path)

    async def get_profile(self, profile):
        print(f"Downloading {profile.username} ")
        # response = await asyncio.get_event_loop().run_in_executor(None, requests.get, profile.profile_pic_url)
        response = requests.get(profile.profile_pic_url, allow_redirects=True)
        np_arr = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        json_profile = {
            'id': profile.userid,
            'image': b64encode(response.content).decode(),
            'detected_faces': [],
            'username': profile.username,
            'full_name': profile.full_name,
        }
        # detected_faces = await asyncio.get_event_loop().run_in_executor(None, self.face_detector.detect_faces, img)
        detected_faces = detected_faces = self.face_detector.detect_faces(img)

        for detected_face in detected_faces:
            starting_point = detected_face['box']['start_point']
            ending_point = detected_face['box']['end_point']

            # if (ending_point[0] - starting_point[0]) > 0 and (ending_point[1] - starting_point[1]) > 0:
            try:
                face = img[starting_point[0]:ending_point[0], starting_point[1]:ending_point[1]]
                is_success, im_buf_arr = cv2.imencode('.jpg', face)
                json_profile['detected_faces'].append(b64encode(im_buf_arr).decode())
                frame = poi(img, detected_face['box']['start_point'], detected_face['box']['end_point'], text='404')

            except:

                print(len(detected_face), profile.username, detected_face)

        cv2.imshow("Frame", frame)

        cv2.waitKey(1) & 0xFF

        done_task_count = len([task for task in asyncio.Task.all_tasks() if task.done()])
        all_tasks_count = len(asyncio.Task.all_tasks())
        print(f"{profile.username} downloaded  ({done_task_count}/{all_tasks_count})")

        return json_profile

    @staticmethod
    async def __print_progress():
        with yaspin(text="Downloading images", color="cyan") as sp:
            while True:
                done_task_count = len([task for task in asyncio.Task.all_tasks() if task.done()])
                all_tasks_count = len(asyncio.Task.all_tasks())
                sp.write(f"{done_task_count}/{all_tasks_count}")
                if all_tasks_count - 1 == done_task_count:
                    break
                else:
                    await asyncio.sleep(0.1)

            sp.ok("Profiles downloaded.")

    def get_followers_profile_from_user(self, user):
        profile = Profile.from_username(self.L.context, user)
        followings = profile.get_followees()

        tasks = list()
        # tasks.append(self.__print_progress())
        print('here')
        for following in followings:
            if len(tasks) > 10:
                break
            print(f'\r{len(tasks) - 1}/{followings.count}', end='')
            tasks.append(self.get_profile(following))
        print()

        profiles = asyncio.get_event_loop().run_until_complete(asyncio.gather(*tasks))

        # profiles.remove(None)

        self.dataset.add_profiles(profiles)

        self.dataset.save()
