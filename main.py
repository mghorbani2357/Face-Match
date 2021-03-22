from face_detection import Detector
from face_verification.OneShotFaceVerification import Verifier
from data_source import CCTV
import cv2
from utils import *
import os
from dotenv import load_dotenv
from menu_pages import *
import json
from threading import Thread
import time
from multiprocessing import Process

detector = Detector()
verifier = Verifier('instagram.json')


def view_camera(camera_config):
    # cctvs = list()
    # for camera_config in camera_configs:
    ip_camera_url = f"{camera_config['protocol']}://{camera_config['ip']}:{camera_config['port']}/{camera_config['path']}"
    print(ip_camera_url)
    cctv = CCTV(ip_camera_url)

    # verifier = Verifier('instagram.json')

    for frame in cctv.start_streaming():
        for detected_face in detector.detect_faces(frame):
            # if real_face.verify(detected_face['face']):
            # if True:
            # identity = verifier.who_is_it(detector.align(detected_face['face']))

            frame = poi(frame, detected_face['box']['start_point'], detected_face['box']['end_point'], text='')
        # print(len(frames))

        cv2.imshow(f'Camera ({ip_camera_url})', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return


cv2.destroyAllWindows()

with open('config.json') as config_file:
    config = json.load(config_file)

while True:
    # view_camera(config['cameras'])
    camera_processes = list()
    for camera in config['cameras']:
        camera_processes.append(Process(target=view_camera, args=(camera,)))
        camera_processes[-1].start()

    for camera_process in camera_processes:
        camera_process.join()

    command = get_command(first_page_commands)
    if command == 'cctv':
        command = get_command(cctv_page_commands)
        if command == 'add':
            new_camera = dict()
            new_camera['protocol'] = input('protocol>')
            new_camera['ip'] = input('Protocol>')
            new_camera['port'] = input('port>')
            new_camera['path'] = input('path>')

            config['cameras'].append(new_camera)

            with open('config.json', 'w') as config_file:
                json.dump(config, config_file)

        elif command == 'view':
            pass
