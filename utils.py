import cv2
import os
from dotenv import load_dotenv

load_dotenv()


def poi(frame, starting_point, ending_point, padding=0.7, text=''):
    l = ending_point[0] - starting_point[0]
    h = ending_point[1] - starting_point[1]

    square_length = max(l, h)
    square_length += square_length * padding
    square_length += square_length % 22

    sc = (int((ending_point[0] + starting_point[0]) / 2), int((ending_point[1] + starting_point[1]) / 2))
    r = int(square_length / 2 / 2 ** (1 / 2)) + 10

    starting_point = (sc[0] - r, sc[1] - r)
    ending_point = (sc[0] + r, sc[1] + r)

    box = frame[starting_point[1]:ending_point[1], starting_point[0]:ending_point[0]]

    line_length = (ending_point[0] - starting_point[0]) / 22

    thickness_1_positions = [2, 4, 6, 8, 13, 15, 17, 19]
    thickness_2_positions = [10, 11]
    thickness_3_positions = [0, 21]

    for rotation in range(4):
        thickness = 1
        for position in thickness_1_positions:
            pt1 = (int(position * line_length), 2)
            pt2 = (int((position + 1) * line_length), 2)

            cv2.line(box, pt1, pt2, (255, 255, 255), thickness)

        # for position in thickness_2_positions:
        #     pt1 = (int(position * line_length), 0)
        #     pt2 = (int((position + 1) * line_length), 0)
        #
        #     cv2.line(box, tuple(pt1), tuple(pt2), (0, 255, 255), 2)

        pt1 = (int(11 * line_length), 2)
        pt2 = (int(11 * line_length), int(line_length))

        cv2.line(box, tuple(pt1), tuple(pt2), (0, 0, 200), 2)

        for position in thickness_3_positions:
            pt1 = (int(position * line_length), 2)
            pt2 = (int((position + 1) * line_length), 2)

            cv2.line(box, tuple(pt1), tuple(pt2), (0, 0, 200), 4)

        box = cv2.rotate(box, cv2.ROTATE_90_CLOCKWISE)

    # text = 'XXX-XXX-6354'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    font_thickness = 2
    text_color = (255, 255, 255)
    text_color = (0, 100, 200)
    text_background_color = (0, 0, 0)
    text_size = list(cv2.getTextSize(text, font, font_scale, font_thickness)[0])

    text_position = (
        int((starting_point[0] + ending_point[0] - text_size[0]) / 2), starting_point[1] - text_size[1] - 10)

    # text_size[0] += 14
    # text_size[1] += 10

    # rectangle_starting = (text_position[0] - 14, text_position[1] - int(text_size[1] )-7 )
    # rectangle_ending = (text_position[0] + int(text_size[0] / 1)+7, text_position[1] + int(text_size[1] / 1)+2)

    # cv2.rectangle(frame, rectangle_starting, rectangle_ending, text_background_color, -1)

    cv2.putText(frame, text, text_position, font, font_scale, text_color, font_thickness)

    if box is None:
        return frame
    else:
        frame[starting_point[1]:ending_point[1], starting_point[0]:ending_point[0]] = box
        return frame


def ip_camera_url_builder():
    return f"{os.getenv('IP_CAMERA_PROTOCOL')}://{os.getenv('IP_CAMERA_IP')}:{os.getenv('IP_CAMERA_PORTL')}/{os.getenv('IP_CAMERA_PATH')}"
