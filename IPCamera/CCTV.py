import imutils
from imutils.video import FPS
from imutils.video import VideoStream
import time


class CCTV:
    def __init__(self, ip_camera_url, *args, **kwargs):
        self.ip_camera_url = ip_camera_url
        self.streaming = False

    def start_streaming(self):
        self.streaming = True

        vs = VideoStream(src=self.ip_camera_url).start()
        time.sleep(2.0)

        # start the FPS throughput estimator
        fps = FPS().start()

        while self.streaming:
            # grab the frame from the threaded video stream
            frame = vs.read()

            # resize the frame to have a width of 600 pixels (while maintaining the aspect ratio),
            # and then grab the image dimensions
            frame = imutils.resize(frame, width=1080)
            fps.update()

            yield frame

        # stop the timer and display FPS information
        fps.stop()
        print("Elapsed time: {:.2f}".format(fps.elapsed()))
        print("Approx. FPS: {:.2f}".format(fps.fps()))

        # cleanup
        vs.stop()

    def stop_streaming(self):
        self.streaming = False
