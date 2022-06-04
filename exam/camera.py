from imutils.video import VideoStream
import imutils
import cv2

class Cam_detect(object):
    def __init__(self):
        self.vs = VideoStream(src=0).start()

    def __del__(self):
        self.vs.stop()

    def get_frame(self):
        frame = self.vs.read()
        frame = imutils.resize(frame, width=650)
        frame = cv2.flip(frame, 1)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
