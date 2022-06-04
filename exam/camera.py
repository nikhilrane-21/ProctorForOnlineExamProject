from imutils.video import VideoStream
import imutils
import cv2

class Cam_detect(object):
    def __init__(self):
        self.vs = VideoStream(src=0).start()

    def __del__(self):
        cv2.destroyAllWindows()

    def detect_and_predict_mask(self, frame, faceNet):
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        locs = []
        preds = []
        return (locs, preds)

    def get_frame(self):
        frame = self.vs.read()
        frame = imutils.resize(frame, width=450)
        frame = cv2.flip(frame, 1)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
