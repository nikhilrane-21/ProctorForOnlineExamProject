from PIL.Image import fromarray
from imutils.video import VideoStream
import imutils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
#######################################

from student.proctor.cheating_detector import detect_cheating_frame, segment_count, print_stats
from student.proctor.face_detector import detect_faces
from student.proctor.face_landmarks import detect_landmarks
from student.proctor.face_recognition import verify_faces
from student.proctor.models.Facenet512 import loadFaceNet512Model
from student.proctor.plot_graphs import plot_main, plot_segments
from student.proctor.utils import register_user, print_faces

class Cam_detect(object):
    def __init__(self):
        self.vs = VideoStream(src=0).start()

    def __del__(self):
        self.vs.stop()
        # cap.release()
        cv2.destroyAllWindows()

    def get_frame(self, request):

        font = cv2.FONT_HERSHEY_SIMPLEX
        frmodel = loadFaceNet512Model()
        input_embeddings, input_im_list = register_user(frmodel, request)
        frames = []

        while (True):
            cap = self.vs.read()
            cap = imutils.resize(cap, width=650)
            cap = cv2.flip(cap, 1)
            faces = detect_faces(cap, confidence=0.7)
            detect_landmarks(cap, faces, det_conf=0.7, track_conf=0.7)
            verify_faces(faces, frmodel, input_embeddings)

            print_faces(cap, faces)

            cheat_temp = detect_cheating_frame(faces, frames)
            frames.append(cheat_temp)
            if cheat_temp.cheat > 0:
                cv2.putText(cap, "Cheating suspected", (15, 130), font, 0.5, (0, 0, 255), 2)

            if cv2.waitKey(1) & 0xFF == 27:
                break
        ret, jpeg = cv2.imencode('.jpg', cap)
        return jpeg.tobytes()
        cap.release()
        cv2.destroyAllWindows()

