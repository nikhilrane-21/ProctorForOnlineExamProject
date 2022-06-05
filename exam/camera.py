from imutils.video import VideoStream
import imutils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
#######################################

from student.proctor.cheating_detector import detect_cheating_frame, segment_count, print_stats
from student.proctor.face_detector import detect_faces
from student.proctor.face_landmarks import detect_landmarks
from student.proctor.face_recognition import verify_faces
from student.proctor.models.Facenet512 import loadFaceNet512Model
from student.proctor.plot_graphs import plot_main, plot_segments
from student.proctor.utils import register_user, print_faces

class Cam_detect(object):
    def __init__(self, request):
        self.vs = VideoStream(src=0).start()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.frmodel = loadFaceNet512Model()
        self.input_embeddings, self.input_im_list = register_user(self.frmodel, request)
        self.frames = []

    def __del__(self):
        print("destructor")
        i=0
        while (i<5):
            try:
                self.vs.stop()
                self.vs = VideoStream(src=0).stop()
                i+=1
            except:
                break
        self.fps_assumed = 5
        self.segment_time = 5
        plot_main(self.frames, self.segment_time, self.fps_assumed)
        segments = segment_count(self.frames, self.segment_time, self.fps_assumed)
        print_stats(segments)
        plot_segments(segments, self.segment_time, [])


    def get_frame(self, request):

        while (True):
            self.cap = self.vs.read()
            self.cap = imutils.resize(self.cap, width=650)
            self.cap = cv2.flip(self.cap, 1)

            faces = detect_faces(self.cap, confidence=0.7)
            detect_landmarks(self.cap, faces, det_conf=0.7, track_conf=0.7)
            verify_faces(faces, self.frmodel, self.input_embeddings)

            print_faces(self.cap, faces)

            cheat_temp = detect_cheating_frame(faces, self.frames)
            self.frames.append(cheat_temp)
            if cheat_temp.cheat > 0:
                cv2.putText(self.cap, "Cheating suspected", (15, 130), self.font, 0.5, (0, 0, 255), 2)
            ret, jpeg = cv2.imencode('.jpg', self.cap)
            return jpeg.tobytes()

        # self.vs.release()
        # cv2.destroyAllWindows()

