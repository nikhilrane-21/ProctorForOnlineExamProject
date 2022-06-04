
from student.proctor.cheating_detector import detect_cheating_frame, segment_count, print_stats
from student.proctor.face_detector import detect_faces
from student.proctor.face_landmarks import detect_landmarks
from student.proctor.face_recognition import verify_faces
from student.proctor.models.Facenet512 import loadFaceNet512Model
from student.proctor.plot_graphs import plot_main, plot_segments
from student.proctor.utils import register_user, print_faces

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2


def run_proctor(request):
    font = cv2.FONT_HERSHEY_SIMPLEX
    frmodel = loadFaceNet512Model()
    input_embeddings, input_im_list = register_user(frmodel, request)
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('PROCTORING ON')
    frames = []
    while (True):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        faces = detect_faces(frame, confidence=0.7)
        detect_landmarks(frame, faces, det_conf=0.7, track_conf=0.7)
        verify_faces(faces, frmodel, input_embeddings)

        print_faces(frame, faces)

        cheat_temp = detect_cheating_frame(faces, frames)
        frames.append(cheat_temp)
        if cheat_temp.cheat > 0:
            cv2.putText(frame, "Cheating suspected", (15, 130), font, 0.5, (0, 0, 255), 2)

        cv2.imshow('PROCTORING ON', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # plot_main(frames, segment_time, fps_assumed)
    # segments = segment_count(frames, segment_time, fps_assumed)
    # print_stats(segments)
    # plot_segments(segments, segment_time, [])



