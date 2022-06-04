import os
import time
import cv2

from face_detector import detect_faces
from face_landmarks import detect_landmarks
from face_recognition import verify_faces

curr_path = os.getcwd()

font = cv2.FONT_HERSHEY_SIMPLEX


def print_fps(frame, pTime):
    cTime = time.time()
    fps = 1 / (cTime - pTime[0])
    pTime[0] = cTime
    cv2.putText(frame, f"FPS : {int(fps)}", (15, 30), font, 0.5, (255, 0, 0), 2)


def print_faces(frame, faces):
    if not faces:
        return

    for face in faces:
        x, y, w, h = face.origbbox

        # Face detection
        cv2.rectangle(frame, face.origbbox, (0, 0, 153), 2)
        cv2.putText(frame, "c:" + str(round(face.confidence[0], 4)), (x + w + 5, y + 28), cv2.FONT_HERSHEY_PLAIN, 1,
                    (153, 0, 0), 1)
        cv2.putText(frame, str(face.id + 1), (x + w + 5, y + h), cv2.FONT_HERSHEY_PLAIN, 1, (153, 0, 0), 1)

        # Face recognition
        if face.name:
            cv2.putText(frame, face.name, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (204, 0, 0), 2)
            cv2.putText(frame, "d:" + str(round(face.distance, 4)), (x + w + 5, y + 46), cv2.FONT_HERSHEY_PLAIN, 1,
                        (153, 0, 0), 1)
            # cv2.imshow("Best Match-{}".format(face.id), input_im_list[face.best_index])

        # Face landmarks
        if face.landmarks:
            for (x1, y1, _) in face.landmarks[:468]:
                cv2.circle(frame, (x1, y1), 1, (0, 255, 0), -1)
            for (x1, y1, _) in face.landmarks[468:]:
                cv2.circle(frame, (x1, y1), 1, (0, 128, 255), -1)


def register_user(frmodel, input_dir):
    cam = cv2.VideoCapture(input_dir)
    cv2.namedWindow('Face registration')
    input_embeddings = []
    input_im_list = []

    while cam.isOpened():

        path = 'static/profile_pic/Student'
        ret, frame = cam.read()
        myList = os.listdir(path)
        for cl in myList:
            frame = cv2.imread(f'{path}/{cl}')
            break;

        if ret:
            cv2.putText(frame, 'Press r to capture image', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow('Face registration', frame)

            if cv2.waitKey(1) & 0xFF == ord('r'):
                # capturing image
                faces = detect_faces(frame, confidence=0.7)
                if not faces or len(faces) != 1:
                    print('No face detected or Multiple faces detected. Please try again.')
                    # cv2.putText(frame, 'No face detected or Multiple faces detected. Please try again.', (30,85),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)
                else:
                    detect_landmarks(frame, faces, det_conf=0.7, track_conf=0.7)
                    verify_faces(faces, frmodel)
                    face_img = faces[0].img
                    input_im_list.append(face_img)
                    input_embeddings.append(faces[0].embedding)
                    break
        else:
            # print("Camera not available, close any other apps using the webcam and try again")
            continue
    cam.release()
    cv2.destroyAllWindows()

    return input_embeddings, input_im_list


def convert_bbox(bbox, small_frame):  ## from relative bbox to css order
    real_h, real_w, c = small_frame.shape
    x, y, w, h = bbox
    y1 = 0 if y < 0 else y
    x1 = 0 if x < 0 else x
    y2 = real_h if y1 + h > real_h else y + h
    x2 = real_w if x1 + w > real_w else x + w
    return (y1, x2, y2, x1)
