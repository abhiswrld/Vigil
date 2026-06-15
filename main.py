import cv2 as cv
import mediapipe as mp
import numpy as np
import time
from scipy.spatial import distance as dist

# constants
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EAR_THRESHOLD = 0.25
FRAME_THRESHOLD = 20

def calculate_EAR(eye_points, frame_width, frame_height):
    points = []
    for lm in eye_points:
        x = int(lm.x * frame_width)
        y = int(lm.y * frame_height)
        points.append((x, y))
    
    A = dist.euclidean(points[1], points[5])  # p2 to p6
    B = dist.euclidean(points[2], points[4])  # p3 to p5

    C = dist.euclidean(points[0], points[3])  # p1 to p4
    
    ear = (A + B) / (2.0 * C)
    
    return ear

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_faces = 1
)

capture = cv.VideoCapture(1)
prev_time = 0
frame_counter = 0
alert_active = False

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        isTrue, frame = capture.read()

        if not isTrue:
            print("Can't receive frame. Exiting...")
            break

        rgb_frame = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        )

        result = landmarker.detect(rgb_frame)

        if result.face_landmarks:
            for face in result.face_landmarks:
                for landmark in face:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv.circle(frame, (x, y), 3, (0, 255, 0), -1)
                
                left_eye_points = [face[i] for i in LEFT_EYE]
                right_eye_points = [face[i] for i in RIGHT_EYE]
                calculated_ear_left = calculate_EAR(left_eye_points, frame.shape[1], frame.shape[0])
                calculated_ear_right = calculate_EAR(right_eye_points, frame.shape[1], frame.shape[0])
                ear = (calculated_ear_left + calculated_ear_right) / 2
                cv.putText(frame, f'EAR: {ear:.2f}', (20, 110), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if ear < EAR_THRESHOLD:
                    frame_counter += 1
                    if frame_counter >= FRAME_THRESHOLD:
                        alert_active = True
                else:
                    frame_counter = 0
                    alert_active = False

        curr_time = time.time()
        fps = 1/(curr_time - prev_time)
        prev_time = curr_time

        cv.putText(frame, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        height, width, _ = frame.shape
        cv.putText(frame, 'Vigil v0.1', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        border_color = (0, 0, 255) if alert_active else (0, 255, 0)
        cv.rectangle(frame, (0, 0), (width, height), border_color, 4)

        if alert_active:
            cv.putText(frame, 'DROWSINESS ALERT!', (width//2 - 200, height//2), 
                        cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        cv.imshow('Video', frame)

        if cv.waitKey(20) & 0xFF == ord('d'):
            break

capture.release()
cv.destroyAllWindows()