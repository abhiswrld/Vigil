import cv2 as cv
import mediapipe as mp
import numpy as np

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

        height, width, _ = frame.shape
        cv.putText(frame, 'Vigil v0.1', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.rectangle(frame, (0, 0), (width, height), (0, 255, 0), 2)

        cv.imshow('Video', frame)

        if cv.waitKey(20) & 0xFF == ord('d'):
            break

capture.release()
cv.destroyAllWindows()