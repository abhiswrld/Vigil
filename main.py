import cv2 as cv

capture = cv.VideoCapture(1)

while True:
    isTrue, frame = capture.read()

    if not isTrue:
        print("Can't receive frame from video stream. (Possibly the video has ended.) Exiting ...")
        break

    height, width, _ = frame.shape
    cv.putText(frame, 'Vigil v0.1', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.rectangle(frame, (0, 0), (width, height), (0, 255, 0), 2)

    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()