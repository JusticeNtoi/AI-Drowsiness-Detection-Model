import numpy as np
from keras.models import load_model
import cv2

# Dlib for deep learning based modules and face landmark detection
import dlib

# For importing sound player
from pygame import mixer

# Loading the model to predict eyes open and eyes closed
model = load_model('models/InceptionV3_2_model.h5')

# haarcascade to detect face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Loading a video to test the Detection with
video = 'Test_Videos/TestVideo1.webm'

# Loading the alarm audio
mixer.init()
sound = mixer.Sound(r'alarm/mixkit-alarm-990.wav')

# Capturing the video / Loading the video
cap = cv2.VideoCapture(video)

counter = 0

while True:
    # reading the frames from the video stream
    ret, frame = cap.read()
    height, width = frame.shape[0:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Draw a rectangle
    x1, y1, w1, h1 = 0, 0, 175, 75
    cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)

    # for detecting faces and eyes
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(frame, pt1=(fx, fy), pt2=(fx+fw, fy+fh), color=(255, 0, 0), thickness=2)

        # Preprocessing: crop the face image
        roi_face_gray = gray[fy: fy + fh, fx: fx + fw]
        roi_face = frame[fy: fy + fh, fx: fx + fw]

        # For Drawing the Rectangles on the eyes
        drawEyes = eye_cascade.detectMultiScale(roi_face_gray)
        for (x, y, w, h) in drawEyes:
            cv2.rectangle(roi_face, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

    eyes = eye_cascade.detectMultiScale(gray)

    for (x, y, w, h) in eyes:
        # cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

        # Preprocessing: cropping the eye image
        # ROI Region Of Interest
        roi_eyes_gray = gray[y: y + h, x: x + w]
        roi_eyes_color = frame[y: y + h, x: x + w]

        font = cv2.FONT_HERSHEY_SIMPLEX

        eye = cv2.resize(roi_eyes_color, (80, 80))
        eye = eye / 255  # Rescaling the image pixel to a normalized range 0, 1
        eye = eye.reshape(80, 80, 3)  # Adding color channels
        eye = np.expand_dims(eye, axis=0)  # Adding fourth dimension

        # Predictions
        prediction = model.predict(eye)
        print(prediction)

        if prediction > 0.5:
            status = "Eyes Open"
            # Inserting text into the video
            cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), font,
                        fontScale=0.8,
                        color=(0, 255, 0),
                        thickness=2)

        else:
            status = "Eyes Closed"

            counter = counter + 1

            if counter > 10:
                # Inserting text into the video
                cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), font,
                            fontScale=0.8,
                            color=(0, 0, 255),
                            thickness=2)
                try:
                    sound.play()
                except Exception:
                    pass
                counter = 0

    cv2.imshow('Drowsiness Detection', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
