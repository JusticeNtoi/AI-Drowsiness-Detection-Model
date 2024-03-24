import numpy as np
from keras.models import load_model
import cv2

# Dlib for deep learning based modules and face landmark detection
import dlib

# face_utils for basic operations of conversion
from imutils import face_utils

# For importing sound player
from pygame import mixer

# Loading the model to predict eyes open and eyes closed
model = load_model('models/InceptionV3_2_model.h5')

# Dlib face detection
detector = dlib.get_frontal_face_detector()

# predict where the eyes are located/ predict eyes face landmarks(face, eyes, nose, mouth)
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Loading a video to test the Detection with
video = 'Test_Videos/TestVideo1.webm'
# video = 'Test_Videos/TestVideo2.mp4'

# Loading the alarm audio
mixer.init()
sound = mixer.Sound(r'alarm/mixkit-alarm-990.wav')


def get_eye_landmarks(shape):
    # Define the landmark indices for the left and right eyes
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # Extract the landmarks for the left and right eyes
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    return leftEye, rightEye


# Capturing the video / Loading the video
cap = cv2.VideoCapture(video)

# check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open WebCam")

counter = 0

while True:
    # reading the frames from the video stream
    ret, frame = cap.read()
    height, width = frame.shape[0:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Draw a rectangle
    x1, y1, w1, h1 = 0, 0, 175, 75
    # cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # creating a rectangle defined by four floating point numbers from the frame.
    rects = detector(gray, 0)

    for rect in rects:
        # Getting the coordinates of the face rectangle
        (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
        cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye, rightEye = get_eye_landmarks(shape)

        # Define the size of the square region
        square_size = 40  # Adjust the size as needed

        # Extract the central point of the left and right eyes
        left_eye_center = (int((leftEye[0, 0] + leftEye[3, 0]) / 2), int((leftEye[1, 1] + leftEye[4, 1]) / 2))
        right_eye_center = (int((rightEye[0, 0] + rightEye[3, 0]) / 2), int((rightEye[1, 1] + rightEye[4, 1]) / 2))

        # Regions of interest (ROIs) for the left and right eyes
        left_eye_roi = frame[
                       left_eye_center[1] - square_size // 2: left_eye_center[1] + square_size // 2,
                       left_eye_center[0] - square_size // 2: left_eye_center[0] + square_size // 2,
                       ]

        right_eye_roi = frame[
                        right_eye_center[1] - square_size // 2: right_eye_center[1] + square_size // 2,
                        right_eye_center[0] - square_size // 2: right_eye_center[0] + square_size // 2,
                        ]

        # # Draw rectangles around the left and right eyes
        cv2.rectangle(frame, pt1=(left_eye_center[0] - square_size // 2, left_eye_center[1] - square_size // 2),
                      pt2=(left_eye_center[0] + square_size // 2, left_eye_center[1] + square_size // 2),
                      color=(0, 255, 0), thickness=2)

        cv2.rectangle(frame, pt1=(right_eye_center[0] - square_size // 2, right_eye_center[1] - square_size // 2),
                      pt2=(right_eye_center[0] + square_size // 2, right_eye_center[1] + square_size // 2),
                      color=(0, 255, 0), thickness=2)

        # Preprocessing: cropping the eye images
        left_roi = cv2.resize(left_eye_roi, (80, 80))
        left_roi / 255  # Rescaling the image pixel to a normalized range 0, 1
        left_roi = left_roi.reshape(80, 80, 3)  # Adding color channels
        left_roi = np.expand_dims(left_roi, axis=0)

        right_roi = cv2.resize(right_eye_roi, (80, 80))
        right_roi / 255  # Rescaling the image pixel to a normalized range 0, 1
        right_roi = right_roi.reshape(80, 80, 3)  # Adding color channels
        right_roi = np.expand_dims(right_roi, axis=0)

        # Make predictions for both eyes using your model
        prediction = model.predict(left_roi)
        predictions = model.predict(right_roi)

        # print(predictions)

        if predictions == 1:
            status = "Eyes Open"
            text_color = (0, 255, 0)
        else:
            counter = counter + 1
            if counter > 15:
                status = "Eyes Closed"
                text_color = (0, 0, 255)
                try:
                    sound.play()
                except Exception:
                    pass
                counter = 0

        # Inserting text into the video
        cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), font,
                    fontScale=0.8,
                    color=text_color,
                    thickness=2)

    cv2.imshow('Drowsiness Detection', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
