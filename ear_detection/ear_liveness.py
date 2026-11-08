
import os
import cv2
import dlib
import numpy as np
from scipy.spatial import distance

# 1. Load Dlib models

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

predictor_path = os.path.join(
    CURRENT_DIR,
    "shape_predictor_68_face_landmarks.dat"
)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


# 2. Calculate EAR

def calculate_ear(eye):

    vertical_1 = distance.euclidean(eye[1], eye[5])
    vertical_2 = distance.euclidean(eye[2], eye[4])
    horizontal = distance.euclidean(eye[0], eye[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


# 3. Blink settings

EAR_THRESHOLD = 0.23
CONSECUTIVE_FRAMES = 2

blink_counter = 0
closed_frames = 0


# 4. Webcam

def start_webcam():

    global blink_counter, closed_frames

    print("Starting EAR detection...", flush=True)

    # Same initialization as camera_test.py
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Camera could not open!")
        return

    print("Camera opened!", flush=True)

    while True:

        # Same frame-reading logic as test_landmarks.py
        ret, frame = cap.read()

        if not ret:
            print("Frame unavailable", flush=True)
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Same Dlib detector used in test_landmarks.py
        faces = detector(gray, 0)

        ear = 0.0
        status = "FACE NOT DETECTED"
        color = (255, 255, 255)

        if len(faces) > 0:

            face = faces[0]

            # Same landmark prediction used in test_landmarks.py
            landmarks = predictor(gray, face)

            points = np.array([
                (landmarks.part(i).x, landmarks.part(i).y)
                for i in range(68)
            ])

            # Extract eyes
            left_eye = points[36:42]
            right_eye = points[42:48]

            # Calculate EAR
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)

            ear = (left_ear + right_ear) / 2.0

            # Draw all eye landmarks
            for point in left_eye:
                cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)

            for point in right_eye:
                cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)

            # Count blink
            if ear < EAR_THRESHOLD:

                closed_frames += 1
                status = "EYES CLOSED"
                color = (0, 0, 255)

            else:

                if closed_frames >= CONSECUTIVE_FRAMES:
                    blink_counter += 1

                closed_frames = 0
                status = "EYES OPEN"
                color = (0, 255, 0)

            cv2.rectangle(
                frame,
                (face.left(), face.top()),
                (face.right(), face.bottom()),
                (255, 0, 0),
                2
            )

        else:
            closed_frames = 0

        # Display information
        cv2.putText(
            frame, "EAR: {:.3f}".format(ear),
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 0), 2
        )

        cv2.putText(
            frame, "Blinks: {}".format(blink_counter),
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        cv2.putText(
            frame, status,
            (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, color, 2
        )

        cv2.imshow("EAR Blink Detection", frame)

        # Same exit condition as test_landmarks.py
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Total blinks:", blink_counter, flush=True)


if __name__ == "__main__":
    start_webcam()
