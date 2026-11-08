
import cv2
import dlib
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

predictor_path = os.path.join(
    CURRENT_DIR,
    "shape_predictor_68_face_landmarks.dat"
)

print("Loading face detector...", flush=True)
detector = dlib.get_frontal_face_detector()

print("Loading landmark model...", flush=True)
predictor = dlib.shape_predictor(predictor_path)

print("Both models loaded!", flush=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera failed to open")
    exit()

print("Camera started!", flush=True)

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    print("Frame captured", flush=True)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    print("Detecting face...", flush=True)
    faces = detector(gray, 0)

    print("Number of faces:", len(faces), flush=True)

    if len(faces) > 0:

        print("Selecting face...", flush=True)

        face = faces[0]

        print("Calling landmark predictor...", flush=True)

        landmarks = predictor(gray, face)

        print("Landmarks predicted successfully!", flush=True)

        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y

            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Dlib Landmark Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Test completed!")
