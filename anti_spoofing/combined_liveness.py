
import os
import sys
import cv2
import dlib
import torch
import numpy as np
import torch.nn.functional as F

from collections import deque
from scipy.spatial import distance


# =====================================================
# 1. PROJECT PATHS
# =====================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

REPO_DIR = os.path.join(CURRENT_DIR, "Silent-Face-Anti-Spoofing")

sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
from src.data_io import transform as trans


MODEL_DIR = os.path.join(
    REPO_DIR, "resources", "anti_spoof_models"
)

EAR_DIR = os.path.join(
    os.path.dirname(CURRENT_DIR), "ear_detection"
)

LANDMARK_MODEL = os.path.join(
    EAR_DIR, "shape_predictor_68_face_landmarks.dat"
)


# =====================================================
# 2. LOAD DLIB MODELS
# =====================================================

if not os.path.isfile(LANDMARK_MODEL):
    raise FileNotFoundError(
        "Landmark model not found: " + LANDMARK_MODEL
    )

print("Loading Dlib models...", flush=True)

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(LANDMARK_MODEL)

print("Dlib models loaded!", flush=True)


# =====================================================
# 3. EAR SETTINGS
# =====================================================

EAR_THRESHOLD = 0.23
CONSECUTIVE_FRAMES = 2

blink_counter = 0
closed_frames = 0


def calculate_ear(eye):

    vertical_1 = distance.euclidean(eye[1], eye[5])
    vertical_2 = distance.euclidean(eye[2], eye[4])
    horizontal = distance.euclidean(eye[0], eye[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def estimate_head_direction(points):

    # 3D reference points for the face.
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye corner
        (225.0, 170.0, -135.0),   # Right eye corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ], dtype=np.float64)

    # Corresponding 2D facial landmarks.
    image_points = np.array([
        points[30],  # Nose tip
        points[8],   # Chin
        points[36],  # Left eye corner
        points[45],  # Right eye corner
        points[48],  # Left mouth corner
        points[54]   # Right mouth corner
    ], dtype=np.float64)

    height, width = 480, 640

    focal_length = width

    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return "CENTER"

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    yaw = np.degrees(
        np.arctan2(
            rotation_matrix[1, 0],
            rotation_matrix[0, 0]
        )
    )

    if yaw < -15:
        return "LEFT"

    elif yaw > 15:
        return "RIGHT"

    return "CENTER"

# =====================================================
# 4. LOAD MINIFASNET MODELS
# =====================================================

def load_models(predictor):

    loaded_models = []

    model_files = [
        f for f in os.listdir(MODEL_DIR)
        if f.endswith(".pth")
    ]

    if not model_files:
        raise FileNotFoundError("No MiniFASNet models found!")

    for model_name in sorted(model_files):

        model_path = os.path.join(MODEL_DIR, model_name)

        h_input, w_input, model_type, scale = (
            parse_model_name(model_name)
        )

        predictor._load_model(model_path)

        model = predictor.model
        model.eval()

        loaded_models.append({
            "model": model,
            "height": h_input,
            "width": w_input,
            "scale": scale,
            "name": model_name
        })

        print("Loaded MiniFASNet:", model_name, flush=True)

    return loaded_models


# =====================================================
# 5. MINIFASNET PREDICTION
# =====================================================

def predict_face(frame, bbox, loaded_models, predictor, cropper):

    prediction = np.zeros(3)

    image_transform = trans.Compose([
        trans.ToTensor()
    ])

    for item in loaded_models:

        face_image = cropper.crop(
            org_img=frame,
            bbox=bbox,
            scale=item["scale"],
            out_w=item["width"],
            out_h=item["height"],
            crop=(item["scale"] is not None)
        )

        face_tensor = image_transform(face_image)
        face_tensor = face_tensor.unsqueeze(0).to(predictor.device)

        with torch.no_grad():
            output = item["model"](face_tensor)

            probabilities = (
                F.softmax(output, dim=1)
                .cpu()
                .numpy()[0]
            )

        prediction += probabilities

    avg_probabilities = prediction / len(loaded_models)

    # Existing implementation's assumed class mapping:
    # Class 0 = Fake
    # Class 1 = Real
    # Class 2 = Fake

    fake_score = (
        avg_probabilities[0] +
        avg_probabilities[2]
    )

    real_score = avg_probabilities[1]

    return real_score, fake_score


# =====================================================
# 6. MAIN COMBINED WEBCAM
# =====================================================

def start_combined_webcam():

    global blink_counter, closed_frames

    print("\nInitializing MiniFASNet...", flush=True)

    anti_spoof_predictor = AntiSpoofPredict(0)
    cropper = CropImage()

    loaded_models = load_models(anti_spoof_predictor)

    print("\nAll MiniFASNet models loaded!", flush=True)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Combined webcam started!", flush=True)
    print("Press Q to exit.", flush=True)

    frame_count = 0

    prediction_history = deque(maxlen=10)

    minifasnet_result = "WAITING FOR FACE"
    minifasnet_color = (255, 255, 255)

    minifasnet_real_score = 0.0
    minifasnet_fake_score = 0.0

    ear_status = "WAITING FOR FACE"
    ear_color = (255, 255, 255)

    current_ear = 0.0

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                print("Could not read webcam frame.", flush=True)
                break

            frame_count += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ---------------------------------------------
            # A. DETECT ALL FACES
            # ---------------------------------------------

            faces = face_detector(gray, 0)
            number_of_faces = len(faces)

            # ---------------------------------------------
            # B. NO FACE DETECTED
            # ---------------------------------------------

            if number_of_faces == 0:

                current_ear = 0.0

                ear_status = "FACE NOT DETECTED"
                ear_color = (255, 255, 255)

                minifasnet_result = "FACE NOT DETECTED"
                minifasnet_color = (255, 255, 255)

                minifasnet_real_score = 0.0
                minifasnet_fake_score = 0.0

                closed_frames = 0
                prediction_history.clear()

            # ---------------------------------------------
            # C. MULTIPLE FACES DETECTED
            # ---------------------------------------------

            elif number_of_faces > 1:

                # Draw a rectangle around EVERY detected face.
                for detected_face in faces:

                    cv2.rectangle(
                        frame,
                        (detected_face.left(), detected_face.top()),
                        (detected_face.right(), detected_face.bottom()),
                        (0, 0, 255),
                        2
                    )

                # Do not run EAR or MiniFASNet.
                current_ear = 0.0

                ear_status = "ONLY ONE PERSON ALLOWED"
                ear_color = (0, 0, 255)

                minifasnet_result = "MULTIPLE FACES DETECTED"
                minifasnet_color = (0, 0, 255)

                minifasnet_real_score = 0.0
                minifasnet_fake_score = 0.0

                # Reset temporary detection history.
                closed_frames = 0
                prediction_history.clear()

            # ---------------------------------------------
            # D. EXACTLY ONE FACE DETECTED
            # ---------------------------------------------

            else:

                face = faces[0]

                # -----------------------------------------
                # D1. EAR BLINK DETECTION
                # -----------------------------------------

                landmarks = landmark_predictor(gray, face)

                points = np.array([
                    (landmarks.part(i).x, landmarks.part(i).y)
                    for i in range(68)
                ])

                left_eye = points[36:42]
                right_eye = points[42:48]

                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)

                current_ear = (left_ear + right_ear) / 2.0

                for point in left_eye:
                    cv2.circle(
                        frame, tuple(point), 2, (0, 255, 0), -1
                    )

                for point in right_eye:
                    cv2.circle(
                        frame, tuple(point), 2, (0, 255, 0), -1
                    )

                if current_ear < EAR_THRESHOLD:

                    closed_frames += 1

                    ear_status = "EYES CLOSED"
                    ear_color = (0, 0, 255)

                else:

                    if closed_frames >= CONSECUTIVE_FRAMES:
                        blink_counter += 1

                    closed_frames = 0

                    ear_status = "EYES OPEN"
                    ear_color = (0, 255, 0)

                cv2.rectangle(
                    frame,
                    (face.left(), face.top()),
                    (face.right(), face.bottom()),
                    (255, 0, 0),
                    2
                )

                # -----------------------------------------
                # D2. MINIFASNET ANTI-SPOOFING
                # -----------------------------------------

                if frame_count % 5 == 0:

                    try:

                        bbox = anti_spoof_predictor.get_bbox(frame)

                        if bbox[2] > 0 and bbox[3] > 0:

                            real_score, fake_score = predict_face(
                                frame,
                                bbox,
                                loaded_models,
                                anti_spoof_predictor,
                                cropper
                            )

                            prediction_history.append(
                                (real_score, fake_score)
                            )

                            if len(prediction_history) >= 5:

                                avg_real = sum(
                                    item[0] for item in prediction_history
                                ) / len(prediction_history)

                                avg_fake = sum(
                                    item[1] for item in prediction_history
                                ) / len(prediction_history)

                                minifasnet_real_score = avg_real
                                minifasnet_fake_score = avg_fake

                                if avg_real >= 0.70:

                                    minifasnet_result = "LIVE CANDIDATE"
                                    minifasnet_color = (0, 255, 0)

                                elif avg_fake >= 0.70:

                                    minifasnet_result = "SPOOF SUSPECTED"
                                    minifasnet_color = (0, 0, 255)

                                else:

                                    minifasnet_result = "INCONCLUSIVE"
                                    minifasnet_color = (0, 165, 255)

                            else:

                                minifasnet_result = "ANALYSING..."
                                minifasnet_color = (255, 255, 255)

                        else:

                            minifasnet_result = "FACE NOT DETECTED"
                            minifasnet_color = (255, 255, 255)

                            minifasnet_real_score = 0.0
                            minifasnet_fake_score = 0.0

                            prediction_history.clear()

                    except Exception as error:

                        print(
                            "MiniFASNet prediction error:",
                            error,
                            flush=True
                        )

                        minifasnet_result = "PREDICTION ERROR"
                        minifasnet_color = (0, 0, 255)

                        minifasnet_real_score = 0.0
                        minifasnet_fake_score = 0.0

                        prediction_history.clear()

            # =================================================
            # E. DISPLAY RESULTS
            # =================================================

            cv2.putText(
                frame,
                "MINIFASNET: " + minifasnet_result,
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                minifasnet_color,
                2
            )

            cv2.putText(
                frame,
                "REAL: {:.2f}  FAKE: {:.2f}".format(
                    minifasnet_real_score,
                    minifasnet_fake_score
                ),
                (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "EAR: {:.3f}".format(current_ear),
                (15, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "BLINKS: {}".format(blink_counter),
                (15, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                ear_status,
                (15, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                ear_color,
                2
            )

            # Prominent alert for multiple people.
            if number_of_faces > 1:

                cv2.rectangle(
                    frame,
                    (5, 245),
                    (635, 320),
                    (0, 0, 180),
                    -1
                )

                cv2.putText(
                    frame,
                    "MULTIPLE PEOPLE IN FRONT OF CAMERA!",
                    (15, 275),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Please stand in line - ONE PERSON ONLY",
                    (15, 305),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    2
                )

            cv2.putText(
                frame,
                "Press Q to exit",
                (15, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )

            cv2.imshow(
                "Passpotron - Combined Liveness Detection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:

        cap.release()
        cv2.destroyAllWindows()

        print("\nWebcam closed.", flush=True)
        print("Total blinks detected:", blink_counter, flush=True)


# =====================================================
# 7. RUN APPLICATION
# =====================================================

if __name__ == "__main__":
    start_combined_webcam()
