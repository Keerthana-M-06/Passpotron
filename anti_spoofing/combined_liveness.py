
import os
import sys
import time
import cv2
import dlib
import torch
import numpy as np
import torch.nn.functional as F

from collections import deque
from scipy.spatial import distance


# =========================================================
# 1. PROJECT PATHS AND IMPORTS
# =========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

REPO_DIR = os.path.join(
    CURRENT_DIR,
    "Silent-Face-Anti-Spoofing"
)

sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, REPO_DIR)

os.chdir(REPO_DIR)

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
from src.data_io import transform as trans

from random_challenge.challenge_detection import RandomChallenge
from decision_engine.result_prediction import predict_final_result


# =========================================================
# 2. PATHS AND SETTINGS
# =========================================================

MODEL_DIR = os.path.join(
    REPO_DIR, "resources", "anti_spoof_models"
)

LANDMARK_MODEL = os.path.join(
    PROJECT_DIR,
    "ear_detection",
    "shape_predictor_68_face_landmarks.dat"
)

EAR_THRESHOLD = 0.23
CONSECUTIVE_FRAMES = 2

PREDICTION_INTERVAL = 5
MIN_PREDICTIONS = 5
PREDICTION_HISTORY_SIZE = 10

# Do not use an old MiniFASNet prediction.
MAX_PREDICTION_AGE = 3.0

# Camera freeze check.
MAX_IDENTICAL_FRAMES = 30


# =========================================================
# 3. LOAD DLIB
# =========================================================

if not os.path.isfile(LANDMARK_MODEL):
    raise FileNotFoundError(
        "Dlib landmark model not found: " + LANDMARK_MODEL
    )

print("Loading Dlib models...", flush=True)

face_detector = dlib.get_frontal_face_detector()

landmark_predictor = dlib.shape_predictor(
    LANDMARK_MODEL
)

print("Dlib models loaded!", flush=True)


# =========================================================
# 4. EAR CALCULATION
# =========================================================

def calculate_ear(eye):

    vertical_1 = distance.euclidean(
        eye[1], eye[5]
    )

    vertical_2 = distance.euclidean(
        eye[2], eye[4]
    )

    horizontal = distance.euclidean(
        eye[0], eye[3]
    )

    if horizontal == 0:
        return 0.0

    return (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)


# =========================================================
# 5. HEAD DIRECTION
# =========================================================

def estimate_head_direction(points, frame_shape):

    height, width = frame_shape[:2]

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ], dtype=np.float64)

    image_points = np.array([
        points[30],
        points[8],
        points[36],
        points[45],
        points[48],
        points[54]
    ], dtype=np.float64)

    focal_length = width

    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion = np.zeros((4, 1))

    try:

        success, rotation_vector, translation_vector = (
            cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        )

        if not success:
            return "UNKNOWN"

        rotation_matrix, _ = cv2.Rodrigues(
            rotation_vector
        )

        yaw = np.degrees(
            np.arctan2(
                rotation_matrix[1, 0],
                rotation_matrix[0, 0]
            )
        )

        # These directions were calibrated for this webcam.
        if yaw < -15:
            return "RIGHT"

        elif yaw > 15:
            return "LEFT"

        return "CENTER"

    except cv2.error:
        return "UNKNOWN"


# =========================================================
# 6. LOAD MINIFASNET MODELS
# =========================================================

def load_models(predictor):

    loaded_models = []

    model_files = [
        name for name in os.listdir(MODEL_DIR)
        if name.endswith(".pth")
    ]

    if not model_files:
        raise FileNotFoundError(
            "No MiniFASNet models found!"
        )

    for model_name in sorted(model_files):

        model_path = os.path.join(
            MODEL_DIR,
            model_name
        )

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

        print("Loaded:", model_name, flush=True)

    return loaded_models


# =========================================================
# 7. MINIFASNET PREDICTION
# =========================================================

def predict_face(
    frame,
    bbox,
    loaded_models,
    predictor,
    cropper
):

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

        face_tensor = face_tensor.unsqueeze(0).to(
            predictor.device
        )

        with torch.no_grad():

            output = item["model"](face_tensor)

            probabilities = (
                F.softmax(output, dim=1)
                .cpu()
                .numpy()[0]
            )

        prediction += probabilities

    avg_probabilities = prediction / len(loaded_models)

    # IMPORTANT:
    # This assumes class 0 = fake, class 1 = real,
    # class 2 = fake. Verify against the repository's
    # class labels before relying on these scores.

    fake_score = (
        avg_probabilities[0] +
        avg_probabilities[2]
    )

    real_score = avg_probabilities[1]

    return real_score, fake_score


# =========================================================
# 8. RESET VERIFICATION ATTEMPT
# =========================================================

def reset_attempt():

    return {
        "challenge": RandomChallenge(),
        "challenge_started": False,
        "challenge_status": "NOT_STARTED",
        "attempt_start_blink_count": 0,
        "ear_passed": False,
        "final_result": "WAITING",
        "prediction_history": deque(
            maxlen=PREDICTION_HISTORY_SIZE
        ),
        "minifasnet_result": "WAITING FOR FACE",
        "real_score": 0.0,
        "fake_score": 0.0,
        "last_prediction_time": None,
        "closed_frames": 0,
        "current_ear": 0.0,
        "ear_status": "WAITING FOR FACE",
        "head_direction": "NOT AVAILABLE"
    }


# =========================================================
# 9. WEBCAM APPLICATION
# =========================================================

def start_combined_webcam():

    print("\nInitializing MiniFASNet...", flush=True)

    anti_spoof_predictor = AntiSpoofPredict(0)

    cropper = CropImage()

    loaded_models = load_models(
        anti_spoof_predictor
    )

    print("MiniFASNet loaded!", flush=True)

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        print("ERROR: Could not open webcam.")

        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Webcam started!", flush=True)
    print("Press Q to exit.", flush=True)

    state = reset_attempt()

    blink_counter = 0
    frame_count = 0

    previous_frame = None
    identical_frame_count = 0

    try:

        while True:

            ret, frame = cap.read()

            if not ret:

                print(
                    "Camera disconnected or frame read failed.",
                    flush=True
                )

                break

            frame_count += 1

            # ---------------------------------------------
            # CAMERA FREEZE CHECK
            # ---------------------------------------------

            if previous_frame is not None:

                if np.array_equal(frame, previous_frame):

                    identical_frame_count += 1

                else:

                    identical_frame_count = 0

            previous_frame = frame.copy()

            if identical_frame_count >= MAX_IDENTICAL_FRAMES:

                print(
                    "Camera appears frozen. Stopping verification.",
                    flush=True
                )

                state["final_result"] = "INCONCLUSIVE"

                break

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = face_detector(gray, 0)

            number_of_faces = len(faces)

            # =================================================
            # CASE 1: NO FACE
            # =================================================

            if number_of_faces == 0:

                state = reset_attempt()

                # Keep total blink count for display.
                state["current_ear"] = 0.0

            # =================================================
            # CASE 2: MULTIPLE FACES
            # =================================================

            elif number_of_faces > 1:

                state = reset_attempt()

                state["minifasnet_result"] = (
                    "MULTIPLE FACES DETECTED"
                )

                state["ear_status"] = (
                    "ONLY ONE PERSON ALLOWED"
                )

                for face in faces:

                    cv2.rectangle(
                        frame,
                        (face.left(), face.top()),
                        (face.right(), face.bottom()),
                        (0, 0, 255),
                        2
                    )

            # =================================================
            # CASE 3: EXACTLY ONE FACE
            # =================================================

            else:

                face = faces[0]

                landmarks_valid = False

                eyes_closed = False

                head_direction = "UNKNOWN"

                # ---------------------------------------------
                # 3A. LANDMARK DETECTION AND EAR
                # ---------------------------------------------

                try:

                    landmarks = landmark_predictor(
                        gray,
                        face
                    )

                    points = np.array([
                        (
                            landmarks.part(i).x,
                            landmarks.part(i).y
                        )
                        for i in range(68)
                    ])

                    left_eye = points[36:42]
                    right_eye = points[42:48]

                    left_ear = calculate_ear(left_eye)
                    right_ear = calculate_ear(right_eye)

                    current_ear = (
                        left_ear + right_ear
                    ) / 2.0

                    if not np.isfinite(current_ear):
                        raise ValueError("Invalid EAR value")

                    landmarks_valid = True

                    eyes_closed = (
                        current_ear < EAR_THRESHOLD
                    )

                    if eyes_closed:

                        state["closed_frames"] += 1

                        state["ear_status"] = "EYES CLOSED"

                    else:

                        if (
                            state["closed_frames"]
                            >= CONSECUTIVE_FRAMES
                        ):

                            blink_counter += 1

                        state["closed_frames"] = 0

                        state["ear_status"] = "EYES OPEN"

                    state["current_ear"] = current_ear

                    head_direction = estimate_head_direction(
                        points,
                        frame.shape
                    )

                    state["head_direction"] = head_direction

                    for point in left_eye:

                        cv2.circle(
                            frame,
                            tuple(point),
                            2,
                            (0, 255, 0),
                            -1
                        )

                    for point in right_eye:

                        cv2.circle(
                            frame,
                            tuple(point),
                            2,
                            (0, 255, 0),
                            -1
                        )

                    cv2.rectangle(
                        frame,
                        (face.left(), face.top()),
                        (face.right(), face.bottom()),
                        (255, 0, 0),
                        2
                    )

                except Exception as error:

                    print(
                        "Landmark error:",
                        error,
                        flush=True
                    )

                    state["current_ear"] = 0.0

                    state["closed_frames"] = 0

                    state["ear_status"] = "LANDMARK ERROR"

                    state["head_direction"] = "UNKNOWN"

                # ---------------------------------------------
                # 3B. START CHALLENGE ONLY WITH VALID LANDMARKS
                # ---------------------------------------------

                if (
                    landmarks_valid
                    and not state["challenge_started"]
                ):

                    state["attempt_start_blink_count"] = (
                        blink_counter
                    )

                    state["challenge"].start(
                        blink_counter
                    )

                    state["challenge_started"] = True

                    state["challenge_status"] = "IN_PROGRESS"

                # ---------------------------------------------
                # 3C. EAR REQUIREMENT FOR CURRENT ATTEMPT
                # ---------------------------------------------

                if (
                    landmarks_valid
                    and blink_counter -
                    state["attempt_start_blink_count"] >= 1
                ):

                    state["ear_passed"] = True

                # ---------------------------------------------
                # 3D. UPDATE RANDOM CHALLENGE
                # ---------------------------------------------

                if state["challenge_started"]:

                    # If landmarks fail, do not allow a
                    # blink or eye-closure challenge to pass.
                    if landmarks_valid:

                        state["challenge_status"] = (
                            state["challenge"].update(
                                blink_counter,
                                head_direction,
                                eyes_closed
                            )
                        )

                    else:

                        state["challenge_status"] = (
                            state["challenge"].update(
                                blink_counter,
                                "UNKNOWN",
                                False
                            )
                        )

                # ---------------------------------------------
                # 3E. MINIFASNET PREDICTION
                # ---------------------------------------------

                if frame_count % PREDICTION_INTERVAL == 0:

                    try:

                        bbox = anti_spoof_predictor.get_bbox(
                            frame
                        )

                        if bbox[2] > 0 and bbox[3] > 0:

                            real_score, fake_score = predict_face(
                                frame,
                                bbox,
                                loaded_models,
                                anti_spoof_predictor,
                                cropper
                            )

                            state["prediction_history"].append(
                                (real_score, fake_score)
                            )

                            state["last_prediction_time"] = (
                                time.monotonic()
                            )

                            if (
                                len(state["prediction_history"])
                                >= MIN_PREDICTIONS
                            ):

                                avg_real = sum(
                                    item[0]
                                    for item in state["prediction_history"]
                                ) / len(state["prediction_history"])

                                avg_fake = sum(
                                    item[1]
                                    for item in state["prediction_history"]
                                ) / len(state["prediction_history"])

                                state["real_score"] = avg_real

                                state["fake_score"] = avg_fake

                                if avg_real >= 0.70:

                                    state["minifasnet_result"] = (
                                        "LIVE CANDIDATE"
                                    )

                                elif avg_fake >= 0.70:

                                    state["minifasnet_result"] = (
                                        "SPOOF SUSPECTED"
                                    )

                                else:

                                    state["minifasnet_result"] = (
                                        "INCONCLUSIVE"
                                    )

                            else:

                                state["minifasnet_result"] = (
                                    "ANALYSING..."
                                )

                        else:

                            state["minifasnet_result"] = (
                                "PREDICTION ERROR"
                            )

                            state["real_score"] = 0.0
                            state["fake_score"] = 0.0

                            state["last_prediction_time"] = None

                            state["prediction_history"].clear()

                    except Exception as error:

                        print(
                            "MiniFASNet error:",
                            error,
                            flush=True
                        )

                        state["minifasnet_result"] = (
                            "PREDICTION ERROR"
                        )

                        state["real_score"] = 0.0
                        state["fake_score"] = 0.0

                        state["last_prediction_time"] = None

                        state["prediction_history"].clear()

                # ---------------------------------------------
                # 3F. CHECK PREDICTION FRESHNESS
                # ---------------------------------------------

                prediction_is_fresh = (
                    state["last_prediction_time"] is not None
                    and (
                        time.monotonic()
                        - state["last_prediction_time"]
                    ) <= MAX_PREDICTION_AGE
                )

                if not prediction_is_fresh:

                    effective_minifasnet_result = "INCONCLUSIVE"

                else:

                    effective_minifasnet_result = (
                        state["minifasnet_result"]
                    )

                # ---------------------------------------------
                # 3G. FINAL DECISION
                # ---------------------------------------------

                if (
                    effective_minifasnet_result
                    == "SPOOF SUSPECTED"
                ):

                    state["final_result"] = (
                        predict_final_result(
                            effective_minifasnet_result,
                            state["ear_passed"],
                            state["challenge_status"]
                        )
                    )

                elif (
                    state["challenge_status"]
                    in ["PASSED", "FAILED"]
                ):

                    state["final_result"] = (
                        predict_final_result(
                            effective_minifasnet_result,
                            state["ear_passed"],
                            state["challenge_status"]
                        )
                    )

                else:

                    state["final_result"] = (
                        "VERIFICATION IN PROGRESS"
                    )

            # =================================================
            # 4. DISPLAY
            # =================================================

            if state["minifasnet_result"] == "LIVE CANDIDATE":

                minifasnet_color = (0, 255, 0)

            elif state["minifasnet_result"] in [
                "SPOOF SUSPECTED",
                "PREDICTION ERROR",
                "MULTIPLE FACES DETECTED"
            ]:

                minifasnet_color = (0, 0, 255)

            elif state["minifasnet_result"] == "INCONCLUSIVE":

                minifasnet_color = (0, 165, 255)

            else:

                minifasnet_color = (255, 255, 255)

            cv2.putText(
                frame,
                "MINIFASNET: " + state["minifasnet_result"],
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                minifasnet_color,
                2
            )

            cv2.putText(
                frame,
                "REAL: {:.2f} FAKE: {:.2f}".format(
                    state["real_score"],
                    state["fake_score"]
                ),
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "EAR: {:.3f}".format(state["current_ear"]),
                (15, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "BLINKS: {}".format(blink_counter),
                (15, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                state["ear_status"],
                (15, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "HEAD: " + state["head_direction"],
                (15, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            # Challenge instruction
            if state["challenge_started"]:

                instruction = (
                    state["challenge"].get_instruction()
                )

            else:

                instruction = "Waiting for one face"

            cv2.putText(
                frame,
                "CHALLENGE: " + instruction,
                (15, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "CHALLENGE STATUS: "
                + state["challenge_status"],
                (15, 275),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                2
            )

            ear_text = (
                "PASSED"
                if state["ear_passed"]
                else "WAITING FOR BLINK"
            )

            cv2.putText(
                frame,
                "EAR CHECK: " + ear_text,
                (15, 310),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 0)
                if state["ear_passed"]
                else (0, 165, 255),
                2
            )

            # Final result
            final_result = state["final_result"]

            if final_result == "LIVE CANDIDATE":

                final_color = (0, 255, 0)

            elif final_result == "SPOOF SUSPECTED":

                final_color = (0, 0, 255)

            elif final_result == "INCONCLUSIVE":

                final_color = (0, 165, 255)

            else:

                final_color = (255, 255, 255)

            cv2.putText(
                frame,
                "FINAL: " + final_result,
                (15, 355),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                final_color,
                2
            )

            if number_of_faces > 1:

                cv2.rectangle(
                    frame,
                    (5, 380),
                    (635, 445),
                    (0, 0, 180),
                    -1
                )

                cv2.putText(
                    frame,
                    "MULTIPLE PEOPLE DETECTED!",
                    (15, 405),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Only ONE person is allowed.",
                    (15, 435),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    2
                )

            cv2.putText(
                frame,
                "Press Q to exit",
                (15, 470),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1
            )

            cv2.imshow(
                "Passpotron - Combined Liveness Detection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):

                break

    finally:

        cap.release()

        cv2.destroyAllWindows()

        print("\nWebcam closed.", flush=True)

        print(
            "Total blinks detected:",
            blink_counter,
            flush=True
        )

        print(
            "Last verification result:",
            state["final_result"],
            flush=True
        )


# =========================================================
# 10. RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    start_combined_webcam()
