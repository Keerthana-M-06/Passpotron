
import os
import sys
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from collections import deque

# --------------------------------------------------
# 1. Set up repository path
# --------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

REPO_DIR = os.path.join(
    CURRENT_DIR,
    "Silent-Face-Anti-Spoofing"
)

sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
from src.data_io import transform as trans


MODEL_DIR = os.path.join(
    REPO_DIR,
    "resources",
    "anti_spoof_models"
)


# --------------------------------------------------
# 2. Load MiniFASNet models only once
# --------------------------------------------------

def load_models(predictor):

    loaded_models = []

    model_files = [
        f for f in os.listdir(MODEL_DIR)
        if f.endswith(".pth")
    ]

    if not model_files:
        raise FileNotFoundError(
            "No pretrained .pth models found!"
        )

    for model_name in sorted(model_files):

        model_path = os.path.join(
            MODEL_DIR,
            model_name
        )

        h_input, w_input, model_type, scale = (
            parse_model_name(model_name)
        )

        # Load the model weights
        predictor._load_model(model_path)

        # Keep a reference to this loaded model
        model = predictor.model
        model.eval()

        loaded_models.append({
            "model": model,
            "height": h_input,
            "width": w_input,
            "scale": scale,
            "name": model_name
        })

        print("Loaded:", model_name)

    return loaded_models


# --------------------------------------------------
# 3. Run prediction using already-loaded models
# --------------------------------------------------
def predict_face(frame, bbox, loaded_models, predictor, cropper):

    prediction = np.zeros(3)

    transform = trans.Compose([trans.ToTensor()])

    for item in loaded_models:

        scale = item["scale"]

        face = cropper.crop(
            org_img=frame,
            bbox=bbox,
            scale=scale,
            out_w=item["width"],
            out_h=item["height"],
            crop=(scale is not None)
        )

        face_tensor = transform(face)
        face_tensor = face_tensor.unsqueeze(0).to(predictor.device)

        with torch.no_grad():
            output = item["model"](face_tensor)

            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
            print(
        f"{item['name']} -> "
        f"Class 0: {probabilities[0]:.3f}, "
        f"Class 1: {probabilities[1]:.3f}, "
        f"Class 2: {probabilities[2]:.3f}"
        )


        prediction += probabilities

    # Average the predictions from both MiniFASNet models
    avg_probabilities = prediction / len(loaded_models)

    # Original model class mapping:
    # Class 0 = Fake
    # Class 1 = Real
    # Class 2 = Fake

    fake_score = avg_probabilities[0] + avg_probabilities[2]
    real_score = avg_probabilities[1]

    if real_score > fake_score:
        raw_result = "REAL FACE"
    else:
        raw_result = "FAKE FACE"

    return raw_result, real_score, fake_score



# --------------------------------------------------
# 4. Main webcam function
# --------------------------------------------------

def start_webcam():

    print("Initializing MiniFASNet...")

    predictor = AntiSpoofPredict(0)
    cropper = CropImage()

    loaded_models = load_models(predictor)

    print("\nAll models loaded successfully!")
    print("Starting webcam...")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_count = 0

# Store the last 10 predictions
    prediction_history = deque(maxlen=10)

    result_text = "ANALYSING..."
    result_score = 0.0
    result_color = (255, 255, 255)

    current_bbox = None

    

    # --------------------------------------------------
    # 5. Process webcam frames
    # --------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Could not read webcam frame.")
            break

        frame_count += 1

        # Run expensive inference every 5 frames
        if frame_count % 5 == 0:

            try:
                bbox = predictor.get_bbox(frame)

                if bbox[2] > 0 and bbox[3] > 0:

                    raw_result, real_score, fake_score = predict_face(
                        frame,
                        bbox,
                        loaded_models,
                        predictor,
                        cropper
                    )

            # Store both class scores
                    prediction_history.append((real_score, fake_score))

                    current_bbox = bbox

            # Wait until we have enough predictions
                    if len(prediction_history) < 5:

                        result_text = "ANALYSING..."
                        result_color = (255, 255, 255)
                        result_score = real_score

                    else:

                # Average scores over recent predictions
                        avg_real = sum(
                            item[0] for item in prediction_history
                        ) / len(prediction_history)

                        avg_fake = sum(
                            item[1] for item in prediction_history
                        ) / len(prediction_history)

                        result_score = max(avg_real, avg_fake)

                # Make a stable decision
                        if avg_real >= 0.55:

                            result_text = "LIVE CANDIDATE"
                            result_color = (0, 255, 0)

                        elif avg_fake >= 0.70:

                            result_text = "SPOOF SUSPECTED"
                            result_color = (0, 0, 255)

                        else:

                            result_text = "INCONCLUSIVE"
                            result_color = (0, 165, 255)

                else:

                    result_text = "FACE NOT DETECTED"
                    result_color = (255, 255, 255)
                    current_bbox = None
                    prediction_history.clear()

            except Exception as e:

                print("Prediction error:", e)

                result_text = "PREDICTION ERROR"
                result_color = (0, 0, 255)
                current_bbox = None
                prediction_history.clear()

        # Draw face rectangle
        if current_bbox is not None:

            x, y, w, h = current_bbox

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                result_color,
                2
            )

        # Display prediction
        cv2.putText(
            frame,
            result_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            result_color,
            2
        )

        cv2.putText(
            frame,
            "Score: {:.2f}".format(result_score),
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            result_color,
            2
        )

        cv2.putText(
            frame,
            "Press Q to exit",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        cv2.imshow(
            "MiniFASNet - Webcam Anti Spoofing",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam
    cap.release()
    cv2.destroyAllWindows()

    print("Webcam closed.")


# --------------------------------------------------
# 6. Execute
# --------------------------------------------------

if __name__ == "__main__":
    start_webcam()
