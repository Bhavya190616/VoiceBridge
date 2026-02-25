import cv2
import csv
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------------
# Paths
# -----------------------------
DATASET_DIR = "dataset"
CSV_PATH = os.path.join(DATASET_DIR, "isl_landmarks.csv")
MODEL_PATH = os.path.join("models", "hand_landmarker.task")

os.makedirs(DATASET_DIR, exist_ok=True)

# -----------------------------
# MediaPipe setup (ONE HAND ONLY)
# -----------------------------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

# -----------------------------
# CSV setup
# -----------------------------
file_exists = os.path.isfile(CSV_PATH)
csv_file = open(CSV_PATH, "a", newline="")
writer = csv.writer(csv_file)

if not file_exists:
    header = []
    for i in range(21):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    header.append("label")
    writer.writerow(header)

# -----------------------------
# Camera
# -----------------------------
cap = cv2.VideoCapture(0)

print("\n=== PHASE 1A: DATA ACQUISITION ===")
print("Use ONE HAND only")
print("H = HELLO | Y = YES | N = NO | T = THANK YOU | P = PLEASE")
print("Q = Quit\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if result.hand_landmarks and key != 255:
        hand = result.hand_landmarks[0]
        features = []

        for lm in hand:
            features.extend([lm.x, lm.y, lm.z])

        label = chr(key).upper()
        writer.writerow(features + [label])
        print(f"Saved sample for label: {label}")

cap.release()
csv_file.close()
cv2.destroyAllWindows()
