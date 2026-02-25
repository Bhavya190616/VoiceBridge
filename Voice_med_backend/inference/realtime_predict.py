import cv2
import os
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from deepface import DeepFace
from gtts import gTTS
import playsound
import tempfile

# -----------------------------
# Paths
# -----------------------------
from labelmap import LABEL_TO_WORD

MODEL_PATH = os.path.join("models", "hand_landmarker.task")
CLASSIFIER_PATH = os.path.join("models", "isl_classifier.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

# -----------------------------
# Load ML model & scaler
# -----------------------------
classifier = joblib.load(CLASSIFIER_PATH)
scaler = joblib.load(SCALER_PATH)
print("Loaded trained classifier and scaler")

# -----------------------------
# MediaPipe Hand Landmarker
# -----------------------------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

# -----------------------------
# Sentence state
# -----------------------------
sentence_buffer = []
last_prediction = None
last_committed_word = None
stable_counter = 0
WORD_HOLD_FRAMES = 10

# -----------------------------
# Emotion state
# -----------------------------
current_emotion = "neutral"

# -----------------------------
# Camera
# -----------------------------
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # -----------------------------
    # Emotion Detection (Lightweight continuous)
    # -----------------------------
    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )
        current_emotion = result[0]['dominant_emotion']
    except:
        pass

    # -----------------------------
    # Hand Detection
    # -----------------------------
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        features = []

        for lm in hand:
            features.extend([lm.x, lm.y, lm.z])

        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)

        predicted_label = classifier.predict(X_scaled)[0]
        predicted_word = LABEL_TO_WORD.get(predicted_label, predicted_label)

        # Stability logic
        if predicted_word == last_prediction:
            stable_counter += 1
        else:
            stable_counter = 1
            last_prediction = predicted_word

        if stable_counter == WORD_HOLD_FRAMES:

            # ---------- FULL STOP ----------
            if predicted_word == "FULL STOP":

                if sentence_buffer:
                    final_sentence = " ".join(sentence_buffer)
                    print("Sentence completed:", final_sentence)
                    print("Emotion:", current_emotion)

                    # Emotion-based speech modification
                    speech_text = final_sentence

                    if current_emotion == "happy":
                        speech_text += " 😊"
                    elif current_emotion == "sad":
                        speech_text += " 😔"
                    elif current_emotion == "angry":
                        speech_text += " 😠"

                    # Text-to-Speech using gTTS
                    tts = gTTS(text=speech_text, lang='en')
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    tts.save(temp_file.name)
                    playsound.playsound(temp_file.name)

                sentence_buffer = []
                last_committed_word = None

            # ---------- NORMAL WORD ----------
            else:
                if predicted_word != last_committed_word:
                    sentence_buffer.append(predicted_word)
                    print("Sentence:", sentence_buffer)
                    last_committed_word = predicted_word

            stable_counter = 0

    else:
        last_prediction = None
        stable_counter = 0

    # -----------------------------
    # Display sentence
    # -----------------------------
    h, w, _ = frame.shape

    if sentence_buffer:
        sentence_text = " ".join(sentence_buffer)

        (sw, sh), _ = cv2.getTextSize(
            sentence_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            2
        )

        sx = (w - sw) // 2
        sy = int(h * 0.9)

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (sx - 15, sy - sh - 15),
            (sx + sw + 15, sy + 10),
            (0, 0, 0),
            -1
        )

        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(
            frame,
            sentence_text,
            (sx, sy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    # Display emotion
    cv2.putText(
        frame,
        f"Emotion: {current_emotion}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("VoiceBridge – Emotion Aware ISL", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()