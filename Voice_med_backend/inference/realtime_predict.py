import cv2
import os
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyttsx3
import threading

# -----------------------------
# Paths (Absolute for stability)
# -----------------------------
HAND_MODEL = "models/hand_landmarker.task"
FACE_MODEL = "models/face_landmarker.task"
CLASSIFIER_PATH = "models/isl_classifier.pkl"
SCALER_PATH = "models/scaler.pkl"


from labelmap import LABEL_TO_WORD
classifier = joblib.load(CLASSIFIER_PATH)
scaler = joblib.load(SCALER_PATH)

# MediaPipe Initialization (FIXED: Variable name corrected)
base_options_hand = python.BaseOptions(model_asset_path=HAND_MODEL)
hand_detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(base_options=base_options_hand, num_hands=1)
)

base_options_face = python.BaseOptions(model_asset_path=FACE_MODEL)
face_detector = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(base_options=base_options_face, output_face_blendshapes=True, num_faces=1)
)

# -----------------------------
# 1. Humane Offline Voice Engine
# -----------------------------
def speak_humane_offline(text, emotion):
    def run():
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        
        # Humanizing: Selecting a softer 'Natural' style voice if available
        # On Windows, index 1 is often Microsoft Zira (Female/Softer)
        if len(voices) > 1:
            engine.setProperty('voice', voices[1].id) 

        rate = 155
        volume = 1.0
        processed_text = text

        # Tone Modulation via Punctuation
        if emotion == "happy":
            rate = 185
            processed_text = f"Oh! {text}!" # Adds excitatory pitch
        elif emotion == "sad":
            rate = 115
            volume = 0.7
            processed_text = f"{text}..." # Adds trailing 'heavy' pause
        elif emotion == "angry":
            rate = 210
            volume = 1.0
            processed_text = text.upper() # Simulates intensity
        elif emotion == "surprise":
            rate = 190
            processed_text = f"Wait, {text}?" # Adds questioning inflection
        
        engine.setProperty('rate', rate)
        engine.setProperty('volume', volume)
        engine.say(processed_text)
        engine.runAndWait()
        engine.stop()

    threading.Thread(target=run, daemon=True).start()

# -----------------------------
# 2. Main Logic & Loop
# -----------------------------
sentence_buffer = []
last_prediction = None
last_committed_word = None
stable_counter = 0
current_emotion = "neutral"
emotion_last = None
emotion_counter = 0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # A. EMOTION (High Sensitivity Blendshapes)
    face_result = face_detector.detect(mp_image)
    detected_emo = "neutral"
    
    if face_result.face_blendshapes:
        # Access the first detected face's scores
        s = {b.category_name: b.score for b in face_result.face_blendshapes[0]}
        
        smile = (s.get("mouthSmileLeft", 0) + s.get("mouthSmileRight", 0)) / 2
        frown = (s.get("mouthFrownLeft", 0) + s.get("mouthFrownRight", 0)) / 2
        brow_inner_up = s.get("browInnerUp", 0)
        brow_down = (s.get("browDownLeft", 0) + s.get("browDownRight", 0)) / 2
        brow_up_L = s.get("browOuterUpLeft", 0)
        brow_up_R = s.get("browOuterUpRight", 0)
        eye_wide = (s.get("eyeWideLeft", 0) + s.get("eyeWideRight", 0)) / 2

        # Ultra-Sensitive Hierarchy
        if brow_down > 0.15 or abs(brow_up_L - brow_up_R) > 0.25: 
            detected_emo = "angry"
        elif frown > 0.07 or brow_inner_up > 0.12: 
            detected_emo = "sad"
        elif brow_up_L > 0.25 and brow_up_R > 0.25 and eye_wide > 0.3: 
            detected_emo = "surprise"
        elif smile > 0.25: 
            detected_emo = "happy"

    # Smooth emotion transitions
    if detected_emo == emotion_last: emotion_counter += 1
    else: emotion_counter, emotion_last = 1, detected_emo
    if emotion_counter >= 3: current_emotion = detected_emo

    # B. HAND RECOGNITION (Fixed Result Extraction)
    hand_result = hand_detector.detect(mp_image)
    active_word = "None"
    
    if hand_result.hand_landmarks:
        # Task API returns a list of landmarks for each hand
        landmarks = hand_result.hand_landmarks[0]
        features = []
        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z])
        
        # Classifier Prediction
        X_scaled = scaler.transform(np.array(features).reshape(1, -1))
        pred = classifier.predict(X_scaled)
        word = LABEL_TO_WORD.get(pred[0], pred[0])
        active_word = word
        
        if word == last_prediction: stable_counter += 1
        else: stable_counter, last_prediction = 1, word
        
        if stable_counter >= 6:
            if word == "FULL STOP":
                if sentence_buffer: 
                    speak_humane_offline(" ".join(sentence_buffer), current_emotion)
                sentence_buffer = []
            elif word != last_committed_word:
                sentence_buffer.append(word)
                last_committed_word = word
            stable_counter = 0

    # C. INTERFACE
    h, w, _ = frame.shape
    cv2.putText(frame, f"TONE: {current_emotion.upper()}", (20, 50), 2, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"SIGN: {active_word}", (20, 95), 2, 0.8, (255, 165, 0), 2)
    if sentence_buffer:
        cv2.rectangle(frame, (40, h-90), (w-40, h-30), (0, 0, 0), -1)
        cv2.putText(frame, " ".join(sentence_buffer), (60, h-50), 2, 1, (255, 255, 255), 2)

    cv2.imshow("Humane Offline VoiceBridge", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
face_detector.close()
hand_detector.close()
cv2.destroyAllWindows()
