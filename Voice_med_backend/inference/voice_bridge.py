"""
VoiceBridge - Unified GUI
Embeds both Sign→Speech (camera) and Speech→Sign (video player) inside one window.
No pop-up OpenCV windows. Uses tkinter Label widgets as video canvases.
"""

import tkinter as tk
from tkinter import font as tkfont
import customtkinter as ctk
import cv2
import queue
import threading
import os
import json
import time
import numpy as np
from PIL import Image, ImageTk

# ─── Optional heavy imports (graceful fallback if models missing) ────────────
try:
    import joblib
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False

try:
    import pyttsx3
    TTS_OK = True
except ImportError:
    TTS_OK = False

try:
    import sounddevice as sd
    import vosk
    VOSK_OK = True
except ImportError:
    VOSK_OK = False

# ─── Paths ───────────────────────────────────────────────────────────────────
# Walk upward from this script's location to find any file by name.
# This works no matter where the script lives in the project tree.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _find_file(filename, search_roots=None):
    """
    Search for `filename` starting from BASE_DIR and its parents (up to 4 levels).
    Also searches common subdirectory names: models/, inference/models/, data_collection/.
    Returns the absolute path if found, otherwise returns a descriptive placeholder.
    """
    if search_roots is None:
        # Build a list of directories to search: BASE_DIR + up to 4 parent levels
        search_roots = []
        d = BASE_DIR
        for _ in range(5):
            search_roots.append(d)
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent

    subdirs = ["", "models", "inference/models", "inference",
               "data_collection", "../data_collection"]

    for root in search_roots:
        for sub in subdirs:
            candidate = os.path.normpath(os.path.join(root, sub, filename))
            if os.path.exists(candidate):
                return candidate

    # Not found — return a path that will produce a clear error message
    return os.path.join(BASE_DIR, "models", filename)

MODELS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "models"))

HAND_MODEL      = os.path.join(MODELS_DIR, "hand_landmarker.task")
FACE_MODEL      = os.path.join(MODELS_DIR, "face_landmarker.task")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "isl_classifier.pkl")
SCALER_PATH     = os.path.join(MODELS_DIR, "scaler.pkl")
VOSK_MODEL      = os.path.join(MODELS_DIR, "vosk-model-small-en-us-0.15")
VOCAB_PATH      = os.path.normpath(os.path.join(BASE_DIR, "..", "data_collection", "vocabulary.json"))
SIGNMAP_PATH    = os.path.normpath(os.path.join(BASE_DIR, "..", "data_collection", "signmap.json"))
LABEL_MAP_PATH  = os.path.join(BASE_DIR, "labelmap.py")

# ─── Debug: print resolved paths on startup ──────────────────────────────────
print("=== VoiceBridge Path Resolution ===")
for name, path in [("HAND_MODEL", HAND_MODEL), ("FACE_MODEL", FACE_MODEL),
                   ("CLASSIFIER", CLASSIFIER_PATH), ("SCALER", SCALER_PATH),
                   ("VOSK_MODEL", VOSK_MODEL), ("VOCAB", VOCAB_PATH),
                   ("SIGNMAP", SIGNMAP_PATH), ("LABELMAP", LABEL_MAP_PATH)]:
    status = "✓" if os.path.exists(path) else "✗ MISSING"
    print(f"  {status}  {name}: {path}")
print("===================================")

# ─── Label map ───────────────────────────────────────────────────────────────
LABEL_TO_WORD = {}
try:
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("labelmap", LABEL_MAP_PATH)
    lm   = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lm)
    LABEL_TO_WORD = lm.LABEL_TO_WORD
    print(f"  ✓  Loaded {len(LABEL_TO_WORD)} labels")
except Exception as e:
    print(f"  ✗  labelmap load failed: {e}")

DEVICE_INDEX = 9  # change to your mic index

# ═════════════════════════════════════════════════════════════════════════════
# SIGN → SPEECH ENGINE (runs in background thread, posts frames to queue)
# ═════════════════════════════════════════════════════════════════════════════
class SignToSpeechEngine:
    def __init__(self, frame_q: queue.Queue, status_cb, emotion_cb=None):
        self.frame_q   = frame_q
        self.status_cb = status_cb
        self.emotion_cb = emotion_cb  # called with emotion string on change
        self._stop     = threading.Event()
        self._thread   = None

        self.sentence_buffer   = []
        self.last_prediction   = None
        self.last_committed    = None
        self.stable_counter    = 0
        self.current_emotion   = "neutral"
        self.emotion_last      = None
        self.emotion_counter   = 0

        # ── Emotion smoothing ─────────────────────────────────────────────────
        # Exponential moving averages of each blendshape score (α = 0.25)
        # so noisy single-frame spikes don't flip the emotion
        self._ema = {}
        # Frames the NEW emotion must hold CONSECUTIVELY before committing
        self._hold_needed = {
            "neutral" : 10,  # hard to fall back — prevents flicker
            "happy"   : 5,   # smile is clear and strong
            "sad"     : 14,  # subtle, needs long sustained hold
            "angry"   : 10,  # brow_down must sustain
            "surprise": 8,
            "fear"    : 10,
            "disgust" : 10,
        }
        self._min_gap        = 0.08   # winner must beat 2nd place by this margin
        self._candidate      = "neutral"
        self._candidate_count= 0

    # ── Humanized TTS ─────────────────────────────────────────────────────────
    def _speak(self, text, emotion):
        if not TTS_OK:
            return
        def run():
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')

            # Pick warmest available voice (Zira on Windows is softer than David)
            preferred_voice = None
            for v in voices:
                name = v.name.lower()
                if any(k in name for k in ["zira", "hazel", "susan", "female"]):
                    preferred_voice = v.id
                    break
            if preferred_voice:
                engine.setProperty('voice', preferred_voice)
            elif len(voices) > 1:
                engine.setProperty('voice', voices[1].id)

            words = text.strip().rstrip(".!?,")

            # ── Each emotion uses punctuation/rhythm to shape delivery naturally
            # No interjections — pyttsx3 uses punctuation for pacing/pitch hints
            if emotion == "happy":
                rate, volume = 168, 1.0
                # Comma after first word creates a natural lift; ends on high note
                parts = words.split(" ", 1)
                if len(parts) > 1:
                    processed = f"{parts[0]}, {parts[1]}!"
                else:
                    processed = f"{words}!"

            elif emotion == "sad":
                rate, volume = 105, 0.70
                # Commas between words slow pacing; ellipsis trails off naturally
                spaced = ", ".join(words.split())
                processed = f"{spaced}..."

            elif emotion == "angry":
                rate, volume = 188, 1.0
                # No softening — flat declarative, slight pause mid-sentence
                mid = len(words) // 2
                space = words.rfind(" ", 0, mid)
                if space > 0:
                    processed = words[:space] + ". " + words[space+1:] + "."
                else:
                    processed = f"{words}."

            elif emotion == "surprise":
                rate, volume = 175, 1.0
                # Em-dash creates a beat/stutter before the message
                processed = f"— {words}?"

            elif emotion == "fear":
                rate, volume = 178, 0.75
                processed = f"{words}... please."

            elif emotion == "disgust":
                rate, volume = 128, 0.82
                processed = f"{words}."

            else:  # neutral
                rate, volume = 155, 0.90
                processed = f"{words}."

            engine.setProperty('rate', rate)
            engine.setProperty('volume', volume)
            # Small leading silence so first word isn't clipped
            engine.say(f", {processed}")
            engine.runAndWait()
            engine.stop()

        threading.Thread(target=run, daemon=True).start()

    # ── Main loop (runs in its own thread) ───────────────────────────────────
    def _run(self):
        if not MEDIAPIPE_OK:
            self.status_cb("MediaPipe not installed"); return

        # Pre-flight check
        missing = [p for p in [CLASSIFIER_PATH, SCALER_PATH, HAND_MODEL, FACE_MODEL] if not os.path.exists(p)]
        if missing:
            self.status_cb("Missing: " + ", ".join(os.path.basename(p) for p in missing)); return
        try:
            classifier = joblib.load(CLASSIFIER_PATH)
            scaler     = joblib.load(SCALER_PATH)
        except Exception as e:
            self.status_cb(f"Model load error: {e}"); return

        # Hand landmarker
        hand_det = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
                num_hands=1))
        # Face landmarker
        face_det = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL),
                output_face_blendshapes=True, num_faces=1))

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.status_cb("Camera not found"); return

        self.status_cb("Running")
        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # ── Emotion detection with EMA smoothing ──────────────────────────
            face_result = face_det.detect(mp_img)
            detected_emo = "neutral"

            if face_result.face_blendshapes:
                raw = {b.category_name: b.score
                       for b in face_result.face_blendshapes[0]}

                # EMA smoothing — α=0.30 for responsiveness while killing single-frame spikes
                α = 0.30
                for k, v in raw.items():
                    self._ema[k] = α * v + (1 - α) * self._ema.get(k, v)
                s = self._ema

                smile      = (s.get("mouthSmileLeft",0)   + s.get("mouthSmileRight",0))  / 2
                frown      = (s.get("mouthFrownLeft",0)   + s.get("mouthFrownRight",0))  / 2
                brow_inner = s.get("browInnerUp", 0)
                brow_down  = (s.get("browDownLeft",0)     + s.get("browDownRight",0))     / 2
                brow_L     = s.get("browOuterUpLeft",  0)
                brow_R     = s.get("browOuterUpRight", 0)
                eye_wide   = (s.get("eyeWideLeft",0)      + s.get("eyeWideRight",0))      / 2
                jaw_open   = s.get("jawOpen", 0)
                nose_sneer = (s.get("noseSneerLeft",0)    + s.get("noseSneerRight",0))    / 2
                lip_upper  = (s.get("mouthUpperUpLeft",0) + s.get("mouthUpperUpRight",0)) / 2
                lip_stretch= (s.get("mouthStretchLeft",0) + s.get("mouthStretchRight",0)) / 2
                mouth_press= (s.get("mouthPressLeft",0)   + s.get("mouthPressRight",0))   / 2
                mouth_pucker= s.get("mouthPucker", 0)
                mouth_funnel= s.get("mouthFunnel", 0)
                cheek_squint= (s.get("cheekSquintLeft",0) + s.get("cheekSquintRight",0))  / 2
                eye_squint = (s.get("eyeSquintLeft",0)    + s.get("eyeSquintRight",0))    / 2

                # ── DEBUG: print key scores every 30 frames ──────────────────
                self._dbg_frame = getattr(self, "_dbg_frame", 0) + 1
                if self._dbg_frame % 30 == 0:
                    print(
                        f"[EMO] smile={smile:.3f} brow_inner={brow_inner:.3f} "
                        f"brow_down={brow_down:.3f} brow_L={brow_L:.3f} brow_R={brow_R:.3f} "
                        f"eye_wide={eye_wide:.3f} jaw={jaw_open:.3f} "
                        f"mouth_press={mouth_press:.3f} mouth_pucker={mouth_pucker:.3f} "
                        f"sneer={nose_sneer:.3f} "
                        f"  → {self._candidate}({self._candidate_count}) | {self.current_emotion}"
                    )

                # ══════════════════════════════════════════════════════════════
                # FINAL CALIBRATION — based on full debug session analysis
                #
                # NEUTRAL RESTING VALUES (these must NOT trigger anything):
                #   brow_L = 0.35-0.55  ← naturally HIGH, left brow sits elevated
                #   brow_R = 0.15-0.35  ← lower but still drifts up
                #   brow_down = 0.004-0.012  ← nearly zero at rest
                #   brow_inner = 0.002-0.030 ← very low at rest
                #   mouth_press = 0.02-0.07 ← some resting lip tension
                #   smile = 0.000-0.003 ← near zero at rest
                #
                # EXPRESSION SIGNATURES:
                #   SURPRISE: brow_R jumps to 0.30-0.48 (right brow is the reliable one)
                #             brow_L already high so useless alone
                #             → Use brow_R excess above 0.28 as primary gating signal
                #   ANGRY:    brow_down spikes to 0.20-0.37 (squinting hard)
                #             brow_inner barely moves (0.000-0.047) → not useful
                #             → brow_down excess above 0.15 is the angry signal
                #   SAD:      mouth_press 0.10-0.39 + brow_L/R both DROP (brows go flat)
                #             brow knit = brow_L drops from 0.45→0.17, brow_R drops too
                #             → detect brow DROP + mouth_press together
                #   HAPPY:    smile 0.30-0.70 → clean and reliable ✓
                # ══════════════════════════════════════════════════════════════

                # ══ FINAL CALIBRATION ══════════════════════════════════════
                # ANGRY: brow knit = brows pull DOWN + TOGETHER
                #   Your angry brow_down: 0.08-0.35 depending on intensity
                #   Lowered threshold to 0.06 to catch mild angry too
                #   Also use brow_inner as a secondary boost (pulls brows together)
                # ANGRY: brow_down is your main signal (0.028-0.093 when angry)
                # brow_inner stays near 0.001 — not useful for you at all
                # resting brow_down is ~0.002, so threshold of 0.018 is safe
                angry_excess = max(0, brow_down - 0.018)
                angry_score  = (angry_excess * 14.0       # high weight — main signal
                                - smile * 5.0
                                - jaw_open * 2.0)

                # SURPRISE: eye_wide + jaw_open are the real signals
                brow_R_genuine = max(0, brow_R - 0.40)
                eye_jaw_boost  = eye_wide * 4.0 + jaw_open * 3.0
                surprise_score = (eye_jaw_boost
                                  + brow_R_genuine * eye_jaw_boost * 8.0
                                  - brow_down * 5.0)

                # HAPPY: smile is clean and reliable
                happy_score = smile * 3.5 + cheek_squint * 0.8

                # SAD: mouth_pucker + press — but ONLY when brows are flat/resting
                # When angry: brow_down rises → angry_excess rises → sad cancelled
                # mouth_pucker alone should NOT trigger sad if brows are furrowed
                brow_L_drop  = max(0, 0.30 - brow_L)
                brow_scrunch = max(0, brow_down - 0.018)  # same threshold as angry
                sad_score    = (mouth_press * 3.0
                                + mouth_pucker * 2.0       # reduced weight
                                + brow_L_drop * 1.0
                                + frown * 1.5
                                - smile * 5.0
                                - jaw_open * 2.0
                                - brow_scrunch * 12.0      # HARD cancel when brows furrow
                                - angry_excess * 8.0)      # angry firing = definitely not sad

                disgust_score = nose_sneer * 4.0 + lip_upper * 1.5 - smile * 2.0
                fear_score    = (max(0, eye_wide - 0.08) * 3.0
                                 + lip_stretch * 1.0
                                 - brow_down * 2.0 - smile * 2.0)

                scores = {
                    "happy"   : happy_score,
                    "sad"     : sad_score,
                    "angry"   : angry_score,
                    "surprise": surprise_score,
                    "disgust" : disgust_score,
                    "fear"    : fear_score,
                    "neutral" : 0.16,
                }

                best       = max(scores, key=scores.get)
                best_score = scores[best]

                if self._dbg_frame % 30 == 0:
                    print(f"[SCORES] happy={happy_score:.2f} sad={sad_score:.2f} "
                          f"angry={angry_score:.2f} surprise={surprise_score:.2f} "
                          f"brow_down={brow_down:.3f} brow_inner={brow_inner:.3f} "
                          f"  → {best}({best_score:.2f}) | was={self.current_emotion}")

                # Hysteresis: need higher score to ENTER emotion from neutral
                # prevents flickering at boundaries
                if self.current_emotion == "neutral":
                    threshold = 0.18   # lowered — angry needs to break through
                else:
                    threshold = 0.12

                # Dominance gap: winner must beat 2nd place by min margin
                sorted_scores = sorted(scores.values(), reverse=True)
                gap = sorted_scores[0] - sorted_scores[1]
                if best != "neutral" and best_score > threshold and gap >= self._min_gap:
                    detected_emo = best
                else:
                    detected_emo = "neutral"

            # ── Per-emotion hold-count gating ─────────────────────────────────────────────────────
            if detected_emo == self._candidate:
                self._candidate_count += 1
            else:
                # Ignore single-frame blips — need 2+ frames to reset
                if self._candidate_count >= 2:
                    self._candidate       = detected_emo
                    self._candidate_count = 1

            needed = self._hold_needed.get(detected_emo, 8)
            if self._candidate_count >= needed:
                if self._candidate != self.current_emotion:
                    self.current_emotion = self._candidate
                    if self.emotion_cb:
                        self.emotion_cb(self.current_emotion)

            # Hand
            hand_result = hand_det.detect(mp_img)
            active_word = ""
            if hand_result.hand_landmarks:
                landmarks = hand_result.hand_landmarks[0]
                features  = []
                for lm in landmarks: features.extend([lm.x,lm.y,lm.z])
                X_scaled  = scaler.transform(np.array(features).reshape(1,-1))
                pred      = classifier.predict(X_scaled)
                word      = LABEL_TO_WORD.get(pred[0], str(pred[0]))
                active_word = word
                if word==self.last_prediction: self.stable_counter+=1
                else: self.stable_counter,self.last_prediction=1,word
                if self.stable_counter>=6:
                    if word=="FULL STOP":
                        if self.sentence_buffer:
                            self._speak(" ".join(self.sentence_buffer), self.current_emotion)
                        self.sentence_buffer=[]
                    elif word!=self.last_committed:
                        self.sentence_buffer.append(word)
                        self.last_committed=word
                    self.stable_counter=0

            # Overlay on the frame
            h,w,_ = frame.shape
            # Emotion badge
            emo_colors = {
                "neutral" :(140,140,150),
                "happy"   :(0,220,100),
                "sad"     :(100,140,220),
                "angry"   :(60,60,220),
                "surprise":(0,200,230),
                "fear"    :(180,80,220),
                "disgust" :(80,180,80),
            }
            emo_icons = {
                "neutral":"😐","happy":"😊","sad":"😢",
                "angry":"😠","surprise":"😲","fear":"😨","disgust":"🤢"
            }
            ec  = emo_colors.get(self.current_emotion,(140,140,150))
            # Top bar
            cv2.rectangle(frame,(0,0),(w,58),(12,14,22),-1)
            cv2.rectangle(frame,(0,0),(w,3),ec,-1)   # thin color accent line
            cv2.putText(frame,
                        f"EMOTION: {self.current_emotion.upper()}",
                        (14,38), cv2.FONT_HERSHEY_DUPLEX, 0.82, ec, 2)

            # Sign label (right-aligned in top bar)
            if active_word:
                label = f"SIGN: {active_word}"
                (tw,_),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.72, 1)
                cv2.putText(frame, label,
                            (w-tw-14, 38), cv2.FONT_HERSHEY_DUPLEX, 0.72, (255,165,0), 2)

            # Sentence buffer at bottom
            if self.sentence_buffer:
                txt = " ".join(self.sentence_buffer)
                cv2.rectangle(frame,(0,h-54),(w,h),(12,14,22),-1)
                cv2.rectangle(frame,(0,h-3),(w,h),ec,-1)  # bottom accent line
                cv2.putText(frame, txt, (14,h-16),
                            cv2.FONT_HERSHEY_DUPLEX, 0.82, (240,240,255), 2)

            # Push annotated frame (convert BGR→RGB for PIL)
            annotated_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try: self.frame_q.put_nowait(annotated_rgb)
            except queue.Full: pass

        cap.release()
        hand_det.close()
        face_det.close()
        self.status_cb("Stopped")

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()


# ═════════════════════════════════════════════════════════════════════════════
# SPEECH → SIGN ENGINE
# ═════════════════════════════════════════════════════════════════════════════
class SpeechToSignEngine:
    def __init__(self, frame_q: queue.Queue, status_cb):
        self.frame_q   = frame_q
        self.status_cb = status_cb
        self._stop     = threading.Event()
        self._thread   = None

    def _run(self):
        if not VOSK_OK:
            self.status_cb("Vosk not installed"); return
        try:
            with open(VOCAB_PATH)  as f: vocabulary = json.load(f)
            with open(SIGNMAP_PATH) as f: sign_map   = json.load(f)
        except Exception as e:
            self.status_cb(f"Config error: {e}"); return
        if not os.path.exists(VOSK_MODEL):
            self.status_cb(
                f"Vosk model not found at:\n{VOSK_MODEL}\n"
                "Download vosk-model-small-en-us-0.15 and place in models/ folder."
            ); return
        try:
            model = vosk.Model(VOSK_MODEL)
        except Exception as e:
            self.status_cb(f"Vosk model error: {e}"); return

        grammar    = json.dumps(vocabulary)
        audio_q    = queue.Queue()

        try:
            dev_info   = sd.query_devices(DEVICE_INDEX,'input')
            samplerate = int(dev_info['default_samplerate'])
        except Exception as e:
            self.status_cb(f"Mic error: {e}"); return

        def audio_cb(indata,frames,t,status):
            audio_q.put(bytes(indata))

        recognizer   = vosk.KaldiRecognizer(model, samplerate, grammar)
        last_sentence= ""

        self.status_cb("Listening…")
        try:
            with sd.RawInputStream(samplerate=samplerate, blocksize=8000,
                                   dtype='int16', channels=1,
                                   callback=audio_cb, device=DEVICE_INDEX):
                while not self._stop.is_set():
                    try: data = audio_q.get(timeout=0.5)
                    except queue.Empty: continue

                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        text   = result.get("text","").strip()
                        if text and text!=last_sentence:
                            last_sentence = text
                            self.status_cb(f'Heard: "{text}"')
                            for word in text.split():
                                if self._stop.is_set(): break
                                word = word.lower()
                                if word in sign_map:
                                    vpath = os.path.join(BASE_DIR,"..",sign_map[word])
                                    self._play_video(vpath, word)
                                else:
                                    # Show "no sign" placeholder
                                    self._show_placeholder(word)
        except Exception as e:
            self.status_cb(f"Stream error: {e}")
        self.status_cb("Stopped")

    def _play_video(self, path, word):
        if not os.path.exists(path):
            self._show_placeholder(word); return
        cap = cv2.VideoCapture(path)
        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h,w,_ = frame.shape
            # word label
            cv2.rectangle(frame,(0,h-50),(w,h),(15,15,25),-1)
            cv2.putText(frame, word.upper(),(12,h-14),
                        cv2.FONT_HERSHEY_DUPLEX,1.0,(0,220,180),2)
            try: self.frame_q.put_nowait(frame)
            except queue.Full: pass
            time.sleep(1/30)
        cap.release()

    def _show_placeholder(self, word):
        for _ in range(20):           # ~0.6 s placeholder
            if self._stop.is_set(): return
            img = np.zeros((300,400,3),dtype=np.uint8)
            img[:] = (25,25,35)
            cv2.putText(img, "No sign for:",(30,130),
                        cv2.FONT_HERSHEY_DUPLEX,0.8,(100,100,120),1)
            cv2.putText(img, f'"{word}"',(30,180),
                        cv2.FONT_HERSHEY_DUPLEX,1.1,(0,180,220),2)
            try: self.frame_q.put_nowait(img)
            except queue.Full: pass
            time.sleep(1/30)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()


# ═════════════════════════════════════════════════════════════════════════════
# GUI
# ═════════════════════════════════════════════════════════════════════════════

DARK_BG   = "#0d0f14"
PANEL_BG  = "#13161e"
CARD_BG   = "#1a1d27"
ACCENT1   = "#00d4aa"   # teal  – Sign→Speech
ACCENT2   = "#7b5cf0"   # violet – Speech→Sign
MUTED     = "#4a4f6a"
TEXT_PRI  = "#e8eaf2"
TEXT_SEC  = "#7a7f99"
RED       = "#e05c5c"

CANVAS_W, CANVAS_H = 460, 300

class VoiceBridgeApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VoiceBridge")
        self.geometry("1080x720")
        self.minsize(900,620)
        self.configure(fg_color=DARK_BG)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Video queues
        self.sign_q   = queue.Queue(maxsize=2)
        self.speech_q = queue.Queue(maxsize=2)

        # Engines
        self.sign_engine   = SignToSpeechEngine(self.sign_q,   self._sign_status,
                                                emotion_cb=self._on_emotion_change)
        self.speech_engine = SpeechToSignEngine(self.speech_q, self._speech_status)

        self._sign_running   = False
        self._speech_running = False

        # Placeholder images (drawn once)
        self._sign_placeholder   = self._make_placeholder("📷  Camera feed will appear here",   ACCENT1)
        self._speech_placeholder = self._make_placeholder("🎤  Sign animation will appear here", ACCENT2)

        self._build_ui()
        self._poll_frames()

    # ── Placeholder ──────────────────────────────────────────────────────────
    def _make_placeholder(self, msg, color_hex):
        img = Image.new("RGB",(CANVAS_W,CANVAS_H),(20,22,32))
        # simple grid lines for style
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        r,g,b = int(color_hex[1:3],16),int(color_hex[3:5],16),int(color_hex[5:7],16)
        for x in range(0,CANVAS_W,40):
            draw.line([(x,0),(x,CANVAS_H)], fill=(r//6,g//6,b//6,255), width=1)
        for y in range(0,CANVAS_H,40):
            draw.line([(0,y),(CANVAS_W,y)], fill=(r//6,g//6,b//6,255), width=1)
        draw.text((CANVAS_W//2, CANVAS_H//2), msg, fill=(r//2,g//2,b//2),
                  anchor="mm")
        return ImageTk.PhotoImage(img)

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header ──
        hdr = ctk.CTkFrame(self, fg_color=PANEL_BG, corner_radius=0, height=64)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        ctk.CTkLabel(hdr, text="Voice", font=("Trebuchet MS",28,"bold"),
                     text_color=ACCENT1).place(x=28, y=14)
        ctk.CTkLabel(hdr, text="Bridge", font=("Trebuchet MS",28,"bold"),
                     text_color=ACCENT2).place(x=94, y=14)
        ctk.CTkLabel(hdr, text="Indian Sign Language ↔ Speech Bridge",
                     font=("Trebuchet MS",12), text_color=MUTED).place(x=29, y=42)

        # status dot right side
        self._hdr_dot = ctk.CTkLabel(hdr, text="● IDLE", font=("Courier",11),
                                     text_color=MUTED)
        self._hdr_dot.place(relx=1.0, x=-20, y=22, anchor="e")

        # ── Main 2-column grid ──
        body = ctk.CTkFrame(self, fg_color=DARK_BG)
        body.pack(fill="both", expand=True, padx=18, pady=14)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self._build_sign_panel(body)
        self._build_speech_panel(body)

    # ── Sign → Speech panel (right) ──────────────────────────────────────────
    def _build_sign_panel(self, parent):
        card = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=14)
        card.grid(row=0, column=1, padx=(8,0), pady=0, sticky="nsew")

        # Title bar
        tbar = ctk.CTkFrame(card, fg_color=PANEL_BG, corner_radius=10, height=44)
        tbar.pack(fill="x", padx=10, pady=(10,0))
        tbar.pack_propagate(False)
        ctk.CTkLabel(tbar, text="Sign  →  Speech",
                     font=("Trebuchet MS",15,"bold"),
                     text_color=ACCENT1).pack(side="left", padx=14, pady=8)
        self._sign_badge = ctk.CTkLabel(tbar, text="● OFF",
                                        font=("Courier",10), text_color=MUTED)
        self._sign_badge.pack(side="right", padx=14)

        # Video canvas
        self._sign_canvas = tk.Label(card, bg="#14161f",
                                     image=self._sign_placeholder,
                                     cursor="crosshair")
        self._sign_canvas.pack(padx=10, pady=(10,4))

        # ── Emotion bar ──────────────────────────────────────────────────────
        emo_bar = ctk.CTkFrame(card, fg_color=PANEL_BG, corner_radius=8, height=36)
        emo_bar.pack(fill="x", padx=10, pady=(0,6))
        emo_bar.pack_propagate(False)

        ctk.CTkLabel(emo_bar, text="MOOD",
                     font=("Courier",10,"bold"), text_color=MUTED).pack(side="left", padx=10)

        self._emo_icons = {
            "neutral" : ("😐", MUTED),
            "happy"   : ("😊", "#00dc6a"),
            "sad"     : ("😢", "#6496dc"),
            "angry"   : ("😠", "#e05c5c"),
            "surprise": ("😲", "#00c8e6"),
            "fear"    : ("😨", "#b464dc"),
            "disgust" : ("🤢", "#50b450"),
        }

        self._emo_dot   = ctk.CTkLabel(emo_bar, text="●",
                                       font=("Arial",14), text_color=MUTED)
        self._emo_dot.pack(side="left", padx=(4,2))

        self._emo_label = ctk.CTkLabel(emo_bar, text="neutral",
                                       font=("Trebuchet MS",13,"bold"), text_color=MUTED)
        self._emo_label.pack(side="left")

        # Emotion pill indicators (all 7)
        pills_frame = ctk.CTkFrame(emo_bar, fg_color="transparent")
        pills_frame.pack(side="right", padx=8)
        self._emo_pills = {}
        for emo, (icon, color) in self._emo_icons.items():
            pill = ctk.CTkLabel(pills_frame, text=icon,
                                font=("Arial",16), text_color=MUTED)
            pill.pack(side="left", padx=2)
            self._emo_pills[emo] = (pill, color)

        # Status
        self._sign_status_var = tk.StringVar(value="Ready")
        ctk.CTkLabel(card, textvariable=self._sign_status_var,
                     font=("Courier",11), text_color=TEXT_SEC,
                     wraplength=CANVAS_W-20).pack(padx=10)

        # Sentence readout
        self._sign_sentence = tk.StringVar(value="")
        ctk.CTkLabel(card, textvariable=self._sign_sentence,
                     font=("Trebuchet MS",13), text_color=TEXT_PRI,
                     wraplength=CANVAS_W-20).pack(padx=10, pady=(4,8))

        # Buttons
        btns = ctk.CTkFrame(card, fg_color="transparent")
        btns.pack(pady=(0,14))
        self._sign_start_btn = ctk.CTkButton(
            btns, text="▶  Start Camera",
            fg_color=ACCENT1, hover_color="#00b892",
            text_color="#0d0f14", font=("Trebuchet MS",13,"bold"),
            width=160, height=38, corner_radius=8,
            command=self._toggle_sign)
        self._sign_start_btn.pack(side="left", padx=6)

        ctk.CTkButton(
            btns, text="🔊  Speak Buffer",
            fg_color=PANEL_BG, hover_color=CARD_BG,
            text_color=ACCENT1, border_width=1, border_color=ACCENT1,
            font=("Trebuchet MS",12), width=140, height=38, corner_radius=8,
            command=self._speak_now).pack(side="left", padx=6)

    # ── Speech → Sign panel (left) ───────────────────────────────────────────
    def _build_speech_panel(self, parent):
        card = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=14)
        card.grid(row=0, column=0, padx=(0,8), pady=0, sticky="nsew")

        tbar = ctk.CTkFrame(card, fg_color=PANEL_BG, corner_radius=10, height=44)
        tbar.pack(fill="x", padx=10, pady=(10,0))
        tbar.pack_propagate(False)
        ctk.CTkLabel(tbar, text="Speech  →  Sign",
                     font=("Trebuchet MS",15,"bold"),
                     text_color=ACCENT2).pack(side="left", padx=14, pady=8)
        self._speech_badge = ctk.CTkLabel(tbar, text="● OFF",
                                          font=("Courier",10), text_color=MUTED)
        self._speech_badge.pack(side="right", padx=14)

        self._speech_canvas = tk.Label(card, bg="#14161f",
                                       image=self._speech_placeholder)
        self._speech_canvas.pack(padx=10, pady=10)

        self._speech_status_var = tk.StringVar(value="Ready")
        ctk.CTkLabel(card, textvariable=self._speech_status_var,
                     font=("Courier",11), text_color=TEXT_SEC,
                     wraplength=CANVAS_W-20).pack(padx=10)

        self._speech_word = tk.StringVar(value="")
        ctk.CTkLabel(card, textvariable=self._speech_word,
                     font=("Trebuchet MS",13), text_color=TEXT_PRI,
                     wraplength=CANVAS_W-20).pack(padx=10, pady=(4,8))

        btns = ctk.CTkFrame(card, fg_color="transparent")
        btns.pack(pady=(0,14))
        self._speech_start_btn = ctk.CTkButton(
            btns, text="▶  Start Listening",
            fg_color=ACCENT2, hover_color="#6448d4",
            text_color="#ffffff", font=("Trebuchet MS",13,"bold"),
            width=160, height=38, corner_radius=8,
            command=self._toggle_speech)
        self._speech_start_btn.pack(side="left", padx=6)

    def _on_emotion_change(self, emotion):
        """Called from engine thread — schedule UI update on main thread."""
        self.after(0, lambda: self._update_emotion_bar(emotion))

    def _update_emotion_bar(self, emotion):
        icon, color = self._emo_icons.get(emotion, ("😐", MUTED))
        self._emo_dot.configure(text_color=color)
        self._emo_label.configure(text=emotion, text_color=color)
        # Highlight active pill, dim all others
        for emo, (pill, pill_color) in self._emo_pills.items():
            if emo == emotion:
                pill.configure(text_color=pill_color)
            else:
                pill.configure(text_color=MUTED)

    # ── Toggle handlers ───────────────────────────────────────────────────────
    def _toggle_sign(self):
        if not self._sign_running:
            self.sign_engine.start()
            self._sign_running = True
            self._sign_start_btn.configure(text="■  Stop Camera",
                                           fg_color=RED, hover_color="#c04040")
            self._sign_badge.configure(text="● ON", text_color=ACCENT1)
            self._update_header_dot()
        else:
            self.sign_engine.stop()
            self._sign_running = False
            self._sign_start_btn.configure(text="▶  Start Camera",
                                           fg_color=ACCENT1, hover_color="#00b892",
                                           text_color="#0d0f14")
            self._sign_badge.configure(text="● OFF", text_color=MUTED)
            self._sign_canvas.configure(image=self._sign_placeholder)
            self._update_header_dot()

    def _toggle_speech(self):
        if not self._speech_running:
            self.speech_engine.start()
            self._speech_running = True
            self._speech_start_btn.configure(text="■  Stop Listening",
                                             fg_color=RED, hover_color="#c04040")
            self._speech_badge.configure(text="● ON", text_color=ACCENT2)
            self._update_header_dot()
        else:
            self.speech_engine.stop()
            self._speech_running = False
            self._speech_start_btn.configure(text="▶  Start Listening",
                                             fg_color=ACCENT2, hover_color="#6448d4")
            self._speech_badge.configure(text="● OFF", text_color=MUTED)
            self._speech_canvas.configure(image=self._speech_placeholder)
            self._update_header_dot()

    def _speak_now(self):
        """Manually trigger TTS for the current sign buffer."""
        if not self.sign_engine.sentence_buffer: return
        self.sign_engine._speak(
            " ".join(self.sign_engine.sentence_buffer),
            self.sign_engine.current_emotion)

    def _update_header_dot(self):
        if self._sign_running and self._speech_running:
            self._hdr_dot.configure(text="● ACTIVE", text_color=ACCENT1)
        elif self._sign_running or self._speech_running:
            self._hdr_dot.configure(text="● RUNNING", text_color=ACCENT2)
        else:
            self._hdr_dot.configure(text="● IDLE", text_color=MUTED)

    # ── Status callbacks (called from worker threads) ─────────────────────────
    def _sign_status(self, msg):
        self.after(0, lambda: self._sign_status_var.set(msg))

    def _speech_status(self, msg):
        self.after(0, lambda: self._speech_status_var.set(msg))

    # ── Frame polling (runs every 30 ms on main thread) ───────────────────────
    def _poll_frames(self):
        # Sign → Speech frame
        try:
            frame = self.sign_q.get_nowait()
            img   = Image.fromarray(frame).resize((CANVAS_W, CANVAS_H), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._sign_canvas.configure(image=photo)
            self._sign_canvas._photo = photo          # keep reference
            # update sentence readout
            buf = self.sign_engine.sentence_buffer
            self._sign_sentence.set(" ".join(buf) if buf else "")
        except queue.Empty:
            pass

        # Speech → Sign frame
        try:
            frame = self.speech_q.get_nowait()
            img   = Image.fromarray(frame).resize((CANVAS_W, CANVAS_H), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._speech_canvas.configure(image=photo)
            self._speech_canvas._photo = photo
        except queue.Empty:
            pass

        self.after(30, self._poll_frames)

    # ── Clean shutdown ────────────────────────────────────────────────────────
    def destroy(self):
        self.sign_engine.stop()
        self.speech_engine.stop()
        super().destroy()


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = VoiceBridgeApp()
    app.mainloop()