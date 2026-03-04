import queue
import sounddevice as sd
import vosk
import json
import os
import cv2

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "models/vosk-model-small-en-us-0.15"
DEVICE_INDEX = 9   # your microphone index

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

vocab_path = os.path.join(BASE_DIR, "..", "data_collection", "vocabulary.json")
sign_map_path = os.path.join(BASE_DIR, "..", "data_collection", "signmap.json")

# -----------------------------
# Load Vocabulary
# -----------------------------
with open(vocab_path, "r") as f:
    vocabulary = json.load(f)

grammar = json.dumps(vocabulary)

# -----------------------------
# Load Sign Map
# -----------------------------
with open(sign_map_path, "r") as f:
    sign_map = json.load(f)

# -----------------------------
# Load Vosk Model
# -----------------------------
model = vosk.Model(MODEL_PATH)

q = queue.Queue()

device_info = sd.query_devices(DEVICE_INDEX, 'input')
samplerate = int(device_info['default_samplerate'])

# -----------------------------
# Audio Callback
# -----------------------------
def callback(indata, frames, time, status):

    if status:
        print(status)

    q.put(bytes(indata))

# -----------------------------
# Sign Player
# -----------------------------
def play_sign(video_path):

    if not os.path.exists(video_path):
        print("Video not found:", video_path)
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Cannot open video:", video_path)
        return

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow("ISL Interpreter", frame)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()

# -----------------------------
# Speech Recognition Loop
# -----------------------------
with sd.RawInputStream(
        samplerate=samplerate,
        blocksize=8000,
        dtype='int16',
        channels=1,
        callback=callback,
        device=DEVICE_INDEX):
    cv2.namedWindow("ISL Interpreter", cv2.WINDOW_NORMAL)
    recognizer = vosk.KaldiRecognizer(model, samplerate, grammar)

    last_sentence = ""
    try:

        print("🎤 Speak now... (Press Ctrl+C to stop)")

        while True:

            data = q.get()

            if recognizer.AcceptWaveform(data):

                result = json.loads(recognizer.Result())

                text = result.get("text", "").strip()

                if text != "" and text != last_sentence:

                    last_sentence = text

                    print("Recognized:", text)

                    tokens = text.split()

                    for word in tokens:

                        word = word.lower()

                        if word in sign_map:

                            video_path = os.path.join(BASE_DIR, "..", sign_map[word])
                            print("Video path:", video_path)

                            print("Playing:", word)

                            play_sign(video_path)

                        else:

                            print("No sign available for:", word)

    except KeyboardInterrupt:

        print("\nStopped.")