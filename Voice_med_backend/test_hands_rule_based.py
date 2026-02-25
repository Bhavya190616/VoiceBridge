import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# -----------------------------
# Thresholds
# -----------------------------
HELLO_THRESHOLD = 5
YES_THRESHOLD = 5

# -----------------------------
# Counters (per hand)
# -----------------------------
hello_counter = {"Left": 0, "Right": 0}
yes_counter = {"Left": 0, "Right": 0}

# -----------------------------
# Segmentation state
# -----------------------------
sentence_buffer = []
waiting_for_rest = False
active_hand = None   # <-- key change

# -----------------------------
# MediaPipe setup
# -----------------------------
MODEL_PATH = os.path.join("models", "hand_landmarker.task")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)

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

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for i, hand in enumerate(result.hand_landmarks):

            handedness = result.handedness[i][0].category_name

            # Fix handedness because frame is mirrored
            if handedness.lower() == "left":
                handedness = "Right"
            elif handedness.lower() == "right":
                handedness = "Left"

            # -----------------------------
            # Landmarks
            # -----------------------------
            thumb_tip = hand[4]
            thumb_ip = hand[3]

            index_tip = hand[8]
            index_pip = hand[6]

            middle_tip = hand[12]
            middle_pip = hand[10]

            ring_tip = hand[16]
            ring_pip = hand[14]

            pinky_tip = hand[20]
            pinky_pip = hand[18]

            # -----------------------------
            # Finger states
            # -----------------------------
            thumb_open_side = abs(thumb_tip.x - index_tip.x) > 0.03   # HELLO
            thumb_open_up = thumb_tip.y < thumb_ip.y                  # YES

            index_open = index_tip.y < index_pip.y + 0.02
            middle_open = middle_tip.y < middle_pip.y + 0.02
            ring_open = ring_tip.y < ring_pip.y + 0.02
            pinky_open = pinky_tip.y < pinky_pip.y + 0.02

            # -----------------------------
            # Rest state (true neutral)
            # -----------------------------
            is_rest = (
                not index_open and
                not middle_open and
                not ring_open and
                not pinky_open and
                not thumb_open_up
            )

            # -----------------------------
            # HELLO (open palm)
            # -----------------------------
            is_hello = (
                thumb_open_side and
                index_open and
                middle_open and
                ring_open and
                pinky_open
            )

            # -----------------------------
            # YES (thumbs up)
            # -----------------------------
            is_yes = (
                thumb_open_up and
                not index_open and
                not middle_open and
                not ring_open and
                not pinky_open
            )

            # -----------------------------
            # ACTIVE-HAND SEGMENTATION
            # -----------------------------
            if active_hand is None or handedness == active_hand:

                # Wait until rest before new word
                if waiting_for_rest:
                    if is_rest:
                        waiting_for_rest = False
                        active_hand = None
                        hello_counter[handedness] = 0
                        yes_counter[handedness] = 0
                    continue

                # ---------- HELLO ----------
                if is_hello:
                    hello_counter[handedness] += 1
                else:
                    hello_counter[handedness] = 0

                if hello_counter[handedness] == HELLO_THRESHOLD:
                    sentence_buffer.append("HELLO")
                    print("WORD ADDED: HELLO")
                    print("CURRENT SENTENCE:", sentence_buffer)
                    active_hand = handedness
                    waiting_for_rest = True
                    continue

                # ---------- YES ----------
                if is_yes:
                    yes_counter[handedness] += 1
                else:
                    yes_counter[handedness] = 0

                if yes_counter[handedness] == YES_THRESHOLD:
                    sentence_buffer.append("YES")
                    print("WORD ADDED: YES")
                    print("CURRENT SENTENCE:", sentence_buffer)
                    active_hand = handedness
                    waiting_for_rest = True

    cv2.imshow("VoiceBridge – ISL Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
