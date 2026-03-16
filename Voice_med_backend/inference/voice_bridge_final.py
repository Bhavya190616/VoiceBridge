"""
VoiceBridge — Professional UI
Emotion-Aware Bidirectional ISL ↔ Speech System
"""

import tkinter as tk
import customtkinter as ctk
import cv2, queue, threading, os, json, time, re
import numpy as np
from PIL import Image, ImageTk, ImageFont, ImageDraw

# ─── Optional heavy imports ───────────────────────────────────────────────────
try:
    import joblib, mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False

try:
    import pyttsx3; TTS_OK = True
except ImportError:
    TTS_OK = False

try:
    import sounddevice as sd, vosk; VOSK_OK = True
except ImportError:
    VOSK_OK = False

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "models"))
HAND_MODEL      = os.path.join(MODELS_DIR, "hand_landmarker.task")
FACE_MODEL      = os.path.join(MODELS_DIR, "face_landmarker.task")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "isl.pkl")
SCALER_PATH     = os.path.join(MODELS_DIR, "scaler.pkl")
VOSK_MODEL      = os.path.join(MODELS_DIR, "vosk-model-small-en-us-0.15")
VOCAB_PATH      = os.path.normpath(os.path.join(BASE_DIR, "..", "data_collection", "vocabulary.json"))
SIGNMAP_PATH    = os.path.normpath(os.path.join(BASE_DIR, "..", "data_collection", "signmap.json"))
LABEL_MAP_PATH  = os.path.join(BASE_DIR, "labelmap.py")

print("=== VoiceBridge Path Resolution ===")
for name, path in [("HAND_MODEL",HAND_MODEL),("FACE_MODEL",FACE_MODEL),
                   ("CLASSIFIER",CLASSIFIER_PATH),("SCALER",SCALER_PATH),
                   ("VOSK_MODEL",VOSK_MODEL),("VOCAB",VOCAB_PATH),
                   ("SIGNMAP",SIGNMAP_PATH),("LABELMAP",LABEL_MAP_PATH)]:
    print(f"  {'✓' if os.path.exists(path) else '✗ MISSING'}  {name}: {path}")
print("===================================")

LABEL_TO_WORD = {}
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("labelmap", LABEL_MAP_PATH)
    lm   = importlib.util.module_from_spec(spec); spec.loader.exec_module(lm)
    LABEL_TO_WORD = lm.LABEL_TO_WORD
    print(f"  ✓  Loaded {len(LABEL_TO_WORD)} labels")
except Exception as e:
    print(f"  ✗  labelmap load failed: {e}")

DEVICE_INDEX = None

# ═════════════════════════════════════════════════════════════════════════════
# SPEECH INTELLIGENCE — Pipeline 2 helper
# Converts natural spoken English → list of signmap keys
# Tier 1: synonym/variant mapping  |  Tier 2: filler word removal
# Fully offline, zero extra dependencies
# ═════════════════════════════════════════════════════════════════════════════
class SpeechIntelligence:

    WORD_MAP = {
        # Greetings
        "hi":"hello","hey":"hello","greetings":"hello",
        "goodbye":"bye","good bye":"bye","farewell":"bye","see you":"bye",
        # Thank you
        "thanks":"thank you","ty":"thank you","cheers":"thank you",
        # You/Your
        "you're":"you","ur":"you","yours":"your",
        # Me/My
        "i'm":"me","i am":"me","myself":"me","mine":"my",
        # We/Our
        "we're":"we","ours":"our",
        # They
        "their":"they","them":"they","they're":"they",
        # Negation
        "don't":"do not","dont":"do not",
        "doesn't":"does not","doesnt":"does not",
        "can't":"cannot","cant":"cannot",
        # Come/Go
        "coming":"come","came":"come",
        "going":"go","went":"go","gone":"go","leave":"go","leaving":"go",
        # Help
        "helping":"help","helped":"help","assist":"help","assistance":"help",
        # Work
        "working":"work","worked":"work","job":"work",
        # Learn/Study
        "learning":"learn","learned":"learn",
        "studying":"study","studied":"study","education":"study",
        "school":"study","class":"college","university":"college","institution":"college",
        # Talk/Sing
        "talking":"talk","talked":"talk","speak":"talk","speaking":"talk",
        "spoke":"talk","say":"talk","said":"talk",
        "singing":"sing","sang":"sing","song":"sing",
        # See/TV
        "seeing":"see","saw":"see","look":"see","looking":"see",
        "watch":"see","watching":"see","tv":"television","telly":"television",
        # Eat/Food
        "eating":"eat","ate":"eat","food":"eat","hungry":"eat",
        "meal":"eat","lunch":"eat","dinner":"eat","breakfast":"eat",
        # Walk
        "walking":"walk","walked":"walk","run":"walk","running":"walk","move":"walk",
        # Wash
        "washing":"wash","washed":"wash","clean":"wash","cleaning":"wash",
        # Stay/Home
        "staying":"stay","stayed":"stay","remain":"stay",
        "house":"home","place":"home",
        # Keep
        "keeping":"keep","kept":"keep","hold":"keep","maintain":"keep",
        # Ask
        "asking":"ask","asked":"ask","question":"ask","request":"ask",
        # Change
        "changing":"change","changed":"change","different":"change",
        # Finish
        "finished":"finish","done":"finish","complete":"finish",
        "completed":"finish","end":"finish","over":"finish",
        # Fight
        "fighting":"fight","fought":"fight","argue":"fight","argument":"fight",
        # Laugh
        "laughing":"laugh","laughed":"laugh","funny":"laugh",
        # Emotions
        "unhappy":"sad","upset":"sad","cry":"sad","crying":"sad",
        "glad":"happy","joy":"happy","joyful":"happy",
        "wonderful":"great","fantastic":"great","excellent":"best",
        "nice":"good","fine":"good","okay":"good","ok":"good",
        "lovely":"beautiful","bad":"wrong","incorrect":"wrong",
        "correct":"right","secure":"safe","lonely":"alone","occupied":"busy",
        # Time
        "today":"now","currently":"now","later":"after",
        "earlier":"before","following":"next","daily":"day","repeat":"again",
        # Location
        "there":"here","somewhere":"where","everywhere":"where","far":"distance",
        # Quantity
        "many":"more","much":"more","extra":"more","additional":"more",
        "every":"all","everyone":"all","everything":"all","entire":"whole",
        # Tech
        "pc":"computer","laptop":"computer","phone":"computer",
        "device":"computer","coding":"computer","programming":"computer",
        # Misc
        "earth":"world","global":"world","word":"words",
        "called":"name","method":"way","shall":"will",
        "noise":"sound","voice":"sound","outside":"out",
        "signing":"sign","invented":"invent","create":"invent",
        "typing":"type","engineering":"engineer",
    }

    DROP_WORDS = {
        # Articles
        "the","an",
        # Auxiliary verbs not in signmap
        "is","are","am","was","were","been","being",
        "have","has","had","having","did","does",
        "would","could","should","might","may","must",
        # Prepositions not in signmap
        "in","into","onto","about","above","below",
        "between","through","across","around","near",
        "by","for","up","down","under","off",
        # Filler words
        "please","just","really","very","quite","rather",
        "actually","basically","literally","like","well",
        "um","uh","hmm","oh","ah","yeah",
        "i","i've","i'd","i'll",
        # Subordinate
        "if","because","since","although","while",
        "whether","unless","until","as",
        # Generic verbs not in signmap
        "get","got","getting","give","giving","given",
        "take","taking","taken","took","make","made","making",
        "put","putting","know","knew","known",
        "think","thought","want","wanted","need","needed",
        "try","tried","trying","use","used","using",
        "let","lets","let's","able","back","still",
        "even","only","never","always","already",
        "some","any","few","most","other","new","old",
        "big","small","long","little","first","last",
        "same","thing","things","person","people",
        "man","woman","tell","told","telling",
        "show","showing","showed",
    }

    def process(self, sentence: str) -> list:
        if not sentence:
            return []
        sentence = sentence.lower().strip()
        sentence = re.sub(r"[^\w\s']", " ", sentence)

        # Handle multi-word signs before splitting
        sentence = sentence.replace("do not",    "__DO_NOT__")
        sentence = sentence.replace("don't",     "__DO_NOT__")
        sentence = sentence.replace("dont",      "__DO_NOT__")
        sentence = sentence.replace("does not",  "__DOES_NOT__")
        sentence = sentence.replace("doesn't",   "__DOES_NOT__")
        sentence = sentence.replace("doesnt",    "__DOES_NOT__")
        sentence = sentence.replace("thank you", "__THANK_YOU__")
        sentence = sentence.replace("thanks",    "__THANK_YOU__")

        tokens  = sentence.split()
        result  = []

        for token in tokens:
            if token == "__DO_NOT__":
                result.append("do not"); continue
            if token == "__DOES_NOT__":
                result.append("does not"); continue
            if token == "__THANK_YOU__":
                result.append("thank you"); continue

            # Tier 1 — synonym map
            if token in self.WORD_MAP:
                result.append(self.WORD_MAP[token]); continue

            # Keep if directly in signmap (checked at runtime via sign_map)
            # We keep the token and let the caller decide
            if token not in self.DROP_WORDS:
                result.append(token)

        # Remove consecutive duplicates
        deduped = []
        for w in result:
            if not deduped or w != deduped[-1]:
                deduped.append(w)

        return deduped


# ═════════════════════════════════════════════════════════════════════════════
# SIGN → SPEECH ENGINE
# ═════════════════════════════════════════════════════════════════════════════
class SignToSpeechEngine:
    def __init__(self, frame_q, status_cb, emotion_cb=None):
        self.frame_q=frame_q; self.status_cb=status_cb; self.emotion_cb=emotion_cb
        self._stop=threading.Event(); self._thread=None
        self.sentence_buffer=[]; self.last_prediction=None
        self.last_committed=None; self.stable_counter=0
        self.current_emotion="neutral"; self._ema={}
        self._hold_needed={"neutral":10,"happy":5,"sad":10,"angry":8,"surprise":20}
        self._min_gap=0.08; self._candidate="neutral"; self._candidate_count=0

    def _speak(self, text, emotion):
        if not TTS_OK: return
        def run():
            engine=pyttsx3.init(); voices=engine.getProperty('voices')
            preferred=next((v.id for v in voices if any(k in v.name.lower() for k in ["zira","hazel","susan","female"])),None)
            if preferred: engine.setProperty('voice',preferred)
            elif len(voices)>1: engine.setProperty('voice',voices[1].id)
            words=text.strip().rstrip(".!?,")
            if emotion=="happy":
                rate,vol=168,1.0
                p=f"{words.split(' ',1)[0]}, {words.split(' ',1)[1]}!" if ' ' in words else f"{words}!"
            elif emotion=="sad":     rate,vol,p=105,0.70,f"{', '.join(words.split())}..."
            elif emotion=="angry":
                rate,vol=188,1.0; mid=len(words)//2; sp=words.rfind(" ",0,mid)
                p=words[:sp]+". "+words[sp+1:]+"." if sp>0 else f"{words}."
            elif emotion=="surprise": rate,vol,p=185,1.0,f"— {words}?"
            else:                     rate,vol,p=155,0.90,f"{words}."
            engine.setProperty('rate',rate); engine.setProperty('volume',vol)
            engine.say(f", {p}"); engine.runAndWait(); engine.stop()
        threading.Thread(target=run,daemon=True).start()

    def _run(self):
        if not MEDIAPIPE_OK: self.status_cb("MediaPipe not installed"); return
        missing=[p for p in [CLASSIFIER_PATH,SCALER_PATH,HAND_MODEL,FACE_MODEL] if not os.path.exists(p)]
        if missing: self.status_cb("Missing: "+", ".join(os.path.basename(p) for p in missing)); return
        try: classifier=joblib.load(CLASSIFIER_PATH); scaler=joblib.load(SCALER_PATH)
        except Exception as e: self.status_cb(f"Model load error: {e}"); return

        hand_det=mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),num_hands=1))
        face_det=mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL),
                output_face_blendshapes=True,num_faces=1))
        cap=cv2.VideoCapture(0)
        if not cap.isOpened(): self.status_cb("Camera not found"); return
        self.status_cb("Running")

        while not self._stop.is_set():
            ret,frame=cap.read()
            if not ret: break
            frame=cv2.flip(frame,1); rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)

            face_result=face_det.detect(mp_img); detected_emo="neutral"
            if face_result.face_blendshapes:
                raw={b.category_name:b.score for b in face_result.face_blendshapes[0]}
                α=0.30
                for k,v in raw.items(): self._ema[k]=α*v+(1-α)*self._ema.get(k,v)
                s=self._ema
                smile=(s.get("mouthSmileLeft",0)+s.get("mouthSmileRight",0))/2
                frown=(s.get("mouthFrownLeft",0)+s.get("mouthFrownRight",0))/2
                brow_inner=s.get("browInnerUp",0)
                brow_down=(s.get("browDownLeft",0)+s.get("browDownRight",0))/2
                brow_L=s.get("browOuterUpLeft",0); brow_R=s.get("browOuterUpRight",0)
                eye_wide=(s.get("eyeWideLeft",0)+s.get("eyeWideRight",0))/2
                jaw_open=s.get("jawOpen",0)
                nose_sneer=(s.get("noseSneerLeft",0)+s.get("noseSneerRight",0))/2
                lip_upper=(s.get("mouthUpperUpLeft",0)+s.get("mouthUpperUpRight",0))/2
                lip_stretch=(s.get("mouthStretchLeft",0)+s.get("mouthStretchRight",0))/2
                mouth_press=(s.get("mouthPressLeft",0)+s.get("mouthPressRight",0))/2
                mouth_pucker=s.get("mouthPucker",0)
                cheek_squint=(s.get("cheekSquintLeft",0)+s.get("cheekSquintRight",0))/2

                self._dbg_frame=getattr(self,"_dbg_frame",0)+1
                if self._dbg_frame%30==0:
                    print(f"[EMO] smile={smile:.3f} brow_inner={brow_inner:.3f} "
                          f"brow_down={brow_down:.3f} brow_L={brow_L:.3f} brow_R={brow_R:.3f} "
                          f"eye_wide={eye_wide:.3f} jaw={jaw_open:.3f} "
                          f"mouth_press={mouth_press:.3f} mouth_pucker={mouth_pucker:.3f} "
                          f"  → {self._candidate}({self._candidate_count}) | {self.current_emotion}")

                angry_excess=max(0,brow_down-0.018)
                angry_score=angry_excess*14.0-smile*5.0-jaw_open*2.0
                brow_R_genuine=max(0,brow_R-1.0)
                eye_jaw=eye_wide*4.0+jaw_open*3.0
                surprise_score=eye_jaw+brow_R_genuine*eye_jaw*10.0-brow_down*3.0
                happy_score=smile*3.5+cheek_squint*0.8
                brow_L_drop=max(0,0.30-brow_L)
                brow_scrunch=max(0,brow_down-0.018)
                sad_score=(mouth_press*3.0+mouth_pucker*2.0+brow_L_drop*1.0+frown*1.5
                           -smile*5.0-jaw_open*2.0-brow_scrunch*12.0-angry_excess*8.0)
                
                if self._dbg_frame%30==0:
                    print(f"[SCORES] happy={happy_score:.2f} sad={sad_score:.2f} "
                          f"angry={angry_score:.2f} surprise={surprise_score:.2f} "
                          f"  → {self.current_emotion}")

                scores={"happy":happy_score,"sad":sad_score,"angry":angry_score,
                        "surprise":surprise_score,"neutral":0.16}
                best=max(scores,key=scores.get); best_score=scores[best]
                threshold=0.18 if self.current_emotion=="neutral" else 0.12
                sorted_scores=sorted(scores.values(),reverse=True)
                gap=sorted_scores[0]-sorted_scores[1]
                if best!="neutral" and best_score>threshold and gap>=self._min_gap:
                    detected_emo=best

            if detected_emo==self._candidate: self._candidate_count+=1
            else:
                if self._candidate_count>=2: self._candidate,self._candidate_count=detected_emo,1
            if self._candidate_count>=self._hold_needed.get(detected_emo,8):
                if self._candidate!=self.current_emotion:
                    self.current_emotion=self._candidate
                    if self.emotion_cb: self.emotion_cb(self.current_emotion)

            hand_result=hand_det.detect(mp_img); active_word=""
            if hand_result.hand_landmarks:
                lms=hand_result.hand_landmarks[0]; features=[]
                for lm in lms: features.extend([lm.x,lm.y,lm.z])
                X_sc=scaler.transform(np.array(features).reshape(1,-1))
                pred=classifier.predict(X_sc); word=LABEL_TO_WORD.get(pred[0],str(pred[0]))
                active_word=word
                if word==self.last_prediction: self.stable_counter+=1
                else: self.stable_counter,self.last_prediction=1,word
                if self.stable_counter>=6:
                    if word=="FULL STOP":
                        if self.sentence_buffer: self._speak(" ".join(self.sentence_buffer),self.current_emotion)
                        self.sentence_buffer=[]
                    elif word!=self.last_committed:
                        self.sentence_buffer.append(word); self.last_committed=word
                    self.stable_counter=0

            h,w,_=frame.shape
            emo_colors={"neutral":(140,140,150),"happy":(0,220,100),"sad":(100,140,220),
                        "angry":(60,60,220),"surprise":(0,200,230),}
            ec=emo_colors.get(self.current_emotion,(140,140,150))
            cv2.rectangle(frame,(0,0),(w,58),(12,14,22),-1)
            cv2.rectangle(frame,(0,0),(w,3),ec,-1)
            cv2.putText(frame,f"EMOTION: {self.current_emotion.upper()}",(14,38),
                        cv2.FONT_HERSHEY_DUPLEX,0.82,ec,2)
            if active_word:
                lbl=f"SIGN: {active_word}"; (tw,_),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_DUPLEX,0.72,1)
                cv2.putText(frame,lbl,(w-tw-14,38),cv2.FONT_HERSHEY_DUPLEX,0.72,(255,165,0),2)
            if self.sentence_buffer:
                txt=" ".join(self.sentence_buffer)
                cv2.rectangle(frame,(0,h-54),(w,h),(12,14,22),-1)
                cv2.rectangle(frame,(0,h-3),(w,h),ec,-1)
                cv2.putText(frame,txt,(14,h-16),cv2.FONT_HERSHEY_DUPLEX,0.82,(240,240,255),2)

            try: self.frame_q.put_nowait(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            except queue.Full: pass

        cap.release(); hand_det.close(); face_det.close(); self.status_cb("Stopped")

    def start(self):
        self._stop.clear()
        self._thread=threading.Thread(target=self._run,daemon=True); self._thread.start()
    def stop(self): self._stop.set()


# Module-level singleton — shared across the app
_speech_intel = SpeechIntelligence()

# ═════════════════════════════════════════════════════════════════════════════
# SPEECH → SIGN ENGINE
# ═════════════════════════════════════════════════════════════════════════════
class SpeechToSignEngine:
    def __init__(self,frame_q,status_cb):
        self.frame_q=frame_q; self.status_cb=status_cb
        self._stop=threading.Event(); self._thread=None

    def _run(self):
        if not VOSK_OK: self.status_cb("Vosk not installed"); return
        try:
            with open(VOCAB_PATH)  as f: vocabulary=json.load(f)
            with open(SIGNMAP_PATH) as f: sign_map=json.load(f)
        except Exception as e: self.status_cb(f"Config error: {e}"); return
        if not os.path.exists(VOSK_MODEL): self.status_cb("Vosk model not found"); return
        try: model=vosk.Model(VOSK_MODEL)
        except Exception as e: self.status_cb(f"Vosk model error: {e}"); return
        grammar=json.dumps(vocabulary); audio_q=queue.Queue()
        try:
            dev_info=sd.query_devices(kind='input'); samplerate=int(dev_info['default_samplerate'])
        except Exception as e: self.status_cb(f"Mic error: {e}"); return
        recognizer=vosk.KaldiRecognizer(model,samplerate,grammar); last_sent=""
        self.status_cb("Listening…")
        try:
            with sd.RawInputStream(samplerate=samplerate,blocksize=8000,dtype='int16',
                                   channels=1,callback=lambda i,f,t,s:audio_q.put(bytes(i))):
                while not self._stop.is_set():
                    try: data=audio_q.get(timeout=0.5)
                    except queue.Empty: continue
                    if recognizer.AcceptWaveform(data):
                        result=json.loads(recognizer.Result()); text=result.get("text","").strip()
                        if text and text!=last_sent:
                            last_sent=text; self.status_cb(f'Heard: "{text}"')
                            # ── SpeechIntelligence: map to signable words ──
                            sign_words = _speech_intel.process(text)
                            for word in sign_words:
                                if self._stop.is_set(): break
                                if word in sign_map:
                                    # Known word — play sign video
                                    self._play_video(os.path.join(BASE_DIR,"..",sign_map[word]),word)
                                else:
                                    # Unknown word — spell it letter by letter
                                    self._spell_word(word, sign_map)
        except Exception as e: self.status_cb(f"Stream error: {e}")
        self.status_cb("Stopped")

    def _play_video(self,path,word):
        if not os.path.exists(path): self._show_placeholder(word); return
        cap=cv2.VideoCapture(path)
        while not self._stop.is_set():
            ret,frame=cap.read()
            if not ret: break
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); h,w,_=frame.shape
            cv2.rectangle(frame,(0,h-50),(w,h),(15,15,25),-1)
            cv2.putText(frame,word.upper(),(12,h-14),cv2.FONT_HERSHEY_DUPLEX,1.0,(0,220,180),2)
            try: self.frame_q.put_nowait(frame)
            except queue.Full: pass
            time.sleep(1/30)
        cap.release()

    def _show_placeholder(self,word):
        for _ in range(20):
            if self._stop.is_set(): return
            img=np.zeros((300,400,3),dtype=np.uint8); img[:]=(25,25,35)
            cv2.putText(img,"No sign for:",(30,130),cv2.FONT_HERSHEY_DUPLEX,0.8,(100,100,120),1)
            cv2.putText(img,f'"{word}"',(30,180),cv2.FONT_HERSHEY_DUPLEX,1.1,(0,180,220),2)
            try: self.frame_q.put_nowait(img)
            except queue.Full: pass
            time.sleep(1/30)

    def _spell_word(self, word, sign_map):
        """
        Spell an unknown word letter by letter using a-z sign videos.
        Fast mode: plays every 3rd frame at 2x speed so spelling feels snappy.
        Header card shown for only 0.3 sec before letters begin.
        Example: 'water' → W → A → T → E → R
        """
        # Show brief header card — 0.3 sec (9 frames)
        for _ in range(9):
            if self._stop.is_set(): return
            img = np.zeros((300,400,3), dtype=np.uint8); img[:] = (20,20,32)
            cv2.putText(img, "SPELLING:",
                        (30,110), cv2.FONT_HERSHEY_DUPLEX, 0.85, (100,100,140), 1)
            cv2.putText(img, f'"{word.upper()}"',
                        (30,165), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,200,180), 2)
            try: self.frame_q.put_nowait(img)
            except queue.Full: pass
            time.sleep(1/30)

        # Play each letter fast
        for letter in word.lower():
            if self._stop.is_set(): return
            if letter in sign_map:
                video_path = os.path.join(BASE_DIR, "..", sign_map[letter])
                self._play_letter_fast(video_path, letter.upper())
            # Non-alpha characters skipped silently

    def _play_letter_fast(self, path, letter):
        """
        Play a letter sign video at 3x speed by reading every 3rd frame.
        Keeps the sign recognisable but moves quickly during spelling.
        """
        if not os.path.exists(path): return
        cap = cv2.VideoCapture(path)
        frame_idx = 0
        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            if frame_idx % 3 != 0:   # skip 2 out of every 3 frames → 3x speed
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            # Letter label overlay — bright cyan so user can track progress
            cv2.rectangle(frame, (0,h-46), (w,h), (10,10,22), -1)
            cv2.putText(frame, letter,
                        (12, h-10), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0,230,200), 2)
            try: self.frame_q.put_nowait(frame)
            except queue.Full: pass
            time.sleep(1/30)   # display rate stays 30fps — only content is sparser
        cap.release()

    def start(self):
        self._stop.clear()
        self._thread=threading.Thread(target=self._run,daemon=True); self._thread.start()
    def stop(self): self._stop.set()


# ═════════════════════════════════════════════════════════════════════════════
# THEME
# ═════════════════════════════════════════════════════════════════════════════
DARK_BG  = "#0d0f14"
PANEL_BG = "#13161e"
CARD_BG  = "#1a1d27"
ACCENT1  = "#00d4aa"   # teal  — Sign→Speech
ACCENT2  = "#7b5cf0"   # violet — Speech→Sign
MUTED    = "#8892cb"
TEXT_PRI = "#e8eaf2"
TEXT_SEC = "#d6ce39"
RED      = "#e05c5c"

CANVAS_W, CANVAS_H = 570, 420

# Glow border colour per panel
GLOW1 = "#00d4aa"   # teal glow — Sign→Speech
GLOW2 = "#7b5cf0"   # violet glow — Speech→Sign
HDR_LINE = "#f0a500"  # gold/amber accent line under header


# ═════════════════════════════════════════════════════════════════════════════
# APP
# ═════════════════════════════════════════════════════════════════════════════
class VoiceBridgeApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VoiceBridge")
        self.geometry("1340x820")
        self.minsize(1140, 720)
        self.configure(fg_color=DARK_BG)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.sign_q   = queue.Queue(maxsize=2)
        self.speech_q = queue.Queue(maxsize=2)
        self.sign_engine   = SignToSpeechEngine(self.sign_q,   self._sign_status,
                                                emotion_cb=self._on_emotion_change)
        self.speech_engine = SpeechToSignEngine(self.speech_q, self._speech_status)
        self._sign_running=False; self._speech_running=False

        self._sign_ph   = self._make_placeholder("  Camera feed will appear here",   TEXT_SEC)
        self._speech_ph = self._make_placeholder("  Sign animation will appear here", TEXT_SEC)

        self._build_ui()
        self._poll_frames()

    # ── Placeholder ───────────────────────────────────────────────────────────
    def _make_placeholder(self, msg, color_hex):
        img  = Image.new("RGB", (CANVAS_W, CANVAS_H), (18, 20, 30))
        draw = ImageDraw.Draw(img)
        r,g,b = int(color_hex[1:3],16),int(color_hex[3:5],16),int(color_hex[5:7],16)
        # subtle dot grid
        for x in range(22, CANVAS_W, 36):
            for y in range(22, CANVAS_H, 36):
                draw.ellipse([x-1,y-1,x+1,y+1], fill=(r//8, g//8, b//8))
        # Multi-layer glow border — outermost dim → innermost bright
        glow_layers = [
            (8, (r//12, g//12, b//12), 20),
            (6, (r//7,  g//7,  b//7),  18),
            (4, (r//4,  g//4,  b//4),  16),
            (2, (r//2,  g//2,  b//2),  14),
            (1, (r,     g,     b    ),  12),
        ]
        for i, (offset, color, radius) in enumerate(glow_layers):
            draw.rounded_rectangle(
                [(offset, offset), (CANVAS_W-1-offset, CANVAS_H-1-offset)],
                radius=radius, outline=color, width=1)
        # Main label text — bright and large
        try:    font_big  = ImageFont.truetype("arialbd.ttf", 20)
        except Exception:
            try: font_big  = ImageFont.truetype("arial.ttf", 20)
            except Exception: font_big = ImageFont.load_default()
        try:    font_hint = ImageFont.truetype("arial.ttf", 18)
        except Exception: font_hint = ImageFont.load_default()

        # Bright main text
        draw.text((CANVAS_W//2, CANVAS_H//2 - 14), msg,
                  fill=(min(r+180,255), min(g+180,255), min(b+180,255)),
                  font=font_big, anchor="mm")
        # Hint text below
        hint_map = {
            "  Camera feed will appear here":   "Press  ▶ Start Camera  to begin",
            "  Sign animation will appear here": "Press  ▶ Start Listening  to begin",
        }
        hint_text = hint_map.get(msg, "")
        draw.text((CANVAS_W//2, CANVAS_H//2 + 18), hint_text,
                  fill=(r//2, g//2, b//2),
                  font=font_hint, anchor="mm")
        return ImageTk.PhotoImage(img)

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = ctk.CTkFrame(self, fg_color=PANEL_BG, corner_radius=0, height=70)
        hdr.pack(fill="x"); hdr.pack_propagate(False)

        # Gold/amber accent line at bottom of header
        
        ctk.CTkLabel(hdr, text="Voice",  font=("Trebuchet MS",30,"bold"), text_color=ACCENT1).place(x=28,y=12)
        ctk.CTkLabel(hdr, text="Bridge", font=("Trebuchet MS",30,"bold"), text_color=ACCENT2).place(x=102,y=12)

        # Subtitle — professional italic, spaced lettering feel
        ctk.CTkLabel(hdr,
                     text="Emotion-Aware  ·  Bidirectional  ·  Indian Sign Language  ↔  Speech",
                     font=("Georgia", 12, "italic"),
                     text_color="#e9e450").place(x=29, y=48)

        self._hdr_dot = ctk.CTkLabel(hdr, text="● IDLE", font=("Courier",14), text_color=MUTED)
        self._hdr_dot.place(relx=1.0, x=-24, y=26, anchor="e")

        # Body — 2 equal columns, both expand
        body = ctk.CTkFrame(self, fg_color=DARK_BG)
        body.pack(fill="both", expand=True, padx=18, pady=14)
        body.grid_columnconfigure(0, weight=1, uniform="col")
        body.grid_columnconfigure(1, weight=1, uniform="col")
        body.grid_rowconfigure(0, weight=1)

        self._build_sign_panel(body)
        self._build_speech_panel(body)

    # ── Sign → Speech (right) ─────────────────────────────────────────────────
    def _build_sign_panel(self, parent):
        card = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=14)
        card.grid(row=0, column=1, padx=(8,0), sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=1)   # canvas row expands

        # Title bar
        tbar = ctk.CTkFrame(card, fg_color=PANEL_BG, corner_radius=10, height=46)
        tbar.grid(row=0, column=0, sticky="ew", padx=10, pady=(10,0))
        tbar.pack_propagate(False)
        ctk.CTkLabel(tbar, text="Sign  →  Speech",
                     font=("Trebuchet MS",15,"bold"), text_color=ACCENT1).pack(side="left",padx=14,pady=10)
        self._sign_badge = ctk.CTkLabel(tbar, text="● OFF", font=("Courier",13), text_color=MUTED)
        self._sign_badge.pack(side="right", padx=14)

        # Canvas — sticky nsew so it fills the card, glowing teal border
        canvas_wrap = tk.Frame(card, bg="#14161f",
                               highlightthickness=2, highlightbackground=GLOW1)
        canvas_wrap.grid(row=1, column=0, sticky="nsew", padx=10, pady=(8,0))
        canvas_wrap.grid_rowconfigure(0, weight=1)
        canvas_wrap.grid_columnconfigure(0, weight=1)

        self._sign_canvas = tk.Label(canvas_wrap, bg="#14161f",
                                     image=self._sign_ph, cursor="crosshair")
        self._sign_canvas.grid(row=0, column=0, sticky="nsew")

        # Emotion bar
        emo_bar = ctk.CTkFrame(card, fg_color=PANEL_BG, corner_radius=8, height=38)
        emo_bar.grid(row=2, column=0, sticky="ew", padx=10, pady=(6,0))
        emo_bar.pack_propagate(False)
        ctk.CTkLabel(emo_bar, text="MOOD", font=("Courier",12,"bold"), text_color=MUTED).pack(side="left",padx=10)

        self._emo_icons = {
            "neutral" :("😐",MUTED),       "happy":("😊","#00dc6a"),
            "sad"     :("😢","#6496dc"),   "angry":("😠","#e05c5c"),
            "surprise":("😲","#00c8e6")
        }
        self._emo_dot = ctk.CTkLabel(emo_bar, text="●", font=("Arial",15), text_color=MUTED)
        self._emo_dot.pack(side="left", padx=(4,2))
        self._emo_label = ctk.CTkLabel(emo_bar, text="neutral",
                                       font=("Trebuchet MS",14,"bold"), text_color=MUTED)
        self._emo_label.pack(side="left")
        pf = ctk.CTkFrame(emo_bar, fg_color="transparent"); pf.pack(side="right", padx=8)
        self._emo_pills = {}
        for emo,(icon,col) in self._emo_icons.items():
            p = ctk.CTkLabel(pf, text=icon, font=("Arial",17), text_color=MUTED)
            p.pack(side="left", padx=2)
            self._emo_pills[emo] = (p, col)

        # Status + sentence
        self._sign_status_var = tk.StringVar(value="Ready")
        ctk.CTkLabel(card, textvariable=self._sign_status_var,
                     font=("Courier",13), text_color=TEXT_SEC,
                     wraplength=CANVAS_W).grid(row=3, column=0, padx=12, pady=(4,0), sticky="ew")

        self._sign_sentence = tk.StringVar(value="")
        ctk.CTkLabel(card, textvariable=self._sign_sentence,
                     font=("Trebuchet MS",15,"bold"), text_color=TEXT_PRI,
                     wraplength=CANVAS_W).grid(row=4, column=0, padx=12, pady=(2,0), sticky="ew")

        # Buttons
        btns = ctk.CTkFrame(card, fg_color="transparent")
        btns.grid(row=5, column=0, pady=(8,14))
        self._sign_start_btn = ctk.CTkButton(
            btns, text="▶  Start Camera",
            fg_color=ACCENT1, hover_color="#00b892",
            text_color="#0d0f14", font=("Trebuchet MS",14,"bold"),
            width=168, height=40, corner_radius=8, command=self._toggle_sign)
        self._sign_start_btn.pack(side="left", padx=6)
        ctk.CTkButton(
            btns, text="🔊  Speak Buffer",
            fg_color=PANEL_BG, hover_color=CARD_BG,
            text_color=ACCENT1, border_width=1, border_color=ACCENT1,
            font=("Trebuchet MS",14), width=148, height=40, corner_radius=8,
            command=self._speak_now).pack(side="left", padx=6)

    # ── Speech → Sign (left) ──────────────────────────────────────────────────
    def _build_speech_panel(self, parent):
        card = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=14)
        card.grid(row=0, column=0, padx=(0,8), sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=1)

        tbar = ctk.CTkFrame(card, fg_color=PANEL_BG, corner_radius=10, height=46)
        tbar.grid(row=0, column=0, sticky="ew", padx=10, pady=(10,0))
        tbar.pack_propagate(False)
        ctk.CTkLabel(tbar, text="Speech  →  Sign",
                     font=("Trebuchet MS",15,"bold"), text_color=ACCENT2).pack(side="left",padx=14,pady=10)
        self._speech_badge = ctk.CTkLabel(tbar, text="● OFF", font=("Courier",13), text_color=MUTED)
        self._speech_badge.pack(side="right", padx=14)

        canvas_wrap = tk.Frame(card, bg="#14161f",
                               highlightthickness=2, highlightbackground=GLOW2)
        canvas_wrap.grid(row=1, column=0, sticky="nsew", padx=10, pady=(8,0))
        canvas_wrap.grid_rowconfigure(0, weight=1)
        canvas_wrap.grid_columnconfigure(0, weight=1)

        self._speech_canvas = tk.Label(canvas_wrap, bg="#14161f", image=self._speech_ph)
        self._speech_canvas.grid(row=0, column=0, sticky="nsew")

        self._speech_status_var = tk.StringVar(value="Ready")
        ctk.CTkLabel(card, textvariable=self._speech_status_var,
                     font=("Courier",13), text_color=TEXT_SEC,
                     wraplength=CANVAS_W).grid(row=2, column=0, padx=12, pady=(8,0), sticky="ew")

        self._speech_word = tk.StringVar(value="")
        ctk.CTkLabel(card, textvariable=self._speech_word,
                     font=("Trebuchet MS",15,"bold"), text_color=TEXT_PRI,
                     wraplength=CANVAS_W).grid(row=3, column=0, padx=12, pady=(2,0), sticky="ew")

        btns = ctk.CTkFrame(card, fg_color="transparent")
        btns.grid(row=4, column=0, pady=(8,14))
        self._speech_start_btn = ctk.CTkButton(
            btns, text="▶  Start Listening",
            fg_color=ACCENT2, hover_color="#6448d4",
            text_color="#ffffff", font=("Trebuchet MS",14,"bold"),
            width=178, height=40, corner_radius=8, command=self._toggle_speech)
        self._speech_start_btn.pack(padx=6)

    # ── Emotion ───────────────────────────────────────────────────────────────
    def _on_emotion_change(self, emotion):
        self.after(0, lambda: self._update_emotion_bar(emotion))

    def _update_emotion_bar(self, emotion):
        _,col = self._emo_icons.get(emotion,("😐",MUTED))
        self._emo_dot.configure(text_color=col)
        self._emo_label.configure(text=emotion, text_color=col)
        for emo,(pill,pc) in self._emo_pills.items():
            pill.configure(text_color=pc if emo==emotion else MUTED)

    # ── Toggles ───────────────────────────────────────────────────────────────
    def _toggle_sign(self):
        if not self._sign_running:
            self.sign_engine.start(); self._sign_running=True
            self._sign_start_btn.configure(text="■  Stop Camera",
                                           fg_color=RED,hover_color="#c04040",text_color="#ffffff")
            self._sign_badge.configure(text="● ON", text_color=ACCENT1)
        else:
            self.sign_engine.stop(); self._sign_running=False
            self._sign_start_btn.configure(text="▶  Start Camera",
                                           fg_color=ACCENT1,hover_color="#00b892",text_color="#0d0f14")
            self._sign_badge.configure(text="● OFF", text_color=MUTED)
            self._sign_canvas.configure(image=self._sign_ph)
        self._update_header_dot()

    def _toggle_speech(self):
        if not self._speech_running:
            self.speech_engine.start(); self._speech_running=True
            self._speech_start_btn.configure(text="■  Stop Listening",
                                             fg_color=RED,hover_color="#c04040")
            self._speech_badge.configure(text="● ON", text_color=ACCENT2)
        else:
            self.speech_engine.stop(); self._speech_running=False
            self._speech_start_btn.configure(text="▶  Start Listening",
                                             fg_color=ACCENT2,hover_color="#6448d4")
            self._speech_badge.configure(text="● OFF", text_color=MUTED)
            self._speech_canvas.configure(image=self._speech_ph)
        self._update_header_dot()

    def _speak_now(self):
        if not self.sign_engine.sentence_buffer: return
        self.sign_engine._speak(" ".join(self.sign_engine.sentence_buffer),
                                self.sign_engine.current_emotion)

    def _update_header_dot(self):
        if self._sign_running and self._speech_running:
            self._hdr_dot.configure(text="● ACTIVE",  text_color=ACCENT1)
        elif self._sign_running or self._speech_running:
            self._hdr_dot.configure(text="● RUNNING", text_color=ACCENT2)
        else:
            self._hdr_dot.configure(text="● IDLE",    text_color=MUTED)

    def _sign_status(self,msg):   self.after(0,lambda: self._sign_status_var.set(msg))
    def _speech_status(self,msg): self.after(0,lambda: self._speech_status_var.set(msg))

    def _poll_frames(self):
        try:
            frame=self.sign_q.get_nowait()
            img=Image.fromarray(frame).resize((CANVAS_W,CANVAS_H),Image.LANCZOS)
            photo=ImageTk.PhotoImage(img)
            self._sign_canvas.configure(image=photo); self._sign_canvas._photo=photo
            buf=self.sign_engine.sentence_buffer
            self._sign_sentence.set(" ".join(buf) if buf else "")
        except queue.Empty: pass
        try:
            frame=self.speech_q.get_nowait()
            img=Image.fromarray(frame).resize((CANVAS_W,CANVAS_H),Image.LANCZOS)
            photo=ImageTk.PhotoImage(img)
            self._speech_canvas.configure(image=photo); self._speech_canvas._photo=photo
        except queue.Empty: pass
        self.after(30, self._poll_frames)

    def destroy(self):
        self.sign_engine.stop(); self.speech_engine.stop(); super().destroy()


if __name__ == "__main__":
    app = VoiceBridgeApp()
    app.mainloop()