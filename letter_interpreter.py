import hand_detector2 as hdm
import cv2
import pandas as pd
import numpy as np
import time
import warnings
import serial
import serial.tools.list_ports
import ollama
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# =====================================================
# ARDUINO (auto-detect, fallback safe mode)
# =====================================================
arduino = None

for port in serial.tools.list_ports.comports():
    try:
        arduino = serial.Serial(port.device, 9600, timeout=1)
        time.sleep(2)
        print(f"[OK] Arduino connected: {port.device}")
        break
    except:
        pass

if arduino is None:
    print("[WARN] No Arduino found → simulation mode")

def send_to_robot(text):
    print("SEND:", text)

    if arduino is None:
        return

    try:
        arduino.reset_input_buffer()
        arduino.write((text + "\n").encode())
        arduino.flush()
    except Exception as e:
        print("[WARN] Serial error:", e)

# =====================================================
# OLLAMA
# =====================================================
SYSTEM_PROMPT = """
You are an AI assistant for trying to infer the user's need based on little context.

RULES:
- up to 20 words
- no punctuation
- no numbers
- no J or Z
"""

def clean_text(text):
    text = ''.join(c if c.isalpha() or c == ' ' else ' ' for c in text)
    words = text.upper().split()
    words = [w for w in words if 'J' not in w and 'Z' not in w]
    return " ".join(words[:3]).strip()

def process_word(word):
    word = word.strip()
    if not word:
        return

    print("\nUSER:", word)

    try:
        res = ollama.chat(
            model="gemma3:4b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": word}
            ]
        )

        reply = clean_text(res["message"]["content"])
        if not reply:
            reply = "YES"

        send_to_robot(reply)

    except Exception as e:
        print("[OLLAMA ERROR]", e)

# =====================================================
# LOAD MODEL
# =====================================================
print("Loading classifier...")

data = pd.read_csv("hand_signals.csv")
data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

X = data.drop("letter", axis=1)
y = data["letter"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

print("Classifier ready.")

# =====================================================
# MAIN
# =====================================================
def main():

    cap = cv2.VideoCapture(0)
    detector = hdm.handDetector()

    word = ""

    current_pred = None
    stable_frames = 0

    cooldown_until = 0

    no_hand_start = None
    two_hand_start = None

    flash_until = 0
    last_added = ""

    CONFIDENCE = 0.7

    print("\nREADY")

    while True:

        success, img = cap.read()
        if not success:
            continue

        img = cv2.flip(img, 1)
        key = cv2.waitKey(1) & 0xFF

        img = detector.find_hands(img, draw=False)
        landmarks = detector.find_position(img)

        h, w, _ = img.shape

        # =================================================
        # VISUAL STATUS
        # =================================================
        color = (0, 0, 255)
        if time.time() < flash_until:
            color = (0, 255, 0)

        cv2.circle(img, (40, 40), 20, color, -1)

        cv2.putText(img, f"TEXT: {word}", (80, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

        cv2.putText(img, f"LAST: {last_added}", (20, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

        # =================================================
        # NO HANDS → SEND AFTER 2.5s
        # =================================================
        if not landmarks:

            current_pred = None
            stable_frames = 0
            two_hand_start = None

            if no_hand_start is None:
                no_hand_start = time.time()

            if time.time() - no_hand_start >= 5 and word.strip():

                process_word(word)
                word = ""

                no_hand_start = None

        # =================================================
        # TWO HANDS → SPACE (2 sec hold)
        # =================================================
        elif len(landmarks) == 2:

            no_hand_start = None

            if two_hand_start is None:
                two_hand_start = time.time()

            if time.time() - two_hand_start >= 2:

                if time.time() > cooldown_until:

                    word += " "
                    last_added = "SPACE"

                    flash_until = time.time() + 0.5
                    cooldown_until = time.time() + 1

                    print(word)

                two_hand_start = None

        # =================================================
        # ONE HAND → LETTER
        # =================================================
        elif len(landmarks) == 1:

            no_hand_start = None
            two_hand_start = None

            if time.time() < cooldown_until:
                continue

            lmlist = landmarks[0][1]

            p1 = (min(x[1] for x in lmlist) - 25,
                  min(x[2] for x in lmlist) - 25)
            p2 = (max(x[1] for x in lmlist) + 25,
                  max(x[2] for x in lmlist) + 25)

            cv2.rectangle(img, p1, p2, (255, 255, 255), 2)

            vec = np.array([c for lm in lmlist for c in lm[1:3]]).reshape(1, -1)

            probs = model.predict_proba(vec)
            conf = np.max(probs)

            if conf > CONFIDENCE:

                pred = model.predict(vec)[0].upper()

                if pred == current_pred:
                    stable_frames += 1
                else:
                    current_pred = pred
                    stable_frames = 1

                if stable_frames >= 10:

                    word += pred
                    last_added = pred

                    flash_until = time.time() + 0.5
                    cooldown_until = time.time() + 1

                    stable_frames = 0
                    current_pred = None

                    print("WORD:", word)

        # =================================================
        # EXIT
        # =================================================
        cv2.imshow("ASL CHAT", img)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if arduino:
        arduino.close()

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    main()