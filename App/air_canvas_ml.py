import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import pytesseract
from collections import deque
import mediapipe as mp
import PIL.Image, PIL.ImageTk
from googletrans import Translator

# Global language target
selected_language = 'hi'  # Default: Hindi

# Available languages for translation
languages = {
    "Hindi": "hi",
    "Tamil": "ta",
    "Bengali": "bn",
    "Telugu": "te",
    "Malayalam": "ml",
    "English": "en"
}

def launch_air_canvas():
    global selected_language

    wpoints = [deque(maxlen=1024)]
    white_index = 0
    colors = [(255, 255, 255)]
    colorIndex = 0
    paintWindow = np.zeros((471, 636, 3))

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
        frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 255, 255), 2)
        cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, "WHITE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        result = hands.process(framergb)

        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * 640)
                    lmy = int(lm.y * 480)
                    landmarks.append([lmx, lmy])
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            fore_finger = (landmarks[8][0], landmarks[8][1])
            center = fore_finger
            thumb = (landmarks[4][0], landmarks[4][1])
            cv2.circle(frame, center, 3, (0, 255, 0), -1)

            if (thumb[1] - center[1] < 30):
                wpoints.append(deque(maxlen=512))
                white_index += 1

            elif center[1] <= 65:
                if 40 <= center[0] <= 140:
                    wpoints = [deque(maxlen=512)]
                    white_index = 0
                    paintWindow[:, :, :] = 0
                elif 160 <= center[0] <= 255:
                    colorIndex = 0
            else:
                if colorIndex == 0:
                    wpoints[white_index].appendleft(center)

        else:
            wpoints.append(deque(maxlen=512))
            white_index += 1

        for j in range(len(wpoints)):
            for k in range(1, len(wpoints[j])):
                if wpoints[j][k - 1] is None or wpoints[j][k] is None:
                    continue
                cv2.line(frame, wpoints[j][k - 1], wpoints[j][k], colors[0], 15)
                cv2.line(paintWindow, wpoints[j][k - 1], wpoints[j][k], colors[0], 15)

        cv2.imshow("Output", frame)
        cv2.imshow("Paint", paintWindow)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            resized = cv2.resize(paintWindow, (200, 200))
            cv2.imwrite("airdrawn_text.png", resized)
            break

    cap.release()
    cv2.destroyAllWindows()

    extract_and_display_text("airdrawn_text.png", selected_language)

def extract_and_display_text(img_path, lang_code):
    translator = Translator()
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    config = "--psm 6"
    text = pytesseract.image_to_string(thresh, config=config, lang='eng').strip()

    if text:
        translated = translator.translate(text, dest=lang_code).text
        output_label.config(text=f"Translated to :{translated}")
    else:
        output_label.config(text="No text detected!")

# GUI Setup
root = tk.Tk()
root.title("Airwriting Language Translator")
root.geometry("500x350")

tk.Label(root, text="Choose Output Language:", font=('Arial', 12)).pack(pady=10)

lang_names = list(languages.keys())
lang_dropdown = ttk.Combobox(root, values=lang_names, state='readonly', font=('Arial', 11))
lang_dropdown.set("Hindi")
lang_dropdown.pack()

def update_language(event):
    global selected_language
    selected_language = languages[lang_dropdown.get()]

lang_dropdown.bind("<<ComboboxSelected>>", update_language)

tk.Button(root, text="Launch Air Canvas", command=launch_air_canvas, font=('Arial', 12), bg="#4CAF50", fg="white").pack(pady=20)

output_label = tk.Label(root, text="", wraplength=450, justify="left", font=('Arial', 11))
output_label.pack(pady=10)

root.mainloop()
