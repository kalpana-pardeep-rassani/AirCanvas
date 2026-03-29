import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Canvas
canvas = None

# Drawing settings
draw_color = (0, 0, 255)  # Red
brush_thickness = 5
eraser_thickness = 50

prev_x, prev_y = 0, 0

# Webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Get finger tip (index finger tip = 8)
            h, w, _ = frame.shape
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            # Draw line if previous exists
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y

            cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, brush_thickness)

            prev_x, prev_y = x, y

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        prev_x, prev_y = 0, 0

    # Merge canvas and frame
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    # UI Buttons (Screen Friendly)
    cv2.rectangle(frame, (10, 10), (100, 60), (0, 0, 255), -1)
    cv2.putText(frame, "RED", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.rectangle(frame, (120, 10), (220, 60), (0, 255, 0), -1)
    cv2.putText(frame, "GREEN", (130, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.rectangle(frame, (230, 10), (330, 60), (255, 0, 0), -1)
    cv2.putText(frame, "BLUE", (250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.rectangle(frame, (340, 10), (450, 60), (0, 0, 0), -1)
    cv2.putText(frame, "CLEAR", (360, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Finger Drawing App", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC to exit
        break

    # Simple button detection using finger position
    if results.multi_hand_landmarks:
        if 10 < x < 100 and 10 < y < 60:
            draw_color = (0, 0, 255)  # Red
        elif 120 < x < 220 and 10 < y < 60:
            draw_color = (0, 255, 0)  # Green
        elif 230 < x < 330 and 10 < y < 60:
            draw_color = (255, 0, 0)  # Blue
        elif 340 < x < 450 and 10 < y < 60:
            canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()