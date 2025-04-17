import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time

# 🔁 Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# 🎥 Webcam
cap = cv2.VideoCapture(0)

# 🖐 MediaPipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# 🧠 TTS engine
engine = pyttsx3.init()

# 📌 Labels
labels_dict = {0: 'A', 1: 'B', 2: 'L'}

# 🧱 Word construction
word = ''
last_letter = ''
last_time = time.time()

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                x_.append(x)
                y_.append(y)

            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        prediction = model.predict([np.asarray(data_aux)])[0]
        predicted_letter = labels_dict[int(prediction)]

        # Add letter to word only if it's new & after delay
        if predicted_letter != last_letter and time.time() - last_time > 1.5:
            word += predicted_letter
            last_letter = predicted_letter
            last_time = time.time()

            print("You signed:", word)
            engine.say("You signed: " + predicted_letter)
            engine.runAndWait()

        # Show letter on screen
        cv2.putText(frame, f'Letter: {predicted_letter}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 0), 3, cv2.LINE_AA)

    # Show constructed word
    cv2.putText(frame, f'Word: {word}', (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

    # Display
    cv2.imshow("Sign Language Detector", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):  # Quit
        break
    elif key == ord('c'):  # Clear word
        word = ''
        last_letter = ''
        print("Word cleared.")

cap.release()
cv2.destroyAllWindows()
