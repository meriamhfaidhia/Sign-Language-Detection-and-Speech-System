import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands  #a model that can detect and track 21 hand landmarks in real-time using a webcam.
mp_drawing = mp.solutions.drawing_utils #helper module for drawing the hand landmarks and connections
mp_drawing_styles = mp.solutions.drawing_styles #predefined styles/colors when drawing landmarks and connections

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
#static_image_mode=True  => analyzing individual images, not video streams
#min_detection_confidence=0.3  => If detection confidence is below 0.3, the model will ignore the hand


DATA_DIR = './Data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb) #Pass img_rgb to the hands detector

        if results.multi_hand_landmarks:
            #This checks whether MediaPipe actually found any hands in the image
            for hand_landmarks in results.multi_hand_landmarks:
                #You’re looping through each detected hand
                #🧮 Normalize Coordinates
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                    #These are normalized values between 0 and 1 relative to the image size

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

        #🗃️ Add Features and Labels to Dataset
        data.append(data_aux)
        labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
