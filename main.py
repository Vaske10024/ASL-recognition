import os

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from tensorflow.keras.models import load_model


class RealTimeRecognizer:
    def __init__(self, model_path, class_labels):
        self.model = load_model(model_path)  #ucitava model sa load_model
        self.class_labels = class_labels   #A,B,C,...
        self.mp_hands = mp.solutions.hands  # Koristeci mediaPipe detektuje ruke
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils  # Crte landmarkove na ekran

    def preprocess_with_landmarks(self, image):

        #isto kao u train, Uzima sliku i pravi landmarkove i
        #procesuje ih na isti nacin kao slike iz dataseta za kasnije poredjenje
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.hands.process(image_rgb)
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            normalized_landmarks = []
            for lm in hand_landmarks.landmark:
                normalized_landmarks.extend([lm.x, lm.y, lm.z])
            return np.array(normalized_landmarks)
        return None

    def predict_class(self, landmark_vector):

        #saljemo landmarks
        landmark_vector = landmark_vector.reshape(1, -1)

        #prediktujemo na osnovu modela koje slovo pokazujemo
        predictions = self.model.predict(landmark_vector)

        #pronalazi index sa najvecom verovatnocom i vraca naziv te klase
        predicted_class = np.argmax(predictions, axis=1)[0]
        return self.class_labels[predicted_class]


    #paljenje kamere
    def recognize_from_camera(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # uzima landmarks sa kamera frame-a
            landmarks = self.preprocess_with_landmarks(frame)

            #ako nadje landmarkove onda zove predict_class metodu
            if landmarks is not None:
                predicted_class = self.predict_class(landmarks)

                #Ispisuje nasu predikciju na ekran
                cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            2, cv2.LINE_AA)

            #Iscrtava landmarkove na ruci
            result = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

             #Prikazivanje ekrana
            cv2.imshow('ASL Recognition', frame)

            if cv2.waitKey(5) & 0xFF == 27:  #Izlaz na ESC
                break

            #izlaz na X
            if cv2.getWindowProperty('ASL Recognition', cv2.WND_PROP_VISIBLE) < 1:  # Klik na 'X'
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    model_path = "asl_alphabet_model.h5"   #Path do istreniran model
    dataset_path = "asl_alphabet_train"   #path do dataset.
    class_labels = os.listdir(dataset_path) # A,B,C,D,..

    recognizer = RealTimeRecognizer(model_path, class_labels)
    recognizer.recognize_from_camera()
