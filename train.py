import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class ASLModelTrainer:
    def __init__(self, dataset_path, model_save_path="asl_alphabet_model.h5"):
        self.dataset_path = dataset_path
        self.model_save_path = model_save_path
        self.class_labels = os.listdir(dataset_path)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)


    #metoda za preprocesiranje sa hand trackeri sa mediapipe
    def preprocess_with_landmarks(self, image_path):  #path do sliku

        image = cv2.imread(image_path)
        if image is None:
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.hands.process(image_rgb)  #trazi ruku

        if result.multi_hand_landmarks:  # Ako detektuje ruku
            hand_landmarks = result.multi_hand_landmarks[0]  #cuva landmarks
            normalized_landmarks = []
            for lm in hand_landmarks.landmark:
                normalized_landmarks.extend([lm.x, lm.y, lm.z])
            return np.array(normalized_landmarks)    # vraca niz sa koordinate od landmarks
        return None

    def load_dataset_landmarks(self):
        features = []
        labels = []

        # Prolazi kroz sve slike i zove im preprocces_with_landmarks

        for label_index, class_name in enumerate(self.class_labels):
            class_folder = os.path.join(self.dataset_path, class_name)
            for image_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_name)
                feature_vector = self.preprocess_with_landmarks(image_path)
                if feature_vector is not None:
                    features.append(feature_vector)
                    labels.append(label_index)
        return np.array(features), np.array(labels) #Vraca numpy nizove od features i labels.

    def train_model(self):
        #Ucitavanje dataseta
        print("Loading dataset...")
        X, y = self.load_dataset_landmarks()
        y = to_categorical(y, num_classes=len(self.class_labels)) #konvertuje u one hot encoding format

        # Delimo na train i test 80% 20%
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Dataset loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples.")


        #pravimo neuronsku mrezu

        #Sekvencijalni model sa 3 sloja.

        model = Sequential([
            Dense(128, activation='relu', input_shape=(X.shape[1],)), #Ulazni sloj 128 neurona ReLU aktivacija
            Dropout(0.5),  # da sprecimo overfiting
            Dense(64, activation='relu'), #Srednji sloj 64 neurona
            Dropout(0.5),  #opet overfiting
            Dense(len(self.class_labels), activation='softmax')  #Izlazni sloj sa 26 "neurona" jer 26 slova abecede.
        ])


        #treniranje
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Ako se validaciona greska ne smanji nakon 5 epoha, zaustavlja se obuka
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        #smanjuje stopu ucenja ako validaciona greska stagnira
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

        # Zapravo treniranje
        print("Starting training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr]
        )

        # Saveovanje
        model.save(self.model_save_path)
        print(f"Model saved as '{self.model_save_path}'")

        # poziva se istreniran model na onih 20% dataseta za testiranje
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
        print(f"Test accuracy: {test_accuracy:.2f}")


         #Graficki prikaz:
        '''
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        cm = confusion_matrix(y_true_classes, y_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_labels, yticklabels=self.class_labels)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.show()
      
        
        print(classification_report(y_true_classes, y_pred_classes, target_names=self.class_labels))
        '''



if __name__ == "__main__":
    trainer = ASLModelTrainer(dataset_path="asl_alphabet_train")
    trainer.train_model()
