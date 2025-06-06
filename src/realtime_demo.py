import argparse
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyttsx3
import os


class ASLInterpreter:
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)
        class_file = os.path.splitext(model_path)[0] + "_classes.txt"
        if os.path.exists(class_file):
            with open(class_file) as f:
                self.labels = [line.strip() for line in f if line.strip()]
        else:
            self.labels = None
        self.engine = pyttsx3.init()
        self.mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1)

    def predict(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mp_hands.process(img_rgb)
        if not result.multi_hand_landmarks:
            return None
        hand_landmarks = result.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        landmark_array = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark]).flatten()
        # Normalize to 0-1
        landmark_array = landmark_array / np.array([w, h] * 21)
        landmark_array = landmark_array.reshape(1, -1)
        # Convert to image-like input expected by the model
        # This assumes model was trained on 64x64 images
        img = np.zeros((1, 64, 64, 3), dtype=np.float32)
        img[0, :21, 0] = landmark_array[0, ::2]
        img[0, :21, 1] = landmark_array[0, 1::2]
        preds = self.model.predict(img, verbose=0)[0]
        pred_idx = np.argmax(preds)
        if self.labels:
            return self.labels[pred_idx]
        return str(pred_idx)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()


def main():
    parser = argparse.ArgumentParser(description="ASL real-time interpreter")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--no-speech", action="store_true", help="Disable text-to-speech")
    args = parser.parse_args()

    interpreter = ASLInterpreter(args.model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        label = interpreter.predict(frame)
        if label:
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if not args.no_speech:
                interpreter.speak(label)
        cv2.imshow("ASL Interpreter", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
