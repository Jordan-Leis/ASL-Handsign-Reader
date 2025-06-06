# ASL Handsign Reader

This project is a starting point for building a real-time American Sign Language interpreter using a webcam. It demonstrates how to capture hand landmarks with MediaPipe and classify them with a simple neural network in TensorFlow. The current code provides a basic training script and a real-time prediction demo that overlays the detected sign on the video feed.

## Features
- Hand detection and landmark extraction using MediaPipe
- Convolutional Neural Network for sign classification
- Real-time video inference with optional text-to-speech output

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the ASL alphabet dataset and place the images inside `data/asl_alphabet` so that each sign has its own folder (`A/`, `B/`, ...).
   You can obtain the dataset from <https://github.com/ardamavi/Sign-Language-Digits-Dataset> or any other ASL alphabet dataset.
3. Train the model:
   ```bash
   python src/train.py --data-dir data/asl_alphabet --epochs 5
   ```
4. Run the real-time demo:
   ```bash
   python src/realtime_demo.py --model-path models/asl_model.h5
   ```

The UI uses OpenCV to display the webcam feed with the predicted sign label. If `pyttsx3` is installed, the label will also be spoken aloud.

## Notes
This is an initial implementation meant for experimentation. Accuracy will depend heavily on the quality of the dataset and the training process. Feel free to modify the model architecture and parameters to improve performance.
