# 🖐️ Real-Time Face and Hand Gesture Detection using OpenCV & MediaPipe

This project performs **real-time face detection**, **hand tracking**, and **gesture recognition** using **OpenCV** and **MediaPipe**. It also logs the results (gestures, landmarks, bounding boxes) to a CSV file and displays the output in fullscreen.

---

## 🚀 Features

- 👤 Face detection with confidence score and bounding box
   - Neutral/Sad
   - Smiling
- ✋ Hand tracking for up to 2 hands
- 🤙 Gesture recognition:
  - Fist
  - Thumbs Up
  - Open Palm
  - Peace ✌️
- 📊 Real-time logging of all data to `detection_log.csv`
- 📺 Fullscreen output with FPS counter

---

## 🛠️ Requirements

- Python 3.7+
- OpenCV
- MediaPipe

### 📦 Install Dependencies

```bash
pip install opencv-python mediapipe
