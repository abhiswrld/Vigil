# Vigil
Real-time driver drowsiness detection using computer vision and facial landmark analysis.

## What it does
Vigil is a real-time driver drowsiness detection system that runs on a standard webcam. It monitors the driver's eye state continuously and triggers a visual and audio alert when drowsiness is detected. Before each session, Vigil runs a personalized calibration to establish the user's unique eye baseline — making detection more accurate than hardcoded threshold systems.

## How it works
Vigil uses Google's MediaPipe Face Landmarker to track 478 facial landmarks in real time. It extracts the 6 key points around each eye and computes the Eye Aspect Ratio (EAR) — a metric that measures how open the eyes are based on the ratio of vertical to horizontal eye distances. When EAR drops below 75% of the user's calibrated baseline for 20 consecutive frames (~1.5 seconds), Vigil triggers a red border alert and audio alarm.

## Tech Stack
- Python 3.11
- OpenCV — video capture and frame rendering
- MediaPipe — facial landmark detection (478 points)
- NumPy — EAR calculations and array operations
- SciPy — Euclidean distance calculations
- Pygame — audio alert playback

## Installation

### Requirements
- Python 3.11
- Webcam

### Setup
```bash
# Clone the repo
git clone https://github.com/abhiswrld/Vigil.git
cd Vigil

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download MediaPipe face landmarker model
curl -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## Usage
```bash
python3.11 main.py
```

1. Look directly at the camera during the calibration screen
2. Keep your eyes open naturally — Vigil is measuring your baseline
3. Once calibration completes, normal monitoring begins
4. Close your eyes for ~1.5 seconds to trigger a test alert

## Author
Abhinav Khanna — [linkedin.com/in/abhinav-khanna06](https://linkedin.com/in/abhinav-khanna06)
