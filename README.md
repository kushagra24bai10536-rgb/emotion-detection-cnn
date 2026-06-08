# Emotion Detection - Real Time CNN

Real-Time Facial Emotion Recognition using CNN, OpenCV & Twilio Alert System

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)

## Team - VIT Bhopal University
| Name | Enrollment |
|------|-----------|
| Kiranjeet Mishra | 24BAI10127 |
| Kushagra Yadav | 24BAI10536 |
| Ashutosh Dora | 24BAI10916 |
| Prakhar Gupta | 24BAI10868 |
| Aditya Pradeep Khod | 24BAI10366 |

## Tech Stack
- Python 3.9+
- TensorFlow / Keras (CNN Model)
- OpenCV (Face Detection)
- Twilio API (Real-time Alerts)
- FER2013 Dataset (48x48 grayscale)

## Project Structure
```
emotion-detection-cnn/
├── emotion_detection.py   # Main real-time detection
├── train_model.py         # CNN training script
├── requirements.txt       # Dependencies
├── model/
│   └── emotion_model.h5   # Trained model
├── dataset/
│   ├── train/
│   └── test/
└── docs/
    ├── presentation.pptx
    └── report.docx
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (optional)
```bash
python train_model.py
```

### 3. Run real-time detection
```bash
python emotion_detection.py
```

## Emotions Detected
Angry | Disgust | Fear | Happy | Sad | Surprise | Neutral

## Dataset
FER2013 - 35,887 grayscale 48x48 images