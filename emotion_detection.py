import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from twilio.rest import Client
from dotenv import load_dotenv
import time

load_dotenv()  # reads .env into environment variables

# Emotion labels (FER2013)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Twilio config — loaded from environment variables, never hardcoded
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM_NUMBER")
TWILIO_TO = os.getenv("TWILIO_TO_NUMBER")

ALERT_EMOTIONS = ["Angry", "Fear", "Sad"]
last_alert_time = 0
ALERT_COOLDOWN = 30  # seconds

print("Loading emotion detection model...")
MODEL = load_model("model/emotion_model.h5")

def send_alert(emotion):
    if not all([TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, TWILIO_TO]):
        print("Twilio credentials not configured — skipping alert.")
        return
    try:
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        client.messages.create(
            body=f"[Emotion Alert] Detected: {emotion}",
            from_=TWILIO_FROM,
            to=TWILIO_TO
        )
        print(f"Alert sent for: {emotion}")
    except Exception as e:
        print(f"Twilio error: {e}")

def detect_emotions():
    global last_alert_time

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Default backend — works on Windows, Linux, and Mac
    cap = cv2.VideoCapture(0)
    time.sleep(1.0)

    if not cap.isOpened():
        print("Could not access webcam. Exiting.")
        return

    print("Starting emotion detection... Press 'Q' on the video window to quit.")
    # ... rest of the function stays exactly the same ...
