import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from twilio.rest import Client
import time

# Emotion labels (FER2013)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Twilio config (apni values daalo)
TWILIO_SID = "your_account_sid"
TWILIO_TOKEN = "your_auth_token"
TWILIO_FROM = "+1xxxxxxxxxx"
TWILIO_TO = "+91xxxxxxxxxx"

ALERT_EMOTIONS = ["Angry", "Fear", "Sad"]
last_alert_time = 0
ALERT_COOLDOWN = 30  # seconds

def send_alert(emotion):
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

    # Load trained model
    model = load_model("model/emotion_model.h5")

    # Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    print("Starting emotion detection... Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float32") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi, verbose=0)[0]
            emotion = EMOTIONS[np.argmax(preds)]
            confidence = np.max(preds) * 100

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{emotion}: {confidence:.1f}%"
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Send alert for negative emotions
            current_time = time.time()
            if emotion in ALERT_EMOTIONS:
                if current_time - last_alert_time > ALERT_COOLDOWN:
                    send_alert(emotion)
                    last_alert_time = current_time

        cv2.imshow("Emotion Detection - VIT Bhopal", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_emotions()