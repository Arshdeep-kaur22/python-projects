import cv2
import numpy as np
import os
from datetime import datetime
import yagmail
import time

# ==========================================
# 1. CONFIGURATION & CREDENTIALS
# ==========================================
print("[INFO] Initializing system...")

SENDER_EMAIL = "customerservicepointshop@gmail.com"
APP_PASSWORD = "fvba iyqb livl jqjj"
RECIPIENTS = ["arshdeepkaurhakwan@gmail.com"]

# Path setup
BASE_INTRUDER_PATH = "intruders"
KNOWN_FACES_DIR = "known_faces"
HAAR_PATH = "haarcascade_frontalface_default.xml"
SSD_PROTOTXT = "MobileNetSSD_deploy.prototxt"
SSD_MODEL = "MobileNetSSD_deploy.caffemodel"

yag = yagmail.SMTP(SENDER_EMAIL, APP_PASSWORD)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def is_security_time():
    """Checks if current time is between 11 PM (23) and 6 AM (6)"""
    now = datetime.now().hour
    return now >= 23 or now < 6

def get_save_path():
    """Creates a date-based folder inside 'intruders'"""
    today = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(BASE_INTRUDER_PATH, today)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# ==========================================
# 3. MAIN SYSTEM
# ==========================================

def main():
    # Load Models
    try:
        net = cv2.dnn.readNetFromCaffe(SSD_PROTOTXT, SSD_MODEL)
        face_cascade = cv2.CascadeClassifier(HAAR_PATH)
        print("[SUCCESS] AI Models loaded.")
    except Exception as e:
        print(f"[ERROR] Could not load model files: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam not detected.")
        return

    last_alert_time = 0
    print("--- SYSTEM LIVE | MONITORING MODE ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        person_detected = False
        security_mode = is_security_time()
        
        # Determine Status for Overlay
        status_text = "SECURITY ACTIVE" if security_mode else "DAY MONITORING"
        status_color = (0, 0, 255) if security_mode else (0, 255, 0)

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                idx = int(detections[0, 0, i, 1])
                if idx == 15: # It's a person
                    person_detected = True
                    
                    # Detect Face inside the person area
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR_GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    for (x, y, fw, fh) in faces:
                        # Placeholder for Face Recognition logic
                        cv2.rectangle(frame, (x, y), (x+fw, y+fh), (255, 255, 255), 2)
                        cv2.putText(frame, "Checking ID...", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # TRIGGER ALERT LOGIC (Only if it's night time)
        if person_detected and security_mode:
            current_time = time.time()
            if current_time - last_alert_time > 60: # Send once per minute max
                timestamp = datetime.now().strftime('%H-%M-%S')
                save_dir = get_save_path()
                file_path = os.path.join(save_dir, f"night_alert_{timestamp}.jpg")
                
                cv2.imwrite(file_path, frame)
                print(f"[ALERT] Intruder detected at {timestamp}!")

                try:
                    yag.send(
                        to=RECIPIENTS,
                        subject="⚠️ NIGHT SECURITY BREACH",
                        contents=[f"Intruder detected in shop at {timestamp}. Check attached photo.", file_path]
                    )
                    last_alert_time = current_time
                    print("[INFO] Security Email Sent.")
                except:
                    print("[ERROR] Could not send email.")

        # ATTRACIVE OVERLAY
        # Top Bar
        cv2.rectangle(frame, (0,0), (w, 50), (30, 30, 30), -1)
        cv2.putText(frame, f"SHOP: {status_text}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, datetime.now().strftime('%H:%M:%S'), (w-150, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow("AI Shop Shield", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "_main_":
    main()