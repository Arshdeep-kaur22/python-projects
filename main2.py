import cv2
import numpy as np
import os
from datetime import datetime
import yagmail
import time

# ==========================================
# 1. SETTINGS & PATHS
# ==========================================
SENDER_EMAIL = "customerservicepointshop@gmail.com"
APP_PASSWORD = "fvba iyqb livl jqjj"
RECIPIENTS = ["arshdeepkaurhakwan@gmail.com", "kiran24012004@gmail.com"]

# Paths
BASE_INTRUDER_PATH = "intruders"
KNOWN_FACES_DIR = "known_faces" # Store images in folders named after the person
HAAR_PATH = "haarcascade_frontalface_default.xml"
SSD_PROTOTXT = "MobileNetSSD_deploy.prototxt"
SSD_MODEL = "MobileNetSSD_deploy.caffemodel"

# Initialize Email
yag = yagmail.SMTP(SENDER_EMAIL, APP_PASSWORD)

# ==========================================
# 2. FACE RECOGNITION SETUP (LBPH)
# ==========================================
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(HAAR_PATH)
label_map = {} # To map IDs back to Names

def train_known_faces():
    print("[INFO] Training known faces...")
    faces = []
    ids = []
    
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print("[WARNING] 'known_faces' folder created. Add photos and restart!")
        return False

    current_id = 0
    for name in os.listdir(KNOWN_FACES_DIR):
        person_path = os.path.join(KNOWN_FACES_DIR, name)
        if os.path.isdir(person_path):
            label_map[current_id] = name
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Detect face in the training image
                    detected_faces = face_cascade.detectMultiScale(img, 1.3, 5)
                    for (x, y, w, h) in detected_faces:
                        faces.append(img[y:y+h, x:x+w])
                        ids.append(current_id)
            current_id += 1
    
    if len(faces) > 0:
        face_recognizer.train(faces, np.array(ids))
        print(f"[SUCCESS] Trained on {len(label_map)} people.")
        return True
    return False

# ==========================================
# 3. MAIN SECURITY SYSTEM
# ==========================================
def main():
    # Load SSD Model
    net = cv2.dnn.readNetFromCaffe(SSD_PROTOTXT, SSD_MODEL)
    is_trained = train_known_faces()
    
    cap = cv2.VideoCapture(0)
    last_alert_time = 0
    ALERT_INTERVAL = 30 # Seconds

    print("--- Security System Active ---")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        display_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # 1. DETECT PEOPLE (SSD)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        intruder_detected = False

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                idx = int(detections[0, 0, i, 1])
                if idx == 15: # 15 is the class ID for 'person'
                    person_detected = True
                    
                    # 2. DETECT FACES (Haar Cascade)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR_GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    person_name = "Unknown Intruder"
                    color = (0, 0, 255) # Red for Unknown

                    for (x, y, fw, fh) in faces:
                        if is_trained:
                            id_num, confidence_score = face_recognizer.predict(gray[y:y+fh, x:x+fw])
                            
                            # LBPH confidence: lower is better (distance)
                            if confidence_score < 70: 
                                person_name = f"Known: {label_map[id_num]}"
                                color = (0, 255, 0) # Green for Known
                            else:
                                intruder_detected = True
                        else:
                            intruder_detected = True

                        # Draw Face Box
                        cv2.rectangle(display_frame, (x, y), (x+fw, y+fh), color, 2)
                        cv2.putText(display_frame, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 3. DATE-BASED STORAGE & ALERTS
        if intruder_detected and (time.time() - last_alert_time > ALERT_INTERVAL):
            # Create date folder: intruders/2026-03-01/
            today = datetime.now().strftime("%Y-%m-%d")
            save_dir = os.path.join(BASE_INTRUDER_PATH, today)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            timestamp = datetime.now().strftime("%H-%M-%S")
            file_path = os.path.join(save_dir, f"intruder_{timestamp}.jpg")
            cv2.imwrite(file_path, frame)
            
            try:
                yag.send(to=RECIPIENTS, subject="⚠️ SHOP ALERT: UNKNOWN PERSON",
                         contents=[f"Security breach at {timestamp}. Unknown face detected.", file_path])
                print(f"[ALERT] Intruder photo saved and email sent.")
                last_alert_time = time.time()
            except:
                print("[ERROR] Email failed.")

        # Attractive Overlay
        cv2.rectangle(display_frame, (0, 0), (w, 40), (50, 50, 50), -1)
        cv2.putText(display_frame, f"SHOP SECURITY | {datetime.now().strftime('%H:%M:%S')}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Advanced AI Security", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "_main_":
    main()