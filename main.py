import cv2
import numpy as np
import os
import time
from datetime import datetime
import yagmail

# ================= CONFIGURATION ================= #

DAY_START = 6
NIGHT_START = 21
MAX_IMAGES_PER_PERSON = 5
STAY_TIME_LIMIT = 20      # seconds
SAVE_COOLDOWN = 15        # seconds

EMAIL_USER = "customerservicepointshop@gmail.com"
EMAIL_PASS = "fvba iyab livl jqjj"
EMAIL_RECEIVER = [  "arshdeepkaurhakwan@gmail.com",
                    "jarnailsingh162@gmail.com",
                    "kiran24012004@gmail.com",
                    "harbhajankaur40001@gmail.com"
]


# ================================================= #

# ----------- Load Models ----------- #

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

prototxt_path = "MobileNetSSD_deploy.prototxt"
model_path = "MobileNetSSD_deploy.caffemodel"

if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
    print("person detection model files not found ")
    print("make sure these files exist in project folder: ")
    print(" - MobileNetSSD_deploy.prototxt")
    print(" - MobileNetSSD_deploy.caffemodel ")
    exit()
person_net = cv2.dnn.readNetFromCaffe(prototxt_path,model_path)

recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "trained_faces.yml")

if os.path.exists(model_path):
    recognizer.read(model_path)
    print("face model loaded successfully.")
else:
    print("trained_faces.yml not found! run train_faces.py first.")
    exit()
# ----------- Initialize ----------- #

cap = cv2.VideoCapture(0)

intruder_count = {}
person_timer = {}
last_saved_time = 0


# ================= FUNCTIONS ================= #

def is_night_time():
    current_hour = datetime.now().hour
    return current_hour >= NIGHT_START or current_hour <= DAY_START


def send_email_alert(image_path):
    try:
        yag = yagmail.SMTP(EMAIL_USER, EMAIL_PASS)
        yag.send(
            EMAIL_RECEIVER,
            subject="⚠ Night Intruder Alert",
            contents="Unknown person detected in shop.",
            attachments=image_path
        )
        print("📧 Email Alert Sent")
    except Exception as e:
        print("Email Error:", e)


def save_intruder_image(frame, person_id):
    global last_saved_time

    if person_id not in intruder_count:
        intruder_count[person_id] = 0

    if intruder_count[person_id] >= MAX_IMAGES_PER_PERSON:
        return None

    current_time = time.time()

    if current_time - last_saved_time < SAVE_COOLDOWN:
        return None

    if not os.path.exists("intruders"):
        os.makedirs("intruders")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"intruders/intruder_{timestamp}.jpg"

    cv2.imwrite(filename, frame)

    intruder_count[person_id] += 1
    last_saved_time = current_time

    print("📸 Intruder Image Saved:", filename)
    return filename


# ================= MAIN LOOP ================= #

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        0.007843,
        (300, 300),
        127.5
    )

    person_net.setInput(blob)
    detections = person_net.forward()

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])

        # Class ID 15 = Person
        if confidence > 0.6 and class_id == 15:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            person_region = gray[startY:endY, startX:endX]

            faces = face_cascade.detectMultiScale(
                person_region,
                scaleFactor=1.3,
                minNeighbors=5
            )

            for (x, y, fw, fh) in faces:

                face_img = person_region[y:y+fh, x:x+fw]

                try:
                    label, confidence_score = recognizer.predict(face_img)
                except:
                    continue

                name = "Unknown"
                color = (0, 0, 255)

                if confidence_score < 70:
                    name = f"Person_{label}"
                    color = (0, 255, 0)

                person_id = f"{startX}_{startY}"

                # -------- Suspicious Stay Timer -------- #
                if person_id not in person_timer:
                    person_timer[person_id] = time.time()

                stay_duration = time.time() - person_timer[person_id]

                if stay_duration > STAY_TIME_LIMIT:
                    print("⚠ Suspicious Stay Detected")

                # -------- Unknown Handling -------- #
                if name == "Unknown":

                    image_path = save_intruder_image(frame, person_id)

                    if image_path and is_night_time():
                        send_email_alert(image_path)

                # -------- Draw Box -------- #
                cv2.rectangle(frame,
                              (startX, startY),
                              (endX, endY),
                              color, 2)

                cv2.putText(frame,
                            name,
                            (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color, 2)

    cv2.imshow("Smart Shop Security System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()