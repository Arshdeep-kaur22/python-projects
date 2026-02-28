import cv2
import numpy as np
import os
import time
from datetime import datetime
import yagmail

# ==============================
# EMAIL CONFIGURATION
# ==============================

sender_email = "customerservicepointshop@gmail.com"
app_password = "fvba iyqb livl jqjj "
recipients = ["arshdeepkaurhakwan@gmail.com"]

yag = yagmail.SMTP(user=sender_email, password=app_password)

# ==============================
# FOLDER SETUP
# ==============================

intruder_folder = "intruders"

if not os.path.exists(intruder_folder):
    os.makedirs(intruder_folder)

# ==============================
# LOAD SSD MODEL
# ==============================

print("[INFO] Loading MobileNet SSD Model...")

prototxt_path = "MobileNetSSD_deploy.prototxt"
model_path = "MobileNetSSD_deploy.caffemodel"

if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
    print("Model files missing!")
    exit()

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

print("[SUCCESS] Model loaded successfully!")

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"
]

PERSON_CLASS_ID = CLASSES.index("person")

confidence_threshold = 0.6
save_interval = 15
alert_interval = 15

last_save_time = 0
last_alert_time = 0

print("[INFO] Detection confidence threshold set to:", confidence_threshold)
print("[INFO] Alert interval set to:", alert_interval, "seconds")

# ==============================
# START CAMERA
# ==============================

cap = cv2.VideoCapture(0)

print("AI Shop Security System Started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Create blob
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        0.007843,
        (300, 300),
        127.5
    )

    net.setInput(blob)
    detections = net.forward()

    person_count = 0

    # Loop over detections
    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence < confidence_threshold:
            continue

        class_id = int(detections[0, 0, i, 1])

        if class_id != PERSON_CLASS_ID:
            continue

        person_count += 1

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        label = f"Person: {confidence:.2f}"

        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ==============================
    # IF PERSON DETECTED
    # ==============================

    if person_count > 0:

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")

        # Create date folder
        date_folder = os.path.join(intruder_folder, date_str)
        if not os.path.exists(date_folder):
            os.makedirs(date_folder)

        current_time = time.time()

        # ==============================
        # SAVE IMAGE EVERY 15 SECONDS
        # ==============================

        if current_time - last_save_time > save_interval:

            filename = f"{time_str}.jpg"
            intruder_path = os.path.join(date_folder, filename)

            cv2.imwrite(intruder_path, frame)
            print("Image saved!")

            last_save_time = current_time

        # ==============================
        # NIGHT ALERT LOGIC
        # ==============================

        hour = now.hour
        night_alert = (hour >= 23 or hour < 6)

        if night_alert:

            if current_time - last_alert_time > alert_interval:
                try:
                    yag.send(
                        to=recipients,
                        subject="🚨 Shop Security Alert - Person Detected",
                        contents=f"Person detected at {time_str}",
                        attachments=intruder_path
                    )

                    print("Email sent!")
                    last_alert_time = current_time

                except Exception as e:
                    print("Email failed:", e)

    # Show person count on screen
    cv2.putText(frame, f"Persons: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("AI Shop Security System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("System Stopped.")