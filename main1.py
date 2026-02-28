import cv2
import numpy as np
import os
from datetime import datetime
import yagmail
import time

# ==============================
# EMAIL CONFIGURATION
# ==============================
intruder_folder = "intruders"
sender_email = "customerservicepointshop@gmail.com"
app_password = "fvba iyqb livl jqjj"
recipients = [
    "arshdeepkaurhakwan@gmail.com",
    #"jarnailsingh62@gmail.com",
    #"harbhajankaur40001@gmail.com",
    "kiran24012004@gmail.com"
]
# Initialize Yagmail
yag = yagmail.SMTP(sender_email, app_password)
# main function

# ==============================
# FOLDER SETUP
# ==============================

if not os.path.exists(intruder_folder):
    
        os.makedirs(intruder_folder)


# ==============================
# LOAD SSD MODEL
# ==============================

print(" Loading MobileNet SSD Model...")

prototxt_path = "MobileNetSSD_deploy.prototxt"
model_path = "MobileNetSSD_deploy.caffemodel"

if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
    print("Model files missing!")
    exit()

try:
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    print("[SUCCESS] Model loaded successfully! ")
except Exception as e:
    print("[ERROR] Failed to load model: ",e )
    exit()

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


#CONFIDENCE_THRESHOLD = 0.6
#ALERT_INTERVAL = 15  # seconds between alerts to prevent repeated email notifications
#last_alert_time = 0

#print("[INFO] Detection confidence threshold set to: ", CONFIDENCE_THRESHOLD)
#print("[INFO] Alert interval set to: ", ALERT_INTERVAL, "seconds")

# ==============================
# START CAMERA
# ==============================
def main():

    cap = cv2.VideoCapture(0)
    print(" Warming up camera...")
    time.sleep(2.0)

    if not cap.isOpened():
        print("Camera not opened! ")
        return

last_alert_time = 0
ALERT_INTERVAL = 20  # Increased interval to prevent email spam
print("AI Shop Security System Started...")

# MAIN LOOP

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame , retrying...")
        continue
    (h,w)=frame.shape[:2]

    # Prepare for frame for the Neutral Network
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)),0.007843,(300,300),127.5)
    net.setInput(blob)
    detections = net.forward()

    person_detected = False
    for i in range(0,detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6: 
            idx= int(detections[0, 0, i, 1])

            # We only care if the detected object is a person 
            if CLASSES[idx] == "person":
                person_detected = True
                box = detections[0,0,i,3:7]* np.array([w, h, w, h])
                (startX, startY, endX, endY)= box.astype("int")

                # draw rectangle around the person
                label = f"Intruder:{confidence*100:.2f}%"
                cv2.rectangle(frame, (startX, startY),(endX, endY), (0,0,255), 2)
                cv2.putText(frame, "INTRUDER", (startX,startY - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

                # Alert Logic
                current_time = time.time()
                if person_detected and (current_time - last_alert_time > ALERT_INTERVAL):
                    print(" Intruder detected! saving image and sending alert...")

                    #Save the image 
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    file_name = f"intruder_{int(current_time)}.jpg"
                    file_path = os.path.join(intruder_folder,file_name)
                    cv2.imwrite(file_path,frame)

                    #Send Email
                    try:
                        yag.send(to=recipients,subject="Security Alert: Person Detected! ", contents=[f"Intruder detected at {timestamp}.Image attached. ", file_path])

                        print("[SUCCESS] Email alert sent. ")
                        last_alert_time = current_time
                    except Exception as e:
                        print(f"[ERROR] Failed to send email: {e}")

                #Show the video feed 
                cv2.imshow("Security Feed", frame )

                # Press "q" to exit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                cap.release()
                cv2.destroyAllWindows()
            if __name__=="__main__":
                main()
