import cv2
import mediapipe as mp
import paho.mqtt.client as mqtt
import time
import json
import math
import numpy as np

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
USER_ID = "user1"
BEHAVIOR_TOPIC = f"behavior/{USER_ID}"

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

# Simplified EAR calculation
def get_ear(eye_landmarks, all_landmarks, img_w, img_h):
    # This is a simplified estimation using a few points.
    # In a full implementation, you'd calculate Euclidean distances.
    p2_p6 = math.hypot(all_landmarks[eye_landmarks[1]].x - all_landmarks[eye_landmarks[5]].x, all_landmarks[eye_landmarks[1]].y - all_landmarks[eye_landmarks[5]].y)
    p3_p5 = math.hypot(all_landmarks[eye_landmarks[2]].x - all_landmarks[eye_landmarks[4]].x, all_landmarks[eye_landmarks[2]].y - all_landmarks[eye_landmarks[4]].y)
    p1_p4 = math.hypot(all_landmarks[eye_landmarks[0]].x - all_landmarks[eye_landmarks[3]].x, all_landmarks[eye_landmarks[0]].y - all_landmarks[eye_landmarks[3]].y)
    if p1_p4 == 0: return 0
    return (p2_p6 + p3_p5) / (2.0 * p1_p4)

# Landmark indices for eyes
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

cap = cv2.VideoCapture(0)

blink_count = 0
incomplete_blink_count = 0
last_blink_time = time.time()
is_blinking = False
EAR_THRESH = 0.2
EAR_INCOMPLETE_THRESH = 0.25

last_publish = time.time()

print("Starting Vision Tracker...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    img_h, img_w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    cam_conf = 0.0
    gaze_dev = 0.0
    head_pose = 0.0
    
    if results.multi_face_landmarks:
        cam_conf = 1.0 # Detection confident
        landmarks = results.multi_face_landmarks[0].landmark
        
        left_ear = get_ear(LEFT_EYE, landmarks, img_w, img_h)
        right_ear = get_ear(RIGHT_EYE, landmarks, img_w, img_h)
        ear = (left_ear + right_ear) / 2.0
        
        # Blink detection
        if ear < EAR_THRESH:
            if not is_blinking:
                is_blinking = True
        else:
            if is_blinking:
                is_blinking = False
                blink_count += 1
            elif ear < EAR_INCOMPLETE_THRESH: # Just a crude way to simulate incomplete blinks for the example
                incomplete_blink_count += 1
                
        # Simulate head pose based on nose tip and face edges
        nose_x = landmarks[1].x
        left_edge_x = landmarks[234].x
        right_edge_x = landmarks[454].x
        gaze_dev = abs((nose_x - left_edge_x) - (right_edge_x - nose_x)) * 100 # arbitrary metric for deviation
        head_pose = gaze_dev 
        
        # Draw on frame for local debugging only - NOT saved or transmitted
        cv2.putText(frame, f'EAR: {ear:.2f} Blinks: {blink_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    cv2.imshow('Companion App (Local debug)', frame)
    
    # Publish every 5 seconds
    if time.time() - last_publish > 5.0:
        duration = time.time() - last_publish
        blink_rate = (blink_count / duration) * 60
        inc_ratio = incomplete_blink_count / max(1, (blink_count + incomplete_blink_count))
        
        payload = {
            "user_id": USER_ID,
            "blink_rate": round(blink_rate, 2),
            "incomplete_blink_ratio": round(inc_ratio, 2),
            "gaze_dev": round(gaze_dev, 2),
            "head_pose": round(head_pose, 2),
            "cam_conf": cam_conf,
            "timestamp": int(time.time() * 1000)
        }
        client.publish(BEHAVIOR_TOPIC, json.dumps(payload))
        print(f"Published behavior: {payload}")
        
        # Reset counters
        blink_count = 0
        incomplete_blink_count = 0
        last_publish = time.time()
        
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
client.loop_stop()
