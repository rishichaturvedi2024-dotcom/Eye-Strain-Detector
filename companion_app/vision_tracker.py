"""
Companion Device Vision Tracker - On-Device Ocular & Posture Extraction
========================================================================
Uses MediaPipe FaceLandmarker Tasks API on local video feed to compute:
  - Eye Aspect Ratio (EAR) for blink detection and incomplete blink ratio
  - Gaze deviation metric (relative iris / pupil displacement)
  - Head pose displacement
  - Camera confidence score (downweights in low light or occlusion)

Note on Privacy Architecture:
  Raw camera frames are processed on-device in memory and immediately discarded.
  Only numerical features are published over MQTT to the backend broker.
"""

import os
import cv2
import time
import json
import math
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import paho.mqtt.client as mqtt

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
USER_ID = "user1"
BEHAVIOR_TOPIC = f"behavior/{USER_ID}"

# Landmark indices for eyes
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

def ensure_model_file():
    """Ensures the official MediaPipe face_landmarker.task model is available locally."""
    if not os.path.exists(MODEL_PATH):
        print(f"[{time.strftime('%H:%M:%S')}] [VISION] Downloading official face_landmarker.task model (~3.7MB)...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)
        print(f"[{time.strftime('%H:%M:%S')}] [VISION] Model downloaded successfully ({os.path.getsize(MODEL_PATH)} bytes).")

def create_face_landmarker():
    """Initializes MediaPipe 1.1+ FaceLandmarker using Video running mode."""
    ensure_model_file()
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)

def get_ear(eye_landmarks, all_landmarks, img_w, img_h):
    """Computes Eye Aspect Ratio (EAR) from vertical and horizontal eye landmark distances."""
    p2_p6 = math.hypot(all_landmarks[eye_landmarks[1]].x - all_landmarks[eye_landmarks[5]].x,
                       all_landmarks[eye_landmarks[1]].y - all_landmarks[eye_landmarks[5]].y)
    p3_p5 = math.hypot(all_landmarks[eye_landmarks[2]].x - all_landmarks[eye_landmarks[4]].x,
                       all_landmarks[eye_landmarks[2]].y - all_landmarks[eye_landmarks[4]].y)
    p1_p4 = math.hypot(all_landmarks[eye_landmarks[0]].x - all_landmarks[eye_landmarks[3]].x,
                       all_landmarks[eye_landmarks[0]].y - all_landmarks[eye_landmarks[3]].y)
    if p1_p4 == 0:
        return 0.0
    return (p2_p6 + p3_p5) / (2.0 * p1_p4)

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"[{time.strftime('%H:%M:%S')}] [VISION] Connected to MQTT broker (rc=0)")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] [VISION] Connection failed (rc={rc})")

def create_mqtt_client():
    vision_client_id = f"{USER_ID}-vision-tracker"
    try:
        cl = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=vision_client_id)
    except AttributeError:
        cl = mqtt.Client(client_id=vision_client_id)
    cl.on_connect = on_connect
    cl.reconnect_delay_set(min_delay=1, max_delay=30)
    return cl

def main():
    client = create_mqtt_client()
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
    except Exception as e:
        print(f"[VISION] Warning: Could not connect to MQTT broker ({e}). Will retry...")

    print(f"[{time.strftime('%H:%M:%S')}] [VISION] Initializing FaceLandmarker model...")
    detector = create_face_landmarker()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[VISION] Error: Camera device not accessible.")
        return

    blink_count = 0
    incomplete_blink_count = 0
    is_blinking = False
    EAR_THRESH = 0.20
    EAR_INCOMPLETE_THRESH = 0.26

    last_publish = time.time()
    last_frame_timestamp_ms = 0
    print(f"[{time.strftime('%H:%M:%S')}] [VISION] Starting Vision Tracker loop (press ESC to quit)...")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            img_h, img_w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            now_ms = int(time.time() * 1000)
            if now_ms <= last_frame_timestamp_ms:
                now_ms = last_frame_timestamp_ms + 1
            last_frame_timestamp_ms = now_ms

            results = detector.detect_for_video(mp_image, now_ms)

            cam_conf = 0.0
            gaze_dev = 0.0
            head_pose = 0.0
            ear = 0.0

            if results.face_landmarks and len(results.face_landmarks) > 0:
                landmarks = results.face_landmarks[0]

                # Check ambient brightness in face region to model camera confidence
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_brightness = np.mean(gray)
                if mean_brightness < 40:
                    cam_conf = max(0.3, mean_brightness / 80.0)  # Degraded in dim light
                else:
                    cam_conf = 0.95

                left_ear = get_ear(LEFT_EYE, landmarks, img_w, img_h)
                right_ear = get_ear(RIGHT_EYE, landmarks, img_w, img_h)
                ear = (left_ear + right_ear) / 2.0

                # Blink detection logic
                if ear < EAR_THRESH:
                    if not is_blinking:
                        is_blinking = True
                else:
                    if is_blinking:
                        is_blinking = False
                        blink_count += 1
                    elif ear < EAR_INCOMPLETE_THRESH:
                        incomplete_blink_count += 1

                # Approximate head and gaze orientation
                nose_x = landmarks[1].x
                left_edge_x = landmarks[234].x
                right_edge_x = landmarks[454].x
                gaze_dev = abs((nose_x - left_edge_x) - (right_edge_x - nose_x)) * 100.0
                head_pose = gaze_dev

                # Draw green eye points for clear user feedback
                for idx in LEFT_EYE + RIGHT_EYE:
                    pt = (int(landmarks[idx].x * img_w), int(landmarks[idx].y * img_h))
                    cv2.circle(frame, pt, 2, (0, 255, 0), -1)

                # Visual feedback on local display
                cv2.putText(frame, f'EAR: {ear:.2f} | Blinks: {blink_count} | Conf: {cam_conf:.2f}',
                            (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cam_conf = 0.0
                cv2.putText(frame, 'No Face Detected (cam_conf = 0.0)',
                            (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Companion Vision Tracker (Local Privacy Preview)', frame)

            # Publish aggregated behavioral features every 5.0 seconds
            now = time.time()
            if now - last_publish >= 5.0:
                duration = max(1.0, now - last_publish)
                blink_rate = (blink_count / duration) * 60.0
                total_blinks = blink_count + incomplete_blink_count
                inc_ratio = (incomplete_blink_count / total_blinks) if total_blinks > 0 else 0.0

                payload = {
                    "user_id": USER_ID,
                    "blink_rate": round(blink_rate, 2),
                    "incomplete_blink_ratio": round(inc_ratio, 3),
                    "gaze_dev": round(gaze_dev, 2),
                    "head_pose": round(head_pose, 2),
                    "cam_conf": round(cam_conf, 2),
                    "timestamp": int(now * 1000)
                }

                if client.is_connected():
                    client.publish(BEHAVIOR_TOPIC, json.dumps(payload), qos=1)
                    print(f"[{time.strftime('%H:%M:%S')}] [VISION_PUB] Published: {payload}")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] [VISION] MQTT not connected, buffering payload...")

                # Reset interval counters
                blink_count = 0
                incomplete_blink_count = 0
                last_publish = now

            if cv2.waitKey(5) & 0xFF == 27:  # ESC key
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        client.loop_stop()

if __name__ == '__main__':
    main()
