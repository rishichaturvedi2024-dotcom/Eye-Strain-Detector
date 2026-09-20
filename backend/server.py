import paho.mqtt.client as mqtt
import json
import time
import database

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
USER_ID = "user1"

# In-memory buffer for windowing
telemetry_buffer = []
behavior_buffer = []

# Simple Baseline
baseline = {
    "d_avg": 40.0,
    "l_avg": 100.0,
    "blink_rate": 15.0
}
alpha = 0.1 # EMA smoothing factor

import joblib
import pandas as pd
import numpy as np
import os

# Load trained ML model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'rf_risk_model.joblib')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.joblib')

try:
    rf_model = joblib.load(model_path)
except Exception as e:
    print(f"Warning: Could not load ML model: {e}")
    rf_model = None

try:
    scaler = joblib.load(scaler_path)
except Exception as e:
    print(f"Warning: Could not load scaler: {e}. Using unscaled features.")
    scaler = None

# ML Risk Scoring with StandardScaler support
def compute_risk_score(features, deviation):
    if rf_model is not None:
        # Prepare feature array: [d_avg, l_avg, blink_rate, d_dev, l_dev, br_dev]
        X = pd.DataFrame([{
            'd_avg': features['d_avg'],
            'l_avg': features['l_avg'],
            'blink_rate': features['blink_rate'],
            'd_dev': deviation['d_dev'],
            'l_dev': deviation['l_dev'],
            'br_dev': deviation['br_dev']
        }])

        # Apply StandardScaler if available (matches training pipeline)
        if scaler is not None:
            X = pd.DataFrame(scaler.transform(X), columns=X.columns)

        # Predict probability of high risk
        prob = rf_model.predict_proba(X)[0][1]
        return prob
    else:
        # Fallback to rule-based simulation
        score = 0.0
        if features.get('d_avg', 40) < 30: score += 0.3
        if features.get('d_avg', 40) < 20: score += 0.3
        if features.get('l_avg', 100) < 50: score += 0.2
        if features.get('blink_rate', 15) < 10: score += 0.2
        return min(1.0, score)

def update_baseline(features):
    global baseline
    if 'd_avg' in features: baseline['d_avg'] = (alpha * features['d_avg']) + ((1 - alpha) * baseline['d_avg'])
    if 'l_avg' in features: baseline['l_avg'] = (alpha * features['l_avg']) + ((1 - alpha) * baseline['l_avg'])
    if 'blink_rate' in features: baseline['blink_rate'] = (alpha * features['blink_rate']) + ((1 - alpha) * baseline['blink_rate'])

def check_window(client):
    global telemetry_buffer, behavior_buffer
    if not telemetry_buffer or not behavior_buffer:
        return

    print("Evaluating session window...")

    # Feature extraction
    d_avg = sum(item['distance'] for item in telemetry_buffer) / len(telemetry_buffer)
    l_avg = sum(item['lux'] for item in telemetry_buffer) / len(telemetry_buffer)
    br_avg = sum(item['blink_rate'] for item in behavior_buffer) / len(behavior_buffer)

    features = {
        "d_avg": d_avg,
        "l_avg": l_avg,
        "blink_rate": br_avg
    }

    deviation = {
        "d_dev": features["d_avg"] - baseline["d_avg"],
        "l_dev": features["l_avg"] - baseline["l_avg"],
        "br_dev": features["blink_rate"] - baseline["blink_rate"]
    }

    risk_score = compute_risk_score(features, deviation)
    print(f"Risk Score: {risk_score:.2f}")

    # Receptivity and Alert Dispatch
    # Simplified receptivity: Only alert if risk > 0.6.
    alert_flag = False
    if risk_score > 0.6:
        print(">>> HIGH RISK DETECTED. DISPATCHING ALERT COMMAND! <<<")
        alert_flag = True
        client.publish(f"alert/{USER_ID}", "TRUE")

    # Store session
    database.insert_session(USER_ID, risk_score, alert_flag, int(time.time() * 1000))

    # Update baseline and clear buffer
    update_baseline(features)
    telemetry_buffer.clear()
    behavior_buffer.clear()

def on_connect(client, userdata, flags, rc):
    print(f"Backend connected with result code {rc}")
    client.subscribe(f"telemetry/+")
    client.subscribe(f"behavior/+")

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())

    if msg.topic.startswith("telemetry"):
        database.insert_telemetry(payload["device_id"], payload["distance"], payload["lux"], payload["timestamp"])
        telemetry_buffer.append(payload)
    elif msg.topic.startswith("behavior"):
        database.insert_behavior(payload["user_id"], payload["blink_rate"], payload["incomplete_blink_ratio"], payload["gaze_dev"], payload["head_pose"], payload["cam_conf"], payload["timestamp"])
        behavior_buffer.append(payload)

    # Trigger window evaluation every roughly 3 behavior readings (15 seconds)
    if len(behavior_buffer) >= 3:
        check_window(client)

if __name__ == '__main__':
    database.init_db()
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    print("Starting Backend Server...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()