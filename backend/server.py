"""
Backend Server - IoT & ML Digital Eye Strain Risk Assessment System
====================================================================
Responsibilities:
  - Ingests environmental telemetry from ESP32 (HC-SR04 distance + BH1750 lux).
  - Ingests behavioral telemetry from companion vision tracker (MediaPipe Face Mesh).
  - Maintains per-user sliding window buffers and performs feature fusion:
      F = [d_avg, d_min, I_avg, t_dur, blink_rate, incomplete_blink_ratio, gaze_dev, head_pose, cam_conf]
  - If camera is not active, gracefully scores telemetry using user's baseline (cam_conf=0.0).
  - Uses server epoch receive time for all database records (ignores device uptime millis).
  - Retrieves persistent per-user EWMA baseline B(U) and computes deviation Δ = F - B(U).
  - Executes ML classification R = f_θ(F, Δ) via pre-trained Random Forest and Scaler.
  - Gated alerting via Receptivity-Estimation Module (immediate vs bounded deferred dispatch).
  - Publishes alert commands back to ESP32 (alert/<user_id>).
  - Unique MQTT client ID: "eye-strain-backend".
"""

import os
import sys
import time
import json
import uuid
import logging
import joblib
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt

# Ensure backend directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import database

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("BackendServer")

# MQTT Configuration
MQTT_BROKER = os.environ.get("MQTT_BROKER", "localhost")
MQTT_PORT   = int(os.environ.get("MQTT_PORT", 1883))
MQTT_KEEPALIVE = 60
MQTT_CLIENT_ID = "eye-strain-backend"

# Model Feature Schema (18 features: 9 raw + 9 deviations)
RAW_FEATURE_NAMES = [
    'd_avg', 'd_min', 'I_avg', 't_dur',
    'blink_rate', 'incomplete_blink_ratio',
    'gaze_dev', 'head_pose', 'cam_conf'
]
DEV_FEATURE_NAMES = [f"{feat}_dev" for feat in RAW_FEATURE_NAMES]
ALL_FEATURE_COLUMNS = RAW_FEATURE_NAMES + DEV_FEATURE_NAMES

# EWMA update coefficient
ALPHA = 0.10

# In-memory per-user state tracking
user_buffers = {}
user_receptivity_state = {}

# Load ML Model and StandardScaler
MODEL_PATH  = os.path.join(os.path.dirname(__file__), 'rf_risk_model.joblib')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.joblib')

try:
    rf_model = joblib.load(MODEL_PATH)
    logger.info("Successfully loaded ML model from %s", MODEL_PATH)
except Exception as e:
    logger.error("Failed to load ML model from %s: %s", MODEL_PATH, e)
    rf_model = None

try:
    scaler = joblib.load(SCALER_PATH)
    logger.info("Successfully loaded StandardScaler from %s", SCALER_PATH)
except Exception as e:
    logger.warning("Could not load scaler from %s: %s. Using unscaled fallback.", SCALER_PATH, e)
    scaler = None


def get_user_buffer(user_id):
    """Initializes or returns sliding buffers for a given user."""
    if user_id not in user_buffers:
        user_buffers[user_id] = {
            'telemetry': [],
            'behavior': [],
            'window_start_time': time.time()
        }
    return user_buffers[user_id]


def get_receptivity_tracker(user_id):
    """Tracks alert history, deferral counts, and holding window."""
    if user_id not in user_receptivity_state:
        user_receptivity_state[user_id] = {
            'last_immediate_alert_time': 0,
            'deferred_count': 0,
            'last_deferred_time': 0
        }
    return user_receptivity_state[user_id]


def evaluate_receptivity(user_id, risk_score, features):
    """
    Receptivity-Estimation Module:
    Determines whether to dispatch an alert immediately or defer it for a bounded interval.
    
    Factors considered:
      1. Alert Fatigue / Refractory Period: At least 45s since last immediate alert.
      2. Head / Gaze Transient Instability: If the user is currently moving head violently
         or looking away (>28 deg deviation), defer briefly.
      3. Bounded Holding Interval: If deferred >= 2 consecutive times (~30 seconds),
         force immediate dispatch to prevent indefinite alert suppression.
    """
    tracker = get_receptivity_tracker(user_id)
    now = time.time()
    
    time_since_last_alert = now - tracker['last_immediate_alert_time']
    head_pose = features.get('head_pose', 0.0)
    gaze_dev = features.get('gaze_dev', 0.0)
    
    # Check bounded holding constraint: max 2 deferrals before forced dispatch
    if tracker['deferred_count'] >= 2:
        logger.info("[Receptivity] User %s: Bounded holding interval reached (%d deferrals). Forcing immediate alert.",
                    user_id, tracker['deferred_count'])
        tracker['deferred_count'] = 0
        tracker['last_immediate_alert_time'] = now
        return True, "immediate"

    # Cooldown check (prevent barrage of buzzes within 45s)
    if time_since_last_alert < 45.0:
        logger.info("[Receptivity] User %s: Within refractory cooldown (%.1fs < 45s). Deferring alert.",
                    user_id, time_since_last_alert)
        tracker['deferred_count'] += 1
        tracker['last_deferred_time'] = now
        return False, "deferred"

    # Check momentary task-switch / severe posture movement
    if head_pose > 28.0 or gaze_dev > 28.0:
        logger.info("[Receptivity] User %s: Transient posture/gaze instability (head=%.1f, gaze=%.1f). Deferring alert.",
                    user_id, head_pose, gaze_dev)
        tracker['deferred_count'] += 1
        tracker['last_deferred_time'] = now
        return False, "deferred"

    # Otherwise: user is seated, stable, and receptive
    logger.info("[Receptivity] User %s: User estimated RECEPTIVE. Immediate dispatch approved.", user_id)
    tracker['deferred_count'] = 0
    tracker['last_immediate_alert_time'] = now
    return True, "immediate"


def compute_ml_risk_score(features, deviation):
    """
    Computes risk score R = f_θ(F, Δ) using the trained Random Forest classifier.
    Falls back to a clinical rule-based formula if the model artifact is missing.
    """
    if rf_model is not None:
        # Construct feature row with exact columns
        feature_dict = {}
        for feat in RAW_FEATURE_NAMES:
            feature_dict[feat] = features.get(feat, 0.0)
        for feat in RAW_FEATURE_NAMES:
            feature_dict[f"{feat}_dev"] = deviation.get(f"{feat}_dev", 0.0)

        X = pd.DataFrame([feature_dict])[ALL_FEATURE_COLUMNS]

        if scaler is not None:
            X_scaled = pd.DataFrame(scaler.transform(X), columns=ALL_FEATURE_COLUMNS)
        else:
            X_scaled = X

        # Predict probability of class 1 (High Risk)
        prob = rf_model.predict_proba(X_scaled)[0][1]
        return float(prob)
    else:
        # Clinical heuristic fallback
        d = features.get('d_avg', 50.0)
        lux = features.get('I_avg', 300.0)
        br = features.get('blink_rate', 15.0)
        score = 0.15
        if d < 35.0: score += 0.30
        if d < 25.0: score += 0.25
        if lux < 60.0: score += 0.20
        if br < 8.0: score += 0.20
        return min(1.0, score)


def process_session_window(client, user_id):
    """
    Fuses buffered telemetry and behavioral readings into F,
    computes baseline deviation Δ = F - B(U), executes ML inference,
    applies receptivity check, dispatches alert if needed, and logs to SQLite.
    
    If camera is not running (no behavior packets), gracefully scores telemetry
    using user's baseline for behavioral features with cam_conf=0.0.
    """
    buf = user_buffers.get(user_id)
    if not buf or not buf['telemetry']:
        return

    # Use server epoch timestamp for session record
    now_ts = int(time.time() * 1000)
    t_dur = max(5.0, time.time() - buf['window_start_time'])

    # 1. Environmental Feature Aggregation
    distances = [t['distance'] for t in buf['telemetry'] if t.get('distance', -1) > 0]
    lux_values = [t['lux'] for t in buf['telemetry'] if t.get('lux', -1) >= 0]
    
    d_avg = float(np.mean(distances)) if distances else 50.0
    d_min = float(np.min(distances)) if distances else d_avg
    I_avg = float(np.mean(lux_values)) if lux_values else 300.0

    # 2. Retrieve user baseline B(U) and personalized threshold T(U)
    baseline, threshold = database.get_user_baseline(user_id)

    # 3. Behavioral Features handling (Sensible fallback if camera is not running)
    has_vision_data = len(buf['behavior']) > 0

    if not has_vision_data:
        logger.warning("[Vision Pipeline] No vision data for %s, using baseline for behavioral features (cam_conf=0.0).", user_id)
        raw_br   = float(baseline.get('blink_rate', 16.0))
        raw_ibr  = float(baseline.get('incomplete_blink_ratio', 0.12))
        raw_gaze = float(baseline.get('gaze_dev', 8.0))
        raw_pose = float(baseline.get('head_pose', 8.0))
        cam_conf = 0.0
    else:
        blink_rates = [b['blink_rate'] for b in buf['behavior']]
        ibr_values  = [b['incomplete_blink_ratio'] for b in buf['behavior']]
        gaze_devs   = [b['gaze_dev'] for b in buf['behavior']]
        head_poses  = [b['head_pose'] for b in buf['behavior']]
        cam_confs   = [b['cam_conf'] for b in buf['behavior']]

        raw_br   = float(np.mean(blink_rates)) if blink_rates else 16.0
        raw_ibr  = float(np.mean(ibr_values)) if ibr_values else 0.12
        raw_gaze = float(np.mean(gaze_devs)) if gaze_devs else 8.0
        raw_pose = float(np.mean(head_poses)) if head_poses else 8.0
        cam_conf = float(np.mean(cam_confs)) if cam_confs else 1.0

    # Low-Light Vision Adaptation:
    # If camera confidence is low (< 0.6), downweight behavioral features by blending toward baseline
    if cam_conf < 0.6:
        w_cam = max(0.0, cam_conf)
        br   = (w_cam * raw_br) + ((1.0 - w_cam) * baseline.get('blink_rate', 16.0))
        ibr  = (w_cam * raw_ibr) + ((1.0 - w_cam) * baseline.get('incomplete_blink_ratio', 0.12))
        gaze = (w_cam * raw_gaze) + ((1.0 - w_cam) * baseline.get('gaze_dev', 8.0))
        pose = (w_cam * raw_pose) + ((1.0 - w_cam) * baseline.get('head_pose', 8.0))
        if has_vision_data:
            logger.info("[Vision Adaptation] Low cam_conf (%.2f). Blended behavioral features toward baseline.", cam_conf)
    else:
        br, ibr, gaze, pose = raw_br, raw_ibr, raw_gaze, raw_pose

    # Construct complete 9-feature vector F
    F = {
        'd_avg': round(d_avg, 2),
        'd_min': round(d_min, 2),
        'I_avg': round(I_avg, 2),
        't_dur': round(t_dur, 2),
        'blink_rate': round(br, 2),
        'incomplete_blink_ratio': round(ibr, 3),
        'gaze_dev': round(gaze, 2),
        'head_pose': round(pose, 2),
        'cam_conf': round(cam_conf, 2)
    }

    # 4. Compute Baseline Deviation: Δ = F - B(U)
    Delta = {
        'd_avg_dev': round(F['d_avg'] - baseline.get('d_avg', 55.0), 2),
        'd_min_dev': round(F['d_min'] - baseline.get('d_min', 45.0), 2),
        'I_avg_dev': round(F['I_avg'] - baseline.get('I_avg', 350.0), 2),
        't_dur_dev': round(F['t_dur'] - baseline.get('t_dur', 15.0), 2),
        'blink_rate_dev': round(F['blink_rate'] - baseline.get('blink_rate', 16.0), 2),
        'incomplete_blink_ratio_dev': round(F['incomplete_blink_ratio'] - baseline.get('incomplete_blink_ratio', 0.12), 3),
        'gaze_dev_dev': round(F['gaze_dev'] - baseline.get('gaze_dev', 8.0), 2),
        'head_pose_dev': round(F['head_pose'] - baseline.get('head_pose', 8.0), 2),
        'cam_conf_dev': round(F['cam_conf'] - baseline.get('cam_conf', 0.95), 2),
    }

    # 5. Compute Risk Score R = f_θ(F, Δ)
    risk_score = compute_ml_risk_score(F, Delta)
    session_id = f"sess_{user_id}_{now_ts}_{str(uuid.uuid4())[:6]}"
    
    logger.info("Session %s [%s] -> Fused Distance: %.1f cm (min %.1f) | Lux: %.1f | Blink: %.1f/min | CamConf: %.2f",
                session_id, user_id, F['d_avg'], F['d_min'], F['I_avg'], F['blink_rate'], F['cam_conf'])
    logger.info("Risk Score: %.3f (Threshold: %.2f)", risk_score, threshold)

    # 6. Threshold Comparison & Receptivity-Gated Alerting
    alert_dispatched = False
    if risk_score >= threshold:
        is_receptive, dispatch_type = evaluate_receptivity(user_id, risk_score, F)
        
        if is_receptive and dispatch_type == "immediate":
            alert_topic = f"alert/{user_id}"
            
            logger.warning(">>> HIGH RISK DETECTED (%.3f >= %.2f). DISPATCHING IMMEDIATE ALERT to %s <<<",
                           risk_score, threshold, alert_topic)
            # Publishes "TRUE" string strictly so firmware callback fires buzzer
            client.publish(alert_topic, "TRUE", qos=1)
            alert_dispatched = True
            database.insert_alert(session_id, user_id, risk_score, "immediate", True, now_ts)
        else:
            logger.info(">>> Alert deferred by Receptivity Module (dispatch_type: %s, receptive: %s) <<<",
                        dispatch_type, is_receptive)
            database.insert_alert(session_id, user_id, risk_score, "deferred", False, now_ts)

    # 7. Persist Fused Session in SQLite (using server epoch timestamp)
    database.insert_session(
        session_id=session_id,
        user_id=user_id,
        risk_score=risk_score,
        alert_dispatched=alert_dispatched,
        ts=now_ts,
        features=F,
        deviation=Delta
    )

    # 8. Adaptive EWMA Baseline Update
    # Only update baseline on healthy or moderate sessions so baseline is not contaminated by high strain
    if risk_score < threshold and has_vision_data:
        for k in baseline.keys():
            if k in F:
                baseline[k] = round((ALPHA * F[k]) + ((1.0 - ALPHA) * baseline[k]), 3)
        database.update_user_baseline(user_id, baseline)
        logger.debug("Updated EWMA baseline for user %s: %s", user_id, baseline)

    # 9. Reset Window Buffer
    buf['telemetry'].clear()
    buf['behavior'].clear()
    buf['window_start_time'] = time.time()


# ================= Paho MQTT Callbacks =================
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        logger.info("Connected to MQTT broker at %s:%d as client '%s' (rc=0)", MQTT_BROKER, MQTT_PORT, MQTT_CLIENT_ID)
        # Subscribe to all incoming telemetry and behavior streams
        client.subscribe("telemetry/+", qos=1)
        client.subscribe("behavior/+", qos=1)
        logger.info("Subscribed to telemetry/+ and behavior/+")
    else:
        logger.error("Failed to connect to MQTT broker, return code: %s", rc)


def on_disconnect(client, userdata, *args, **kwargs):
    rc = args[1] if len(args) > 1 else (args[0] if len(args) > 0 else 0)
    if rc != 0:
        logger.warning("MQTT disconnection (rc=%s). Client will auto-reconnect.", rc)
    else:
        logger.info("Disconnected from MQTT broker.")


def on_message(client, userdata, msg):
    try:
        raw_payload = msg.payload.decode('utf-8')
        payload = json.loads(raw_payload)
        topic = msg.topic
        server_receive_time = int(time.time() * 1000)

        if topic.startswith("telemetry/"):
            device_id = payload.get("device_id") or topic.split("/")[-1]
            distance  = float(payload.get("distance", -1.0))
            lux       = float(payload.get("lux", -1.0))
            # Store device millis as metadata, use server receive time as canonical timestamp
            device_millis = payload.get("timestamp")

            if distance > 0 and lux >= 0:
                database.get_or_create_user(device_id)
                database.insert_telemetry(device_id, distance, lux, ts=server_receive_time, device_millis=device_millis)
                buf = get_user_buffer(device_id)
                buf['telemetry'].append({
                    'distance': distance,
                    'lux': lux,
                    'timestamp': server_receive_time,
                    'device_millis': device_millis
                })
                logger.debug("[Telemetry] %s -> Dist: %.1f cm, Lux: %.1f lx (ServerTS: %d)",
                             device_id, distance, lux, server_receive_time)

                # Trigger window evaluation if 3 telemetry samples buffered (~15s)
                # This guarantees evaluation even when vision_tracker is not running!
                if len(buf['telemetry']) >= 3:
                    process_session_window(client, device_id)

        elif topic.startswith("behavior/"):
            user_id  = payload.get("user_id") or topic.split("/")[-1]
            br       = float(payload.get("blink_rate", 15.0))
            ibr      = float(payload.get("incomplete_blink_ratio", 0.1))
            gd       = float(payload.get("gaze_dev", 5.0))
            hp       = float(payload.get("head_pose", 5.0))
            conf     = float(payload.get("cam_conf", 1.0))

            database.get_or_create_user(user_id)
            database.insert_behavior(user_id, br, ibr, gd, hp, conf, ts=server_receive_time)
            buf = get_user_buffer(user_id)
            buf['behavior'].append({
                'blink_rate': br,
                'incomplete_blink_ratio': ibr,
                'gaze_dev': gd,
                'head_pose': hp,
                'cam_conf': conf,
                'timestamp': server_receive_time
            })
            logger.debug("[Behavior] %s -> BR: %.1f, IBR: %.2f, Gaze: %.1f, Pose: %.1f, Conf: %.2f",
                         user_id, br, ibr, gd, hp, conf)

            # Trigger window evaluation if 3 behavior samples buffered and we have telemetry
            if len(buf['behavior']) >= 3 and len(buf['telemetry']) >= 1:
                process_session_window(client, user_id)

    except json.JSONDecodeError as jde:
        logger.error("Malformed JSON payload on topic %s: %s", msg.topic, jde)
    except Exception as ex:
        logger.error("Error processing MQTT message on topic %s: %s", msg.topic, ex, exc_info=True)


def create_mqtt_client():
    """Instantiates Paho MQTT client with cross-version compatibility (v1 and v2)."""
    try:
        # Paho MQTT v2 API with unique client ID
        cl = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=MQTT_CLIENT_ID)
    except AttributeError:
        # Paho MQTT v1 API with unique client ID
        cl = mqtt.Client(client_id=MQTT_CLIENT_ID)

    cl.on_connect = on_connect
    cl.on_disconnect = on_disconnect
    cl.on_message = on_message
    cl.reconnect_delay_set(min_delay=1, max_delay=30)
    return cl


def main():
    logger.info("Initializing SQLite database...")
    database.init_db()

    client = create_mqtt_client()
    logger.info("Connecting to MQTT broker at %s:%d as client '%s'...", MQTT_BROKER, MQTT_PORT, MQTT_CLIENT_ID)
    
    connected = False
    retry_delay = 2
    while not connected:
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE)
            connected = True
        except Exception as e:
            logger.warning("Could not connect to broker (%s). Retrying in %ds...", e, retry_delay)
            time.sleep(retry_delay)
            retry_delay = min(30, retry_delay * 2)

    logger.info("Starting MQTT network loop...")
    client.loop_forever()


if __name__ == '__main__':
    main()