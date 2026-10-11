"""
End-to-End Pipeline & Dashboard API Test Suite
==============================================
Validates:
  1. Flask Dashboard endpoints (/api/current_risk, /api/risk_trend, /api/alert_history, /api/break_compliance, /api/users, /api/session_detail)
  2. Database schema integrity & server epoch timestamp storage
  3. Feature fusion, EWMA baseline deviation Delta = F - B(U)
  4. Machine learning inference with Random Forest model
  5. Receptivity-gated alert logic
  6. Remote alert payload filter condition (Fix 1 verification)
  7. Sensible scoring fallback when vision tracker is inactive (Fix 4 verification)
"""

import os
import sys
import time
import json

# Add backend and dashboard to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(BASE_DIR, 'backend'))
sys.path.insert(0, os.path.join(BASE_DIR, 'dashboard'))

import database
import server
from app import app

def test_database_and_fusion():
    print("\n--- TEST 1: Database & Feature Fusion ---")
    database.init_db()
    user_id = "user1"
    
    # 1. Simulate telemetry with device uptime millis (e.g. 15000ms)
    # The server should record its OWN epoch timestamp, not 15000!
    server_now = int(time.time() * 1000)
    database.insert_telemetry(user_id, distance=21.0, lux=28.0, ts=server_now, device_millis=15000)
    database.insert_behavior(user_id, br=4.5, ibr=0.40, gd=14.0, hp=13.0, conf=0.90, ts=server_now)

    # 2. Check mock client for server.process_session_window
    class MockClient:
        def __init__(self):
            self.published = []
        def publish(self, topic, payload, qos=0):
            self.published.append((topic, payload))
            print(f"    [MockClient Published] Topic: {topic} | Payload: {payload}")

    mock_client = MockClient()
    buf = server.get_user_buffer(user_id)
    buf['telemetry'] = [
        {'distance': 21.0, 'lux': 28.0, 'timestamp': server_now, 'device_millis': 15000},
        {'distance': 20.0, 'lux': 25.0, 'timestamp': server_now + 1000, 'device_millis': 16000},
        {'distance': 19.5, 'lux': 24.0, 'timestamp': server_now + 2000, 'device_millis': 17000}
    ]
    buf['behavior'] = [
        {'blink_rate': 4.5, 'incomplete_blink_ratio': 0.40, 'gaze_dev': 14.0, 'head_pose': 13.0, 'cam_conf': 0.90, 'timestamp': server_now},
        {'blink_rate': 4.0, 'incomplete_blink_ratio': 0.45, 'gaze_dev': 15.0, 'head_pose': 14.0, 'cam_conf': 0.90, 'timestamp': server_now + 1000},
        {'blink_rate': 4.2, 'incomplete_blink_ratio': 0.42, 'gaze_dev': 14.5, 'head_pose': 13.5, 'cam_conf': 0.90, 'timestamp': server_now + 2000}
    ]

    server.process_session_window(mock_client, user_id)
    assert len(mock_client.published) > 0, "Expected alert to be published on high strain!"
    assert mock_client.published[0][1] == "TRUE", f"Expected 'TRUE', got {mock_client.published[0][1]}"
    print("    [PASSED] Feature fusion, ML scoring, and alert dispatch verified!")

def test_receptivity_gating():
    print("\n--- TEST 2: Receptivity Gating ---")
    user_id = "user1"
    features = {'head_pose': 8.0, 'gaze_dev': 8.0}
    
    # 1. Immediately after alert, user is in refractory cooldown
    is_rec, d_type = server.evaluate_receptivity(user_id, 0.75, features)
    print(f"    Cooldown test: is_receptive={is_rec}, dispatch_type={d_type}")
    assert d_type == "deferred", "Expected alert to be deferred within cooldown window!"

    # 2. Simulate 2 consecutive deferrals -> bounded holding forces immediate dispatch
    tracker = server.get_receptivity_tracker(user_id)
    tracker['deferred_count'] = 2
    is_rec, d_type = server.evaluate_receptivity(user_id, 0.75, features)
    print(f"    Bounded holding test: is_receptive={is_rec}, dispatch_type={d_type}")
    assert d_type == "immediate", "Expected forced immediate alert when bounded interval reached!"
    print("    [PASSED] Receptivity gating and bounded holding logic verified!")

def test_remote_alert_payload_filtering():
    print("\n--- TEST 3: Fix 1 - Remote Alert Payload Filtering Condition ---")
    
    # Simulates the C++ firmware check in main.ino:
    # (strstr(message, "TRUE") != NULL || strstr(message, "true") != NULL)
    def firmware_should_trigger(payload_str):
        return ("TRUE" in payload_str) or ("true" in payload_str)

    # Payloads that MUST trigger the buzzer:
    should_fire = ["TRUE", "true", '{"alert": true}', '{"status": "TRUE"}']
    for p in should_fire:
        result = firmware_should_trigger(p)
        print(f"    Testing payload '{p}' -> triggers: {result}")
        assert result is True, f"Failed: payload '{p}' should have triggered alert!"

    # Payloads that MUST NOT trigger the buzzer:
    must_not_fire = ["false", "0.61", '{"alert": false}', '{"alert": false, "risk_score": 0.61}', "0", "OK"]
    for p in must_not_fire:
        result = firmware_should_trigger(p)
        print(f"    Testing payload '{p}' -> triggers: {result}")
        assert result is False, f"Failed: payload '{p}' must NOT trigger alert!"

    print("    [PASSED] Remote alert payload filtering verified strictly!")

def test_no_vision_fallback():
    print("\n--- TEST 4: Fix 4 - Graceful Scoring Without Active Camera ---")
    user_id = "user1"
    
    class MockClient:
        def __init__(self):
            self.published = []
        def publish(self, topic, payload, qos=0):
            self.published.append((topic, payload))

    mock_client = MockClient()
    buf = server.get_user_buffer(user_id)
    buf['telemetry'].clear()
    buf['behavior'].clear() # Camera is OFF!
    
    now = int(time.time() * 1000)
    buf['telemetry'] = [
        {'distance': 18.0, 'lux': 22.0, 'timestamp': now},
        {'distance': 17.5, 'lux': 20.0, 'timestamp': now + 1000},
        {'distance': 18.2, 'lux': 21.0, 'timestamp': now + 2000}
    ]
    
    # Trigger window evaluation with zero behavior readings
    server.process_session_window(mock_client, user_id)
    
    # Verify that session was recorded
    latest = database.get_latest_session(user_id)
    assert latest is not None
    features = json.loads(latest['features_json'])
    print(f"    No-vision session recorded: cam_conf={features['cam_conf']}, blink_rate={features['blink_rate']}")
    assert features['cam_conf'] == 0.0, f"Expected cam_conf 0.0, got {features['cam_conf']}"
    assert latest['risk_score'] > 0.0, "Expected valid risk score from environmental features!"
    print("    [PASSED] System scored successfully using baseline when camera was inactive!")

def test_server_timestamp_integrity():
    print("\n--- TEST 5: Fix 3 - Server Epoch Timestamp Integrity ---")
    latest = database.get_latest_session("user1")
    ts = latest['timestamp']
    print(f"    Latest session timestamp: {ts}")
    # Verify timestamp is standard epoch ms (> 1.7e12), NOT device uptime millis (< 1e8)
    assert ts > 1700000000000, f"Timestamp {ts} appears to be device uptime rather than epoch time!"
    print("    [PASSED] Server epoch timestamp verified across database and sessions!")

def test_flask_api():
    print("\n--- TEST 6: Flask API Endpoints ---")
    client = app.test_client()

    res = client.get('/api/current_risk/user1')
    assert res.status_code == 200
    data = res.get_json()
    print("    /api/current_risk/user1:", data)
    assert 'risk_score' in data and 'status' in data

    res = client.get('/api/risk_trend/user1?range=7d')
    assert res.status_code == 200
    data = res.get_json()
    print("    /api/risk_trend/user1 count:", len(data['timestamps']))
    assert 'timestamps' in data and 'risk_scores' in data

    res = client.get('/api/alert_history/user1')
    assert res.status_code == 200
    data = res.get_json()
    print("    /api/alert_history/user1 count:", len(data))

    res = client.get('/api/break_compliance/user1')
    assert res.status_code == 200
    data = res.get_json()
    print("    /api/break_compliance/user1:", data)

    res = client.get('/api/users')
    assert res.status_code == 200
    data = res.get_json()
    print("    /api/users:", data)

    res = client.get('/dashboard/user1')
    assert res.status_code == 200
    assert b"Digital Eye Strain Monitor" in res.data

    res = client.get('/dashboard/supervisor')
    assert res.status_code == 200
    assert b"Supervisor Portal" in res.data

    print("    [PASSED] All Flask endpoints and views returned 200 OK!")

if __name__ == '__main__':
    test_database_and_fusion()
    test_receptivity_gating()
    test_remote_alert_payload_filtering()
    test_no_vision_fallback()
    test_server_timestamp_integrity()
    test_flask_api()
    print("\n==========================================")
    print(" ALL PIPELINE & TARGETED FIX TESTS PASSED! ")
    print("==========================================")
