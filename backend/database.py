import sqlite3
import os
import json
import time

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_strain.db'))

DEFAULT_BASELINE = {
    "d_avg": 55.0,
    "d_min": 45.0,
    "I_avg": 350.0,
    "t_dur": 15.0,
    "blink_rate": 16.0,
    "incomplete_blink_ratio": 0.12,
    "gaze_dev": 8.0,
    "head_pose": 8.0,
    "cam_conf": 0.95
}
DEFAULT_THRESHOLD = 0.60

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    c = conn.cursor()

    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
    ''')

    # Telemetry table (timestamp is server receive time, device_millis is device uptime)
    c.execute('''
        CREATE TABLE IF NOT EXISTS telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            distance REAL,
            lux REAL,
            timestamp INTEGER,
            device_millis INTEGER
        )
    ''')

    # Check and migrate telemetry columns
    c.execute("PRAGMA table_info(telemetry)")
    tel_cols = [r['name'] for r in c.fetchall()]
    if 'device_millis' not in tel_cols:
        c.execute("ALTER TABLE telemetry ADD COLUMN device_millis INTEGER")

    # Behavioral table
    c.execute('''
        CREATE TABLE IF NOT EXISTS behavior (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            blink_rate REAL,
            incomplete_blink_ratio REAL,
            gaze_dev REAL,
            head_pose REAL,
            cam_conf REAL,
            timestamp INTEGER
        )
    ''')

    # Sessions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE,
            user_id TEXT,
            risk_score REAL,
            alert_dispatched BOOLEAN,
            timestamp INTEGER,
            features_json TEXT,
            deviation_json TEXT,
            d_avg REAL,
            d_min REAL,
            l_avg REAL,
            t_dur REAL,
            blink_rate REAL,
            incomplete_blink_ratio REAL,
            gaze_dev REAL,
            head_pose REAL,
            cam_conf REAL
        )
    ''')

    # Check and migrate sessions table columns
    c.execute("PRAGMA table_info(sessions)")
    existing_cols = [r['name'] for r in c.fetchall()]
    migration_cols = [
        ("session_id", "TEXT"),
        ("features_json", "TEXT"),
        ("deviation_json", "TEXT"),
        ("d_avg", "REAL"),
        ("d_min", "REAL"),
        ("l_avg", "REAL"),
        ("t_dur", "REAL"),
        ("blink_rate", "REAL"),
        ("incomplete_blink_ratio", "REAL"),
        ("gaze_dev", "REAL"),
        ("head_pose", "REAL"),
        ("cam_conf", "REAL")
    ]
    for col_name, col_type in migration_cols:
        if col_name not in existing_cols:
            c.execute(f"ALTER TABLE sessions ADD COLUMN {col_name} {col_type}")

    # User Baselines
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_baselines (
            user_id TEXT PRIMARY KEY,
            baseline_json TEXT NOT NULL,
            threshold REAL NOT NULL DEFAULT 0.60,
            last_updated INTEGER NOT NULL
        )
    ''')

    # Alerts history
    c.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_id TEXT,
            risk_score REAL,
            dispatch_type TEXT,
            receptive BOOLEAN,
            timestamp INTEGER
        )
    ''')

    # Migrate alerts table columns if needed
    c.execute("PRAGMA table_info(alerts)")
    alert_cols = [r['name'] for r in c.fetchall()]
    alert_migration_cols = [
        ("session_id", "TEXT"),
        ("user_id", "TEXT"),
        ("risk_score", "REAL"),
        ("dispatch_type", "TEXT"),
        ("receptive", "BOOLEAN"),
        ("timestamp", "INTEGER")
    ]
    for col_name, col_type in alert_migration_cols:
        if col_name not in alert_cols:
            c.execute(f"ALTER TABLE alerts ADD COLUMN {col_name} {col_type}")

    # Break Compliance stats per user
    c.execute('''
        CREATE TABLE IF NOT EXISTS break_compliance (
            user_id TEXT PRIMARY KEY,
            breaks_taken INTEGER DEFAULT 0,
            breaks_recommended INTEGER DEFAULT 0,
            compliance_pct REAL DEFAULT 100.0,
            last_break_time INTEGER DEFAULT 0
        )
    ''')

    # Seed default user if not exists
    c.execute('SELECT user_id FROM users WHERE user_id = ?', ('user1',))
    if not c.fetchone():
        now = int(time.time() * 1000)
        c.execute('INSERT INTO users (user_id, name, created_at) VALUES (?, ?, ?)',
                  ('user1', 'Student User (Primary)', now))
        c.execute('INSERT INTO users (user_id, name, created_at) VALUES (?, ?, ?)',
                  ('user2', 'Alex Taylor (Lab Station 2)', now))
        c.execute('INSERT INTO users (user_id, name, created_at) VALUES (?, ?, ?)',
                  ('user3', 'Maya Chen (Lab Station 3)', now))

    # Seed default baselines if not exists
    for uid in ['user1', 'user2', 'user3']:
        c.execute('SELECT user_id FROM user_baselines WHERE user_id = ?', (uid,))
        if not c.fetchone():
            now = int(time.time() * 1000)
            c.execute('INSERT INTO user_baselines (user_id, baseline_json, threshold, last_updated) VALUES (?, ?, ?, ?)',
                      (uid, json.dumps(DEFAULT_BASELINE), DEFAULT_THRESHOLD, now))

    # Seed default break compliance if not exists
    for uid in ['user1', 'user2', 'user3']:
        c.execute('SELECT user_id FROM break_compliance WHERE user_id = ?', (uid,))
        if not c.fetchone():
            c.execute('INSERT INTO break_compliance (user_id, breaks_taken, breaks_recommended, compliance_pct, last_break_time) VALUES (?, 4, 5, 80.0, ?)',
                      (uid, int(time.time() * 1000)))

    conn.commit()
    conn.close()

def get_or_create_user(user_id, name=None):
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
    row = c.fetchone()
    if not row:
        now = int(time.time() * 1000)
        user_name = name or f"Student {user_id}"
        c.execute('INSERT INTO users (user_id, name, created_at) VALUES (?, ?, ?)', (user_id, user_name, now))
        c.execute('INSERT INTO user_baselines (user_id, baseline_json, threshold, last_updated) VALUES (?, ?, ?, ?)',
                  (user_id, json.dumps(DEFAULT_BASELINE), DEFAULT_THRESHOLD, now))
        c.execute('INSERT INTO break_compliance (user_id, breaks_taken, breaks_recommended, compliance_pct, last_break_time) VALUES (?, 0, 1, 0.0, ?)',
                  (user_id, now))
        conn.commit()
        c.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
        row = c.fetchone()
    res = dict(row)
    conn.close()
    return res

def get_all_users():
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT user_id, name FROM users ORDER BY user_id ASC')
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def insert_telemetry(user_id, distance, lux, ts=None, device_millis=None):
    conn = get_connection()
    c = conn.cursor()
    # Always use server epoch time as primary timestamp
    server_ts = int(time.time() * 1000) if ts is None else int(ts)
    dev_ms = int(device_millis) if device_millis is not None else None
    c.execute('INSERT INTO telemetry (user_id, distance, lux, timestamp, device_millis) VALUES (?, ?, ?, ?, ?)',
              (user_id, float(distance), float(lux), server_ts, dev_ms))
    conn.commit()
    conn.close()

def insert_behavior(user_id, br, ibr, gd, hp, conf, ts=None):
    conn = get_connection()
    c = conn.cursor()
    server_ts = int(time.time() * 1000) if ts is None else int(ts)
    c.execute('INSERT INTO behavior (user_id, blink_rate, incomplete_blink_ratio, gaze_dev, head_pose, cam_conf, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)',
              (user_id, float(br), float(ibr), float(gd), float(hp), float(conf), server_ts))
    conn.commit()
    conn.close()

def get_user_baseline(user_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT baseline_json, threshold FROM user_baselines WHERE user_id = ?', (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        try:
            baseline = json.loads(row['baseline_json'])
            return baseline, float(row['threshold'])
        except Exception:
            pass
    return DEFAULT_BASELINE.copy(), DEFAULT_THRESHOLD

def update_user_baseline(user_id, baseline_dict, threshold=None):
    conn = get_connection()
    c = conn.cursor()
    now = int(time.time() * 1000)
    baseline_json = json.dumps(baseline_dict)
    if threshold is not None:
        c.execute('''
            INSERT INTO user_baselines (user_id, baseline_json, threshold, last_updated)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET baseline_json = ?, threshold = ?, last_updated = ?
        ''', (user_id, baseline_json, threshold, now, baseline_json, threshold, now))
    else:
        c.execute('''
            INSERT INTO user_baselines (user_id, baseline_json, threshold, last_updated)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET baseline_json = ?, last_updated = ?
        ''', (user_id, baseline_json, DEFAULT_THRESHOLD, now, baseline_json, now))
    conn.commit()
    conn.close()

def insert_session(session_id, user_id, risk_score, alert_dispatched, ts, features, deviation):
    conn = get_connection()
    c = conn.cursor()
    features_json = json.dumps(features)
    deviation_json = json.dumps(deviation)
    server_ts = int(time.time() * 1000) if ts is None else int(ts)
    c.execute('''
        INSERT INTO sessions (
            session_id, user_id, risk_score, alert_dispatched, timestamp,
            features_json, deviation_json,
            d_avg, d_min, l_avg, t_dur,
            blink_rate, incomplete_blink_ratio, gaze_dev, head_pose, cam_conf
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        session_id, user_id, float(risk_score), bool(alert_dispatched), server_ts,
        features_json, deviation_json,
        float(features.get('d_avg', 0)), float(features.get('d_min', 0)),
        float(features.get('I_avg', features.get('l_avg', 0))), float(features.get('t_dur', 15)),
        float(features.get('blink_rate', 0)), float(features.get('incomplete_blink_ratio', 0)),
        float(features.get('gaze_dev', 0)), float(features.get('head_pose', 0)),
        float(features.get('cam_conf', 1.0))
    ))
    conn.commit()
    conn.close()

def insert_alert(session_id, user_id, risk_score, dispatch_type, receptive, ts):
    conn = get_connection()
    c = conn.cursor()
    server_ts = int(time.time() * 1000) if ts is None else int(ts)
    c.execute('''
        INSERT INTO alerts (session_id, user_id, risk_score, dispatch_type, receptive, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (session_id, user_id, float(risk_score), dispatch_type, bool(receptive), server_ts))

    if dispatch_type == 'immediate':
        c.execute('''
            UPDATE break_compliance
            SET breaks_recommended = breaks_recommended + 1,
                compliance_pct = ROUND((CAST(breaks_taken AS REAL) / MAX(1, breaks_recommended + 1)) * 100.0, 1)
            WHERE user_id = ?
        ''', (user_id,))
    conn.commit()
    conn.close()

def record_break_taken(user_id):
    conn = get_connection()
    c = conn.cursor()
    now = int(time.time() * 1000)
    c.execute('''
        UPDATE break_compliance
        SET breaks_taken = breaks_taken + 1,
            last_break_time = ?,
            compliance_pct = ROUND((CAST(breaks_taken + 1 AS REAL) / MAX(1, breaks_recommended)) * 100.0, 1)
        WHERE user_id = ?
    ''', (now, user_id))
    conn.commit()
    conn.close()

def get_latest_session(user_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT * FROM sessions
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT 1
    ''', (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def get_risk_trend(user_id, range_param='7d'):
    conn = get_connection()
    c = conn.cursor()
    limit = 100
    if range_param == '24h':
        limit = 50
    elif range_param == '7d':
        limit = 100
    elif range_param == '30d':
        limit = 200

    c.execute('''
        SELECT timestamp, risk_score FROM sessions
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (user_id, limit))
    rows = c.fetchall()
    conn.close()
    
    rows = rows[::-1]
    timestamps = [r['timestamp'] for r in rows]
    risk_scores = [round(float(r['risk_score']), 3) for r in rows]
    return timestamps, risk_scores

def get_alert_history(user_id, limit=20):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT timestamp, risk_score, dispatch_type, receptive
        FROM alerts
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (user_id, limit))
    rows = c.fetchall()
    conn.close()
    return [{
        'timestamp': r['timestamp'],
        'risk_score': round(float(r['risk_score']), 2),
        'dispatch_type': r['dispatch_type'],
        'receptive': bool(r['receptive'])
    } for r in rows]

def get_break_compliance(user_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT breaks_taken, breaks_recommended, compliance_pct FROM break_compliance WHERE user_id = ?', (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        pct = min(100.0, max(0.0, float(row['compliance_pct'])))
        return {
            'compliance_pct': round(pct, 1),
            'breaks_taken': int(row['breaks_taken']),
            'breaks_recommended': int(row['breaks_recommended'])
        }
    return {
        'compliance_pct': 85.0,
        'breaks_taken': 4,
        'breaks_recommended': 5
    }

def get_session_detail(session_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM sessions WHERE session_id = ? OR id = ?', (str(session_id), str(session_id)))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    
    row_dict = dict(row)
    features = {}
    deviation = {}
    if row_dict.get('features_json'):
        try:
            features = json.loads(row_dict['features_json'])
        except Exception:
            pass
    if row_dict.get('deviation_json'):
        try:
            deviation = json.loads(row_dict['deviation_json'])
        except Exception:
            pass
            
    return {
        'session_id': row_dict.get('session_id'),
        'user_id': row_dict.get('user_id'),
        'timestamp': row_dict.get('timestamp'),
        'risk_score': round(float(row_dict.get('risk_score', 0)), 3),
        'alert_dispatched': bool(row_dict.get('alert_dispatched', False)),
        'features': features,
        'deviation': deviation
    }

def get_recent_sessions(limit=50):
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT risk_score, alert_dispatched, timestamp FROM sessions ORDER BY timestamp DESC LIMIT ?', (limit,))
    rows = [(r['risk_score'], r['alert_dispatched'], r['timestamp']) for r in c.fetchall()]
    conn.close()
    return rows

def get_all_students_summary():
    users = get_all_users()
    summaries = []
    for u in users:
        uid = u['user_id']
        latest = get_latest_session(uid)
        compliance = get_break_compliance(uid)
        conn = get_connection()
        c = conn.cursor()
        c.execute('SELECT COUNT(*) as alert_count FROM alerts WHERE user_id = ? AND dispatch_type = "immediate"', (uid,))
        alert_count = c.fetchone()['alert_count']
        conn.close()
        
        risk_score = round(float(latest['risk_score']), 2) if latest else 0.25
        status = 'high' if risk_score >= 0.6 else ('moderate' if risk_score >= 0.4 else 'low')
        summaries.append({
            'user_id': uid,
            'name': u['name'],
            'risk_score': risk_score,
            'status': status,
            'last_active': latest['timestamp'] if latest else None,
            'total_alerts': alert_count,
            'compliance_pct': compliance['compliance_pct']
        })
    return summaries
