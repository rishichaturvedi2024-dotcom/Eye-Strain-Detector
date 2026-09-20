import sqlite3
import os

DB_PATH = 'eye_strain.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            distance REAL,
            lux REAL,
            timestamp INTEGER
        )
    ''')
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
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            risk_score REAL,
            alert_dispatched BOOLEAN,
            timestamp INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def insert_telemetry(user_id, distance, lux, ts):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO telemetry (user_id, distance, lux, timestamp) VALUES (?, ?, ?, ?)',
              (user_id, distance, lux, ts))
    conn.commit()
    conn.close()

def insert_behavior(user_id, br, ibr, gd, hp, conf, ts):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO behavior (user_id, blink_rate, incomplete_blink_ratio, gaze_dev, head_pose, cam_conf, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)',
              (user_id, br, ibr, gd, hp, conf, ts))
    conn.commit()
    conn.close()

def insert_session(user_id, risk_score, alert, ts):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO sessions (user_id, risk_score, alert_dispatched, timestamp) VALUES (?, ?, ?, ?)',
              (user_id, risk_score, alert, ts))
    conn.commit()
    conn.close()

def get_recent_sessions(limit=50):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT risk_score, alert_dispatched, timestamp FROM sessions ORDER BY timestamp DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()
    return rows
