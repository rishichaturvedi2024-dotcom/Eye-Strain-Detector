"""
Seeds realistic historical sessions into eye_strain.db
so the trend chart, alert history, and supervisor cohort summary
display realistic real-time and historical data.
"""

import os
import sys
import time
import random
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
import database

def seed():
    database.init_db()
    users = ['user1', 'user2', 'user3']
    now = int(time.time() * 1000)
    day_ms = 86400 * 1000
    
    print("Seeding historical session records...")
    
    for uid in users:
        base, thresh = database.get_user_baseline(uid)
        # Generate 15 historical sessions over last 7 days
        for i in range(15, 0, -1):
            ts = now - (i * 3600 * 6 * 1000) + random.randint(-60000, 60000)
            is_strain = random.random() < 0.28
            
            if is_strain:
                dist = random.uniform(18.0, 28.0)
                lux = random.uniform(20.0, 50.0)
                blink = random.uniform(4.0, 7.5)
                ibr = random.uniform(0.30, 0.55)
                gaze = random.uniform(14.0, 22.0)
                pose = random.uniform(12.0, 20.0)
                conf = random.uniform(0.70, 0.95)
                score = round(random.uniform(0.68, 0.92), 3)
                alert = True
            else:
                dist = random.uniform(48.0, 65.0)
                lux = random.uniform(280.0, 480.0)
                blink = random.uniform(13.5, 18.0)
                ibr = random.uniform(0.08, 0.16)
                gaze = random.uniform(4.0, 8.5)
                pose = random.uniform(4.0, 8.0)
                conf = 0.96
                score = round(random.uniform(0.12, 0.38), 3)
                alert = False

            F = {
                'd_avg': round(dist, 2),
                'd_min': round(dist - random.uniform(2, 6), 2),
                'I_avg': round(lux, 2),
                't_dur': 15.0,
                'blink_rate': round(blink, 2),
                'incomplete_blink_ratio': round(ibr, 3),
                'gaze_dev': round(gaze, 2),
                'head_pose': round(pose, 2),
                'cam_conf': round(conf, 2)
            }
            Delta = {f"{k}_dev": round(F[k] - base.get(k, 0), 2) for k in F}
            
            sess_id = f"sess_{uid}_{ts}_{random.randint(100, 999)}"
            database.insert_session(sess_id, uid, score, alert, ts, F, Delta)
            
            if alert:
                d_type = 'immediate' if random.random() < 0.8 else 'deferred'
                database.insert_alert(sess_id, uid, score, d_type, d_type == 'immediate', ts)

    print("Historical data seeded successfully!")

if __name__ == '__main__':
    seed()
