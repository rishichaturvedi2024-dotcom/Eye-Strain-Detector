from flask import Flask, render_template, jsonify
import sys
import os

# Add backend to path to import database
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
import database

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    # Get last 50 sessions
    rows = database.get_recent_sessions(50)
    # rows: [(risk_score, alert_dispatched, timestamp), ...]
    # reverse to chronological
    rows = rows[::-1]
    
    timestamps = [r[2] for r in rows]
    scores = [r[0] for r in rows]
    alerts = [r[1] for r in rows]
    
    return jsonify({
        'timestamps': timestamps,
        'scores': scores,
        'alerts': alerts
    })

if __name__ == '__main__':
    database.init_db()
    app.run(debug=True, port=5000)
