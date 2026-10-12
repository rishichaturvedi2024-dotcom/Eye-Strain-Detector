# =========================================================================
# Author: Tiya Agrawal (24BDS0004) - Dashboard & REST API Architecture
# Route Handlers: Student view, supervisor cohort monitoring, and session drill-down
# =========================================================================


import os
import sys
import time
from flask import Flask, render_template, jsonify, request, redirect, url_for

# Ensure backend directory is in path to import database module
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import database

app = Flask(__name__)

# ================= Page Views =================

@app.route('/')
def index():
    return redirect(url_for('student_view', user_id='user1'))

@app.route('/dashboard/<user_id>')
def student_view(user_id):
    user = database.get_or_create_user(user_id)
    all_users = database.get_all_users()
    return render_template('dashboard.html', user=user, all_users=all_users)

@app.route('/dashboard/supervisor')
def supervisor_view():
    all_users = database.get_all_users()
    summaries = database.get_all_students_summary()
    return render_template('supervisor.html', all_users=all_users, summaries=summaries)

@app.route('/dashboard/session/<session_id>')
def session_detail_view(session_id):
    detail = database.get_session_detail(session_id)
    return render_template('session_detail.html', session=detail, session_id=session_id)


# ================= Required API Endpoints =================

@app.route('/api/current_risk/<user_id>')
def api_current_risk(user_id):
    latest = database.get_latest_session(user_id)
    if not latest:
        # Default starting state if no session recorded yet
        return jsonify({
            'user_id': user_id,
            'risk_score': 0.15,
            'timestamp': int(time.time() * 1000),
            'status': 'low',
            'alert_dispatched': False,
            'features': {}
        })

    score = float(latest['risk_score'])
    status = 'high' if score >= 0.60 else ('moderate' if score >= 0.40 else 'low')
    
    return jsonify({
        'user_id': user_id,
        'session_id': latest.get('session_id'),
        'risk_score': round(score, 3),
        'timestamp': latest['timestamp'],
        'status': status,
        'alert_dispatched': bool(latest['alert_dispatched']),
        'd_avg': latest.get('d_avg'),
        'l_avg': latest.get('l_avg'),
        'blink_rate': latest.get('blink_rate'),
        'cam_conf': latest.get('cam_conf')
    })

@app.route('/api/risk_trend/<user_id>')
def api_risk_trend(user_id):
    range_param = request.args.get('range', '7d')
    timestamps, risk_scores = database.get_risk_trend(user_id, range_param)
    return jsonify({
        'user_id': user_id,
        'range': range_param,
        'timestamps': timestamps,
        'risk_scores': risk_scores
    })

@app.route('/api/alert_history/<user_id>')
def api_alert_history(user_id):
    limit = int(request.args.get('limit', 20))
    alerts = database.get_alert_history(user_id, limit=limit)
    return jsonify(alerts)

@app.route('/api/break_compliance/<user_id>')
def api_break_compliance(user_id):
    compliance = database.get_break_compliance(user_id)
    return jsonify(compliance)

@app.route('/api/session_detail/<session_id>')
def api_session_detail(session_id):
    detail = database.get_session_detail(session_id)
    if not detail:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify(detail)

@app.route('/api/users')
def api_users():
    users = database.get_all_users()
    return jsonify(users)

@app.route('/api/supervisor_summary')
def api_supervisor_summary():
    summaries = database.get_all_students_summary()
    return jsonify(summaries)

@app.route('/api/record_break/<user_id>', methods=['POST'])
def api_record_break(user_id):
    database.record_break_taken(user_id)
    compliance = database.get_break_compliance(user_id)
    return jsonify({'success': True, 'compliance': compliance})


# ================= Legacy Compatibility Endpoint =================

@app.route('/api/data')
def api_data():
    rows = database.get_recent_sessions(50)
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
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Eye Strain Monitoring Dashboard on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
