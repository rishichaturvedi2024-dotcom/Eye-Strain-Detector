.\start_all.bat
.\start_all.bat
@echo off
title Eye Strain Monitor - System Launcher
echo ========================================================
echo   Starting Eye Strain Monitoring System
echo ========================================================
echo.

cd /d "%~dp0"

echo [1/4] Starting Mosquitto MQTT Broker...
start "Mosquitto MQTT Broker" cmd /k ""C:\Program Files\Mosquitto\mosquitto.exe" -c mosquitto.conf -v"
timeout /t 2 /nobreak >nul

echo [2/4] Starting Backend Server (ML & Fusion Engine)...
start "Backend Server" cmd /k ".\venv\Scripts\python.exe backend\server.py"
timeout /t 2 /nobreak >nul

echo [3/4] Starting Flask Dashboard...
start "Flask Dashboard" cmd /k ".\venv\Scripts\python.exe dashboard\app.py"
timeout /t 2 /nobreak >nul

echo [4/4] Starting Companion Vision Tracker (Webcam)...
start "Vision Tracker (Webcam)" cmd /k ".\venv\Scripts\python.exe companion_app\vision_tracker.py"

echo.
echo ========================================================
echo   All 4 services launched in separate windows!
echo   Dashboard URL: http://localhost:5000
echo ========================================================
echo.
pause
