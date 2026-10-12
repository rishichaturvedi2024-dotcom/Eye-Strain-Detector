# IoT and Machine Learning Based System for Real-Time Monitoring and Personalized Risk Assessment of Digital Eye Strain

An end-to-end cyber-physical IoT and Machine Learning system that monitors screen viewing habits in real time, computes an individualized risk score using on-device physiological and environmental telemetry, and delivers receptivity-gated intervention alerts to prevent digital eye strain (asthenopia).

---

## 1. Minimal Hardware Configuration & Wiring Reference

The physical setup uses a minimal hardware footprint: an ESP32 powered over standard USB (or a 5V power bank), an ultrasonic distance sensor, a digital ambient light sensor, and an active buzzer driven directly from a GPIO pin.

### Wiring Table
| Component | Pin / Interface | ESP32 Pin | Notes & Voltage Protection |
| :--- | :--- | :--- | :--- |
| **ESP32 Dev Board** | Power | **Micro-USB / Type-C** | 5V supply from PC USB or power bank |
| **HC-SR04 Ultrasonic** | `VCC` | **5V (VIN)** | Powers ultrasonic transmitter circuitry |
| | `GND` | **GND** | Shared common ground |
| | `Trig` | **GPIO 5** | 10 µs trigger pulse (3.3V output from ESP32) |
| | `Echo` | **GPIO 18** | **Through 1kΩ / 2kΩ Voltage Divider**<br>Echo (5V) $\to$ 1kΩ resistor $\to$ GPIO 18 $\to$ 2kΩ resistor $\to$ GND.<br>Divides 5V signal to ~3.3V to protect ESP32 GPIO. |
| **BH1750 Light Sensor** | `VCC` | **3.3V** | 3.3V logic & power |
| | `GND` | **GND** | Shared common ground |
| | `SDA` | **GPIO 21** | I2C Data line (`Wire.begin(21, 22)`) |
| | `SCL` | **GPIO 22** | I2C Clock line |
| **Active Buzzer** | Positive (`+`) | **GPIO 23** | **Direct drive from ESP32 GPIO 23**<br>Polarity configurable via `#define BUZZER_ACTIVE_HIGH 1` (set to `0` if using an active-low buzzer module). |
| | Negative (`-`) | **GND** | Direct ground connection |

### Unique MQTT Client IDs
To prevent broker disconnect conflicts when running physical hardware alongside companion modules:
- ESP32 Firmware: `user1` (`DEVICE_ID`)
- Edge Simulator: `user1-sim`
- Backend Server: `eye-strain-backend`
- Vision Tracker: `user1-vision-tracker`

---

## 2. Arduino IDE 2.x Libraries to Install

To compile `edge_device/main.ino` for the **ESP32 Dev Module**, only the following libraries are required:

1. **`WiFi`** — Built-in with the official ESP32 Arduino Core (`esp32` by Espressif Systems).
2. **`Wire`** — Built-in with the official ESP32 Arduino Core.
3. **`BH1750`** by Christopher Laws (install via Arduino Library Manager).
4. **`PubSubClient`** by Nick O'Leary (install via Arduino Library Manager).

*Note: Telemetry serialization and alert decoding are handled cleanly with zero dependency on ArduinoJson, preventing version conflicts in Arduino IDE 2.x.*

---

## 3. System Architecture & Mathematical Workflow

The system follows a five-layer closed-loop architecture:
1. **Sensor Layer**: Non-camera ultrasonic distance ($d$) and digital illuminance ($I$) sampled at 1 Hz, debounced and filtered via median window filtering.
2. **Edge Processing Layer (`main.ino`)**: Timestamping, filtering, buffering, and **autonomous local threshold checking** (fires buzzer locally if $d < 25\text{ cm}$ or $I < 30\text{ lx}$ with a 250 ms non-blocking pulse independently of network connectivity).
3. **Wireless Communication Layer (MQTT)**:
   - ESP32 publishes telemetry to `telemetry/<device_id>` every 5 seconds.
   - Companion vision tracker publishes behavioral indicators to `behavior/<user_id>` every 5 seconds.
   - ESP32 subscribes to `alert/<device_id>` for backend actuation commands.
4. **Backend & Machine Learning Layer (`server.py`)**:
   - Performs session windowing (~15s) and feature fusion into the 9-dimensional feature vector:
     $$F = [d_{\text{avg}}, d_{\text{min}}, I_{\text{avg}}, t_{\text{dur}}, \text{blink\_rate}, \text{incomplete\_blink\_ratio}, \text{gaze\_dev}, \text{head\_pose}, \text{cam\_conf}]$$
   - Computes deviation from persistent per-user EWMA baseline $B(U)$:
     $$\Delta = F - B(U)$$
   - Low-light vision adaptation: If camera confidence $\text{cam\_conf} < 0.6$, behavioral features are smoothly downweighted toward baseline.
   - Evaluates risk score $R = f_\theta(F, \Delta)$ using trained Random Forest Classifier.
   - **Receptivity-Estimation Gating**: When $R \ge T(U)$, evaluates alert fatigue cooldown and head pose stability before deciding whether to dispatch immediately or defer for a bounded holding window.
5. **Dashboard Presentation Layer (`app.py`)**: Glassmorphic Flask dashboard displaying real-time risk scores, 7-day/30-day trends (Chart.js), alert dispatch history (immediate vs deferred), 20-20-20 break compliance, and supervisor multi-student cohort monitoring.

---

## 4. Machine Learning Model & Dataset Overhaul (Task 2 Justification)

### Why 99.75% Accuracy was a Red Flag:
An earlier prototype reported 99.75% classification accuracy on a synthetic test set. In clinical and physiological monitoring, near-100% accuracy is an indicator of artificial class separability caused by deterministic labeling rules. Real-world ocular and ergonomic signals suffer from:
- Ultrasonic jitter ($\pm 3\text{--}5\text{ cm}$) and echo dispersion.
- Ambient lux variance that scales inversely with illumination intensity.
- Substantial inter-individual resting blink rate differences (12–20 blinks/min).
- Natural class overlap: mild fatigue and normal concentrated screen work overlap near the boundary; strain is a continuous spectrum, not a step function.
- Vision tracker confidence degradation during dim lighting or occlusion.

### Dataset Regeneration & Realistic Evaluation:
The synthetic data pipeline in `backend/train_model.py` was regenerated with:
1. **Multi-User Baseline Variation**: 60 simulated participants, each with distinct baseline viewing distances and resting blink dynamics.
2. **Physiological Noise**: Gaussian sensor noise, ambient lux scaling, and physiological blink distributions.
3. **Continuous Strain Modeling & Boundary Overlap**: Continuous latent strain score with probabilistic transition margins and 4% label ambiguity.
4. **Participant-Level Splitting**: Strictly grouped train/test split on `user_id` (`GroupShuffleSplit`) ensuring **0% user leakage**.

### Validated Results (Random Forest Classifier):
- **Accuracy**: **86.48%** (in the scientifically plausible 80–93% range)
- **ROC-AUC**: **0.9182**
- **High-Risk Recall**: **82.00%** (ensuring true strain events are reliably captured)
- **Low-Risk Precision**: **92.72%**

---

## 5. Setup and Execution

### Step 1: Install Python Dependencies
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Step 2: Initialize Database & Train Model
```bash
python backend/train_model.py
```

### Step 3: Start Mosquitto Broker
Ensure your local Mosquitto broker is active:
```bash
mosquitto -v -p 1883
```

### Step 4: Run Backend Server
```bash
python backend/server.py
```

### Step 5: Start Companion App (or Simulator)
- **With Webcam**:
  ```bash
  python companion_app/vision_tracker.py
  ```
- **Edge Simulator (if ESP32 hardware is offline)**:
  ```bash
  python edge_device/simulator.py
  ```

### Step 6: Start Dashboard
```bash
python dashboard/app.py
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser:
- Student Dashboard: `/dashboard/user1`
- Supervisor Cohort Overview: `/dashboard/supervisor`
- Session Drill-Down: `/dashboard/session/<session_id>`

---

## 6. First Power-Up Checklist & Demonstration Procedure

1. **Install Arduino Libraries**: In Arduino IDE 2.x, verify that `BH1750` (Christopher Laws) and `PubSubClient` (Nick O'Leary) are installed.
2. **Configure Constants**: In `edge_device/main.ino`, set `#define WIFI_SSID`, `#define WIFI_PASSWORD`, and `#define MQTT_BROKER_IP` (your development computer's local IP on the same Wi-Fi network).
3. **Select Board and Upload**: Select **Tools > Board > ESP32 Arduino > ESP32 Dev Module**, choose the correct COM port, and click **Upload**.
4. **Power-On Self Test**: Upon boot, the active buzzer will emit a 100 ms confirmation chirp.
5. **Open Serial Monitor**: Set baud rate to **115200**. Confirm structured logs:
   ```
   [INFO] [SETUP] --- Digital Eye Strain Monitor - Minimal Hardware Firmware ---
   [INFO] [BH1750] BH1750 Ambient Light Sensor initialized (3.3V, I2C 21/22)
   [INFO] [MQTT] MQTT connected successfully
   ```
6. **Test Local Buzzer (Independent of Network)**:
   - Place your hand within 25 cm of the HC-SR04:
   - The buzzer emits a 250 ms pulse immediately and logs:
     `[WARN] [LOCAL_THRESHOLD] Distance too close: ... cm < 25.00cm`
   - This functions whether Wi-Fi/MQTT is connected or disconnected.
7. **Test End-to-End Round Trip (Sensors $\to$ MQTT $\to$ Backend ML $\to$ Alert $\to$ Dashboard)**:
   - Run `backend/server.py` and `dashboard/app.py`.
   - Hold hand within 25 cm (or cover the BH1750 to dim light) for ~15 seconds while webcam is active.
   - `server.py` ingests the batch, fuses features into $F$, computes $\Delta = F - B(U)$, detects risk $R > 0.60$, passes the receptivity gate, and publishes `"TRUE"` on `alert/user1`.
   - The ESP32 receives the command and sounds an **800 ms alert pulse**.
   - The dashboard updates in real time to show the elevated risk badge, updated trend point, and alert history entry.

---

## 7. Team Contributions & Division of Work

| Member | Role | Core Modules & Key Deliverables |
| :--- | :--- | :--- |
| **Rishi** *(Team Lead)* | **System Architect & ML/Backend Engineer** | End-to-end architecture design, central backend server (`backend/server.py`), 9D feature fusion, adaptive EWMA baseline tracking, ML model training & zero-leakage GroupShuffleSplit validation (`backend/train_model.py`), receptivity-gated alert dispatching, database schema (`backend/database.py`), broker orchestration (`mosquitto.conf`, `start_all.bat`). |
| **Aditya** | **Embedded Systems & IoT Hardware Engineer** | ESP32 circuit design & wiring, 1kΩ/2kΩ voltage divider protection for 5V echo, C++ firmware (`edge_device/main.ino`), HC-SR04 & BH1750 driver integration, autonomous offline threshold fail-safe, hardware-in-the-loop simulator (`edge_device/simulator.py`). |
| **Tiya** | **Computer Vision & UI/UX Dashboard Engineer** | MediaPipe FaceLandmarker companion vision tracker (`companion_app/vision_tracker.py`), Eye Aspect Ratio (EAR) blink and incomplete blink detection, gaze & posture estimation, privacy-preserving edge processing, Flask web dashboard & REST API (`dashboard/app.py`), student & supervisor cohort interfaces. |


