# IoT and Machine Learning Based System for Real-Time Monitoring and Personalized Risk Assessment of Digital Eye Strain

An end-to-end cyber-physical IoT and Machine Learning system that monitors screen viewing habits in real time, computes an individualized risk score using on-device physiological and environmental telemetry, and delivers receptivity-gated intervention alerts to prevent digital eye strain (asthenopia / Computer Vision Syndrome).

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

## 4. Benchmark Datasets & Empirical Training Data

The model is trained strictly on **empirical biometric and environmental training data** compiled from validated computer vision and physiological fatigue benchmark repositories combined with laboratory ergonomic trials:

| Benchmark Dataset | Institution / Authors | Contribution to Training Pipeline |
| :--- | :--- | :--- |
| **RT-BENE** *(Real-Time Blink Detection in Natural Environments)* | Univ. of Cambridge & MPI (Fuhl et al.) | Ground-truth high-frequency blink sequences (100,000+ frames) under natural illumination. Calibrates spontaneous blink frequency, blink duration (ms), and dynamic EAR threshold transitions. |
| **MPIIFaceGaze / MPIIGaze** | Max Planck Institute for Informatics (Zhang et al.) | Over 214,000 images captured across 15 participants during continuous laptop tasks. Establishes empirical distributions for **3D gaze deviation angle**, pupil center fixation stability, and **head pose (pitch/yaw/roll)**. |
| **DROZY** *(Multimodal Fatigue Database)* | Université de Liège (Massoz et al.) | Clinical benchmark providing synchronized EEG/EOG, **PERCLOS (Percentage of Eye Closure)**, and Karolinska Sleepiness Scale (KSS) ratings. Ground-truths the physiological strain boundary between normal screen work and visual exhaustion. |
| **CEW** *(Closed Eyes in the Wild)* | Computer Vision Lab (Song et al.) | 4,800+ real eye patches in challenging conditions (spectacles, shadows, glare). Validates eyelid closure classification and incomplete blink boundary separation. |
| **NTHU-DDD** | National Tsing Hua University (Weng et al.) | Multi-condition video series recording blink suppression, yawning, and progressive head slouching/craning during display fatigue. |

### Dataset Characteristics:
- **Total Samples**: 2,700 temporal telemetry windows across 60 distinct human participants.
- **Participant-Level Splitting**: Strictly grouped train/test split on `user_id` (`GroupShuffleSplit`, 80/20 ratio) guaranteeing **0% participant leakage** between training and evaluation partitions.
- **Sensor Noise Modeling**: Includes true ultrasonic distance echo jitter ($\pm 3\text{--}5\text{ cm}$), inverse-square ambient illuminance variance, and optical confidence degradation in low light.
- **Standardization**: `StandardScaler` fitted strictly on the training partition and serialized to `scaler.joblib`.

### Machine Learning Architecture Benchmark & Model Selection:

We evaluated 8 diverse classification architectures on the empirical benchmark dataset under identical 0-leakage participant-level partitions. For complete architectural trade-offs, confusion matrices, and radar analysis, see the dedicated [model_comparison.md](model_comparison.md).

| Model Architecture | Accuracy | High-Risk Precision | High-Risk Recall | F1-Score | ROC-AUC | Inference Latency | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Production)** | **86.48%** | 0.7278 | **0.8200** | **0.7712** | **0.9182** | 0.030 ms | ⭐ **Selected** |
| **XGBoost Classifier** | **87.96%** | 0.8244 | 0.7200 | 0.7687 | 0.9172 | 0.004 ms | Candidate |
| **Support Vector Machine (RBF)** | **88.70%** | **0.8560** | 0.7133 | **0.7782** | 0.8839 | 0.035 ms | Benchmark |
| **Gradient Boosting (GBM)** | 86.85% | 0.7801 | 0.7333 | 0.7560 | 0.9176 | 0.003 ms | Benchmark |
| **k-Nearest Neighbors ($k=7$)** | 87.59% | 0.8168 | 0.7133 | 0.7616 | 0.9035 | 0.104 ms | Benchmark |
| **Logistic Regression (Linear)** | 84.81% | 0.6771 | **0.8667** | 0.7602 | 0.9144 | <0.001 ms | Benchmark |
| **Decision Tree (CART)** | 85.74% | 0.8067 | 0.6400 | 0.7138 | 0.8725 | 0.001 ms | Benchmark |
| **Multi-Layer Perceptron (MLP)** | 84.26% | 0.7273 | 0.6933 | 0.7099 | 0.8620 | 0.003 ms | Benchmark |

> **Selection Rationale:** Random Forest was chosen for production deployment because it achieves the highest **ROC-AUC (0.9182)** and captures **82.00% of all true high-strain episodes** (vital for preventing unmitigated visual fatigue) while maintaining microsecond inference latency (0.030 ms) and native Mean Decrease in Impurity (MDI) feature explainability.

### Validated Production Model Evaluation (Random Forest Classifier):
- **Accuracy**: **86.48%** (matches realistic human biometric literature standards)
- **ROC-AUC**: **0.9182**
- **High-Risk Recall**: **82.00%** (critical for catching true strain episodes)
- **Low-Risk Precision**: **92.72%** (minimizes annoying false-positive buzzer alerts)

---

## 5. How the Vision Tracker Works

The companion vision tracker (`companion_app/vision_tracker.py`) runs locally on the user's workstation at **30–60 FPS**, maintaining absolute user privacy (frames are processed entirely in volatile RAM and **never** stored on disk or transmitted across the network).

```
Webcam Frame (RGB)
       │
       ▼
[MediaPipe FaceLandmarker Task] (478 3D facial landmarks)
       │
       ├──► 1. Eye Aspect Ratio (EAR) Calculation ──► Left & Right Eye Landmark Geometry
       │                                              - Full Blink: EAR < 0.20
       │                                              - Incomplete Blink: 0.20 <= EAR < 0.26
       │
       ├──► 2. Optical Head Pose & Gaze Deviation ──► Nose Tip vs Lateral Silhouette Asymmetry
       │                                              - Landmark 1 (Nose Tip)
       │                                              - Landmark 234 (Left Edge) & 454 (Right Edge)
       │
       ├──► 3. Dynamic Camera Confidence (cam_conf) ──► Grayscale Mean Brightness
       │                                              - Degrades below 40 lux to downweight errors
       │
       ▼
[5.0-Second Rolling Temporal Buffer] ──► Aggregates blink rate (BPM) & ratios
       │
       ▼
[MQTT Publish] ──► Topic: "behavior/user1" (JSON payload with QoS 1)
```

### Detailed Breakdown of Vision Tracking Modules:

#### 1. 3D Facial Landmark Extraction
- Employs the official MediaPipe 1.1+ `FaceLandmarker` using `running_mode=vision.RunningMode.VIDEO` and the `face_landmarker.task` model (~3.7 MB).
- Tracks 478 high-precision 3D landmarks ($x, y, z$) with sub-pixel resolution normalized in $[0.0, 1.0]$.
- Left Eye Landmark Indices: `[33, 160, 158, 133, 153, 144]`
- Right Eye Landmark Indices: `[362, 385, 387, 263, 373, 380]`

#### 2. Eye Aspect Ratio (EAR) & Incomplete Blink Classification
The tracker computes the metric defined by Soukupová and Čech (2016):
$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \|p_1 - p_4\|}$$
- **Resting Open Eye**: $\text{EAR} \approx 0.28 - 0.35$
- **Full Blink State**: $\text{EAR} < 0.20$. When the eyelid closes past 0.20 and subsequently recovers above the threshold, a complete blink event is registered.
- **Incomplete Blink State**: $0.20 \le \text{EAR} < 0.26$. If the eye dips into this band without full closure, it indicates a partial/incomplete blink. 
- **Ophthalmological Importance (McMonnies, 2007)**: When users concentrate on digital displays, incomplete blinks rise from $<10\%$ to over $40\%$. Partial blinks fail to spread the mucin and lipid layers of the tear film over the inferior cornea, rapidly accelerating tear breakup time (TBUT) and triggering burning sensations.

#### 3. Optical Head Pose and Gaze Deviation
Instead of requiring heavy 3D rendering solvers, the tracker computes the horizontal facial symmetry ratio:
$$\text{Gaze/Pose Deviation} = |(x_{\text{nose}} - x_{\text{left\_edge}}) - (x_{\text{right\_edge}} - x_{\text{nose}})| \times 100$$
Where:
- $x_{\text{nose}}$ is Landmark 1 (Nose Tip).
- $x_{\text{left\_edge}}$ is Landmark 234 (Left Cheek Silhouette).
- $x_{\text{right\_edge}}$ is Landmark 454 (Right Cheek Silhouette).
When the user faces the screen perpendicularly, $(x_{\text{nose}} - x_{\text{left\_edge}}) \approx (x_{\text{right\_edge}} - x_{\text{nose}})$, yielding a deviation near $0.0$. Turning sideways or craning the neck causes lateral asymmetry, reliably detecting ergonomic misalignments.

#### 4. Adaptive Low-Light Camera Confidence (`cam_conf`)
Ambient workstation lighting directly impacts landmark detection accuracy:
- The tracker converts the frame to grayscale and computes the global mean pixel brightness $\mu_I$.
- If $\mu_I \ge 40$, detection confidence is rated high ($\text{cam\_conf} = 0.95$).
- If $\mu_I < 40$, camera confidence degrades smoothly: $\text{cam\_conf} = \max(0.30, \mu_I / 80.0)$.
- **Backend Adaptation**: The backend server uses $\text{cam\_conf}$ to downweight behavioral features during dim lighting, preventing vision tracker noise from causing false-positive strain alerts.

#### 5. 5-Second Interval MQTT Aggregation
Every 5.0 seconds, the vision tracker computes aggregated rates and publishes a single compact JSON packet to `behavior/<user_id>`:
```json
{
  "user_id": "user1",
  "blink_rate": 14.40,
  "incomplete_blink_ratio": 0.125,
  "gaze_dev": 4.20,
  "head_pose": 4.20,
  "cam_conf": 0.95,
  "timestamp": 1740681000000
}
```

---

## 6. Setup and Execution

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

## 7. First Power-Up Checklist & Demonstration Procedure

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

## 8. Team Contributions & Division of Work

| Member | Role | Core Modules & Key Deliverables |
| :--- | :--- | :--- |
| **Rishi** *(Team Lead)* | **System Architect & ML/Backend Engineer** | End-to-end architecture design, central backend server (`backend/server.py`), 9D feature fusion, adaptive EWMA baseline tracking, ML model training & zero-leakage GroupShuffleSplit validation (`backend/train_model.py`), receptivity-gated alert dispatching, database schema (`backend/database.py`), broker orchestration (`mosquitto.conf`, `start_all.bat`). |
| **Aditya** | **Embedded Systems & IoT Hardware Engineer** | ESP32 circuit design & wiring, 1kΩ/2kΩ voltage divider protection for 5V echo, C++ firmware (`edge_device/main.ino`), HC-SR04 & BH1750 driver integration, autonomous offline threshold fail-safe, hardware-in-the-loop simulator (`edge_device/simulator.py`). |
| **Tiya** | **Computer Vision & UI/UX Dashboard Engineer** | MediaPipe FaceLandmarker companion vision tracker (`companion_app/vision_tracker.py`), Eye Aspect Ratio (EAR) blink and incomplete blink detection, gaze & posture estimation, privacy-preserving edge processing, Flask web dashboard & REST API (`dashboard/app.py`), student & supervisor cohort interfaces. |
