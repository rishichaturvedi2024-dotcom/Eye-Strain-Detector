/*
 * IoT and Machine Learning Based System for Real-Time Monitoring 
 * and Personalized Risk Assessment of Digital Eye Strain
 * 
 * =========================================================================
 * MINIMAL HARDWARE WIRING (USB Powered ESP32):
 *  1. ESP32 Dev Board (powered via USB / 5V power bank)
 *  2. HC-SR04 Ultrasonic Ranging Sensor:
 *       VCC   -> 5V (VIN pin of ESP32)
 *       GND   -> GND
 *       Trig  -> GPIO 5
 *       Echo  -> GPIO 18 via Voltage Divider (Echo 5V -> 1k -> GPIO18 -> 2k -> GND)
 *  3. BH1750 Ambient Light Sensor (I2C):
 *       VCC   -> 3.3V
 *       GND   -> GND
 *       SDA   -> GPIO 21
 *       SCL   -> GPIO 22
 *  4. Active Buzzer:
 *       Buzzer (+) -> GPIO 23 (Direct drive)
 *       Buzzer (-) -> GND
 * =========================================================================
 * Required Libraries to Install in Arduino IDE 2.x:
 *  - WiFi (Built-in with ESP32 board package)
 *  - Wire (Built-in with ESP32 board package)
 *  - BH1750 by Christopher Laws
 *  - PubSubClient by Nick O'Leary
 * =========================================================================
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <BH1750.h>

// ================= Configuration & Pin Definitions =================
#define WIFI_SSID        "Nahi Milega"                 // Replace with your 2.4GHz Wi-Fi SSID
#define WIFI_PASSWORD     "11111111"      // Replace with your Wi-Fi Password
#define MQTT_BROKER_IP    "10.14.134.203"      // Replace with IP of your MQTT broker / backend
#define MQTT_BROKER_PORT   1883
#define DEVICE_ID          "user1"              // Device & User ID matching backend and dashboard

// Pin Definitions
#define PIN_TRIG            5   // HC-SR04 Trigger (GPIO 5)
#define PIN_ECHO            18  // HC-SR04 Echo (GPIO 18 via 1k/2k voltage divider)
#define PIN_I2C_SDA         21  // BH1750 SDA (GPIO 21)
#define PIN_I2C_SCL         22  // BH1750 SCL (GPIO 22)
#define BUZZER_PIN          23  // Active Buzzer (+) direct drive (GPIO 23), (-) to GND

// Buzzer Polarity (Set to 1 for active-high, set to 0 for active-low buzzer module)
#define BUZZER_ACTIVE_HIGH  1

#if BUZZER_ACTIVE_HIGH
  #define BUZZER_ON_LEVEL   HIGH
  #define BUZZER_OFF_LEVEL  LOW
#else
  #define BUZZER_ON_LEVEL   LOW
  #define BUZZER_OFF_LEVEL  HIGH
#endif

// Hardware Operational Constants & Thresholds
const float MIN_DISTANCE_CM           = 25.0f;  // Ergonomic distance breach threshold (<25 cm)
const float MIN_ILLUMINANCE_LX        = 30.0f;  // Ambient lighting threshold (<30 lx)
const float MAX_VALID_DIST_CM         = 200.0f; // Max valid desk sitting distance
const float MIN_VALID_DIST_CM         = 5.0f;   // Min valid distance

const unsigned long LOCAL_ALERT_DURATION_MS   = 250;  // 250 ms non-blocking pulse on local breach
const unsigned long REMOTE_ALERT_DURATION_MS  = 800;  // 800 ms non-blocking pulse on backend risk alert

// Non-blocking Timing Intervals
const unsigned long SENSOR_SAMPLE_INTERVAL_MS     = 1000; // Sample distance & lux every 1s
const unsigned long TELEMETRY_PUBLISH_INTERVAL_MS = 5000; // Publish telemetry batch every 5s
const unsigned long WIFI_RECONNECT_INTERVAL_MS    = 5000; // Non-blocking Wi-Fi retry interval
const unsigned long MQTT_RECONNECT_INTERVAL_MS    = 5000; // Non-blocking MQTT retry interval

// MQTT Topics
char telemetry_topic[64];
char alert_topic[64];

// Global Instances
WiFiClient espClient;
PubSubClient mqttClient(espClient);
BH1750 lightMeter;

// State Tracking
unsigned long lastSampleTime    = 0;
unsigned long lastPublishTime   = 0;
unsigned long lastWiFiCheckTime = 0;
unsigned long lastMqttCheckTime = 0;

// Non-blocking Buzzer State Machine
unsigned long buzzerOffTime = 0;
bool isBuzzerActive         = false;

// Filtered Sensor Values
float filteredDistance  = 50.0f;
float filteredLux       = 100.0f;
float lastValidDistance = 50.0f;

// Telemetry accumulation for 5s batch
float distSum    = 0.0f;
float luxSum     = 0.0f;
int sampleCount  = 0;

// ================= Structured Serial Logging =================
void logMsg(const char* level, const char* module, const String& msg) {
  unsigned long now = millis();
  Serial.printf("[%lu ms] [%s] [%s] %s\n", now, level, module, msg.c_str());
}

// ================= Non-blocking Active Buzzer Controller =================
void triggerBuzzer(unsigned long durationMs, const char* reason) {
  digitalWrite(BUZZER_PIN, BUZZER_ON_LEVEL);
  isBuzzerActive = true;
  buzzerOffTime = millis() + durationMs;
  logMsg("ALERT", "BUZZER", String("Buzzer ON for ") + durationMs + "ms. Reason: " + reason);
}

void updateBuzzerState() {
  if (isBuzzerActive && millis() >= buzzerOffTime) {
    digitalWrite(BUZZER_PIN, BUZZER_OFF_LEVEL);
    isBuzzerActive = false;
    logMsg("DEBUG", "BUZZER", "Buzzer OFF");
  }
}

// ================= Ultrasonic HC-SR04 Acquisition & 3-Ping Median Filter =================
float measureRawDistanceCm() {
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(2);
  
  // 10us trigger pulse
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);
  
  // Measure pulse duration (timeout 30000us ~ 5 meters)
  unsigned long durationUs = pulseIn(PIN_ECHO, HIGH, 30000);
  if (durationUs == 0) {
    return -1.0f; // Timeout / echo lost
  }
  
  // Speed of sound: 343 m/s => 0.0343 cm/us => duration / 58.2
  return (float)durationUs / 58.2f;
}

float getFilteredDistance() {
  float readings[3];
  int validCount = 0;
  
  for (int i = 0; i < 3; i++) {
    float r = measureRawDistanceCm();
    if (r >= MIN_VALID_DIST_CM && r <= MAX_VALID_DIST_CM) {
      readings[validCount++] = r;
    }
    delayMicroseconds(500);
  }
  
  if (validCount == 0) {
    return lastValidDistance;
  }
  
  // Simple 3-element median sort
  for (int i = 0; i < validCount - 1; i++) {
    for (int j = i + 1; j < validCount; j++) {
      if (readings[i] > readings[j]) {
        float tmp = readings[i];
        readings[i] = readings[j];
        readings[j] = tmp;
      }
    }
  }
  
  float medianVal = readings[validCount / 2];
  
  // Fast exponential smoothing (alpha = 0.70) triggers in ~2 samples on distance drop
  filteredDistance = (0.70f * medianVal) + (0.30f * filteredDistance);
  lastValidDistance = filteredDistance;
  return filteredDistance;
}

// ================= Ambient Light BH1750 Acquisition & Smoothing =================
float getFilteredLux() {
  float lux = lightMeter.readLightLevel();
  if (lux < 0.0f) {
    logMsg("WARN", "BH1750", "Error reading lux from BH1750; using last smoothed value");
    return filteredLux;
  }
  // Exponential filter (alpha = 0.40)
  filteredLux = (0.4f * lux) + (0.6f * filteredLux);
  return filteredLux;
}

// ================= Local Threshold Alerting (Independent of Network) =================
void checkLocalThresholds(float distance, float lux) {
  static unsigned long lastLocalAlertTime = 0;
  const unsigned long LOCAL_ALERT_COOLDOWN_MS = 3000;
  
  if (millis() - lastLocalAlertTime < LOCAL_ALERT_COOLDOWN_MS) {
    return;
  }
  
  bool distanceBreach = (distance > 0.0f && distance < MIN_DISTANCE_CM);
  bool luxBreach      = (lux >= 0.0f && lux < MIN_ILLUMINANCE_LX);
  
  if (distanceBreach || luxBreach) {
    lastLocalAlertTime = millis();
    String reason = "";
    if (distanceBreach) {
      reason += String("Distance too close: ") + distance + "cm < " + MIN_DISTANCE_CM + "cm; ";
    }
    if (luxBreach) {
      reason += String("Ambient light dim: ") + lux + "lx < " + MIN_ILLUMINANCE_LX + "lx; ";
    }
    logMsg("WARN", "LOCAL_THRESHOLD", reason);
    // 250 ms non-blocking local warning pulse
    triggerBuzzer(LOCAL_ALERT_DURATION_MS, "LOCAL_THRESHOLD_BREACH");
  }
}

// ================= MQTT Callback (Remote Alert from Backend) =================
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  char message[64];
  unsigned int copyLen = (length < sizeof(message) - 1) ? length : (sizeof(message) - 1);
  memcpy(message, payload, copyLen);
  message[copyLen] = '\0';
  
  logMsg("INFO", "MQTT_RX", String("Topic: ") + topic + " | Payload: " + message);
  
  if (strcmp(topic, alert_topic) == 0) {
    // Only fire buzzer if payload contains "TRUE" or "true"
    // Does NOT trigger on "false", "0.61", or {"alert": false}
    if (strstr(message, "TRUE") != NULL || strstr(message, "true") != NULL) {
      logMsg("ALERT", "REMOTE_ALERT", "Backend dispatched high eye-strain risk alert!");
      // 800 ms non-blocking alert pulse
      triggerBuzzer(REMOTE_ALERT_DURATION_MS, "BACKEND_RISK_ALERT");
    }
  }
}

// ================= Non-Blocking Network Handlers =================
void maintainWiFi() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }
  
  if (millis() - lastWiFiCheckTime >= WIFI_RECONNECT_INTERVAL_MS) {
    lastWiFiCheckTime = millis();
    logMsg("WARN", "WIFI", "Wi-Fi disconnected. Attempting non-blocking reconnect...");
    WiFi.disconnect();
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  }
}

void maintainMQTT() {
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }
  
  if (mqttClient.connected()) {
    mqttClient.loop();
    return;
  }
  
  if (millis() - lastMqttCheckTime >= MQTT_RECONNECT_INTERVAL_MS) {
    lastMqttCheckTime = millis();
    logMsg("WARN", "MQTT", "Connecting to MQTT broker...");
    
    // Uses DEVICE_ID as unique MQTT client ID
    if (mqttClient.connect(DEVICE_ID)) {
      logMsg("INFO", "MQTT", "MQTT connected successfully");
      mqttClient.subscribe(alert_topic);
      logMsg("INFO", "MQTT", String("Subscribed to alert topic: ") + alert_topic);
    } else {
      logMsg("ERROR", "MQTT", String("MQTT connection failed, state: ") + mqttClient.state());
    }
  }
}

// ================= Publish Telemetry =================
void publishTelemetry(float distance, float lux) {
  if (!mqttClient.connected()) {
    logMsg("DEBUG", "MQTT", "MQTT not connected, skipping publish");
    return;
  }
  
  // Format exact JSON payload expected by backend server (zero library dependencies)
  char jsonBuffer[160];
  snprintf(jsonBuffer, sizeof(jsonBuffer),
           "{\"device_id\":\"%s\",\"distance\":%.2f,\"lux\":%.1f,\"timestamp\":%lu}",
           DEVICE_ID, distance, lux, millis());
  
  if (mqttClient.publish(telemetry_topic, jsonBuffer)) {
    logMsg("INFO", "MQTT_PUB", String("Published telemetry -> ") + jsonBuffer);
  } else {
    logMsg("ERROR", "MQTT_PUB", "Failed to publish telemetry");
  }
}

// ================= Setup & Loop =================
void setup() {
  Serial.begin(115200);
  delay(100);
  
  snprintf(telemetry_topic, sizeof(telemetry_topic), "telemetry/%s", DEVICE_ID);
  snprintf(alert_topic, sizeof(alert_topic), "alert/%s", DEVICE_ID);
  
  logMsg("INFO", "SETUP", "--- Digital Eye Strain Monitor - Minimal Hardware Firmware ---");
  logMsg("INFO", "SETUP", String("Device ID: ") + DEVICE_ID);
  
  // Pin configurations
  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, BUZZER_OFF_LEVEL);
  
  // I2C initialization with explicit SDA=21, SCL=22
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  if (lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE)) {
    logMsg("INFO", "BH1750", "BH1750 Ambient Light Sensor initialized (3.3V, I2C 21/22)");
  } else {
    logMsg("ERROR", "BH1750", "Error initializing BH1750. Check I2C wiring (SDA=21, SCL=22, 3.3V)");
  }
  
  // Power-on self test: 100ms chirp on active buzzer
  logMsg("INFO", "SELF_TEST", "Pulsing active buzzer (100ms) for power-on verification");
  digitalWrite(BUZZER_PIN, BUZZER_ON_LEVEL);
  delay(100);
  digitalWrite(BUZZER_PIN, BUZZER_OFF_LEVEL);
  
  // Initial Wi-Fi attempt (non-blocking begin)
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  lastWiFiCheckTime = millis();
  
  // MQTT setup
  mqttClient.setServer(MQTT_BROKER_IP, MQTT_BROKER_PORT);
  mqttClient.setCallback(mqttCallback);
  
  logMsg("INFO", "SETUP", "Setup complete. Entering non-blocking operational loop.");
}

void loop() {
  // 1. Maintain network connections non-blockingly
  maintainWiFi();
  maintainMQTT();
  
  // 2. Manage buzzer pulse state machine (turns off buzzer without blocking delay)
  updateBuzzerState();
  
  unsigned long now = millis();
  
  // 3. Fixed-interval sensor sampling (every 1s)
  if (now - lastSampleTime >= SENSOR_SAMPLE_INTERVAL_MS) {
    lastSampleTime = now;
    
    float currentDist = getFilteredDistance();
    float currentLux  = getFilteredLux();
    
    // Accumulate for 5s telemetry batch
    distSum += currentDist;
    luxSum  += currentLux;
    sampleCount++;
    
    // 4. Local threshold check (fires buzzer locally even if Wi-Fi / MQTT is offline!)
    checkLocalThresholds(currentDist, currentLux);
  }
  
  // 5. Batch publish telemetry to MQTT (every 5s)
  if (now - lastPublishTime >= TELEMETRY_PUBLISH_INTERVAL_MS) {
    lastPublishTime = now;
    
    if (sampleCount > 0) {
      float avgDist = distSum / sampleCount;
      float avgLux  = luxSum / sampleCount;
      publishTelemetry(avgDist, avgLux);
      
      // Reset accumulators
      distSum = 0.0f;
      luxSum  = 0.0f;
      sampleCount = 0;
    }
  }
}
