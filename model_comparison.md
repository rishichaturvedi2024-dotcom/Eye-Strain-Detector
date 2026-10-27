# 🔬 Machine Learning Model Evaluation & Comparative Benchmark

This document provides a comparative analysis of **8 distinct Machine Learning classification architectures** evaluated for real-time Digital Eye Strain (Asthenopia) risk assessment.

The models were evaluated on the **Empirical Multi-Modal Benchmark Dataset** (`backend/benchmark_training_data.csv`), comprising **2,700 temporal telemetry windows across 60 distinct human participants**. Evaluation was executed using strict **Participant-Level Splitting** (`GroupShuffleSplit`, 80/20 train/test partition) to guarantee **0% user leakage** across partitions.

---

## 1. Master Model Comparison Table

All models were evaluated using standardized 18-dimensional feature representations ($9\text{ raw features} + 9\text{ personal EWMA baseline deviations}$) preprocessed with `StandardScaler` fitted strictly on the training partition.

| Model Architecture | Test Accuracy | Precision (High Risk) | Recall (High Risk) | F1-Score (High Risk) | ROC-AUC | Inference Latency (ms/sample) | Training Time (ms) | Explainability / Interpretability | Production Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Production)** | **86.48%** | 0.7278 | **0.8200** | **0.7712** | **0.9182** | 0.030 ms | 551 ms | **High** (MDI / Gini Importance) | ⭐ **Selected for Production** |
| **XGBoost (Gradient Boosted Trees)** | **87.96%** | 0.8244 | 0.7200 | 0.7687 | 0.9172 | 0.004 ms | 2,592 ms | High (Gain / SHAP analysis) | Excellent alternative; slightly lower recall |
| **Support Vector Machine (RBF Kernel)** | **88.70%** | **0.8560** | 0.7133 | **0.7782** | 0.8839 | 0.035 ms | 305 ms | Low (Black-box kernel space) | High precision, but lower ROC-AUC |
| **Gradient Boosting (GBM)** | 86.85% | 0.7801 | 0.7333 | 0.7560 | 0.9176 | 0.003 ms | 1,237 ms | Medium-High | Strong performer, longer training time |
| **k-Nearest Neighbors ($k=7$)** | 87.59% | 0.8168 | 0.7133 | 0.7616 | 0.9035 | 0.104 ms | **0.5 ms** | Medium (Distance metric) | Memory-heavy ($O(N)$ inference cost) |
| **Logistic Regression (Linear Baseline)** | 84.81% | 0.6771 | **0.8667** | 0.7602 | 0.9144 | **<0.001 ms** | **9.6 ms** | **Maximum** (Linear Odds Weights) | Fast baseline; high false-positive buzzer rate |
| **Decision Tree (CART)** | 85.74% | 0.8067 | 0.6400 | 0.7138 | 0.8725 | 0.001 ms | 11.8 ms | Maximum (Direct decision rules) | Prone to variance on sensor jitter |
| **Multi-Layer Perceptron (MLP Neural Net)** | 84.26% | 0.7273 | 0.6933 | 0.7099 | 0.8620 | 0.003 ms | 4,464 ms | Low (Hidden layer activations) | Overparameterized for tabular telemetry |

---

## 2. In-Depth Architectural Analysis & Trade-Offs

### 1. Random Forest Classifier (Selected Production Architecture)
- **Hyperparameters**: `n_estimators=120`, `max_depth=9`, `min_samples_split=12`, `min_samples_leaf=6`, `class_weight='balanced'`.
- **Why It Won Production Selection**:
  1. **Highest ROC-AUC (0.9182)**: Demonstrates the strongest separability across varying classification thresholds.
  2. **Superior High-Risk Recall (82.00%)**: In clinical ergonomics and visual health monitoring, **false negatives are hazardous** (failing to alert a user experiencing severe ciliary muscle spasm or extreme blink suppression). Random Forest captured **123 out of 150 true strain episodes** in the unseen test cohort.
  3. **Robustness to Sensor Jitter**: Ensemble bagging averages out high-frequency noise from the HC-SR04 ultrasonic distance sensor and ambient BH1750 lux fluctuations.
  4. **Native Feature Explainability**: Directly ranks features by Mean Decrease in Impurity (MDI), enabling transparent auditability for medical review.
  5. **Ultra-Low Latency**: Inference executes in **0.030 ms per sample**, allowing the backend server to process hundreds of concurrent workstation streams effortlessly.

```
Confusion Matrix (Random Forest Test Partition):
                       Predicted Low Risk (0)    Predicted High Risk (1)
  Actual Low Risk (0)           344                       46
  Actual High Risk (1)           27                      123
```

---

### 2. XGBoost (Extreme Gradient Boosting)
- **Hyperparameters**: `n_estimators=100`, `max_depth=4`, `learning_rate=0.08`, `eval_metric='logloss'`.
- **Performance**:
  - Test Accuracy: **87.96%** | High-Risk Precision: **82.44%** | Recall: **72.00%**.
- **Assessment**:
  - Achieved the highest overall accuracy and precision among tree models.
  - However, it exhibited a **10% drop in high-risk recall** compared to Random Forest (missing 42 strain events vs 27 for RF). For a health intervention system, missing strain episodes is less desirable than occasional conservative alerts.

---

### 3. Support Vector Machine (RBF Kernel)
- **Hyperparameters**: `C=1.0`, `kernel='rbf'`, `gamma='scale'`.
- **Performance**:
  - Test Accuracy: **88.70%** | High-Risk Precision: **85.60%** | Recall: **71.33%** | ROC-AUC: **0.8839**.
- **Assessment**:
  - SVM yielded the highest raw precision (85.60%), minimizing false alarms.
  - **Drawbacks**: Probability calibration via Platt scaling (`probability=True`) added significant training overhead, and ROC-AUC (0.8839) was notably lower than the tree ensemble models. Black-box decision boundaries make it harder to provide users with transparent explanations for why an alert fired.

---

### 4. Logistic Regression (Baseline Model)
- **Hyperparameters**: `max_iter=1000`, `C=1.0`, `class_weight='balanced'`.
- **Performance**:
  - Test Accuracy: **84.81%** | Recall: **86.67%** | Precision: **67.71%** | ROC-AUC: **0.9144**.
- **Assessment**:
  - Excellent linear baseline with sub-millisecond execution (9.6 ms training).
  - High recall (86.67%), but precision was the lowest among all candidates (67.71%). This leads to excessive false-positive buzzer alerts on the ESP32, triggering alert fatigue in users.

---

### 5. Multi-Layer Perceptron (MLP Neural Network)
- **Architecture**: Input (18) $\to$ Dense (64, ReLU) $\to$ Dropout (0.2) $\to$ Dense (32, ReLU) $\to$ Output (1, Sigmoid).
- **Performance**:
  - Test Accuracy: **84.26%** | ROC-AUC: **0.8620** | Training Time: **4,464 ms**.
- **Assessment**:
  - Deep learning underperformed gradient boosting and bagging on this tabular telemetry schema. The model was prone to overfitting on sensor edge cases despite regularization, confirming biomedical literature findings that tree ensembles dominate tabular biometric signals.

---

## 3. Clinical Feature Importance Analysis (Random Forest)

The Random Forest architecture reveals which physical and ocular cues drive Digital Eye Strain risk:

| Rank | Feature Name | MDI Importance | Clinical Significance |
| :---: | :--- | :---: | :--- |
| **1** | `d_avg` | **0.2019** | Average viewing distance (primary physical determinant of ciliary convergence). |
| **2** | `d_avg_dev` | **0.1489** | Deviation from individual's ergonomic baseline ($\Delta d$). |
| **3** | `d_min_dev` | **0.1481** | Sudden forward slouching / screen proximity spikes. |
| **4** | `d_min` | **0.1344** | Closest distance threshold reached during session. |
| **5** | `blink_rate` | **0.0728** | Absolute blink frequency (blinks/min; drops below 6 during strain). |
| **6** | `incomplete_blink_ratio` | **0.0385** | Partial blinks failing to re-coat the cornea with tear lipids. |
| **7** | `head_pose_dev` | **0.0371** | Asymmetrical neck tilt / cervical angle deviation. |
| **8** | `blink_rate_dev` | **0.0357** | Individualized drop relative to personal resting blink rate. |
| **9** | `incomplete_blink_ratio_dev` | **0.0312** | Rise in incomplete blink proportion above user baseline. |
| **10** | `head_pose` | **0.0297** | Absolute lateral head rotation angle. |

---

## 4. Evaluation Methodology: Zero-Leakage Validation

```
All 2,700 Telemetry Windows (60 Distinct Human Users)
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
Training Partition (80%)        Testing Partition (20%)
  48 Unique Users                 12 Unique Users
  2,160 Telemetry Windows         540 Telemetry Windows
  [User Leakage: 0%]              [User Leakage: 0%]
```

In physiological machine learning, standard random train/test splits artificially inflate metrics because the model memorizes an individual's unique resting parameters. By grouping strictly by `user_id`, the reported **86.48% accuracy and 0.9182 ROC-AUC represent true generalization performance on unseen human subjects**.

