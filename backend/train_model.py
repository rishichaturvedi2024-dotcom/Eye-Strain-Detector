"""
Eye Strain Risk Model Training Script
======================================
Inspired by Kaggle: "Dry Eye Disease Prediction" by madhuraatmarambhagat
(https://www.kaggle.com/code/madhuraatmarambhagat/dry-eye-disease-prediction/notebook)

This script implements a multi-model comparison pipeline using:
  1. Logistic Regression (baseline)
  2. Random Forest Classifier
  3. XGBoost Classifier

It uses our medically-grounded synthetic dataset and picks the best
performing model to save as the production model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# Try to import XGBoost; install if missing
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[!] XGBoost not installed. Run: python -m pip install xgboost")
    print("    Skipping XGBoost model. Using Logistic Regression + Random Forest only.\n")


# ============================================================
# 1. REALISTIC SYNTHETIC DATA GENERATION
# ============================================================

def generate_realistic_dataset(num_users=50, sessions_per_user=40):
    """
    Generates a medically-grounded synthetic dataset for digital eye strain
    risk prediction.

    Medical references used:
    - Normal spontaneous blink rate: 15-20 blinks/min (Bentivoglio et al., 1997)
    - Blink rate drops to ~3-4/min during concentrated screen use (Tsubota & Nakamori, 1993)
    - Recommended screen distance: 50-70 cm (American Academy of Ophthalmology)
    - Ideal office lighting: 300-500 lux (OSHA / ISO 8995)
    - 20-20-20 rule: every 20 min, look 20 feet away for 20 seconds
    - Incomplete blinks increase with screen time (McMonnies, 2007)

    The dataset simulates 5 session types per user:
      1. Comfortable (low risk) - good posture, good lighting, healthy blinks
      2. Dim Room (moderate risk) - low ambient light
      3. Too Close (moderate risk) - leaning into the screen
      4. Prolonged Fatigue (high risk) - blink rate degrades over long session
      5. Combined Strain (high risk) - close + dim + low blink rate
    """
    np.random.seed(42)

    session_types = ['comfortable', 'dim_room', 'too_close', 'prolonged_fatigue', 'combined_strain']
    session_weights = [0.35, 0.15, 0.15, 0.20, 0.15]

    data = []

    for user_id in range(num_users):
        # Each user has a personal baseline (varies person to person)
        user_baseline_distance = np.random.uniform(50, 70)       # cm
        user_baseline_light = np.random.uniform(300, 500)         # lux
        user_baseline_blink = np.random.uniform(15, 20)           # blinks/min

        for _ in range(sessions_per_user):
            session_type = np.random.choice(session_types, p=session_weights)

            if session_type == 'comfortable':
                d_avg = np.random.normal(user_baseline_distance, 3)
                l_avg = np.random.normal(user_baseline_light, 30)
                br_avg = np.random.normal(user_baseline_blink, 1.5)
                risk_label = 0

            elif session_type == 'dim_room':
                d_avg = np.random.normal(user_baseline_distance - 5, 5)
                l_avg = np.random.normal(50, 25)
                br_avg = np.random.normal(user_baseline_blink - 3, 2)
                risk_label = 1 if np.random.random() < 0.60 else 0

            elif session_type == 'too_close':
                d_avg = np.random.normal(25, 5)
                l_avg = np.random.normal(user_baseline_light - 50, 40)
                br_avg = np.random.normal(user_baseline_blink - 4, 2)
                risk_label = 1 if np.random.random() < 0.65 else 0

            elif session_type == 'prolonged_fatigue':
                d_avg = np.random.normal(user_baseline_distance - 10, 5)
                l_avg = np.random.normal(user_baseline_light - 80, 50)
                br_avg = np.random.normal(5, 2)
                risk_label = 1 if np.random.random() < 0.85 else 0

            elif session_type == 'combined_strain':
                d_avg = np.random.normal(20, 4)
                l_avg = np.random.normal(30, 15)
                br_avg = np.random.normal(4, 1.5)
                risk_label = 1 if np.random.random() < 0.92 else 0

            # Realistic bounds
            d_avg = max(10, min(100, d_avg))
            l_avg = max(5, min(1000, l_avg))
            br_avg = max(1, min(30, br_avg))

            # Deviations from personal baseline
            d_dev = d_avg - user_baseline_distance
            l_dev = l_avg - user_baseline_light
            br_dev = br_avg - user_baseline_blink

            data.append([
                round(d_avg, 2), round(l_avg, 2), round(br_avg, 2),
                round(d_dev, 2), round(l_dev, 2), round(br_dev, 2),
                risk_label
            ])

    df = pd.DataFrame(data, columns=[
        'd_avg', 'l_avg', 'blink_rate',
        'd_dev', 'l_dev', 'br_dev',
        'risk_label'
    ])
    return df


# ============================================================
# 2. DATA PREPROCESSING (following Kaggle notebook approach)
# ============================================================

def preprocess_data(df):
    """
    Preprocessing pipeline inspired by the Kaggle Dry Eye Disease notebook:
    - Feature/target separation
    - Train/test split (80/20)
    - StandardScaler normalization on features
    """
    X = df[['d_avg', 'l_avg', 'blink_rate', 'd_dev', 'l_dev', 'br_dev']]
    y = df['risk_label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # StandardScaler (as used in the Kaggle notebook)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrames for readability
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    return X_train_scaled, X_test_scaled, y_train.reset_index(drop=True), y_test.reset_index(drop=True), scaler


# ============================================================
# 3. MODEL TRAINING & EVALUATION (Kaggle multi-model approach)
# ============================================================

def evaluate_model(name, model, X_test, y_test):
    """Evaluate a single model and return metrics dict."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk'])

    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  Accuracy : {acc * 100:.2f}%")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Low   High")
    print(f"  Actual Low  [{cm[0][0]:4d}  {cm[0][1]:4d}]")
    print(f"  Actual High [{cm[1][0]:4d}  {cm[1][1]:4d}]")
    print(f"\n  Classification Report:")
    print(report)

    return {'name': name, 'model': model, 'accuracy': acc, 'auc': auc}


def train_all_models(X_train, X_test, y_train, y_test):
    """
    Train and compare multiple models (Kaggle notebook approach):
      1. Logistic Regression
      2. Random Forest
      3. XGBoost (if available)
    """
    results = []

    # --- Model 1: Logistic Regression (baseline) ---
    print("\n[*] Training Logistic Regression...")
    lr = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        class_weight='balanced',
        random_state=42
    )
    lr.fit(X_train, y_train)
    results.append(evaluate_model("Logistic Regression", lr, X_test, y_test))

    # --- Model 2: Random Forest ---
    print("[*] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    )
    rf.fit(X_train, y_train)
    results.append(evaluate_model("Random Forest", rf, X_test, y_test))

    # --- Model 3: XGBoost ---
    if HAS_XGBOOST:
        print("[*] Training XGBoost...")
        # Calculate scale_pos_weight for class imbalance
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        xgb = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        xgb.fit(X_train, y_train)
        results.append(evaluate_model("XGBoost", xgb, X_test, y_test))

    return results


# ============================================================
# 4. MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  Eye Strain Risk Model - Multi-Model Training Pipeline")
    print("  (Inspired by Kaggle Dry Eye Disease Prediction Notebook)")
    print("=" * 60)

    # --- Step 1: Generate or load dataset ---
    csv_path = os.path.join(os.path.dirname(__file__), 'realistic_training_data.csv')
    if os.path.exists(csv_path):
        print(f"\n[1/5] Loading existing dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("\n[1/5] Generating realistic dataset (50 users x 40 sessions = 2000 samples)...")
        df = generate_realistic_dataset(num_users=50, sessions_per_user=40)
        df.to_csv(csv_path, index=False)
        print(f"       Dataset saved to: {csv_path}")

    # Dataset summary
    print(f"\n       Dataset Summary:")
    print(f"       Total samples:    {len(df)}")
    print(f"       Low risk (0):     {len(df[df['risk_label'] == 0])} ({len(df[df['risk_label'] == 0]) / len(df) * 100:.1f}%)")
    print(f"       High risk (1):    {len(df[df['risk_label'] == 1])} ({len(df[df['risk_label'] == 1]) / len(df) * 100:.1f}%)")
    print(f"\n       Feature Ranges:")
    for col in ['d_avg', 'l_avg', 'blink_rate']:
        print(f"         {col:12s}: {df[col].min():.1f} - {df[col].max():.1f}  (mean: {df[col].mean():.1f})")

    # --- Step 2: Preprocess (StandardScaler) ---
    print("\n[2/5] Preprocessing: Train/Test Split + StandardScaler normalization...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    print(f"       Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # --- Step 3: Train all models ---
    print("\n[3/5] Training and evaluating models...")
    results = train_all_models(X_train, X_test, y_train, y_test)

    # --- Step 4: Compare and select best model ---
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<25s} {'Accuracy':>10s} {'ROC-AUC':>10s}")
    print(f"  {'-' * 45}")
    for r in sorted(results, key=lambda x: -x['auc']):
        marker = " <-- BEST" if r == max(results, key=lambda x: x['auc']) else ""
        print(f"  {r['name']:<25s} {r['accuracy'] * 100:>9.2f}% {r['auc']:>9.4f}{marker}")

    best = max(results, key=lambda x: x['auc'])
    print(f"\n  >> Best model: {best['name']} (AUC: {best['auc']:.4f})")

    # --- Step 5: Save the best model + scaler ---
    model_path = os.path.join(os.path.dirname(__file__), 'rf_risk_model.joblib')
    scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.joblib')

    joblib.dump(best['model'], model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n[5/5] Saved artifacts:")
    print(f"       Model:  {model_path}")
    print(f"       Scaler: {scaler_path}")

    # Feature importances (for tree-based models)
    if hasattr(best['model'], 'feature_importances_'):
        importances = best['model'].feature_importances_
        feature_names = ['d_avg', 'l_avg', 'blink_rate', 'd_dev', 'l_dev', 'br_dev']
        print("\n       Feature Importances:")
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
            bar = '#' * int(imp * 40)
            print(f"         {name:12s}: {imp:.4f} {bar}")

    print("\n" + "=" * 60)
    print("  Training complete! Model ready for deployment.")
    print("=" * 60)


if __name__ == "__main__":
    main()
