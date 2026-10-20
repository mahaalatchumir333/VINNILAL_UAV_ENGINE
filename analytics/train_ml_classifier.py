"""
train_ml_classifier.py
-------------------------
Trains the optional Random Forest fault classifier (Phase 2 upgrade path
mentioned in fault_classifier.py's `train_ml_classifier` stub) on the
real labelled datasets from simulator/generate_datasets.py.

This is an OPTIONAL upgrade over the rule-based classifier. The rule-based
classifier remains the default because it is transparent and easy to
validate; this script exists so the upgrade path is no longer just a
docstring -- it's a trained, evaluated model you can compare against it.

Run:
    cd analytics
    python3 train_ml_classifier.py
Output:
    ml_fault_classifier.joblib  (trained model + label encoder)
    Prints a held-out accuracy report so you can compare against the
    rule-based classifier's evaluate_metrics.py numbers honestly.
"""

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
for sub in ["simulator", "digital_twin", "analytics"]:
    sys.path.append(str(ROOT / sub))

from residual_engine import ResidualEngine
from physics_model import DigitalTwin
from feature_engineering import FeatureWindow, extract_features, FEATURE_ORDER, features_to_vector

DATA_DIR = ROOT / "data"
MODEL_PATH = Path(__file__).resolve().parent / "ml_fault_classifier.joblib"

# ground-truth label per dataset (matches evaluate_metrics.py's mapping)
DATASET_LABELS = {
    "01_normal_cruise": "none",
    "02_high_altitude": "none",
    "03_hot_weather_endurance": "none",
    "04_rapid_throttle_transition": "none",
    "05_injector_degradation": "injector_degradation",
    "06_misfire": "misfire",
    "07_lubrication_degradation": "lubrication_degradation",
    "08_cooling_degradation": "cooling_degradation",
    "09_vibration_increase": "vibration_increase",
    "10_egt_sensor_drift": "egt_sensor_drift",
    # 11 and 12 are dropout/network scenarios, not fault-label scenarios --
    # excluded from classifier training (they test resilience, not diagnosis)
}

# only train on windows AFTER a fault has actually ramped in, using the
# per-row ground truth so we don't teach the model "not yet faulty" rows
# as if they were positive examples of that fault
FAULT_START_TICK = {
    "05_injector_degradation": 250,
    "06_misfire": 200,
    "07_lubrication_degradation": 250,
    "08_cooling_degradation": 300,
    "09_vibration_increase": 250,
    "10_egt_sensor_drift": 350,
}


def build_training_table():
    X_rows, y_labels = [], []

    for name, label in DATASET_LABELS.items():
        path = DATA_DIR / f"{name}.csv"
        if not path.exists():
            print(f"WARNING: {path} not found -- run simulator/generate_datasets.py first")
            continue

        rows = list(csv.DictReader(open(path)))
        residual_engine = ResidualEngine(twin=DigitalTwin())
        window = FeatureWindow(maxlen=30)
        start_tick = FAULT_START_TICK.get(name, 0)

        for row in rows:
            t = float(row["t"])
            telemetry = {
                "throttle": float(row["throttle"]), "altitude": float(row["altitude"]),
                "ambient_temperature": float(row["ambient_temperature"]),
                "rpm": float(row["rpm"]), "fuel_flow": float(row["fuel_flow"]),
                "egt": float(row["egt"]), "cht": float(row["cht"]),
                "oil_temperature": float(row["oil_temperature"]),
                "oil_pressure": float(row["oil_pressure"]),
                "vibration": float(row["vibration"]),
            }
            res = residual_engine.compute(telemetry)
            window.push(t, telemetry, res.normalized)
            if not window.is_ready():
                continue

            features = extract_features(window)

            if label == "none":
                X_rows.append(features_to_vector(features))
                y_labels.append("none")
            elif t >= start_tick:
                # only the mature-fault portion is a clean positive example
                X_rows.append(features_to_vector(features))
                y_labels.append(label)
            # rows between t=0 and start_tick for a fault dataset are
            # SKIPPED (ambiguous: fault not yet manifest), not labelled
            # "none" -- mislabelling them would teach the model that early
            # symptoms look like normal operation.

    return np.array(X_rows), np.array(y_labels)


def train_and_evaluate():
    import joblib

    X, y = build_training_table()
    print(f"Training table: {X.shape[0]} rows, {X.shape[1]} features, "
          f"{len(set(y))} classes: {sorted(set(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced", random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n" + "=" * 60)
    print("HELD-OUT TEST SET PERFORMANCE (25% split, not seen in training)")
    print("=" * 60)
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion matrix (rows=true, cols=predicted):")
    labels = sorted(set(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print(f"{'':25s}" + "".join(f"{l[:10]:>12s}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l:25s}" + "".join(f"{v:>12d}" for v in cm[i]))

    # feature importances -- useful for explainability / evidence generation
    print("\nTop feature importances:")
    importances = sorted(zip(FEATURE_ORDER, clf.feature_importances_),
                          key=lambda x: -x[1])
    for name, imp in importances[:6]:
        print(f"  {name:<24s} {imp:.3f}")

    joblib.dump({"model": clf, "feature_order": FEATURE_ORDER}, MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_evaluate()
