"""
evaluate_ml_classifier_generalization.py
-------------------------------------------
The train/test split inside train_ml_classifier.py uses random row
sampling from the SAME 12 simulation runs the model trains on. Because
adjacent rows in a time series are highly correlated (nearly identical
sensor values a second apart), that split is optimistic -- it reports
how well the model memorizes these specific runs, not how well it
generalizes to a new mission.

This script is the honest test: it generates BRAND NEW simulation runs
(different random seeds, never seen during training) for each fault type
and reports real held-out accuracy against them.

Run:
    cd analytics
    python3 train_ml_classifier.py                        # train first
    python3 evaluate_ml_classifier_generalization.py       # then this
"""

import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for sub in ["simulator", "digital_twin", "analytics"]:
    sys.path.append(str(ROOT / sub))

from engine_simulator import EngineSimulator
from mission_profiles import make_endurance_mission
from fault_injection import (
    normal_scenario, injector_degradation_scenario, overheating_scenario,
    lubrication_loss_scenario, misfire_scenario, vibration_increase_scenario,
    egt_sensor_drift_scenario,
)
from residual_engine import ResidualEngine
from physics_model import DigitalTwin
from feature_engineering import FeatureWindow, extract_features, features_to_vector

MODEL_PATH = Path(__file__).resolve().parent / "ml_fault_classifier.joblib"

# NEW seeds (9xx range), never used in generate_datasets.py (1xx range)
# or train_ml_classifier.py's training data -- genuinely unseen runs
HOLDOUT_SCENARIOS = {
    "injector_degradation": (injector_degradation_scenario(start_s=180, ramp_s=350), 901, 250),
    "misfire": (misfire_scenario(start_s=120, ramp_s=250), 902, 200),
    "lubrication_degradation": (lubrication_loss_scenario(start_s=220, ramp_s=380), 903, 250),
    "cooling_degradation": (overheating_scenario(start_s=250, ramp_s=450), 904, 300),
    "vibration_increase": (vibration_increase_scenario(start_s=190, ramp_s=350), 905, 250),
    "egt_sensor_drift": (egt_sensor_drift_scenario(start_s=300, ramp_s=450), 906, 350),
    "none": (normal_scenario(), 907, 0),
}


def evaluate_generalization():
    if not MODEL_PATH.exists():
        print(f"No trained model at {MODEL_PATH} -- run train_ml_classifier.py first.")
        return

    bundle = joblib.load(MODEL_PATH)
    clf = bundle["model"]

    correct = 0
    total = 0
    per_class_correct = {}
    per_class_total = {}

    print("=" * 70)
    print("HONEST GENERALIZATION TEST (new seeds, never seen in training)")
    print("=" * 70)

    for true_label, (fault_state, seed, start_tick) in HOLDOUT_SCENARIOS.items():
        sim = EngineSimulator(seed=seed, fault_state=fault_state)
        mission = make_endurance_mission()
        residual_engine = ResidualEngine(twin=DigitalTwin())
        window = FeatureWindow(maxlen=30)

        preds = []
        for i in range(900):
            pkt = sim.step(mission, dt=1.0)
            res = residual_engine.compute(pkt)
            window.push(i, pkt, res.normalized)
            if not window.is_ready():
                continue
            if true_label != "none" and i < start_tick:
                continue  # skip ambiguous pre-fault rows, same as training
            features = extract_features(window)
            x = features_to_vector(features).reshape(1, -1)
            pred = clf.predict(x)[0]
            preds.append(pred)

        n_correct = sum(1 for p in preds if p == true_label)
        n_total = len(preds)
        correct += n_correct
        total += n_total
        per_class_correct[true_label] = n_correct
        per_class_total[true_label] = n_total

        # most common WRONG prediction, for diagnosis
        wrong = [p for p in preds if p != true_label]
        most_common_wrong = max(set(wrong), key=wrong.count) if wrong else "-"

        print(f"{true_label:<24s} accuracy={n_correct/n_total:.2%}  "
              f"({n_correct}/{n_total})  most common wrong pred: {most_common_wrong}")

    print("=" * 70)
    print(f"OVERALL held-out accuracy on unseen simulation runs: {correct/total:.2%} ({correct}/{total})")
    print("(Compare this to the misleadingly perfect 1.00 from the random-split")
    print(" test in train_ml_classifier.py -- THIS number is the honest one.)")


if __name__ == "__main__":
    evaluate_generalization()
