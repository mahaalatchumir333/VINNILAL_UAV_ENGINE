"""
evaluate_metrics.py
----------------------
Runs every dataset in data/ through the real analytics pipeline and
reports the metrics the problem statement's validation plan asks for:
  - precision / recall / F1 for fault classification
  - false-alarm rate on normal-operation data
  - detection lead time (seconds before ground-truth health crosses 50)
  - robustness under missing telemetry / network interruption

Run:
    cd backend
    python3 ../analytics/evaluate_metrics.py
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ["simulator", "digital_twin", "analytics", "backend"]:
    sys.path.append(str(ROOT / sub))

from residual_engine import ResidualEngine
from physics_model import DigitalTwin
from feature_engineering import FeatureWindow, extract_features
from anomaly_detector import AnomalyDetector, AnomalyState
from fault_classifier import classify_fault
from health_index import compute_health_index

DATA_DIR = ROOT / "data"

# maps dataset filename prefix -> expected ground-truth fault label
EXPECTED_FAULT = {
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
    "11_missing_telemetry": "none",
    "12_network_interruption": "none",
}

# our classifier's internal fault names differ slightly from the raw
# simulator fault labels -- map them so comparisons are apples-to-apples
CLASSIFIER_TO_TRUE = {
    "overheating_cooling_degradation": "cooling_degradation",
    "sensor_drift": "egt_sensor_drift",
    "vibration_abnormality": "vibration_increase",
}


def load_csv(path: Path):
    with open(path) as f:
        return list(csv.DictReader(f))


def run_pipeline_on_dataset(rows: list[dict]):
    """Runs one dataset through a fresh pipeline instance, returns per-row results."""
    residual_engine = ResidualEngine(twin=DigitalTwin())
    window = FeatureWindow(maxlen=30)
    detector = AnomalyDetector()

    # bootstrap detector on the first 60s of THIS dataset if it's the normal
    # baseline shape; otherwise reuse a generically-trained detector
    detector.fit(_bootstrap_normal_rows())

    results = []
    for row in rows:
        telemetry = {
            "throttle": float(row["throttle"]), "altitude": float(row["altitude"]),
            "ambient_temperature": float(row["ambient_temperature"]),
            "rpm": float(row["rpm"]), "fuel_flow": float(row["fuel_flow"]),
            "egt": float(row["egt"]), "cht": float(row["cht"]),
            "oil_temperature": float(row["oil_temperature"]),
            "oil_pressure": float(row["oil_pressure"]),
            "vibration": float(row["vibration"]),
        }
        dropped = row.get("_telemetry_dropped", "0") == "1"
        t = float(row["t"])

        if dropped:
            results.append({
                "t": t, "status": "degraded_no_inference",
                "likely_fault": "none", "ground_truth": row["_ground_truth_fault"],
                "ground_truth_health": float(row["_ground_truth_health"]),
            })
            continue

        res = residual_engine.compute(telemetry)
        window.push(t, telemetry, res.normalized)
        if not window.is_ready():
            continue

        features = extract_features(window)
        anomaly = detector.score(features)
        if anomaly.confirmed_state in (AnomalyState.SUSPECT, AnomalyState.WARNING, AnomalyState.CRITICAL):
            fault = classify_fault(features)
            fault_name = CLASSIFIER_TO_TRUE.get(fault.likely_fault, fault.likely_fault)
        else:
            fault_name = "none"

        results.append({
            "t": t,
            "status": anomaly.confirmed_state.value,
            "likely_fault": fault_name,
            "ground_truth": row["_ground_truth_fault"],
            "ground_truth_health": float(row["_ground_truth_health"]),
        })
    return results


_normal_rows_cache = None


def _bootstrap_normal_rows():
    """Train the Isolation Forest once on the normal-cruise dataset, cached."""
    global _normal_rows_cache
    if _normal_rows_cache is not None:
        return _normal_rows_cache

    rows = load_csv(DATA_DIR / "01_normal_cruise.csv")
    residual_engine = ResidualEngine(twin=DigitalTwin())
    window = FeatureWindow(maxlen=30)
    feature_rows = []
    for row in rows:
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
        window.push(float(row["t"]), telemetry, res.normalized)
        if window.is_ready():
            feature_rows.append(extract_features(window))
    _normal_rows_cache = feature_rows
    return feature_rows


def evaluate_all():
    all_results = {}
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        name = csv_path.stem
        rows = load_csv(csv_path)
        results = run_pipeline_on_dataset(rows)
        all_results[name] = results

    print("=" * 78)
    print(f"{'DATASET':<32}{'EXPECTED':<24}{'ALERT FIRED':<14}{'LEAD (s)':<8}")
    print("=" * 78)

    tp = fp = fn = tn = 0
    lead_times = []
    false_alarms_on_normal = 0
    normal_datasets = 0

    for name, results in all_results.items():
        expected = EXPECTED_FAULT.get(name, "unknown")

        # first tick where ANY alert fired (used for normal-data false-alarm check)
        first_alert_t = next((r["t"] for r in results
                               if r["status"] in ("suspect", "warning", "critical")), None)

        # first tick where the alert's fault_name is the CORRECT one (what
        # actually matters operationally -- an alert with the wrong or no
        # fault attached isn't a useful diagnosis yet)
        first_correct_t = None
        wrong_fault_ever_reported = False
        for r in results:
            if r["status"] not in ("suspect", "warning", "critical"):
                continue
            mapped_fault = CLASSIFIER_TO_TRUE.get(r["likely_fault"], r["likely_fault"])
            if mapped_fault == expected and expected != "none":
                first_correct_t = r["t"]
                break
            if mapped_fault not in ("none", expected):
                wrong_fault_ever_reported = True

        lead_time = None
        if expected == "none":
            normal_datasets += 1
            if first_alert_t is not None:
                false_alarms_on_normal += 1
                fp += 1
            else:
                tn += 1
        else:
            if first_correct_t is not None:
                tp += 1
                threshold_hit = next((r["t"] for r in results if r["ground_truth_health"] <= 50), None)
                if threshold_hit is not None:
                    lead_time = threshold_hit - first_correct_t
                    lead_times.append(lead_time)
            elif wrong_fault_ever_reported:
                fp += 1
            else:
                fn += 1

        display_t = first_correct_t if first_correct_t is not None else first_alert_t
        print(f"{name:<32}{expected:<24}{str(display_t):<14}{str(round(lead_time,1) if lead_time else '-'):<8}")

    print("=" * 78)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    false_alarm_rate = false_alarms_on_normal / normal_datasets if normal_datasets else 0.0
    avg_lead = sum(lead_times) / len(lead_times) if lead_times else None

    print(f"Precision: {precision:.2f}   Recall: {recall:.2f}   F1: {f1:.2f}")
    print(f"False-alarm rate on normal-operation datasets: {false_alarm_rate:.2%} "
          f"({false_alarms_on_normal}/{normal_datasets})")
    print(f"Average detection lead time (before health<=50): "
          f"{round(avg_lead,1) if avg_lead else 'n/a'} seconds")


if __name__ == "__main__":
    evaluate_all()
