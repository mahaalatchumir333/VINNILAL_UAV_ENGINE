"""
fault_classifier.py
----------------------
Maps evidence (features + residuals) to a likely fault type, with
explainable contributing factors. MVP uses a rule-assisted classifier
(transparent, easy to validate); a Random Forest/XGBoost classifier can
be swapped in once enough labelled scenario data exists (see
`train_ml_classifier` below for the upgrade path).
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class FaultResult:
    likely_fault: str
    confidence: float
    evidence: List[str]
    vote_score: float = 0.0
    vote_scores: Dict[str, float] = field(default_factory=dict)
    rule_trace: List[Dict[str, object]] = field(default_factory=list)


# Evidence thresholds -- calibrated against measured feature magnitudes
# from simulator/generate_datasets.py (see analytics/evaluate_metrics.py
# for the measurement harness these were tuned against). Rerun
# evaluate_metrics.py after changing any of these.
EGT_RESIDUAL_THRESHOLD = 1.5      # normalized (sigma units)
CHT_RESIDUAL_THRESHOLD = 3.0      # above observed healthy-scenario peaks
CHT_SLOPE_THRESHOLD = 0.02        # deg C / sec
OIL_PRESSURE_RESIDUAL_THRESHOLD = -1.2
OIL_TEMP_SLOPE_THRESHOLD = 0.015
VIBRATION_RMS_THRESHOLD = 0.30
FUEL_FLOW_RESIDUAL_THRESHOLD = 1.2
RPM_FLUCTUATION_THRESHOLD = 0.8   # std of RPM RESIDUAL (sigma units), not raw RPM
EGT_DRIFT_RESIDUAL_THRESHOLD = 1.5  # lower bar than injector's, since drift is
                                     # the ONLY evidence available (isolated sensor)

FAULT_CODES = {
    "none": "NONE",
    "injector_degradation": "INJECTOR_DEGRADATION",
    "overheating_cooling_degradation": "COOLING_SYSTEM_DEGRADATION",
    "lubrication_degradation": "LUBRICATION_DEGRADATION",
    "misfire": "MISFIRE",
    "sensor_drift": "EGT_SENSOR_DRIFT",
    "vibration_abnormality": "VIBRATION_INCREASE",
}


def classify_fault(features: Dict[str, float]) -> FaultResult:
    """
    Rule-assisted fault classification. Each rule contributes a vote and
    an evidence string; the fault with the highest accumulated vote wins.
    """
    votes: Dict[str, float] = {
        "injector_degradation": 0.0,
        "overheating_cooling_degradation": 0.0,
        "lubrication_degradation": 0.0,
        "misfire": 0.0,
        "sensor_drift": 0.0,
        "vibration_abnormality": 0.0,
    }
    evidence: Dict[str, List[str]] = {k: [] for k in votes}
    rule_trace: List[Dict[str, object]] = []

    def evaluate_rule(
        fault: str,
        rule: str,
        feature: str,
        operator: str,
        threshold: float,
        observed: float,
        vote: float,
        triggered: bool,
        explanation: str,
        conditions: List[Dict[str, object]] | None = None,
    ) -> None:
        if triggered:
            votes[fault] += vote
            evidence[fault].append(explanation)
        rule_trace.append({
            "fault": fault,
            "rule": rule,
            "feature": feature,
            "operator": operator,
            "threshold": threshold,
            "observed": round(float(observed), 4),
            "triggered": triggered,
            "vote": vote if triggered else 0.0,
            "explanation": explanation,
            "conditions": conditions or [{
                "feature": feature,
                "operator": operator,
                "threshold": threshold,
                "observed": round(float(observed), 4),
                "passed": triggered,
            }],
        })

    # Injector degradation: EGT residual up + fuel-flow/RPM mismatch
    evaluate_rule(
        "injector_degradation", "egt_residual_high", "egt_residual", ">",
        EGT_RESIDUAL_THRESHOLD, features["egt_residual"], 1.0,
        features["egt_residual"] > EGT_RESIDUAL_THRESHOLD,
        f"EGT residual elevated ({features['egt_residual']:.2f}\u03c3)",
    )
    evaluate_rule(
        "injector_degradation", "fuel_flow_residual_high", "fuel_flow_residual", ">",
        FUEL_FLOW_RESIDUAL_THRESHOLD, features["fuel_flow_residual"], 1.0,
        features["fuel_flow_residual"] > FUEL_FLOW_RESIDUAL_THRESHOLD,
        "Fuel-flow/RPM mismatch detected",
    )

    # Overheating / cooling degradation: sustained CHT deviation, rising CHT,
    # or a correlated EGT residual.
    evaluate_rule(
        "overheating_cooling_degradation", "cht_residual_high",
        "cht_residual", ">", CHT_RESIDUAL_THRESHOLD,
        features["cht_residual"], 1.0,
        features["cht_residual"] > CHT_RESIDUAL_THRESHOLD,
        f"CHT diverged from the healthy baseline ({features['cht_residual']:.2f}\u03c3)",
    )
    evaluate_rule(
        "overheating_cooling_degradation", "cht_rising", "cht_slope", ">",
        CHT_SLOPE_THRESHOLD, features["cht_slope"], 1.0,
        features["cht_slope"] > CHT_SLOPE_THRESHOLD,
        f"CHT rising ({features['cht_slope']:.3f} \u00b0C/s)",
    )
    evaluate_rule(
        "overheating_cooling_degradation", "egt_residual_trending_up",
        "egt_residual", ">", EGT_RESIDUAL_THRESHOLD * 0.7,
        features["egt_residual"], 0.5,
        features["egt_residual"] > EGT_RESIDUAL_THRESHOLD * 0.7,
        "EGT residual trending up",
    )

    # Lubrication issue: oil pressure down + oil temp up
    evaluate_rule(
        "lubrication_degradation", "oil_pressure_residual_low",
        "oil_pressure_residual", "<", OIL_PRESSURE_RESIDUAL_THRESHOLD,
        features["oil_pressure_residual"], 1.0,
        features["oil_pressure_residual"] < OIL_PRESSURE_RESIDUAL_THRESHOLD,
        f"Oil pressure below expected ({features['oil_pressure_residual']:.2f}\u03c3)",
    )
    evaluate_rule(
        "lubrication_degradation", "oil_temperature_rising",
        "oil_temperature_slope", ">", OIL_TEMP_SLOPE_THRESHOLD,
        features["oil_temperature_slope"], 1.0,
        features["oil_temperature_slope"] > OIL_TEMP_SLOPE_THRESHOLD,
        "Oil temperature rising faster than expected",
    )

    # Misfire: RPM fluctuation + irregular EGT + vibration
    evaluate_rule(
        "misfire", "rpm_fluctuation_high", "rpm_fluctuation", ">",
        RPM_FLUCTUATION_THRESHOLD, features["rpm_fluctuation"], 1.0,
        features["rpm_fluctuation"] > RPM_FLUCTUATION_THRESHOLD,
        f"RPM fluctuation elevated ({features['rpm_fluctuation']:.1f})",
    )
    evaluate_rule(
        "misfire", "vibration_rms_high", "vibration_rms", ">",
        VIBRATION_RMS_THRESHOLD, features["vibration_rms"], 0.5,
        features["vibration_rms"] > VIBRATION_RMS_THRESHOLD,
        "Vibration RMS increasing",
    )

    # Vibration abnormality standalone (imbalance) without other symptoms
    vibration_only = (
        features["vibration_rms"] > VIBRATION_RMS_THRESHOLD
        and features["rpm_fluctuation"] < RPM_FLUCTUATION_THRESHOLD * 0.5
    )
    evaluate_rule(
        "vibration_abnormality", "vibration_high_with_stable_rpm",
        "vibration_rms and rpm_fluctuation", "AND",
        VIBRATION_RMS_THRESHOLD, features["vibration_rms"], 1.0,
        vibration_only,
        "Vibration RMS elevated with stable RPM (possible imbalance)",
        conditions=[
            {
                "feature": "vibration_rms",
                "operator": ">",
                "threshold": VIBRATION_RMS_THRESHOLD,
                "observed": round(features["vibration_rms"], 4),
                "passed": features["vibration_rms"] > VIBRATION_RMS_THRESHOLD,
            },
            {
                "feature": "rpm_fluctuation",
                "operator": "<",
                "threshold": RPM_FLUCTUATION_THRESHOLD * 0.5,
                "observed": round(features["rpm_fluctuation"], 4),
                "passed": features["rpm_fluctuation"] < RPM_FLUCTUATION_THRESHOLD * 0.5,
            },
        ],
    )

    # Sensor drift: a single sensor residual diverges while cross-checks stay normal
    isolated_egt = (
        features["egt_residual"] > EGT_DRIFT_RESIDUAL_THRESHOLD
        and abs(features["cht_slope"]) < CHT_SLOPE_THRESHOLD * 0.3
        and features["fuel_flow_residual"] < FUEL_FLOW_RESIDUAL_THRESHOLD * 0.5
    )
    evaluate_rule(
        "sensor_drift", "isolated_egt_divergence",
        "egt_residual, cht_slope, fuel_flow_residual", "AND",
        EGT_DRIFT_RESIDUAL_THRESHOLD, features["egt_residual"], 1.2,
        isolated_egt,
        "EGT diverges while correlated signals (fuel flow, CHT) remain normal",
        conditions=[
            {
                "feature": "egt_residual",
                "operator": ">",
                "threshold": EGT_DRIFT_RESIDUAL_THRESHOLD,
                "observed": round(features["egt_residual"], 4),
                "passed": features["egt_residual"] > EGT_DRIFT_RESIDUAL_THRESHOLD,
            },
            {
                "feature": "abs(cht_slope)",
                "operator": "<",
                "threshold": CHT_SLOPE_THRESHOLD * 0.3,
                "observed": round(abs(features["cht_slope"]), 4),
                "passed": abs(features["cht_slope"]) < CHT_SLOPE_THRESHOLD * 0.3,
            },
            {
                "feature": "fuel_flow_residual",
                "operator": "<",
                "threshold": FUEL_FLOW_RESIDUAL_THRESHOLD * 0.5,
                "observed": round(features["fuel_flow_residual"], 4),
                "passed": features["fuel_flow_residual"] < FUEL_FLOW_RESIDUAL_THRESHOLD * 0.5,
            },
        ],
    )

    best_fault = max(votes, key=votes.get)
    best_score = votes[best_fault]

    if best_score < 1.0:
        return FaultResult(
            likely_fault="none",
            confidence=0.0,
            evidence=[],
            vote_score=best_score,
            vote_scores=votes,
            rule_trace=rule_trace,
        )

    # simple confidence mapping: normalize vote score to 0-1 range (cap at 3 votes)
    confidence = float(np.clip(best_score / 3.0, 0.1, 0.95))

    return FaultResult(
        likely_fault=best_fault,
        confidence=round(confidence, 2),
        evidence=evidence[best_fault],
        vote_score=best_score,
        vote_scores=votes,
        rule_trace=rule_trace,
    )


# ---------------- Optional ML upgrade path (Phase 2) ----------------

def train_ml_classifier(X: np.ndarray, y: np.ndarray):
    """
    Optional upgrade once enough labelled fault scenarios exist.
    Trains a Random Forest on (feature_vector -> fault_label) pairs
    collected from the simulator's ground-truth labels.
    """
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    clf.fit(X, y)
    return clf
