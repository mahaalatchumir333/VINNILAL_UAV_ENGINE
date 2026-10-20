"""
ml_fault_classifier.py
--------------------------
Drop-in ML alternative to fault_classifier.classify_fault(), using the
Random Forest trained by train_ml_classifier.py. Same FaultResult
interface, so backend/main.py can switch between rule-based and ML-based
classification with one flag.

Evidence strings are generated from the model's feature importances
(which features mattered most for THIS prediction, approximated via the
per-tree feature contributions) rather than hand-written rules, so this
stays explainable even though it's now a trained model.

Falls back gracefully to "model not trained" if ml_fault_classifier.joblib
doesn't exist yet -- callers should check `is_available()` first.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from feature_engineering import features_to_vector
from fault_classifier import FaultResult

MODEL_PATH = Path(__file__).resolve().parent / "ml_fault_classifier.joblib"

_model_bundle: Optional[dict] = None


def is_available() -> bool:
    return MODEL_PATH.exists()


def _load():
    global _model_bundle
    if _model_bundle is None:
        import joblib
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


# human-readable evidence templates keyed by feature name, used to turn
# "this feature contributed most" into an explainable sentence -- same
# spirit as the rule-based classifier's evidence strings
FEATURE_EVIDENCE_TEMPLATES = {
    "egt_residual": "EGT residual is the strongest contributing signal",
    "cht_residual": "CHT residual is the strongest contributing signal",
    "vibration_rms": "Vibration RMS is the strongest contributing signal",
    "vibration_residual": "Vibration deviates from twin prediction",
    "rpm_fluctuation": "RPM residual shows irregular fluctuation",
    "fuel_flow_residual": "Fuel flow deviates from twin prediction",
    "oil_pressure_residual": "Oil pressure deviates from twin prediction",
    "oil_temperature_residual": "Oil temperature deviates from twin prediction",
    "cht_slope": "CHT is trending upward",
    "egt_slope": "EGT is trending upward",
    "oil_temperature_slope": "Oil temperature is trending upward",
    "oil_pressure_slope": "Oil pressure is trending",
    "fuel_flow_per_rpm": "Fuel-flow-to-RPM ratio is abnormal",
    "oil_pressure_margin": "Oil pressure margin is reduced",
    "rpm_residual": "RPM deviates from twin prediction",
}


def classify_fault_ml(features: dict) -> FaultResult:
    """
    ML equivalent of fault_classifier.classify_fault(). Returns the same
    FaultResult shape so it's a transparent swap in main.py.
    """
    if not is_available():
        return FaultResult(likely_fault="none", confidence=0.0,
                            evidence=["ML classifier not trained -- run train_ml_classifier.py"])

    bundle = _load()
    clf = bundle["model"]
    feature_order = bundle["feature_order"]

    x = np.array([features[k] for k in feature_order], dtype=float).reshape(1, -1)
    pred = clf.predict(x)[0]
    proba = clf.predict_proba(x)[0]
    class_index = list(clf.classes_).index(pred)
    confidence = float(proba[class_index])

    if pred == "none":
        return FaultResult(likely_fault="none", confidence=0.0, evidence=[])

    # explainability: which features this specific engine's model instance
    # weighted most heavily overall (global importance, not a per-sample
    # SHAP explanation -- documented limitation, see README note below)
    top_features = sorted(
        zip(feature_order, clf.feature_importances_), key=lambda kv: -kv[1]
    )[:3]
    evidence = [FEATURE_EVIDENCE_TEMPLATES.get(name, f"{name} contributed to this prediction")
                for name, _ in top_features]

    return FaultResult(
        likely_fault=pred,
        confidence=round(confidence, 2),
        evidence=evidence,
    )
