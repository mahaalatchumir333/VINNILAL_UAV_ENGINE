# ml_analytics/anomaly_detector.py
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple, List

import numpy as np
from sklearn.ensemble import IsolationForest

if __package__:
    from .feature_engineering import features_to_vector
else:
    from feature_engineering import features_to_vector


class AnomalyState(str, Enum):
    NORMAL = "NORMAL"
    SUSPECT = "SUSPECT"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# Initial prototype thresholds (must be calibrated on validation test data)
STATE_THRESHOLDS = {
    AnomalyState.SUSPECT: 0.40,
    AnomalyState.WARNING: 0.70,
    AnomalyState.CRITICAL: 0.90,
}
PERSISTENCE_WINDOWS = 3  # Consecutive cycles required to lock an operational state


@dataclass
class AnomalyResult:
    physics_score: float
    ml_score: float
    combined_score: float
    state: AnomalyState
    confirmed_state: AnomalyState
    hard_alarms: List[str] = field(default_factory=list)


class AnomalyDetector:
    def __init__(self, physics_weight: float = 0.6, ml_weight: float = 0.4):
        # Advanced Hybrid Weights
        self.physics_weight = physics_weight
        self.ml_weight = ml_weight
        self.model: Optional[IsolationForest] = None
        self._recent_states = []  # Time-window persistence buffer
        self._confirmed_state = AnomalyState.NORMAL
        
        # Hard Physical Safety Thresholds Configuration
        self.thresholds = {
            "oil_pressure_min": 2.0,
            "oil_temp_max": 110.0,
            "cht_max": 180.0,
            "vibration_max": 2.5,
        }

    @property
    def effective_weights(self) -> Tuple[float, float]:
        """Return the weights actually used for scoring with the current model state."""
        if self.model is None:
            return 1.0, 0.0
        return self.physics_weight, self.ml_weight

    # ---------- Training Pipeline ----------

    def fit(self, normal_feature_rows: List[Dict[str, float]]):
        """Train the Isolation Forest on healthy normal-operation data arrays only."""
        X = np.array([features_to_vector(f) for f in normal_feature_rows])
        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.02,
            random_state=42,
        )
        self.model.fit(X)

    # ---------- Core Advanced Scoring Engines ----------

    def _physics_score(self, residuals: Dict[str, float]) -> float:
        """
        Calculates a blended physics deviation score using a 70/30 weighting 
        of the worst out-of-bounds parameter against the combined system mean.
        """
        residual_keys = [
            "egt_residual", "cht_residual", "oil_pressure_residual",
            "oil_temperature_residual", "vibration_residual", "fuel_flow_residual",
            "rpm_residual",
        ]
        # Safeguard map in case keys omit suffix strings
        available_residuals = [
            abs(residuals.get(k, residuals.get(k.replace("_residual", ""), 0.0)))
            for k in residual_keys
        ]
        
        residuals_arr = np.array(available_residuals)
        worst = float(np.max(residuals_arr))
        mean = float(np.mean(residuals_arr))
        blended = 0.7 * worst + 0.3 * mean
        
        # Non-linear squash mapping bounded between 0.0 and 1.0
        score = 1 - np.exp(-blended / 2.0)
        return float(np.clip(score, 0, 1))

    def _ml_score(self, features: Dict[str, float]) -> float:
        """Calculates multi-dimensional anomaly variance using Isolation Forest."""
        if self.model is None:
            return 0.0
        try:
            x = features_to_vector(features).reshape(1, -1)
            raw = self.model.decision_function(x)[0]
            score = 1 / (1 + np.exp(raw * 5))  # Sigmoid conversion mapping
            return float(np.clip(score, 0, 1))
        except Exception:
            return 0.0

    def _state_for_score(self, score: float) -> AnomalyState:
        """Maps continuous analytical scoring arrays onto absolute operational states."""
        if score >= STATE_THRESHOLDS[AnomalyState.CRITICAL]:
            return AnomalyState.CRITICAL
        if score >= STATE_THRESHOLDS[AnomalyState.WARNING]:
            return AnomalyState.WARNING
        if score >= STATE_THRESHOLDS[AnomalyState.SUSPECT]:
            return AnomalyState.SUSPECT
        return AnomalyState.NORMAL

    # ---------- Unified Detection & API Layer Interface ----------

    def detect_anomalies(
        self,
        telemetry: Dict[str, float],
        residuals: Dict[str, float],
        score_result: Optional[AnomalyResult] = None,
    ) -> Tuple[bool, List[str], str]:
        """
        Performs dual-layer safety auditing: evaluates immediate physical boundaries 
        and updates the persistent mathematical trend tracker.
        """
        alarms = []
        hard_status = "NORMAL"

        # 1. Evaluate Direct Hard Edge-Threshold Constraints
        if telemetry.get("oil_pressure", 4.0) < self.thresholds["oil_pressure_min"]:
            alarms.append("CRITICAL: Low Oil Pressure Loss")
            hard_status = "CRITICAL"

        if telemetry.get("cht", 130.0) > self.thresholds["cht_max"]:
            alarms.append("WARNING: High Cylinder Head Temperature")
            if hard_status != "CRITICAL":
                hard_status = "WARNING"

        if telemetry.get("vibration", 0.1) > self.thresholds["vibration_max"]:
            alarms.append("CRITICAL: Severe Engine Vibration")
            hard_status = "CRITICAL"
            
        if telemetry.get("oil_temperature", 95.0) > self.thresholds["oil_temp_max"]:
            alarms.append("WARNING: High Oil Temperature Profile")
            if hard_status != "CRITICAL":
                hard_status = "WARNING"

        # 2. Process Advanced Hybrid Persistence State Machine
        # Combine telemetry items and residuals to construct the full context features vector
        combined_features = {**telemetry, **residuals}
        scored = score_result or self.score(combined_features)
        
        # 3. Resolve Outliers: Hard limits override analytical persistence equations for flight safety
        if hard_status == "CRITICAL" or scored.confirmed_state == AnomalyState.CRITICAL:
            final_status = "CRITICAL"
        elif hard_status == "WARNING" or scored.confirmed_state == AnomalyState.WARNING:
            final_status = "WARNING"
        elif scored.confirmed_state == AnomalyState.SUSPECT:
            final_status = "SUSPECT"
        else:
            final_status = "NORMAL"

        is_anomaly = (len(alarms) > 0) or (final_status in ["WARNING", "CRITICAL"])
        return is_anomaly, alarms, final_status

    def score(self, features: Dict[str, float]) -> AnomalyResult:
        """Updates mathematical persistence windows to secure noise immunity."""
        physics_score = self._physics_score(features)
        ml_score = self._ml_score(features)
        physics_weight, ml_weight = self.effective_weights
        combined = physics_weight * physics_score + ml_weight * ml_score
        instant_state = self._state_for_score(combined)

        # Append to window queue manager
        self._recent_states.append(instant_state)
        if len(self._recent_states) > PERSISTENCE_WINDOWS:
            self._recent_states.pop(0)

        # Confirmation State-Lock logic
        if len(self._recent_states) == PERSISTENCE_WINDOWS and len(set(self._recent_states)) == 1:
            self._confirmed_state = instant_state
        elif instant_state == AnomalyState.NORMAL and self._recent_states.count(AnomalyState.NORMAL) >= 3:
            self._confirmed_state = AnomalyState.NORMAL

        return AnomalyResult(
            physics_score=physics_score,
            ml_score=ml_score,
            combined_score=combined,
            state=instant_state,
            confirmed_state=self._confirmed_state
        )
