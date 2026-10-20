"""
feature_engineering.py
------------------------
Converts a rolling window of telemetry + Digital Twin residuals into
health-relevant features: slopes, RMS, margins and cross-sensor ratios.

This module owns NO ML models -- it only produces the feature vector
that anomaly_detector.py and fault_classifier.py consume.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List

import numpy as np


@dataclass
class FeatureWindow:
    """Fixed-size rolling buffer of recent (telemetry, residual) samples."""
    maxlen: int = 30  # ~30 seconds at 1 Hz
    timestamps: Deque[float] = field(default_factory=deque)
    egt: Deque[float] = field(default_factory=deque)
    cht: Deque[float] = field(default_factory=deque)
    oil_pressure: Deque[float] = field(default_factory=deque)
    oil_temperature: Deque[float] = field(default_factory=deque)
    vibration: Deque[float] = field(default_factory=deque)
    fuel_flow: Deque[float] = field(default_factory=deque)
    rpm: Deque[float] = field(default_factory=deque)
    egt_residual: Deque[float] = field(default_factory=deque)
    cht_residual: Deque[float] = field(default_factory=deque)
    oil_pressure_residual: Deque[float] = field(default_factory=deque)
    oil_temperature_residual: Deque[float] = field(default_factory=deque)
    vibration_residual: Deque[float] = field(default_factory=deque)
    fuel_flow_residual: Deque[float] = field(default_factory=deque)
    rpm_residual: Deque[float] = field(default_factory=deque)

    def _push(self, buf: Deque[float], value: float):
        buf.append(value)
        if len(buf) > self.maxlen:
            buf.popleft()

    def push(self, t: float, telemetry: dict, normalized_residuals: dict):
        self._push(self.timestamps, t)
        self._push(self.egt, telemetry["egt"])
        self._push(self.cht, telemetry["cht"])
        self._push(self.oil_pressure, telemetry["oil_pressure"])
        self._push(self.oil_temperature, telemetry["oil_temperature"])
        self._push(self.vibration, telemetry["vibration"])
        self._push(self.fuel_flow, telemetry["fuel_flow"])
        self._push(self.rpm, telemetry["rpm"])
        self._push(self.egt_residual, normalized_residuals["egt"])
        self._push(self.cht_residual, normalized_residuals["cht"])
        self._push(self.oil_pressure_residual, normalized_residuals["oil_pressure"])
        self._push(self.oil_temperature_residual, normalized_residuals["oil_temperature"])
        self._push(self.vibration_residual, normalized_residuals["vibration"])
        self._push(self.fuel_flow_residual, normalized_residuals["fuel_flow"])
        self._push(self.rpm_residual, normalized_residuals["rpm"])

    def is_ready(self, min_samples: int = 5) -> bool:
        return len(self.timestamps) >= min_samples


def _slope(values: Deque[float], timestamps: Deque[float]) -> float:
    """Least-squares slope (per second) of a short window."""
    if len(values) < 3:
        return 0.0
    x = np.array(timestamps, dtype=float)
    y = np.array(values, dtype=float)
    x = x - x[0]
    try:
        slope, _ = np.polyfit(x, y, 1)
    except Exception:
        slope = 0.0
    return float(slope)


def _rms(values: Deque[float]) -> float:
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    return float(np.sqrt(np.mean(arr ** 2)))


def extract_features(window: FeatureWindow) -> Dict[str, float]:
    """Build the feature vector consumed by the anomaly detector and
    fault classifier. All features are named explicitly so the fault
    classifier's evidence strings can reference them directly."""

    egt_slope = _slope(window.egt, window.timestamps)
    cht_slope = _slope(window.cht, window.timestamps)
    oil_pressure_slope = _slope(window.oil_pressure, window.timestamps)
    oil_temp_slope = _slope(window.oil_temperature, window.timestamps)
    vibration_rms = _rms(window.vibration)
    # RPM "fluctuation" as std of the RESIDUAL, not raw RPM: the twin already
    # predicts RPM from throttle, so a normal throttle change cancels out of
    # the residual. What's left over is genuine irregularity (e.g. misfire).
    rpm_instability = float(np.std(window.rpm_residual)) if len(window.rpm_residual) >= 3 else 0.0

    fuel_flow_last = window.fuel_flow[-1] if window.fuel_flow else 0.0
    rpm_last = window.rpm[-1] if window.rpm else 1.0
    fuel_flow_per_rpm = fuel_flow_last / max(rpm_last, 1.0)

    oil_pressure_last = window.oil_pressure[-1] if window.oil_pressure else 0.0

    features = {
        "egt_slope": egt_slope,
        "cht_slope": cht_slope,
        "oil_pressure_slope": oil_pressure_slope,
        "oil_temperature_slope": oil_temp_slope,
        "vibration_rms": vibration_rms,
        "rpm_fluctuation": rpm_instability,
        "fuel_flow_per_rpm": fuel_flow_per_rpm,
        "oil_pressure_margin": oil_pressure_last,
        # latest normalized residuals (most recent sample)
        "egt_residual": window.egt_residual[-1] if window.egt_residual else 0.0,
        "cht_residual": window.cht_residual[-1] if window.cht_residual else 0.0,
        "oil_pressure_residual": window.oil_pressure_residual[-1] if window.oil_pressure_residual else 0.0,
        "oil_temperature_residual": window.oil_temperature_residual[-1] if window.oil_temperature_residual else 0.0,
        "vibration_residual": window.vibration_residual[-1] if window.vibration_residual else 0.0,
        "fuel_flow_residual": window.fuel_flow_residual[-1] if window.fuel_flow_residual else 0.0,
        "rpm_residual": window.rpm_residual[-1] if window.rpm_residual else 0.0,
    }
    return features


FEATURE_ORDER: List[str] = [
    "egt_slope", "cht_slope", "oil_pressure_slope", "oil_temperature_slope",
    "vibration_rms", "rpm_fluctuation", "fuel_flow_per_rpm", "oil_pressure_margin",
    "egt_residual", "cht_residual", "oil_pressure_residual",
    "oil_temperature_residual", "vibration_residual", "fuel_flow_residual",
    "rpm_residual",
]


def features_to_vector(features: Dict[str, float]) -> np.ndarray:
    return np.array([features[k] for k in FEATURE_ORDER], dtype=float)
