"""
residual_engine.py
--------------------
Computes measured-vs-predicted residuals, the core Digital Twin signal
used by the AI/analytics layer. A residual is the difference between
what the sensors report and what the twin expects under the same
operating conditions.
"""

from dataclasses import dataclass
from typing import Dict

if __package__:
    from .physics_model import DigitalTwin
else:
    from physics_model import DigitalTwin


SENSOR_KEYS = ["rpm", "fuel_flow", "egt", "cht", "oil_temperature",
               "oil_pressure", "vibration"]

# Typical normal-operation residual scale (sigma) per sensor, used to
# normalize residuals into comparable units. These should be calibrated
# from normal-scenario data, not guessed permanently.
DEFAULT_SIGMA = {
    "rpm": 60.0,
    "fuel_flow": 0.8,
    "egt": 12.0,
    "cht": 6.0,
    "oil_temperature": 4.0,
    "oil_pressure": 0.3,
    "vibration": 0.05,
}


@dataclass
class ResidualResult:
    raw: Dict[str, float]
    normalized: Dict[str, float]
    predicted: Dict[str, float]
    measured: Dict[str, float]


class ResidualEngine:
    def __init__(self, twin: DigitalTwin = None, sigma: dict = None):
        self.twin = twin or DigitalTwin()
        self.sigma = sigma or DEFAULT_SIGMA

    def compute(self, telemetry: dict) -> ResidualResult:
        predicted = self.twin.predict(
            throttle=telemetry["throttle"],
            altitude_ft=telemetry["altitude"],
            ambient_temp_c=telemetry["ambient_temperature"],
        )

        raw = {}
        normalized = {}
        for key in SENSOR_KEYS:
            measured_val = telemetry[key]
            predicted_val = predicted[key]
            residual = measured_val - predicted_val
            raw[key] = residual
            normalized[key] = residual / self.sigma.get(key, 1.0)

        return ResidualResult(
            raw=raw,
            normalized=normalized,
            predicted=predicted,
            measured={k: telemetry[k] for k in SENSOR_KEYS},
        )
