"""
health_index.py
------------------
Computes subsystem health indices (thermal, lubrication, combustion,
vibration, sensor confidence) and a combined overall health score (0-100).

This is a calibrated PROTOTYPE index, not an officially certified engine
health percentage.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np


# subsystem weights for the combined score (must sum to 1.0)
SUBSYSTEM_WEIGHTS = {
    "thermal": 0.30,
    "lubrication": 0.25,
    "combustion": 0.20,
    "vibration": 0.15,
    "sensor": 0.10,
}


@dataclass
class HealthResult:
    thermal: float
    lubrication: float
    combustion: float
    vibration: float
    sensor: float
    overall: float


def _residual_to_degradation(residual_sigma: float, scale: float = 3.0) -> float:
    """Map an absolute normalized residual into a 0-1 degradation score."""
    return float(np.clip(abs(residual_sigma) / scale, 0.0, 1.0))


def _health_components(
    features: Dict[str, float],
    telemetry_quality: str,
) -> Dict[str, Dict[str, float]]:
    thermal_degradation = max(
        _residual_to_degradation(features["egt_residual"]),
        _residual_to_degradation(features["cht_residual"]),
    )
    lubrication_degradation = max(
        _residual_to_degradation(features["oil_pressure_residual"]),
        _residual_to_degradation(features["oil_temperature_residual"]),
    )
    combustion_degradation = _residual_to_degradation(features["fuel_flow_residual"])
    vibration_degradation = float(np.clip(features["vibration_rms"] / 1.0, 0.0, 1.0))

    sensor_degradation = 0.0 if telemetry_quality == "good" else (
        0.3 if telemetry_quality == "degraded" else 0.6
    )

    degradations = {
        "thermal": thermal_degradation,
        "lubrication": lubrication_degradation,
        "combustion": combustion_degradation,
        "vibration": vibration_degradation,
        "sensor": sensor_degradation,
    }
    return {
        name: {
            "health_index": 100.0 * (1.0 - degradation),
            "degradation_fraction": degradation,
        }
        for name, degradation in degradations.items()
    }


def compute_health_index(features: Dict[str, float], telemetry_quality: str = "good") -> HealthResult:
    components = _health_components(features, telemetry_quality)
    health = {
        name: round(details["health_index"], 1)
        for name, details in components.items()
    }
    overall = sum(
        SUBSYSTEM_WEIGHTS[name] * details["health_index"]
        for name, details in components.items()
    )

    return HealthResult(
        thermal=health["thermal"],
        lubrication=health["lubrication"],
        combustion=health["combustion"],
        vibration=health["vibration"],
        sensor=health["sensor"],
        overall=round(overall, 1),
    )


def explain_health_index(
    features: Dict[str, float],
    telemetry_quality: str = "good",
) -> Dict[str, object]:
    """Return the exact weighted components and input drivers of the health index."""
    components = _health_components(features, telemetry_quality)
    drivers = {
        "thermal": {
            "egt_residual_sigma": abs(features["egt_residual"]),
            "cht_residual_sigma": abs(features["cht_residual"]),
        },
        "lubrication": {
            "oil_pressure_residual_sigma": abs(features["oil_pressure_residual"]),
            "oil_temperature_residual_sigma": abs(features["oil_temperature_residual"]),
        },
        "combustion": {
            "fuel_flow_residual_sigma": abs(features["fuel_flow_residual"]),
        },
        "vibration": {
            "window_vibration_rms": features["vibration_rms"],
        },
        "sensor": {
            "telemetry_quality": telemetry_quality,
        },
    }
    for name, details in components.items():
        details["weight"] = SUBSYSTEM_WEIGHTS[name]
        details["weighted_points"] = (
            details["health_index"] * SUBSYSTEM_WEIGHTS[name]
        )
        details["drivers"] = drivers[name]

    return {
        "formula": "Overall health = sum(subsystem health × subsystem weight)",
        "residual_scale_sigma": 3.0,
        "components": components,
        "overall_health_index": round(
            sum(details["weighted_points"] for details in components.values()),
            1,
        ),
    }
