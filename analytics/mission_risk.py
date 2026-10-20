"""Scenario-based mission stress scoring for the simulated engine."""

from dataclasses import asdict, dataclass
from typing import Literal

from digital_twin.physics_model import DigitalTwin

MissionType = Literal[
    "short_cruise",
    "hot_weather_endurance",
    "high_altitude_recon",
]


@dataclass(frozen=True)
class MissionProfile:
    name: str
    description: str
    throttle: float
    altitude_ft: float
    ambient_temperature_c: float
    default_duration_minutes: float


@dataclass(frozen=True)
class MissionRiskResult:
    mission_type: str
    mission_name: str
    duration_minutes: float
    starting_health_index: float
    predicted_max_cht: float
    predicted_max_egt: float
    predicted_fuel_consumption_l: float
    predicted_health_index_end: float
    risk_score: float
    mission_risk: str
    recommendation: str

    def as_dict(self) -> dict:
        return asdict(self)


MISSION_PROFILES: dict[str, MissionProfile] = {
    "short_cruise": MissionProfile(
        name="Short Cruise",
        description="Moderate-temperature, mid-altitude cruise.",
        throttle=0.65,
        altitude_ft=6000.0,
        ambient_temperature_c=25.0,
        default_duration_minutes=60.0,
    ),
    "hot_weather_endurance": MissionProfile(
        name="Hot-Weather Endurance",
        description="Extended cruise in hot ambient conditions.",
        throttle=0.72,
        altitude_ft=8000.0,
        ambient_temperature_c=42.0,
        default_duration_minutes=480.0,
    ),
    "high_altitude_recon": MissionProfile(
        name="High-Altitude Recon",
        description="High-altitude cruise in cool ambient conditions.",
        throttle=0.75,
        altitude_ft=18000.0,
        ambient_temperature_c=5.0,
        default_duration_minutes=240.0,
    ),
}


def available_mission_profiles() -> list[dict]:
    return [
        {"mission_type": key, **asdict(profile)}
        for key, profile in MISSION_PROFILES.items()
    ]


def estimate_mission_risk(
    mission_type: MissionType,
    current_health_index: float,
    twin: DigitalTwin,
    *,
    duration_minutes: float | None = None,
    altitude_ft: float | None = None,
    ambient_temperature_c: float | None = None,
    throttle: float | None = None,
) -> MissionRiskResult:
    """Estimate scenario stress; this is a prototype score, not flight clearance."""
    profile = MISSION_PROFILES[mission_type]
    duration = duration_minutes or profile.default_duration_minutes
    altitude = altitude_ft if altitude_ft is not None else profile.altitude_ft
    ambient = (
        ambient_temperature_c
        if ambient_temperature_c is not None
        else profile.ambient_temperature_c
    )
    throttle_value = throttle if throttle is not None else profile.throttle

    predicted = twin.predict(
        throttle=throttle_value,
        altitude_ft=altitude,
        ambient_temp_c=ambient,
    )
    predicted_cht = float(predicted["cht"])
    predicted_egt = float(predicted["egt"])
    fuel_consumption = float(predicted["fuel_flow"]) * duration / 60.0

    thermal_stress = max(
        0.0,
        (predicted_cht - 160.0) / 35.0,
        (predicted_egt - 750.0) / 150.0,
    )
    duration_stress = min(duration / 480.0, 1.0)
    health_stress = (100.0 - current_health_index) / 100.0
    estimated_health_loss = duration * (0.005 + 0.025 * thermal_stress)
    estimated_end_health = max(0.0, current_health_index - estimated_health_loss)
    risk_score = min(
        100.0,
        max(
            0.0,
            100.0
            * (
                0.50 * health_stress
                + 0.30 * thermal_stress
                + 0.20 * duration_stress
            ),
        ),
    )

    if current_health_index < 30.0 or predicted_cht >= 185.0 or predicted_egt >= 900.0:
        risk = "high"
        recommendation = "Do not treat this profile as cleared; inspect engine health and thermal limits."
    elif risk_score >= 50.0:
        risk = "medium_high"
        recommendation = "Review engine condition and thermal margins before this extended profile."
    elif risk_score >= 25.0:
        risk = "medium"
        recommendation = "Monitor health and thermal trends during the mission."
    else:
        risk = "low"
        recommendation = "No elevated scenario stress identified by this prototype model."

    return MissionRiskResult(
        mission_type=mission_type,
        mission_name=profile.name,
        duration_minutes=duration,
        starting_health_index=current_health_index,
        predicted_max_cht=round(predicted_cht, 1),
        predicted_max_egt=round(predicted_egt, 1),
        predicted_fuel_consumption_l=round(fuel_consumption, 2),
        predicted_health_index_end=round(estimated_end_health, 1),
        risk_score=round(risk_score, 1),
        mission_risk=risk,
        recommendation=recommendation,
    )
