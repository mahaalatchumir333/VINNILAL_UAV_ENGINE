"""
schemas.py
------------
Shared telemetry + analytics-response contract for all four team roles.
Freeze this early -- it prevents frontend/backend integration failure.
"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TelemetryPacket(BaseModel):
    model_config = ConfigDict(extra="allow")

    timestamp: datetime
    engine_id: str
    mission_id: str
    rpm: float = Field(ge=0, le=7000, allow_inf_nan=False)
    throttle: float = Field(ge=0, le=1, allow_inf_nan=False)
    cht: float = Field(ge=-20, le=300, allow_inf_nan=False)
    egt: float = Field(ge=-20, le=1000, allow_inf_nan=False)
    oil_pressure: float = Field(ge=0, le=10, allow_inf_nan=False)
    oil_temperature: float = Field(ge=-20, le=200, allow_inf_nan=False)
    fuel_flow: float = Field(ge=0, le=100, allow_inf_nan=False)
    vibration: float = Field(ge=0, le=20, allow_inf_nan=False)
    battery_voltage: Optional[float] = Field(default=None, ge=0, le=60, allow_inf_nan=False)
    alternator_current: Optional[float] = None
    injection_timing: Optional[float] = None
    altitude: float = Field(ge=0, le=30000, allow_inf_nan=False)
    ambient_temperature: float = Field(ge=-40, le=60, allow_inf_nan=False)
    manifold_pressure: Optional[float] = Field(default=None, ge=0, le=200, allow_inf_nan=False)
    mission_phase: Optional[str] = None

    @field_validator("cht", "egt")
    @classmethod
    def temperature_range_check(cls, v):
        if not -40 <= v <= 1200:
            raise ValueError("temperature reading outside plausible sensor range")
        return v


TelemetryQuality = Literal["good", "degraded", "poor"]
MissionRisk = Literal["low", "medium", "medium_high", "high", "unknown"]
EngineStatus = Literal["normal", "suspect", "warning", "critical"]


class AnalyticsResponse(BaseModel):
    timestamp: datetime
    engine_id: str
    status: EngineStatus
    health_index: float
    thermal_health: float
    lubrication_health: float
    combustion_health: float
    vibration_health: float
    sensor_health: float
    anomaly_score: float
    likely_fault: str
    fault_confidence: float
    estimated_rul_minutes: Optional[float] = None
    rul_lower_bound_minutes: Optional[float] = None
    rul_upper_bound_minutes: Optional[float] = None
    rul_confidence: Optional[str] = None
    rul_assumption: Optional[str] = None
    mission_risk: MissionRisk = "unknown"
    contributing_factors: List[str] = []
    recommendation: str = ""
    telemetry_quality: TelemetryQuality = "good"
    twin_predicted_egt: Optional[float] = None
    twin_predicted_cht: Optional[float] = None


class MissionSimulationRequest(BaseModel):
    mission_type: Literal[
        "short_cruise",
        "hot_weather_endurance",
        "high_altitude_recon",
    ]
    duration_minutes: Optional[float] = Field(default=None, gt=0, le=1440)
    altitude_ft: Optional[float] = Field(default=None, ge=0, le=30000)
    ambient_temperature_c: Optional[float] = Field(default=None, ge=-40, le=60)
    throttle: Optional[float] = Field(default=None, ge=0, le=1)
    starting_health_index: Optional[float] = Field(default=None, ge=0, le=100)


class MissionSimulationResponse(BaseModel):
    mission_type: str
    mission_name: str
    duration_minutes: float
    starting_health_index: float
    predicted_max_cht: float
    predicted_max_egt: float
    predicted_fuel_consumption_l: float
    predicted_health_index_end: float
    risk_score: float
    mission_risk: MissionRisk
    recommendation: str
