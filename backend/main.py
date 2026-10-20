import os
import sys
import asyncio
from contextlib import asynccontextmanager
from os import environ
from typing import Any, AsyncGenerator, Literal, Optional
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

# Ensure root directory is on Python system path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from simulator.engine_simulator import EngineSimulator
from simulator.mission_profiles import MissionPhase, MissionProfile, PhaseSegment
from digital_twin.physics_model import DigitalTwin
from digital_twin.residual_engine import ResidualEngine, ResidualResult
from analytics.anomaly_detector import (
    PERSISTENCE_WINDOWS,
    STATE_THRESHOLDS,
    AnomalyDetector,
    AnomalyResult,
    AnomalyState,
)
from analytics.feature_engineering import FeatureWindow, extract_features
from analytics.fault_classifier import FAULT_CODES, FaultResult, classify_fault
from analytics.health_index import (
    HealthResult,
    compute_health_index,
    explain_health_index,
)
from analytics.rul_estimator import RULResult, RULEstimator
from analytics.mission_risk import (
    MissionType,
    available_mission_profiles,
    estimate_mission_risk,
)
from backend.database import get_history, get_latest, initialize_database, record_tick
from backend.fail_safe_guard import assess_telemetry
from backend.schemas import MissionSimulationRequest, MissionSimulationResponse

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    initialize_database()
    yield


app = FastAPI(
    title="VINNILAL UAV Digital Twin API",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for dashboard communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in environ.get("CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize singletons
simulator = EngineSimulator()
flight_controls = {
    "phase": "cruise",
    "throttle": 0.65,
    "altitude": 6000.0,
    "ambient_temperature": 25.0,
}
mission = MissionProfile(
    mission_id="LIVE-CRUISE",
    ambient_temperature=flight_controls["ambient_temperature"],
    segments=[
        PhaseSegment(MissionPhase.CRUISE, 0.0, 1_000_000_000.0, 0.65, 6000.0)
    ],
)
physics_model = DigitalTwin()
residual_engine = ResidualEngine(twin=physics_model)
feature_window = FeatureWindow()
anomaly_detector = AnomalyDetector()
rul_estimator = RULEstimator()

RESIDUAL_UNITS = {
    "rpm": "RPM",
    "fuel_flow": "telemetry units",
    "egt": "°C",
    "cht": "°C",
    "oil_temperature": "°C",
    "oil_pressure": "bar",
    "vibration": "mm/s",
}


def build_explainability(
    telemetry: dict[str, Any],
    residual_result: ResidualResult,
    features: dict[str, float],
    anomaly_result: AnomalyResult,
    hard_alarms: list[str],
    status: str,
    fault_diagnosis: FaultResult,
    health: HealthResult,
    telemetry_quality: str,
    rul: RULResult,
) -> dict[str, object]:
    normalized_values = [
        abs(features[f"{sensor}_residual"])
        for sensor in RESIDUAL_UNITS
    ]
    worst_residual = max(normalized_values)
    mean_residual = sum(normalized_values) / len(normalized_values)
    blended_residual = 0.7 * worst_residual + 0.3 * mean_residual
    active_thresholds = {
        "suspect": STATE_THRESHOLDS[AnomalyState.SUSPECT],
        "warning": STATE_THRESHOLDS[AnomalyState.WARNING],
        "critical": STATE_THRESHOLDS[AnomalyState.CRITICAL],
    }
    hard_limits = [
        {
            "sensor": "oil_pressure",
            "observed": telemetry["oil_pressure"],
            "operator": "<",
            "threshold": anomaly_detector.thresholds["oil_pressure_min"],
            "unit": "bar",
            "triggered": telemetry["oil_pressure"] < anomaly_detector.thresholds["oil_pressure_min"],
        },
        {
            "sensor": "cht",
            "observed": telemetry["cht"],
            "operator": ">",
            "threshold": anomaly_detector.thresholds["cht_max"],
            "unit": "°C",
            "triggered": telemetry["cht"] > anomaly_detector.thresholds["cht_max"],
        },
        {
            "sensor": "vibration",
            "observed": telemetry["vibration"],
            "operator": ">",
            "threshold": anomaly_detector.thresholds["vibration_max"],
            "unit": "mm/s",
            "triggered": telemetry["vibration"] > anomaly_detector.thresholds["vibration_max"],
        },
        {
            "sensor": "oil_temperature",
            "observed": telemetry["oil_temperature"],
            "operator": ">",
            "threshold": anomaly_detector.thresholds["oil_temp_max"],
            "unit": "°C",
            "triggered": telemetry["oil_temperature"] > anomaly_detector.thresholds["oil_temp_max"],
        },
    ]
    sensor_comparisons = [
        {
            "sensor": sensor,
            "unit": RESIDUAL_UNITS[sensor],
            "measured": residual_result.measured[sensor],
            "twin_predicted": residual_result.predicted[sensor],
            "residual": residual_result.raw[sensor],
            "normal_residual_scale": residual_engine.sigma[sensor],
            "normalized_residual_sigma": residual_result.normalized[sensor],
            "interpretation": (
                "measured above twin baseline"
                if residual_result.raw[sensor] > 0
                else "measured below twin baseline"
                if residual_result.raw[sensor] < 0
                else "matches twin baseline"
            ),
        }
        for sensor in RESIDUAL_UNITS
    ]
    candidates = sorted(
        (
            {"fault": fault, "vote_score": score}
            for fault, score in fault_diagnosis.vote_scores.items()
        ),
        key=lambda candidate: candidate["vote_score"],
        reverse=True,
    )
    return {
        "scope": "Per-tick explanation of the current prototype analytics; not a causal finding.",
        "methods": {
            "digital_twin": "Reduced-order physics model predicts healthy sensor values from throttle, altitude, and ambient temperature.",
            "anomaly_detector": (
                "Physics residual score blended with an Isolation Forest score."
                if anomaly_detector.model is not None
                else "Physics-only score at 100% weight; Isolation Forest is inactive because no fitted model is loaded."
            ),
            "fault_diagnosis": "Transparent threshold rules add weighted votes for each candidate fault; the highest eligible vote is reported.",
            "health_index": "Subsystem indices are combined using fixed prototype weights.",
            "data_basis": "Current simulation and physics-informed scenarios; not validated on operational MALE-UAV engine telemetry.",
        },
        "sensor_comparisons": sensor_comparisons,
        "features": features,
        "anomaly": {
            "physics_score": anomaly_result.physics_score,
            "physics_weight": anomaly_detector.effective_weights[0],
            "ml_score": anomaly_result.ml_score,
            "ml_weight": anomaly_detector.effective_weights[1],
            "configured_physics_weight": anomaly_detector.physics_weight,
            "configured_ml_weight": anomaly_detector.ml_weight,
            "ml_model_loaded": anomaly_detector.model is not None,
            "scoring_mode": "hybrid" if anomaly_detector.model is not None else "physics_only_fallback",
            "physics_residual_summary": {
                "worst_absolute_residual_sigma": worst_residual,
                "mean_absolute_residual_sigma": mean_residual,
                "blend_formula": "0.7 × worst absolute residual + 0.3 × mean absolute residual",
                "blended_value": blended_residual,
                "squash_formula": "1 - exp(-blended value / 2)",
            },
            "combined_score": anomaly_result.combined_score,
            "thresholds": active_thresholds,
            "instant_state": anomaly_result.state.value,
            "persistence_confirmed_state": anomaly_result.confirmed_state.value,
            "persistence_required_consecutive_ticks": PERSISTENCE_WINDOWS,
            "final_status": status.lower(),
            "hard_alarm_limits": hard_limits,
            "hard_alarms": hard_alarms,
            "status_rule": "Hard physical limits override the persistence state; otherwise use the confirmed anomaly state.",
        },
        "fault_diagnosis": {
            "selected_fault": fault_diagnosis.likely_fault,
            "selected_fault_code": FAULT_CODES[fault_diagnosis.likely_fault],
            "system_status": status.lower(),
            "confidence": fault_diagnosis.confidence,
            "winning_vote_score": fault_diagnosis.vote_score,
            "minimum_vote_for_label": 1.0,
            "candidate_vote_scores": candidates,
            "rules": fault_diagnosis.rule_trace,
            "selected_fault_evidence": fault_diagnosis.evidence,
            "confidence_note": "This is a normalized rule-vote heuristic, not a calibrated probability.",
        },
        "health_index": {
            **explain_health_index(features, telemetry_quality),
            "displayed_component_indices": {
                "thermal": health.thermal,
                "lubrication": health.lubrication,
                "combustion": health.combustion,
                "vibration": health.vibration,
                "sensor": health.sensor,
            },
            "displayed_overall_health_index": health.overall,
        },
        "remaining_useful_life": {
            "estimate_minutes": rul.estimated_minutes,
            "lower_bound_minutes": rul.lower_bound_minutes,
            "upper_bound_minutes": rul.upper_bound_minutes,
            "confidence_qualifier": rul.confidence,
            "health_threshold": 30.0,
            "degradation_rate_per_minute": rul.degradation_rate_per_min,
            "slope_standard_error": rul.slope_std_error,
            "assumption": rul.assumption,
        },
        "limitations": [
            "Residual sigma scales and alert thresholds are prototype calibration constants.",
            "Rule-vote confidence is not a probability of failure.",
            "An active Isolation Forest must be trained and loaded before its score contributes.",
            "RUL trend intervals do not cover sudden faults, unmodeled modes, or model error.",
        ],
    }


class FaultInjectionRequest(BaseModel):
    fault_type: Literal[
        "cooling_degradation",
        "injector_clog",
        "lubrication_degradation",
        "misfire",
        "vibration_increase",
        "egt_sensor_drift",
    ]
    severity: float = Field(default=0.5, ge=0.0, le=1.0)


class FlightControlRequest(BaseModel):
    phase: Optional[
        Literal["idle", "takeoff", "climb", "cruise", "loiter", "descent"]
    ] = None
    throttle: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    altitude: Optional[float] = Field(default=None, ge=0.0, le=30000.0)
    ambient_temperature: Optional[float] = Field(default=None, ge=-40.0, le=50.0)


FAULT_ATTRIBUTES = {
    "cooling_degradation": "cooling_efficiency_loss",
    "injector_clog": "injector_degradation",
    "lubrication_degradation": "lubrication_degradation",
    "misfire": "misfire_severity",
    "vibration_increase": "vibration_severity",
    "egt_sensor_drift": "egt_sensor_drift",
}

PHASE_SETTINGS = {
    "idle": (MissionPhase.START, 0.0, 0.0),
    "takeoff": (MissionPhase.TAKEOFF, 0.95, 500.0),
    "climb": (MissionPhase.CLIMB, 0.85, 5000.0),
    "cruise": (MissionPhase.CRUISE, 0.65, 6000.0),
    "loiter": (MissionPhase.LOITER, 0.55, 7500.0),
    "descent": (MissionPhase.DESCENT, 0.4, 1000.0),
}


def update_flight_controls(request: FlightControlRequest) -> dict:
    global mission

    if request.phase is not None:
        phase, throttle, altitude = PHASE_SETTINGS[request.phase]
        flight_controls.update(
            phase=request.phase,
            throttle=throttle,
            altitude=altitude,
        )
    else:
        phase = PHASE_SETTINGS[flight_controls["phase"]][0]

    if request.throttle is not None:
        flight_controls["throttle"] = request.throttle
    if request.altitude is not None:
        flight_controls["altitude"] = request.altitude
    if request.ambient_temperature is not None:
        flight_controls["ambient_temperature"] = request.ambient_temperature

    mission = MissionProfile(
        mission_id=f"LIVE-{flight_controls['phase'].upper()}",
        ambient_temperature=flight_controls["ambient_temperature"],
        segments=[
            PhaseSegment(
                phase,
                0.0,
                1_000_000_000.0,
                flight_controls["throttle"],
                flight_controls["altitude"],
            )
        ],
    )
    return flight_controls.copy()


@app.get("/status")
def get_system_status():
    """System health check endpoint."""
    active_fault = simulator.fault.active_fault_label()
    return {
        "status": "ONLINE",
        "system": "VINNILAL UAV Digital Twin Engine",
        "active_fault": None if active_fault == "none" else active_fault,
        "mode": "LIVE_SIMULATION",
        "flight_controls": flight_controls.copy(),
    }


@app.get("/health")
def get_health():
    """Simple liveness endpoint for local development and deployment probes."""
    return {"status": "ok"}


@app.post("/api/simulator/controls")
def set_flight_controls(request: FlightControlRequest):
    """Set the active simulated flight phase and operator control inputs."""
    return update_flight_controls(request)


@app.get("/faults")
@app.get("/api/simulator/faults")
def get_available_faults():
    """Return available fault types for live toggling."""
    return {
        "available_faults": [
            "cooling_degradation",
            "injector_clog",
            "lubrication_degradation",
            "misfire",
            "vibration_increase",
            "egt_sensor_drift"
        ]
    }

@app.post("/api/simulator/inject")
def inject_fault(request: FaultInjectionRequest):
    """Inject interactive fault mid-flight."""
    attribute = FAULT_ATTRIBUTES.get(request.fault_type)
    if attribute is None:
        raise HTTPException(status_code=400, detail="Unsupported fault type")
    setattr(simulator.fault, attribute, request.severity)
    return {
        "status": "FAULT_INJECTED",
        "fault_type": request.fault_type,
        "severity": request.severity
    }

@app.post("/clear")
@app.post("/api/simulator/clear")
def clear_faults():
    """Reset simulator back to healthy status."""
    simulator.fault.reset()
    return {"status": "FAULTS_CLEARED", "engine_state": "HEALTHY"}


@app.get("/api/simulator/status")
def get_simulator_status():
    return {
        "t": simulator.t,
        "active_fault": simulator.fault.active_fault_label(),
        "true_health_index": simulator.fault.true_health_index(),
        "flight_controls": flight_controls.copy(),
    }


@app.get("/api/mission/profiles")
def get_mission_profiles():
    return {"profiles": available_mission_profiles()}


@app.post(
    "/api/mission/simulate",
    response_model=MissionSimulationResponse,
)
def simulate_mission(request: MissionSimulationRequest):
    profile_type: MissionType = request.mission_type
    latest = get_latest(simulator.engine_id)
    current_health = (
        latest["analytics"]["health_index"]
        if latest is not None
        else 100.0
    )
    if request.starting_health_index is not None:
        current_health = request.starting_health_index

    result = estimate_mission_risk(
        profile_type,
        current_health,
        physics_model,
        duration_minutes=request.duration_minutes,
        altitude_ft=request.altitude_ft,
        ambient_temperature_c=request.ambient_temperature_c,
        throttle=request.throttle,
    )
    return result.as_dict()


@app.get("/api/engine/{engine_id}/latest")
def get_latest_telemetry(engine_id: str):
    latest = get_latest(engine_id)
    if latest is None:
        raise HTTPException(status_code=404, detail="No telemetry history for engine")
    return latest


@app.get("/api/engine/{engine_id}/history")
def get_engine_history(
    engine_id: str,
    limit: int = Query(default=500, ge=1, le=5000),
    before_id: Optional[int] = Query(default=None, ge=1),
):
    return get_history(engine_id, limit=limit, before_id=before_id)


@app.get("/")
def serve_dashboard():
    """Serve the primary GCS dashboard."""
    dashboard_path = os.path.join(ROOT_DIR, "frontend", "vinnilal_dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    return HTMLResponse("<h1>VINNILAL API is Running. Dashboard file missing.</h1>")

@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            telemetry = simulator.step(mission)
            telemetry = {
                key: value for key, value in telemetry.items()
                if not key.startswith("_ground_truth")
            }
            assessment = assess_telemetry(telemetry)
            if assessment.quality == "degraded" or assessment.telemetry is None:
                await websocket.send_json({
                    "telemetry": {},
                    "analytics": {
                        "status": "warning",
                        "telemetry_quality": "degraded",
                        "evidence": list(assessment.issues),
                    },
                })
                await asyncio.sleep(1.0)
                continue
            telemetry = assessment.telemetry
            residual_result = residual_engine.compute(telemetry)
            feature_window.push(simulator.t, telemetry, residual_result.normalized)
            features = extract_features(feature_window)
            anomaly_result = anomaly_detector.score(features)
            _, hard_alarms, status = anomaly_detector.detect_anomalies(
                telemetry,
                features,
                score_result=anomaly_result,
            )
            fault_diagnosis = classify_fault(features)
            health = compute_health_index(
                features,
                telemetry_quality=assessment.quality,
            )
            rul = rul_estimator.update(simulator.t, health.overall)
            explainability = build_explainability(
                telemetry=telemetry,
                residual_result=residual_result,
                features=features,
                anomaly_result=anomaly_result,
                hard_alarms=hard_alarms,
                status=status,
                fault_diagnosis=fault_diagnosis,
                health=health,
                telemetry_quality=assessment.quality,
                rul=rul,
            )

            payload = {
                "telemetry": telemetry,
                "flight_controls": flight_controls.copy(),
                "twin_predicted_egt": residual_result.predicted["egt"],
                "twin_predicted_cht": residual_result.predicted["cht"],
                "residuals": residual_result.normalized,
                "analytics": {
                    "status": status.lower(),
                    "hard_alarms": hard_alarms,
                    "telemetry_quality": assessment.quality,
                    "anomaly_score": anomaly_result.combined_score,
                    "health_index": health.overall,
                    "subsystems": {
                        "thermal": health.thermal,
                        "lubrication": health.lubrication,
                        "combustion": health.combustion,
                        "vibration": health.vibration,
                        "sensor": health.sensor,
                    },
                    "fault_type": fault_diagnosis.likely_fault,
                    "fault_code": FAULT_CODES[fault_diagnosis.likely_fault],
                    "confidence": fault_diagnosis.confidence,
                    "evidence": fault_diagnosis.evidence,
                    "rul_minutes": rul.estimated_minutes,
                    "rul_lower_bound_minutes": rul.lower_bound_minutes,
                    "rul_upper_bound_minutes": rul.upper_bound_minutes,
                    "explainability": explainability,
                }
            }

            persisted_analytics = {
                key: value
                for key, value in payload["analytics"].items()
                if key != "explainability"
            }
            record_tick(telemetry, persisted_analytics)
            await websocket.send_json(payload)
            await asyncio.sleep(1.0)  # 1 Hz tick
    except WebSocketDisconnect:
        print("GCS Dashboard disconnected.")