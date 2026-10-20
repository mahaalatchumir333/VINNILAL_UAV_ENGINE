"""
engine_simulator.py
--------------------
Scenario-based, physics-informed telemetry simulator for the MALE-UAV
aero-piston engine prototype (Person 1 module).

This is NOT a certified thermodynamic model. It is a reduced-order,
physics-informed demonstrator used to generate labelled synthetic
telemetry for the Digital Twin + AI analytics pipeline.
"""

import math
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

if __package__:
    from .mission_profiles import MissionPhase, MissionProfile
    from .fault_injection import FaultState
else:
    from mission_profiles import MissionPhase, MissionProfile
    from fault_injection import FaultState


GAS_CONSTANT = 287.05  # J/(kg*K), dry air


@dataclass
class EngineBaseline:
    base_rpm: float = 3800
    base_egt: float = 560.0
    base_cht: float = 130.0
    base_oil_temp: float = 75.0
    base_oil_pressure: float = 4.5
    base_fuel_flow: float = 14.0
    base_vibration: float = 0.15
    base_manifold_pressure: float = 30.0

    # sensitivity coefficients (illustrative, calibratable)
    egt_load_factor: float = 180.0
    egt_ambient_factor: float = 0.9
    egt_fault_factor: float = 140.0

    cht_load_factor: float = 50.0
    cht_ambient_factor: float = 0.6
    cooling_factor: float = 50.0

    oil_load_factor: float = 25.0
    oil_fault_factor: float = 35.0
    oil_pressure_fault_factor: float = 2.0

    vibration_fault_factor: float = 0.9
    fuel_flow_fault_factor: float = 4.5
    manifold_pressure_throttle_factor: float = 25.0


class EngineSimulator:
    """
    Produces one telemetry packet per call to `step()`.
    Designed to stream via WebSocket at ~1 Hz but can also be
    used to bulk-generate CSV datasets for training/testing.
    """

    def __init__(
        self,
        engine_id: str = "ENG-01",
        baseline: Optional[EngineBaseline] = None,
        seed: Optional[int] = None,
        fault_state: Optional[FaultState] = None,
    ):
        self.engine_id = engine_id
        self.baseline = baseline or EngineBaseline()
        self.t = 0.0
        self.rng = random.Random(seed)

        # hidden degradation state (0 = healthy, 1 = fully degraded)
        # pass a pre-scheduled FaultState (see fault_injection.py factories)
        # to run a specific labelled scenario; defaults to a healthy engine.
        self.fault = fault_state if fault_state is not None else FaultState()

    def air_density(self, altitude_ft: float, ambient_temp_c: float) -> float:
        pressure = 101325 * (1 - 2.25577e-5 * altitude_ft * 0.3048) ** 5.25588
        temp_k = ambient_temp_c + 273.15
        return pressure / (GAS_CONSTANT * temp_k)

    def _expected_values(self, phase: MissionPhase, throttle: float,
                          altitude_ft: float, ambient_temp_c: float):
        b = self.baseline
        density = self.air_density(altitude_ft, ambient_temp_c)
        density_factor = max(density / 1.225, 0.3)  # relative to sea level

        expected_rpm = b.base_rpm + throttle * 900 * density_factor
        expected_manifold_pressure = (
            b.base_manifold_pressure
            + throttle * b.manifold_pressure_throttle_factor * density_factor
        )
        expected_fuel_flow = (
            b.base_fuel_flow
            + throttle * 9.0
            + self.fault.injector_degradation * b.fuel_flow_fault_factor
        )
        expected_egt = (
            b.base_egt
            + throttle * b.egt_load_factor
            + (ambient_temp_c - 25) * b.egt_ambient_factor
            + self.fault.injector_degradation * b.egt_fault_factor
            + self.fault.combustion_instability * 60.0
        )
        expected_cht = (
            b.base_cht
            + throttle * b.cht_load_factor
            + (ambient_temp_c - 25) * b.cht_ambient_factor
            - self.fault.cooling_efficiency_loss * b.cooling_factor * -1
            + self.fault.cooling_efficiency_loss * b.cooling_factor
        )
        expected_oil_temp = (
            b.base_oil_temp
            + throttle * b.oil_load_factor
            + self.fault.lubrication_degradation * b.oil_fault_factor
        )
        expected_oil_pressure = max(
            0.5,
            b.base_oil_pressure
            - self.fault.lubrication_degradation * b.oil_pressure_fault_factor,
        )
        expected_vibration = (
            b.base_vibration
            + self.fault.vibration_severity * b.vibration_fault_factor
            + self.fault.misfire_severity * 0.4
        )
        return {
            "rpm": expected_rpm,
            "manifold_pressure": expected_manifold_pressure,
            "fuel_flow": expected_fuel_flow,
            "egt": expected_egt,
            "cht": expected_cht,
            "oil_temperature": expected_oil_temp,
            "oil_pressure": expected_oil_pressure,
            "vibration": expected_vibration,
        }

    # measurement noise (std dev) per sensor -- illustrative sensor-grade noise
    _NOISE_STD = {
        "rpm": 8.0,
        "manifold_pressure": 0.2,
        "fuel_flow": 0.15,
        "egt": 2.0,
        "cht": 1.0,
        "oil_temperature": 0.5,
        "oil_pressure": 0.05,
        "vibration": 0.01,
    }

    def _noise_std(self, sensor_key: str) -> float:
        return self._NOISE_STD.get(sensor_key, 0.0)

    def step(self, mission: MissionProfile, dt: float = 1.0) -> dict:
        """Advance the simulation by dt seconds and return one telemetry packet."""
        self.t += dt
        phase, throttle = mission.state_at(self.t)
        altitude_ft = mission.altitude_at(self.t)
        ambient_temp_c = mission.ambient_temperature

        self.fault.advance(dt)

        expected = self._expected_values(phase, throttle, altitude_ft, ambient_temp_c)

        # add measurement noise + occasional sensor dropout handled by fault injector
        measured = {
            k: v + self.rng.gauss(0, self._noise_std(k))
            for k, v in expected.items()
        }
        measured = self.fault.apply_sensor_effects(measured, self.rng)

        # Misfire causes genuine cylinder-to-cylinder RPM instability (hunting),
        # not just a sensor-side effect -- this belongs on the "measured" (true
        # engine behaviour) side, not the twin's expected-value prediction,
        # since a healthy twin should NOT anticipate misfire-driven RPM noise.
        if self.fault.misfire_severity > 0:
            measured["rpm"] += self.fault.misfire_severity * self.rng.gauss(0, 140)
            # misfire also causes brief EGT spikes/dips from uneven combustion
            measured["egt"] += self.fault.misfire_severity * self.rng.gauss(0, 25)

        # Misfire's real signature is irregular combustion -- RPM hunting and
        # erratic EGT -- not just higher vibration. This models that directly
        # on the TRUE physical signal (not sensor noise), scaled by severity,
        # matching the evidence table (RPM fluctuation, irregular EGT,
        # vibration increase) the fault classifier looks for.
        if self.fault.misfire_severity > 0:
            measured["rpm"] += self.rng.gauss(0, self.fault.misfire_severity * 120)
            measured["egt"] += self.rng.gauss(0, self.fault.misfire_severity * 15)

        packet = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "engine_id": self.engine_id,
            "mission_id": mission.mission_id,
            "mission_phase": phase.value,
            "rpm": round(measured["rpm"], 1),
            "manifold_pressure": round(measured["manifold_pressure"], 2),
            "throttle": round(throttle, 3),
            "cht": round(measured["cht"], 2),
            "egt": round(measured["egt"], 2),
            "oil_pressure": round(measured["oil_pressure"], 2),
            "oil_temperature": round(measured["oil_temperature"], 2),
            "fuel_flow": round(measured["fuel_flow"], 2),
            "vibration": round(max(0.0, measured["vibration"]), 3),
            "battery_voltage": round(24.0 + self.rng.gauss(0, 0.1), 2),
            "alternator_current": round(12.0 + self.rng.gauss(0, 0.2), 2),
            "injection_timing": round(18.0 + self.rng.gauss(0, 0.1), 2),
            "altitude": round(altitude_ft, 1),
            "ambient_temperature": round(ambient_temp_c, 1),
            # ground truth, kept ONLY for offline evaluation, not sent to AI module
            "_ground_truth_fault": self.fault.active_fault_label(),
            "_ground_truth_health": round(self.fault.true_health_index(), 2),
        }
        return packet


if __name__ == "__main__":
    from mission_profiles import make_endurance_mission

    sim = EngineSimulator(seed=42)
    mission = make_endurance_mission()
    for i in range(10):
        pkt = sim.step(mission)
        print(pkt)
        time.sleep(0.05)
