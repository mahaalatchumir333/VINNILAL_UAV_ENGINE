"""
generate_datasets.py
-----------------------
Generates the 12 labelled scenario datasets required by the validation
plan (Problem Statement 26054): normal, high-altitude, hot-weather,
rapid throttle transition, injector degradation, misfire, lubrication
degradation, cooling degradation, vibration increase, EGT sensor drift,
missing telemetry, and network interruption.

Each row includes the FULL telemetry packet plus ground-truth columns
(_ground_truth_fault, _ground_truth_health) so precision/recall/F1 and
detection-lead-time can be measured honestly against real labels.

Run:
    cd simulator
    python3 generate_datasets.py
Output:
    ../data/<scenario_name>.csv  (one file per scenario)
"""

import csv
import random
import sys
from pathlib import Path

from engine_simulator import EngineSimulator
from mission_profiles import (
    make_normal_cruise_mission, make_endurance_mission, make_hot_high_altitude_mission,
)
from fault_injection import (
    normal_scenario, injector_degradation_scenario, overheating_scenario,
    lubrication_loss_scenario, misfire_scenario, vibration_increase_scenario,
    egt_sensor_drift_scenario,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR.mkdir(exist_ok=True)

FIELDNAMES = [
    "t", "timestamp", "engine_id", "mission_id", "mission_phase",
    "rpm", "throttle", "cht", "egt", "oil_pressure", "oil_temperature",
    "fuel_flow", "vibration", "battery_voltage", "alternator_current",
    "injection_timing", "altitude", "ambient_temperature",
    "_ground_truth_fault", "_ground_truth_health",
    "_telemetry_dropped",  # 1 if this row simulates missing/lost telemetry
]


def _write_csv(name: str, rows: list[dict]):
    path = OUT_DIR / f"{name}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows):>5} rows -> {path}")


def _run_scenario(name: str, sim: EngineSimulator, mission, n_ticks: int,
                   dropout_prob: float = 0.0, seed: int = 0):
    rng = random.Random(seed + 1000)
    rows = []
    for i in range(n_ticks):
        pkt = sim.step(mission, dt=1.0)
        dropped = 1 if rng.random() < dropout_prob else 0
        row = {"t": i, "_telemetry_dropped": dropped}
        row.update({k: pkt[k] for k in FIELDNAMES if k in pkt})
        rows.append(row)
    _write_csv(name, rows)


def main():
    # 1. Normal cruise
    _run_scenario(
        "01_normal_cruise",
        EngineSimulator(seed=101, fault_state=normal_scenario()),
        make_normal_cruise_mission(), n_ticks=600,
    )

    # 2. High-altitude operation
    _run_scenario(
        "02_high_altitude",
        EngineSimulator(seed=102, fault_state=normal_scenario()),
        make_hot_high_altitude_mission(), n_ticks=600,
    )

    # 3. Hot-weather endurance
    _run_scenario(
        "03_hot_weather_endurance",
        EngineSimulator(seed=103, fault_state=normal_scenario()),
        make_endurance_mission(), n_ticks=900,
    )

    # 4. Rapid throttle transition (reuse endurance mission's built-in segment)
    _run_scenario(
        "04_rapid_throttle_transition",
        EngineSimulator(seed=104, fault_state=normal_scenario()),
        make_endurance_mission(), n_ticks=900,
    )

    # 5. Injector degradation
    _run_scenario(
        "05_injector_degradation",
        EngineSimulator(seed=105, fault_state=injector_degradation_scenario(start_s=200, ramp_s=400)),
        make_endurance_mission(), n_ticks=900,
    )

    # 6. Misfire
    _run_scenario(
        "06_misfire",
        EngineSimulator(seed=106, fault_state=misfire_scenario(start_s=150, ramp_s=300)),
        make_endurance_mission(), n_ticks=900,
    )

    # 7. Lubrication degradation
    _run_scenario(
        "07_lubrication_degradation",
        EngineSimulator(seed=107, fault_state=lubrication_loss_scenario(start_s=200, ramp_s=400)),
        make_endurance_mission(), n_ticks=900,
    )

    # 8. Cooling / overheating degradation
    _run_scenario(
        "08_cooling_degradation",
        EngineSimulator(seed=108, fault_state=overheating_scenario(start_s=200, ramp_s=500)),
        make_endurance_mission(), n_ticks=900,
    )

    # 9. Vibration increase
    _run_scenario(
        "09_vibration_increase",
        EngineSimulator(seed=109, fault_state=vibration_increase_scenario(start_s=200, ramp_s=400)),
        make_endurance_mission(), n_ticks=900,
    )

    # 10. EGT sensor drift (sensor fault, not real degradation)
    _run_scenario(
        "10_egt_sensor_drift",
        EngineSimulator(seed=110, fault_state=egt_sensor_drift_scenario(start_s=150, ramp_s=500)),
        make_endurance_mission(), n_ticks=900,
    )

    # 11. Missing telemetry (normal engine, but ~8% of packets flagged dropped)
    _run_scenario(
        "11_missing_telemetry",
        EngineSimulator(seed=111, fault_state=normal_scenario()),
        make_normal_cruise_mission(), n_ticks=600, dropout_prob=0.08,
    )

    # 12. Network interruption (normal engine, one long contiguous dropout block)
    sim12 = EngineSimulator(seed=112, fault_state=normal_scenario())
    mission12 = make_normal_cruise_mission()
    rows = []
    for i in range(600):
        pkt = sim12.step(mission12, dt=1.0)
        dropped = 1 if 250 <= i < 310 else 0  # 60s outage
        row = {"t": i, "_telemetry_dropped": dropped}
        row.update({k: pkt[k] for k in FIELDNAMES if k in pkt})
        rows.append(row)
    _write_csv("12_network_interruption", rows)

    print("\nAll 12 scenario datasets generated in:", OUT_DIR)


if __name__ == "__main__":
    main()
