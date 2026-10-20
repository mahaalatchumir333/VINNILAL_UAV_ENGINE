"""
fault_injection.py
--------------------
Labelled fault-injection state machine. Each fault ramps a hidden
degradation variable (0 = healthy, 1 = fully failed) over a configurable
time window. Ground-truth labels are kept for offline evaluation only
(never sent to the AI module, which must infer faults from residuals).
"""

import random
from dataclasses import dataclass


@dataclass
class FaultConfig:
    fault_name: str
    start_s: float
    ramp_duration_s: float
    max_severity: float = 1.0


class FaultState:
    """
    Tracks up to one primary hidden degradation channel at a time
    (multiple channels are supported simultaneously, e.g. injector +
    sensor drift, to test multi-label fault classification).
    """

    def __init__(self):
        self.t = 0.0
        self.injector_degradation = 0.0
        self.cooling_efficiency_loss = 0.0
        self.lubrication_degradation = 0.0
        self.vibration_severity = 0.0
        self.misfire_severity = 0.0
        self.combustion_instability = 0.0
        self.egt_sensor_drift = 0.0
        self.cht_sensor_dropout_active = False

        self._configs: list[FaultConfig] = []

    def reset(self):
        """
        Clears all active/scheduled degradation in-place (does NOT reset
        the internal clock `t`). Used for LIVE fault control: an engine
        simulator instance already running holds a reference to this
        FaultState object, so mutating it in place (rather than replacing
        it with a new FaultState()) means a running WebSocket stream picks
        up the change on its very next tick.
        """
        self.injector_degradation = 0.0
        self.cooling_efficiency_loss = 0.0
        self.lubrication_degradation = 0.0
        self.vibration_severity = 0.0
        self.misfire_severity = 0.0
        self.combustion_instability = 0.0
        self.egt_sensor_drift = 0.0
        self._configs = []

    def schedule(self, config: FaultConfig):
        self._configs.append(config)

    def advance(self, dt: float):
        self.t += dt
        for cfg in self._configs:
            if self.t < cfg.start_s:
                continue
            progress = min(1.0, (self.t - cfg.start_s) / cfg.ramp_duration_s)
            severity = progress * cfg.max_severity
            if cfg.fault_name == "injector_degradation":
                self.injector_degradation = severity
            elif cfg.fault_name == "cooling_degradation":
                self.cooling_efficiency_loss = severity
            elif cfg.fault_name == "lubrication_degradation":
                self.lubrication_degradation = severity
            elif cfg.fault_name == "vibration_increase":
                self.vibration_severity = severity
            elif cfg.fault_name == "misfire":
                self.misfire_severity = severity
            elif cfg.fault_name == "combustion_instability":
                self.combustion_instability = severity
            elif cfg.fault_name == "egt_sensor_drift":
                self.egt_sensor_drift = severity

    def apply_sensor_effects(self, measured: dict, rng: random.Random) -> dict:
        """Apply drift/dropout effects directly to the *measured* signal
        (as opposed to the true engine state), simulating sensor-level faults
        distinct from real engine degradation."""
        if self.egt_sensor_drift > 0:
            measured["egt"] += self.egt_sensor_drift * 40.0  # slow bias, not real heat
        return measured

    def active_fault_label(self) -> str:
        candidates = {
            "injector_degradation": self.injector_degradation,
            "cooling_degradation": self.cooling_efficiency_loss,
            "lubrication_degradation": self.lubrication_degradation,
            "vibration_increase": self.vibration_severity,
            "misfire": self.misfire_severity,
            "combustion_instability": self.combustion_instability,
            "egt_sensor_drift": self.egt_sensor_drift,
        }
        active = {k: v for k, v in candidates.items() if v > 0.05}
        if not active:
            return "none"
        return max(active, key=active.get)

    def true_health_index(self) -> float:
        """Ground-truth health index for offline evaluation (0-100)."""
        degradation = max(
            self.injector_degradation,
            self.cooling_efficiency_loss,
            self.lubrication_degradation,
            self.vibration_severity,
            self.misfire_severity,
            self.combustion_instability,
        )
        return max(0.0, 100.0 * (1 - degradation))


# ---- convenience factory functions for the 12 required test scenarios ----

def normal_scenario() -> FaultState:
    return FaultState()


def injector_degradation_scenario(start_s=2100, ramp_s=1800) -> FaultState:
    fs = FaultState()
    fs.schedule(FaultConfig("injector_degradation", start_s, ramp_s, max_severity=0.8))
    return fs


def overheating_scenario(start_s=1800, ramp_s=2400) -> FaultState:
    fs = FaultState()
    fs.schedule(FaultConfig("cooling_degradation", start_s, ramp_s, max_severity=0.9))
    return fs


def lubrication_loss_scenario(start_s=1500, ramp_s=1500) -> FaultState:
    fs = FaultState()
    fs.schedule(FaultConfig("lubrication_degradation", start_s, ramp_s, max_severity=0.85))
    return fs


def misfire_scenario(start_s=1000, ramp_s=600) -> FaultState:
    fs = FaultState()
    fs.schedule(FaultConfig("misfire", start_s, ramp_s, max_severity=0.7))
    return fs


def vibration_increase_scenario(start_s=2000, ramp_s=1200) -> FaultState:
    fs = FaultState()
    fs.schedule(FaultConfig("vibration_increase", start_s, ramp_s, max_severity=0.75))
    return fs


def egt_sensor_drift_scenario(start_s=1200, ramp_s=3000) -> FaultState:
    fs = FaultState()
    fs.schedule(FaultConfig("egt_sensor_drift", start_s, ramp_s, max_severity=0.6))
    return fs
