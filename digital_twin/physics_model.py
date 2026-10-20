# digital_twin/physics_model.py
from dataclasses import dataclass
import numpy as np

@dataclass
class TwinCoefficients:
    base_rpm: float = 3800.0
    base_egt: float = 560.0
    base_cht: float = 130.0
    base_oil_temp: float = 75.0
    base_oil_pressure: float = 4.5
    base_fuel_flow: float = 14.0
    base_vibration: float = 0.15
    base_manifold_press: float = 30.0 # kPa

    egt_load_factor: float = 180.0
    egt_ambient_factor: float = 0.9
    cht_load_factor: float = 50.0
    cht_ambient_factor: float = 0.6
    oil_load_factor: float = 25.0
    rpm_throttle_factor: float = 900.0
    fuel_flow_throttle_factor: float = 9.0

class DigitalTwin:
    """Predicts 4-cylinder engine expected baseline telemetry."""

    def __init__(self, coeffs: TwinCoefficients = None):
        self.c = coeffs or TwinCoefficients()

    def _density_factor(self, altitude_ft: float, ambient_temp_c: float) -> float:
        pressure = 101325 * (1 - 2.25577e-5 * altitude_ft * 0.3048) ** 5.25588
        temp_k = ambient_temp_c + 273.15
        density = pressure / (287.05 * temp_k)
        return max(density / 1.225, 0.3)

    def predict(self, throttle: float, altitude_ft: float, ambient_temp_c: float) -> dict:
        c = self.c
        density_factor = self._density_factor(altitude_ft, ambient_temp_c)

        expected_rpm = c.base_rpm + throttle * c.rpm_throttle_factor * density_factor
        expected_fuel_flow = c.base_fuel_flow + throttle * c.fuel_flow_throttle_factor
        expected_egt = c.base_egt + throttle * c.egt_load_factor + (ambient_temp_c - 25) * c.egt_ambient_factor
        expected_cht = c.base_cht + throttle * c.cht_load_factor + (ambient_temp_c - 25) * c.cht_ambient_factor
        expected_oil_temp = c.base_oil_temp + throttle * c.oil_load_factor
        expected_oil_pressure = c.base_oil_pressure
        expected_vibration = c.base_vibration
        expected_manifold_press = c.base_manifold_press + throttle * 25.0 * density_factor

        return {
            "rpm": expected_rpm,
            "fuel_flow": expected_fuel_flow,
            "egt": expected_egt,
            "cht": expected_cht,
            "egt1": expected_egt, "egt2": expected_egt - 2.0, "egt3": expected_egt + 3.0, "egt4": expected_egt + 1.0,
            "cht1": expected_cht, "cht2": expected_cht + 1.0, "cht3": expected_cht + 2.0, "cht4": expected_cht - 1.0,
            "oil_temperature": expected_oil_temp,
            "oil_pressure": expected_oil_pressure,
            "vibration": expected_vibration,
            "manifold_pressure": expected_manifold_press,
        }