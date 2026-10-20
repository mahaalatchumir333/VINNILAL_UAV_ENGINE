import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.fail_safe_guard import assess_telemetry
from backend.main import app
from simulator.engine_simulator import EngineSimulator
from simulator.mission_profiles import make_normal_cruise_mission


def make_packet() -> dict:
    return EngineSimulator(seed=1).step(make_normal_cruise_mission())


class TelemetryQualityGateTests(unittest.TestCase):
    def test_valid_simulator_packet_passes_quality_gate(self):
        assessment = assess_telemetry(make_packet())

        self.assertEqual(assessment.quality, "good")
        self.assertIsNotNone(assessment.telemetry)
        self.assertGreater(assessment.telemetry["rpm"], 0)

    def test_missing_required_field_is_degraded_without_imputation(self):
        packet = make_packet()
        del packet["cht"]

        assessment = assess_telemetry(packet)

        self.assertEqual(assessment.quality, "degraded")
        self.assertIsNone(assessment.telemetry)
        self.assertTrue(any("cht" in issue for issue in assessment.issues))

    def test_non_finite_or_out_of_range_values_are_degraded(self):
        for rpm in (float("nan"), -1, 7001):
            with self.subTest(rpm=rpm):
                packet = make_packet()
                packet["rpm"] = rpm

                assessment = assess_telemetry(packet)

                self.assertEqual(assessment.quality, "degraded")
                self.assertIsNone(assessment.telemetry)
                self.assertTrue(assessment.issues)

    def test_corrupt_payload_is_rejected_without_clamping_or_imputation(self):
        packet = make_packet()
        packet.update({
            "rpm": -500.0,
            "cht": None,
            "egt": "CORRUPTED_VALUE",
            "oil_pressure": 99.0,
        })

        assessment = assess_telemetry(packet)

        self.assertEqual(assessment.quality, "degraded")
        self.assertIsNone(assessment.telemetry)
        self.assertTrue(any(issue.startswith("rpm:") for issue in assessment.issues))
        self.assertTrue(any(issue.startswith("cht:") for issue in assessment.issues))
        self.assertTrue(any(issue.startswith("egt:") for issue in assessment.issues))
        self.assertTrue(any(issue.startswith("oil_pressure:") for issue in assessment.issues))

    def test_cooling_fault_status_and_clear_reset(self):
        with TestClient(app) as client:
            injected = client.post(
                "/api/simulator/inject",
                json={"fault_type": "cooling_degradation", "severity": 0.8},
            )
            self.assertEqual(injected.status_code, 200)
            self.assertEqual(injected.json()["status"], "FAULT_INJECTED")
            self.assertEqual(
                client.get("/status").json()["active_fault"],
                "cooling_degradation",
            )

            cleared = client.post("/api/simulator/clear")
            self.assertEqual(
                cleared.json(),
                {"status": "FAULTS_CLEARED", "engine_state": "HEALTHY"},
            )
            simulator_status = client.get("/api/simulator/status").json()
            self.assertEqual(simulator_status["active_fault"], "none")
            self.assertEqual(simulator_status["true_health_index"], 100.0)

    def test_health_and_legacy_clear_routes_are_available(self):
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").json(), {"status": "ok"})
            self.assertEqual(client.get("/api/simulator/faults").status_code, 200)
            self.assertEqual(
                client.post("/clear").json()["status"],
                "FAULTS_CLEARED",
            )
            self.assertEqual(client.post("/api/simulator/clear").status_code, 200)

    def test_invalid_websocket_packet_is_reported_and_stream_continues(self):
        invalid_packet = make_packet()
        del invalid_packet["cht"]
        with patch(
            "backend.main.simulator.step",
            side_effect=[invalid_packet, make_packet()],
        ):
            with TestClient(app) as client:
                with client.websocket_connect("/ws/telemetry") as websocket:
                    degraded = websocket.receive_json()
                    valid = websocket.receive_json()

        self.assertEqual(degraded["analytics"]["telemetry_quality"], "degraded")
        self.assertIn("cht", degraded["analytics"]["evidence"][0])
        self.assertIn("rpm", valid["telemetry"])
