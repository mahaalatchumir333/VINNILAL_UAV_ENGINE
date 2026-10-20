import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from analytics.fault_classifier import FAULT_CODES, classify_fault
from analytics.anomaly_detector import AnomalyDetector
from analytics.health_index import SUBSYSTEM_WEIGHTS, compute_health_index, explain_health_index
from analytics.feature_engineering import FeatureWindow, extract_features
from digital_twin.physics_model import DigitalTwin
from digital_twin.residual_engine import ResidualEngine
from backend.main import app
import backend.database as database
from simulator.engine_simulator import EngineSimulator
from simulator.mission_profiles import make_normal_cruise_mission


def make_features() -> dict[str, float]:
    return {
        "egt_slope": 0.01,
        "cht_slope": 0.03,
        "oil_pressure_slope": 0.0,
        "oil_temperature_slope": 0.001,
        "vibration_rms": 0.1,
        "rpm_fluctuation": 0.1,
        "fuel_flow_per_rpm": 0.01,
        "oil_pressure_margin": 4.0,
        "egt_residual": 1.2,
        "cht_residual": 0.3,
        "oil_pressure_residual": -0.2,
        "oil_temperature_residual": 0.1,
        "vibration_residual": 0.1,
        "fuel_flow_residual": 0.2,
        "rpm_residual": 0.1,
    }


class FaultExplanationTests(unittest.TestCase):
    def test_classifier_exposes_triggered_and_not_triggered_rule_trace(self):
        diagnosis = classify_fault(make_features())

        self.assertEqual(
            diagnosis.likely_fault,
            "overheating_cooling_degradation",
        )
        self.assertEqual(diagnosis.vote_score, 1.5)
        self.assertGreater(
            diagnosis.vote_scores["overheating_cooling_degradation"],
            diagnosis.vote_scores["injector_degradation"],
        )
        self.assertTrue(any(
            rule["triggered"] and rule["rule"] == "cht_rising"
            for rule in diagnosis.rule_trace
        ))
        self.assertTrue(any(
            not rule["triggered"] and rule["rule"] == "oil_pressure_residual_low"
            for rule in diagnosis.rule_trace
        ))
        self.assertTrue(all("conditions" in rule for rule in diagnosis.rule_trace))

    def test_cooling_injection_label_uses_cht_residual_not_noisy_slope(self):
        simulator = EngineSimulator(seed=108)
        simulator.fault.cooling_efficiency_loss = 0.8
        residual_engine = ResidualEngine(DigitalTwin())
        window = FeatureWindow(maxlen=30)

        for tick in range(1, 13):
            packet = simulator.step(make_normal_cruise_mission())
            residuals = residual_engine.compute(packet)
            window.push(tick, packet, residuals.normalized)
            diagnosis = classify_fault(extract_features(window))

            self.assertEqual(
                diagnosis.likely_fault,
                "overheating_cooling_degradation",
                f"cooling diagnosis was lost on tick {tick}",
            )

        rule = next(
            entry for entry in diagnosis.rule_trace
            if entry["rule"] == "cht_residual_high"
        )
        self.assertTrue(rule["triggered"])
        self.assertGreater(rule["observed"], 3.0)
        self.assertEqual(
            FAULT_CODES[diagnosis.likely_fault],
            "COOLING_SYSTEM_DEGRADATION",
        )


class AnomalyScoringTests(unittest.TestCase):
    def test_absent_isolation_forest_uses_physics_score_at_full_weight(self):
        detector = AnomalyDetector()
        features = make_features()

        result = detector.score(features)

        self.assertEqual(detector.effective_weights, (1.0, 0.0))
        self.assertEqual(result.ml_score, 0.0)
        self.assertEqual(result.combined_score, result.physics_score)


class HealthExplanationTests(unittest.TestCase):
    def test_health_explanation_reconciles_weights_and_displayed_result(self):
        features = make_features()
        result = compute_health_index(features)
        explanation = explain_health_index(features)

        self.assertAlmostEqual(sum(SUBSYSTEM_WEIGHTS.values()), 1.0)
        self.assertEqual(
            explanation["overall_health_index"],
            result.overall,
        )
        self.assertAlmostEqual(
            sum(
                component["weighted_points"]
                for component in explanation["components"].values()
            ),
            result.overall,
            places=1,
        )
        self.assertIn(
            "egt_residual_sigma",
            explanation["components"]["thermal"]["drivers"],
        )


class LiveExplainabilityTests(unittest.TestCase):
    def test_websocket_streams_detailed_explanation_without_expanding_history_rows(self):
        packet = EngineSimulator(seed=19).step(make_normal_cruise_mission())
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "explainability.db"
            with patch.object(database, "DATABASE_PATH", database_path):
                with patch("backend.main.simulator.step", return_value=packet):
                    with TestClient(app) as client:
                        with client.websocket_connect("/ws/telemetry") as websocket:
                            payload = websocket.receive_json()

                explanation = payload["analytics"]["explainability"]
                self.assertEqual(len(explanation["sensor_comparisons"]), 7)
                self.assertIn("physics_residual_summary", explanation["anomaly"])
                self.assertIn("candidate_vote_scores", explanation["fault_diagnosis"])
                self.assertIn("rules", explanation["fault_diagnosis"])
                self.assertEqual(
                    explanation["fault_diagnosis"]["selected_fault_code"],
                    FAULT_CODES[explanation["fault_diagnosis"]["selected_fault"]],
                )
                self.assertEqual(
                    explanation["anomaly"]["scoring_mode"],
                    "physics_only_fallback",
                )
                self.assertEqual(explanation["anomaly"]["physics_weight"], 1.0)
                self.assertEqual(explanation["anomaly"]["ml_weight"], 0.0)
                self.assertIn("components", explanation["health_index"])
                self.assertIn("assumption", explanation["remaining_useful_life"])

                saved = database.get_latest(packet["engine_id"])
                self.assertIsNotNone(saved)
                self.assertNotIn("explainability", saved["analytics"])


if __name__ == "__main__":
    unittest.main()
