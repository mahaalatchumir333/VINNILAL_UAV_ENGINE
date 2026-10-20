import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app


class MissionRiskAndHistoryTests(unittest.TestCase):
    def test_mission_profiles_compare_scenarios_and_use_current_health(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "test.db"
            with patch("backend.database.DATABASE_PATH", database_path):
                with TestClient(app) as client:
                    profiles = client.get("/api/mission/profiles")
                    self.assertEqual(profiles.status_code, 200)
                    self.assertEqual(len(profiles.json()["profiles"]), 3)

                    short = client.post(
                        "/api/mission/simulate",
                        json={
                            "mission_type": "short_cruise",
                            "starting_health_index": 100,
                        },
                    )
                    hot = client.post(
                        "/api/mission/simulate",
                        json={
                            "mission_type": "hot_weather_endurance",
                            "starting_health_index": 40,
                        },
                    )
                    self.assertEqual(short.status_code, 200, short.text)
                    self.assertEqual(hot.status_code, 200, hot.text)
                    self.assertEqual(hot.json()["starting_health_index"], 40)
                    self.assertEqual(hot.json()["mission_risk"], "medium_high")
                    self.assertLess(
                        hot.json()["predicted_health_index_end"],
                        hot.json()["starting_health_index"],
                    )
                    self.assertGreater(
                        hot.json()["risk_score"],
                        short.json()["risk_score"],
                    )

    def test_websocket_ticks_are_persisted_and_available_for_scrubbing(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "test.db"
            with patch("backend.database.DATABASE_PATH", database_path):
                with TestClient(app) as client:
                    with client.websocket_connect("/ws/telemetry") as websocket:
                        websocket.receive_json()
                        second = websocket.receive_json()

                    history = client.get(
                        "/api/engine/ENG-01/history?limit=10"
                    )
                    self.assertEqual(history.status_code, 200)
                    self.assertGreaterEqual(len(history.json()), 2)
                    self.assertLess(
                        history.json()[-2]["id"],
                        history.json()[-1]["id"],
                    )
                    self.assertEqual(
                        history.json()[-1]["telemetry"]["rpm"],
                        second["telemetry"]["rpm"],
                    )

                    latest = client.get("/api/engine/ENG-01/latest")
                    self.assertEqual(latest.status_code, 200)
                    self.assertEqual(
                        latest.json()["id"],
                        history.json()[-1]["id"],
                    )

    def test_history_limit_and_empty_engine_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "test.db"
            with patch("backend.database.DATABASE_PATH", database_path):
                with TestClient(app) as client:
                    self.assertEqual(
                        client.get("/api/engine/ENG-01/history?limit=5001").status_code,
                        422,
                    )
                    self.assertEqual(
                        client.get("/api/engine/NO-SUCH-ENGINE/latest").status_code,
                        404,
                    )


if __name__ == "__main__":
    unittest.main()
