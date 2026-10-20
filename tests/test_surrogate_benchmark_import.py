import csv
import tempfile
import unittest
from pathlib import Path

from analytics.import_surrogate_benchmarks import import_benchmarks


def cmapss_row(engine_id: int, cycle: int) -> str:
    values = [engine_id, cycle] + [0.0] * 24
    return " ".join(str(value) for value in values)


class SurrogateBenchmarkImportTests(unittest.TestCase):
    def test_import_keeps_cmapss_cycles_and_obd_relative_time_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cmapss_dir = root / "cmapss"
            cmapss_dir.mkdir()
            for subset in ("FD001", "FD002", "FD003", "FD004"):
                (cmapss_dir / f"train_{subset}.txt").write_text(
                    f"{cmapss_row(1, 1)}\n{cmapss_row(1, 2)}\n",
                    encoding="utf-8",
                )
                (cmapss_dir / f"test_{subset}.txt").write_text(
                    f"{cmapss_row(1, 1)}\n{cmapss_row(1, 2)}\n",
                    encoding="utf-8",
                )
                (cmapss_dir / f"RUL_{subset}.txt").write_text("5\n", encoding="utf-8")

            obd_csv = root / "obd.csv"
            with obd_csv.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(
                    target,
                    fieldnames=[
                        "timestamp",
                        "dataset_segment_id",
                        "RPM",
                        "COOLANT_TEMP",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "timestamp": "2025-01-01T00:00:00",
                            "dataset_segment_id": "source-a",
                            "RPM": "1000",
                            "COOLANT_TEMP": "",
                        },
                        {
                            "timestamp": "2025-01-01T00:00:00.5",
                            "dataset_segment_id": "source-a",
                            "RPM": "1200",
                            "COOLANT_TEMP": "80",
                        },
                        {
                            "timestamp": "2025-01-01T00:00:01.2",
                            "dataset_segment_id": "source-a",
                            "RPM": "1400",
                            "COOLANT_TEMP": "82",
                        },
                        {
                            "timestamp": "2025-02-01T00:00:00",
                            "dataset_segment_id": "source-b",
                            "RPM": "900",
                            "COOLANT_TEMP": "70",
                        },
                    ]
                )

            output_dir = root / "derived"
            manifest = import_benchmarks(cmapss_dir, obd_csv, output_dir)

            self.assertEqual(manifest["benchmarks"]["cmapss"]["rows"], 16)
            self.assertEqual(manifest["benchmarks"]["cmapss"]["time_unit"], "engine cycles (not seconds)")
            with (output_dir / "cmapss_cycle_summaries.csv").open(
                newline="", encoding="utf-8"
            ) as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(rows[0]["rul_cycles"], "1")
            self.assertEqual(rows[1]["rul_cycles"], "0")
            self.assertIn("sensor_21", rows[0])

            with (output_dir / "obd2_one_second_summaries.csv").open(
                newline="", encoding="utf-8"
            ) as source:
                obd_rows = list(csv.DictReader(source))
            self.assertEqual(len(obd_rows), 3)
            self.assertEqual(obd_rows[0]["rpm_mean"], "1100.0")
            self.assertEqual(obd_rows[-1]["segment"], "2")
            self.assertEqual(obd_rows[-1]["elapsed_second"], "0")
            self.assertNotIn("timestamp", obd_rows[0])

    def test_ims_waveforms_are_summarized_by_record_and_channel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cmapss_dir = root / "cmapss"
            cmapss_dir.mkdir()
            for subset in ("FD001", "FD002", "FD003", "FD004"):
                (cmapss_dir / f"train_{subset}.txt").write_text(
                    f"{cmapss_row(1, 1)}\n",
                    encoding="utf-8",
                )
                (cmapss_dir / f"test_{subset}.txt").write_text(
                    f"{cmapss_row(1, 1)}\n",
                    encoding="utf-8",
                )
                (cmapss_dir / f"RUL_{subset}.txt").write_text("5\n", encoding="utf-8")
            obd_csv = root / "obd.csv"
            obd_csv.write_text(
                "timestamp,segment_id,RPM\n2025-01-01T00:00:00,a,1000\n",
                encoding="utf-8",
            )
            ims_dir = root / "ims"
            for test_name in ("1st_test", "2nd_test", "3rd_test"):
                (ims_dir / test_name / test_name).mkdir(parents=True)
            first_test = ims_dir / "1st_test" / "1st_test"
            (first_test / "2003.10.22.12.00.00").write_text(
                "1 2\n3 4\n5 6\n",
                encoding="utf-8",
            )
            (first_test / "2003.10.22.12.05.00").write_text(
                "2 4\n4 6\n6 8\n",
                encoding="utf-8",
            )
            second_test = ims_dir / "2nd_test" / "2nd_test"
            (second_test / "2004.02.12.10.00.00").write_text(
                "7\n9\n",
                encoding="utf-8",
            )
            third_test = ims_dir / "3rd_test" / "3rd_test"
            (third_test / "2004.03.04.09.00.00").write_text(
                "10\n12\n",
                encoding="utf-8",
            )

            output_dir = root / "derived"
            manifest = import_benchmarks(cmapss_dir, obd_csv, output_dir, ims_dir)

            ims_manifest = manifest["benchmarks"]["ims_bearings"]
            self.assertEqual(ims_manifest["test_files"], 4)
            self.assertEqual(ims_manifest["rows"], 6)
            self.assertEqual(ims_manifest["channels_per_test"]["1st_test"], 2)
            with (output_dir / "ims_bearing_waveform_features.csv").open(
                newline="", encoding="utf-8"
            ) as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(rows[0]["mean"], "3.0")
            self.assertEqual(rows[0]["rms"], "3.41565026")
            self.assertEqual(rows[0]["peak_abs"], "5.0")
            self.assertEqual(rows[2]["elapsed_seconds"], "300")
            self.assertEqual(rows[-1]["test_id"], "3rd_test")
            self.assertNotIn("timestamp", rows[0])


if __name__ == "__main__":
    unittest.main()
