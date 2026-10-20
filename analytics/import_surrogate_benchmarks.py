"""Create provenance-preserving summaries from locally supplied benchmark files.

NASA C-MAPSS remains cycle-indexed; it is deliberately not mislabeled as 1 Hz.
Vehicle OBD-II samples are aggregated into relative one-second feature bins.
IMS Bearing waveforms are summarized per record and channel. None of these
benchmarks is mapped onto the UAV simulator's sensor labels.
"""

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
CMAPSS_SENSOR_COUNT = 21
CMAPSS_COLUMNS = (
    ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{index}" for index in range(1, CMAPSS_SENSOR_COUNT + 1)]
)
OBD_SIGNALS = (
    "RPM",
    "SPEED",
    "THROTTLE_POS",
    "RELATIVE_THROTTLE_POS",
    "ENGINE_LOAD",
    "ABSOLUTE_LOAD",
    "INTAKE_PRESSURE",
    "INTAKE_TEMP",
    "COOLANT_TEMP",
    "FUEL_USAGE_ML_MIN",
)


def _read_cmapss_row(line: str, source: Path, line_number: int) -> list[float]:
    try:
        row = [float(value) for value in line.split()]
    except ValueError as exc:
        raise ValueError(f"{source}:{line_number}: expected numeric C-MAPSS row") from exc
    if len(row) != 26:
        raise ValueError(
            f"{source}:{line_number}: expected 26 fields, found {len(row)}"
        )
    if not row[0].is_integer() or not row[1].is_integer():
        raise ValueError(f"{source}:{line_number}: engine and cycle must be integers")
    return row


def _cmapss_max_cycles(path: Path) -> dict[int, int]:
    max_cycles: dict[int, int] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = _read_cmapss_row(line, path, line_number)
            engine_id, cycle = int(row[0]), int(row[1])
            max_cycles[engine_id] = max(max_cycles.get(engine_id, 0), cycle)
    if not max_cycles:
        raise ValueError(f"No C-MAPSS rows found in {path}")
    return max_cycles


def import_cmapss(cmapss_dir: Path, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    subsets = ["FD001", "FD002", "FD003", "FD004"]
    with output_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=["subset", "split"] + CMAPSS_COLUMNS + ["rul_cycles"],
        )
        writer.writeheader()
        for subset in subsets:
            for split in ("train", "test"):
                source_path = cmapss_dir / f"{split}_{subset}.txt"
                if not source_path.is_file():
                    raise FileNotFoundError(f"Missing C-MAPSS file: {source_path}")
                max_cycles = _cmapss_max_cycles(source_path)
                test_rul: list[int] = []
                if split == "test":
                    rul_path = cmapss_dir / f"RUL_{subset}.txt"
                    if not rul_path.is_file():
                        raise FileNotFoundError(f"Missing C-MAPSS labels: {rul_path}")
                    test_rul = [
                        int(line.strip())
                        for line in rul_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                    if len(test_rul) != len(max_cycles):
                        raise ValueError(
                            f"{rul_path} has {len(test_rul)} labels for "
                            f"{len(max_cycles)} test engines"
                        )
                test_rul_by_engine = (
                    dict(zip(sorted(max_cycles), test_rul))
                    if split == "test"
                    else {}
                )

                with source_path.open(encoding="utf-8") as source:
                    for line_number, line in enumerate(source, 1):
                        if not line.strip():
                            continue
                        values = _read_cmapss_row(line, source_path, line_number)
                        engine_id, cycle = int(values[0]), int(values[1])
                        rul = (
                            max_cycles[engine_id] - cycle
                            if split == "train"
                            else test_rul_by_engine[engine_id] + max_cycles[engine_id] - cycle
                        )
                        writer.writerow(
                            {
                                "subset": subset,
                                "split": split,
                                **dict(zip(CMAPSS_COLUMNS[2:], values[2:])),
                                "engine_id": engine_id,
                                "cycle": cycle,
                                "rul_cycles": rul,
                            }
                        )
                        rows_written += 1
    return {
        "summary_file": output_path.name,
        "rows": rows_written,
        "time_unit": "engine cycles (not seconds)",
        "source_release": "C-MAPSS FD001-FD004",
        "source_citation": (
            "Saxena et al., Damage Propagation Modeling for Aircraft Engine "
            "Run-to-Failure Simulation, PHM 2008."
        ),
        "license_status": "Not specified in the supplied archive; verify source terms.",
    }


def _number_or_none(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def import_obd2(obd_csv: Path, output_path: Path) -> dict[str, Any]:
    first_timestamp: dict[str, datetime] = {}
    segment_numbers: dict[str, int] = {}
    aggregates: dict[tuple[str, int], dict[str, Any]] = {}
    required = {"timestamp", "RPM"}

    with obd_csv.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        headers = set(reader.fieldnames or [])
        if not required.issubset(headers):
            raise ValueError(
                f"{obd_csv} must contain columns: {', '.join(sorted(required))}"
            )
        segment_column = (
            "segment_id"
            if "segment_id" in headers
            else "dataset_segment_id"
            if "dataset_segment_id" in headers
            else None
        )
        if segment_column is None:
            raise ValueError(
                f"{obd_csv} must contain segment_id or dataset_segment_id"
            )
        signal_columns = [column for column in OBD_SIGNALS if column in headers]
        if "RPM" not in signal_columns:
            raise ValueError(f"{obd_csv} does not contain an RPM signal")

        for line_number, row in enumerate(reader, 2):
            try:
                timestamp = datetime.fromisoformat(row["timestamp"].strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{obd_csv}:{line_number}: invalid timestamp"
                ) from exc
            raw_segment = row.get(segment_column, "").strip() or "unsegmented"
            if raw_segment not in segment_numbers:
                segment_numbers[raw_segment] = len(segment_numbers) + 1
                first_timestamp[raw_segment] = timestamp
            segment = segment_numbers[raw_segment]
            elapsed_second = max(
                0,
                int((timestamp - first_timestamp[raw_segment]).total_seconds()),
            )
            key = (raw_segment, elapsed_second)
            accumulator = aggregates.setdefault(
                key,
                {
                    "segment": segment,
                    "elapsed_second": elapsed_second,
                    "sample_count": 0,
                    "sums": defaultdict(float),
                    "counts": defaultdict(int),
                },
            )
            accumulator["sample_count"] += 1
            for signal in signal_columns:
                value = _number_or_none(row.get(signal))
                if value is not None:
                    accumulator["sums"][signal] += value
                    accumulator["counts"][signal] += 1

    if not aggregates:
        raise ValueError(f"No OBD-II rows found in {obd_csv}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_fields = (
        ["segment", "elapsed_second", "sample_count"]
        + [f"{signal.lower()}_mean" for signal in signal_columns]
    )
    with output_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=output_fields)
        writer.writeheader()
        for key in sorted(
            aggregates,
            key=lambda item: (aggregates[item]["segment"], item[1]),
        ):
            accumulator = aggregates[key]
            row: dict[str, int | float] = {
                "segment": accumulator["segment"],
                "elapsed_second": accumulator["elapsed_second"],
                "sample_count": accumulator["sample_count"],
            }
            for signal in signal_columns:
                count = accumulator["counts"][signal]
                if count:
                    row[f"{signal.lower()}_mean"] = round(
                        accumulator["sums"][signal] / count,
                        4,
                    )
            writer.writerow(row)

    return {
        "summary_file": output_path.name,
        "rows": len(aggregates),
        "source_rows": sum(item["sample_count"] for item in aggregates.values()),
        "time_unit": "relative seconds within source segment",
        "signals": signal_columns,
        "license_status": "User-provided source; verify the original dataset terms.",
        "privacy": "Absolute timestamps and source segment identifiers are omitted.",
    }


def _summarize_ims_waveform(
    source_path: Path,
) -> tuple[int, int, list[float], list[float], list[float], list[float], list[float | None]]:
    contents = source_path.read_text(encoding="utf-8")
    first_row = contents.splitlines()[0].split() if contents else []
    if not first_row:
        raise ValueError(f"Empty IMS waveform: {source_path}")
    channel_count = len(first_row)
    values = np.fromstring(contents, dtype=np.float64, sep=" ")
    line_count = contents.count("\n") + int(not contents.endswith("\n"))
    if line_count == 0 or values.size != line_count * channel_count:
        raise ValueError(f"Invalid numeric IMS waveform in {source_path}")
    waveform = values.reshape(line_count, channel_count)
    if not np.isfinite(waveform).all():
        raise ValueError(f"Non-finite IMS samples in {source_path}")

    means = np.mean(waveform, axis=0)
    centered = waveform - means
    variances = np.mean(centered**2, axis=0)
    stds = np.sqrt(variances)
    nonconstant = stds > 0
    kurtoses: list[float | None] = [None] * channel_count
    kurtosis_values = (
        np.mean(centered[:, nonconstant] ** 4, axis=0)
        / variances[nonconstant] ** 2
        - 3
    )
    for index, value in zip(np.flatnonzero(nonconstant), kurtosis_values):
        kurtoses[int(index)] = float(value)
    rms_values = np.sqrt(np.mean(waveform**2, axis=0))
    peak_values = np.max(np.abs(waveform), axis=0)
    return (
        line_count,
        channel_count,
        means.tolist(),
        rms_values.tolist(),
        stds.tolist(),
        peak_values.tolist(),
        kurtoses,
    )


def import_ims_bearings(ims_dir: Path, output_path: Path) -> dict[str, Any]:
    test_names = ("1st_test", "2nd_test", "3rd_test")
    file_pattern = re.compile(r"\d{4}(?:\.\d{2}){5}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    file_count = 0
    channels_per_test: dict[str, int] = {}
    with output_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=[
                "test_id",
                "record_index",
                "elapsed_seconds",
                "channel_index",
                "sample_count",
                "mean",
                "rms",
                "std",
                "peak_abs",
                "excess_kurtosis",
            ],
        )
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as executor:
            for test_name in test_names:
                test_dir = ims_dir / test_name
                if not test_dir.is_dir():
                    raise FileNotFoundError(f"Missing IMS test directory: {test_dir}")
                files = sorted(
                    path
                    for path in test_dir.rglob("*")
                    if path.is_file() and file_pattern.fullmatch(path.name)
                )
                if not files:
                    raise ValueError(
                        f"No timestamped IMS waveform files found in {test_dir}"
                    )

                first_timestamp: datetime | None = None
                channel_count: int | None = None
                sample_count: int | None = None
                summaries = executor.map(_summarize_ims_waveform, files, chunksize=8)
                for record_index, (source_path, summary) in enumerate(
                    zip(files, summaries)
                ):
                    try:
                        timestamp = datetime.strptime(
                            source_path.name,
                            "%Y.%m.%d.%H.%M.%S",
                        )
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid IMS timestamp in {source_path}"
                        ) from exc
                    if first_timestamp is None:
                        first_timestamp = timestamp
                    (
                        record_sample_count,
                        record_channel_count,
                        means,
                        rms_values,
                        stds,
                        peak_values,
                        kurtoses,
                    ) = summary
                    if channel_count is None:
                        channel_count = record_channel_count
                        sample_count = record_sample_count
                    elif (record_sample_count, record_channel_count) != (
                        sample_count,
                        channel_count,
                    ):
                        raise ValueError(
                            f"Inconsistent IMS waveform shape in {source_path}: "
                            f"expected {(sample_count, channel_count)}, found "
                            f"{(record_sample_count, record_channel_count)}"
                        )

                    elapsed_seconds = int((timestamp - first_timestamp).total_seconds())
                    for channel_index in range(channel_count):
                        kurtosis = kurtoses[channel_index]
                        writer.writerow(
                            {
                                "test_id": test_name,
                                "record_index": record_index,
                                "elapsed_seconds": elapsed_seconds,
                                "channel_index": channel_index + 1,
                                "sample_count": sample_count,
                                "mean": round(means[channel_index], 8),
                                "rms": round(rms_values[channel_index], 8),
                                "std": round(stds[channel_index], 8),
                                "peak_abs": round(peak_values[channel_index], 8),
                                "excess_kurtosis": (
                                    round(kurtosis, 8) if kurtosis is not None else ""
                                ),
                            }
                        )
                        rows_written += 1
                    file_count += 1
                channels_per_test[test_name] = channel_count or 0

    return {
        "summary_file": output_path.name,
        "source_release": "IMS Center Bearing Dataset (user-provided archive)",
        "test_files": file_count,
        "rows": rows_written,
        "channels_per_test": channels_per_test,
        "time_unit": "seconds from first filename timestamp within each test",
        "features": ["mean", "rms", "std", "peak_abs", "excess_kurtosis"],
        "labels": "No fault or RUL labels are generated.",
        "license_status": "Not specified in the supplied archive; verify source terms.",
        "privacy": "Absolute timestamps and source file names are omitted.",
    }


def import_benchmarks(
    cmapss_dir: Path,
    obd_csv: Path,
    output_dir: Path,
    ims_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "notice": (
            "These are separate surrogate benchmark domains, not UAV engine "
            "telemetry. Sensor labels and units are not interchangeable."
        ),
        "benchmarks": {
            "cmapss": import_cmapss(
                cmapss_dir,
                output_dir / "cmapss_cycle_summaries.csv",
            ),
            "obd2": import_obd2(
                obd_csv,
                output_dir / "obd2_one_second_summaries.csv",
            ),
        },
    }
    if ims_dir is not None:
        manifest["benchmarks"]["ims_bearings"] = import_ims_bearings(
            ims_dir,
            output_dir / "ims_bearing_waveform_features.csv",
        )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cmapss-dir",
        required=True,
        type=Path,
        help="Directory containing train/test_FD001-FD004.txt and RUL labels",
    )
    parser.add_argument(
        "--obd-csv",
        required=True,
        type=Path,
        help="Locally supplied OBD-II CSV with timestamps and engine signals",
    )
    parser.add_argument(
        "--ims-dir",
        type=Path,
        help="Optional IMS Bearing Dataset folder containing 1st_test, 2nd_test, and 3rd_test",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "external",
    )
    args = parser.parse_args()
    manifest = import_benchmarks(
        args.cmapss_dir,
        args.obd_csv,
        args.output_dir,
        args.ims_dir,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
