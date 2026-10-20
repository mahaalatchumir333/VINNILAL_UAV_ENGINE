"""SQLite persistence for validated telemetry and analytics snapshots."""

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = Path(
    os.environ.get("VINNILAL_DB_PATH", str(PROJECT_ROOT / "telemetry.db"))
)


@contextmanager
def _connect() -> Generator[sqlite3.Connection, None, None]:
    connection = sqlite3.connect(DATABASE_PATH, timeout=10.0)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_database() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS telemetry_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                engine_id TEXT NOT NULL,
                mission_id TEXT NOT NULL,
                telemetry_json TEXT NOT NULL,
                analytics_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_engine_id_id "
            "ON telemetry_history(engine_id, id DESC)"
        )


def record_tick(telemetry: dict[str, Any], analytics: dict[str, Any]) -> int:
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO telemetry_history
                (timestamp, engine_id, mission_id, telemetry_json, analytics_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                telemetry["timestamp"],
                telemetry["engine_id"],
                telemetry["mission_id"],
                json.dumps(telemetry, allow_nan=False),
                json.dumps(analytics, allow_nan=False),
            ),
        )
        return int(cursor.lastrowid)


def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "timestamp": row["timestamp"],
        "engine_id": row["engine_id"],
        "mission_id": row["mission_id"],
        "telemetry": json.loads(row["telemetry_json"]),
        "analytics": json.loads(row["analytics_json"]),
    }


def get_latest(engine_id: str) -> dict[str, Any] | None:
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT id, timestamp, engine_id, mission_id, telemetry_json, analytics_json
            FROM telemetry_history
            WHERE engine_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (engine_id,),
        ).fetchone()
    return _decode_row(row) if row else None


def get_history(
    engine_id: str,
    limit: int = 500,
    before_id: int | None = None,
) -> list[dict[str, Any]]:
    query = """
        SELECT id, timestamp, engine_id, mission_id, telemetry_json, analytics_json
        FROM telemetry_history
        WHERE engine_id = ?
    """
    parameters: list[Any] = [engine_id]
    if before_id is not None:
        query += " AND id < ?"
        parameters.append(before_id)
    query += " ORDER BY id DESC LIMIT ?"
    parameters.append(limit)

    with _connect() as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [_decode_row(row) for row in reversed(rows)]
