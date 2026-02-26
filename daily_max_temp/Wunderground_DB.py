"""
weather_db.py — Lightweight SQLite storage for METAR observations.

Responsibilities:
  - Create and manage the DB schema
  - Store observations received from the poller
  - Query stored data

No API calls are made here; fetching is handled by metar_poller.py.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class WeatherDB:
    def __init__(self, db_path: str = "weather_data.db"):
        self.db_path = Path(db_path)
        self._init_db()

    # ── SETUP ─────────────────────────────────────────────────────────────────
    def _init_db(self):
        """Create table and indexes if they don't exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS weather_observations (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    icaoId      TEXT    NOT NULL,
                    obsTime     TEXT    NOT NULL,
                    reportTime  TEXT,
                    temp        REAL,
                    dewp        REAL,
                    wdir        INTEGER,
                    wspd        INTEGER,
                    qcField     INTEGER,
                    created_at  TEXT    DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(icaoId, obsTime)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_icao_obstime
                ON weather_observations(icaoId, obsTime)
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row   # rows behave like dicts
        return conn

    # ── WRITE ─────────────────────────────────────────────────────────────────
    def store_observation(self, raw: dict) -> bool:
        """
        Store a single raw API observation dict.
        Converts obsTime from Unix timestamp to ISO-8601 string.
        Returns True if inserted, False if it was a duplicate.
        """
        obs_time_iso = datetime.fromtimestamp(
            int(raw["obsTime"]), tz=timezone.utc
        ).isoformat()

        try:
            with self._connect() as conn:
                cursor = conn.execute("""
                    INSERT OR IGNORE INTO weather_observations
                        (icaoId, obsTime, reportTime, temp, dewp, wdir, wspd, qcField)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    raw.get("icaoId"),
                    obs_time_iso,
                    raw.get("reportTime"),
                    raw.get("temp"),
                    raw.get("dewp"),
                    raw.get("wdir"),
                    raw.get("wspd"),
                    raw.get("qcField"),
                ))
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"  ✗ DB error: {e}")
            return False

    def store_many(self, raw_list: list[dict]) -> tuple[int, int]:
        """
        Store a list of raw API observations.
        Returns (inserted_count, duplicate_count).
        """
        inserted = duplicates = 0
        for obs in raw_list:
            if self.store_observation(obs):
                inserted += 1
            else:
                duplicates += 1
        return inserted, duplicates

    # ── READ ──────────────────────────────────────────────────────────────────
    def get_latest(self, icao_id: str, limit: int = 10) -> list[sqlite3.Row]:
        """Return the most recent observations for a station."""
        with self._connect() as conn:
            return conn.execute("""
                SELECT * FROM weather_observations
                WHERE icaoId = ?
                ORDER BY obsTime DESC
                LIMIT ?
            """, (icao_id, limit)).fetchall()

    def get_between(self, icao_id: str, start: str, end: str) -> list[sqlite3.Row]:
        """
        Return observations between two ISO-8601 timestamps (inclusive).
        Example: db.get_between("EGLC", "2024-01-01T00:00:00+00:00", "2024-01-02T00:00:00+00:00")
        """
        with self._connect() as conn:
            return conn.execute("""
                SELECT * FROM weather_observations
                WHERE icaoId = ?
                  AND obsTime BETWEEN ? AND ?
                ORDER BY obsTime ASC
            """, (icao_id, start, end)).fetchall()

    def count(self, icao_id: str | None = None) -> int:
        """Total number of stored observations, optionally filtered by station."""
        with self._connect() as conn:
            if icao_id:
                row = conn.execute(
                    "SELECT COUNT(*) FROM weather_observations WHERE icaoId = ?",
                    (icao_id,)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM weather_observations"
                ).fetchone()
            return row[0]