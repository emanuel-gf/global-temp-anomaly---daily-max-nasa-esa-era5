#!/usr/bin/env python3
"""
metar_poller.py — Polls METAR data every 15s until the observation
matches the nearest reference minute (20 or 50), then stores it in SQLite.

Usage:
    python metar_poller.py
    python metar_poller.py --station KJFK
    python metar_poller.py --station EGLC --interval 10
    python metar_poller.py --station EGLC --ref-minutes 20 50
    python metar_poller.py --station EGLC --db my_weather.db
    python metar_poller.py --station EGLC --output metar.csv   # also export CSV
"""

import asyncio
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from weather_db import WeatherDB

# ── CONFIG ────────────────────────────────────────────────────────────────────
DEFAULT_STATION   = "EGLC"
DEFAULT_INTERVAL  = 15
DEFAULT_DB        = "weather_data.db"
REFERENCE_MINUTES = np.array([20, 50])


# ── API ───────────────────────────────────────────────────────────────────────
def get_metar_data(station: str) -> list[dict]:
    response = requests.get(
        "https://aviationweather.gov/api/data/metar",
        params={"ids": station, "format": "json"},
    )
    response.raise_for_status()
    return response.json()


def make_df_metar(data: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(data)
    df["obsTime"] = df["obsTime"].apply(
        lambda x: datetime.fromtimestamp(int(x), tz=timezone.utc)
    )
    return df[["icaoId", "obsTime", "reportTime", "temp", "dewp", "wdir", "wspd", "qcField"]]


# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_correct_minute(ref_minutes: np.ndarray) -> int:
    return int(ref_minutes[np.argmin(np.abs(ref_minutes - datetime.now().minute))])


def parse_obstime(data: list[dict]) -> datetime:
    return datetime.fromtimestamp(int(data[0]["obsTime"]), tz=timezone.utc)


# ── ASYNC CORE ────────────────────────────────────────────────────────────────
async def fetch_data_nonblocking(station: str) -> list[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_metar_data, station)


async def wait_for_metar(station: str, interval: int, ref_minutes: np.ndarray) -> list[dict]:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Polling METAR for station: {station}")

    correct_min = get_correct_minute(ref_minutes)
    print(f"Target observation minute: {correct_min}")

    attempt = 1
    while True:
        print(f"\n[Attempt {attempt}] Fetching...")
        try:
            data = await fetch_data_nonblocking(station)
        except requests.RequestException as e:
            print(f"  ✗ Request failed: {e}. Retrying in {interval}s...")
            await asyncio.sleep(interval)
            attempt += 1
            continue

        obstime = parse_obstime(data)
        print(f"  obs time  : {obstime.strftime('%H:%M:%S UTC')}")
        print(f"  obs minute: {obstime.minute}  (want: {correct_min})")

        if obstime.minute == correct_min:
            print("  ✓ Correct minute reached!")
            return data

        print(f"  → Not ready. Retrying in {interval}s...")
        await asyncio.sleep(interval)
        attempt += 1


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(
        description="Poll METAR data and store observations in SQLite."
    )
    parser.add_argument("--station",     type=str, default=DEFAULT_STATION,
                        help=f"ICAO station ID (default: {DEFAULT_STATION})")
    parser.add_argument("--interval",    type=int, default=DEFAULT_INTERVAL,
                        help=f"Polling interval in seconds (default: {DEFAULT_INTERVAL})")
    parser.add_argument("--ref-minutes", type=int, nargs="+", default=list(REFERENCE_MINUTES),
                        help="Target observation minutes (default: 20 50)")
    parser.add_argument("--db",          type=str, default=DEFAULT_DB,
                        help=f"SQLite database path (default: {DEFAULT_DB})")
    parser.add_argument("--output",      type=str, default=None,
                        help="Also export results to a CSV file (optional)")
    args = parser.parse_args()

    ref_minutes = np.array(args.ref_minutes)

    # ── Poll until correct observation is available
    data = await wait_for_metar(args.station, args.interval, ref_minutes)

    # ── Store in SQLite
    db = WeatherDB(db_path=args.db)
    inserted, duplicates = db.store_many(data)
    print(f"\n── Stored to {args.db} ──────────────────────")
    print(f"  Inserted : {inserted}")
    print(f"  Duplicates skipped: {duplicates}")
    print(f"  Total rows for {args.station}: {db.count(args.station)}")

    # ── Display result
    df = make_df_metar(data)
    print("\n── Observation ──────────────────────────────")
    print(df.to_string(index=False))

    # ── Optionally export CSV
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\n✓ Also saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())