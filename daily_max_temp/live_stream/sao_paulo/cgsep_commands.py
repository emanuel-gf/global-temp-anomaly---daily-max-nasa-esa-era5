"""
CGESP CLI Commands
==================
Drop these commands into your existing cli.py.

Usage
-----
  # At the top of cli.py, add:
  from daily_max_temp.observations.cgesp_commands import register_cgesp_commands
  register_cgesp_commands(cli)

  # Or paste the @cli.command blocks directly into cli.py.

Expected cli_locations.json structure
--------------------------------------
{
  "sao_paulo": {
    "lat": -23.55,
    "lon": -46.63,
    "timezone": "America/Sao_Paulo",
    "cgesp_stations": {
      "butanta":  { "channel_id": "1000887", "posto_id": "591" },
      "santana":  { "channel_id": "1000462", "posto_id": "592" }
    }
  }
}

Note: Each station entry now carries BOTH channel_id (for the 24h table)
and posto_id (for the current-summary page).  If your existing config only
has bare string IDs, the fetcher falls back gracefully.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import pandas as pd

from daily_max_temp.live_stream.sao_paulo.cgesp_fetcher import (
    fetch_all_stations_history,
    fetch_all_stations_summary,
    fetch_station_history,
    fetch_station_summary,
    CGESPError,
)


# ──────────────────────────────────────────────────────────────────────────────
# Config helper  (mirrors existing load_city pattern)
# ──────────────────────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent.parent.parent / "cli_locations.json"


def _load_city(city: str) -> dict:
    try:
        with open(_CONFIG_PATH) as f:
            config = json.load(f)
    except FileNotFoundError:
        click.echo(f"❌ Config not found: {_CONFIG_PATH}")
        sys.exit(1)

    if city not in config:
        available = ", ".join(sorted(config.keys()))
        click.echo(f"❌ City '{city}' not in config. Available: {available}")
        sys.exit(1)

    return config[city]


def _station_maps(loc: dict, city: str) -> tuple[dict[str, str], dict[str, str]]:
    """
    Returns (channel_map, posto_map) for a city config.

    Handles two config formats:
      • New:  { "butanta": { "channel_id": "...", "posto_id": "..." } }
      • Old:  { "butanta": "1000887" }   (only history, no summary)
    """
    raw = loc.get("cgesp_stations")
    if not raw:
        click.echo(f"❌ No 'cgesp_stations' configured for '{city}'")
        sys.exit(1)

    channel_map: dict[str, str] = {}
    posto_map:   dict[str, str] = {}

    for name, val in raw.items():
        if isinstance(val, dict):
            if cid := val.get("channel_id"):
                channel_map[name] = cid
            if pid := val.get("posto_id"):
                posto_map[name] = pid
        else:
            # Legacy: bare string = channel_id only
            channel_map[name] = str(val)

    return channel_map, posto_map


# ──────────────────────────────────────────────────────────────────────────────
# Command: fetch-observed
# ──────────────────────────────────────────────────────────────────────────────

@click.command("fetch-observed")
@click.argument("city")
@click.option("--station",  default=None,  help="Single station name to fetch")
@click.option("--output",   default=None,  help="Save history DataFrame to this .parquet path")
@click.option("--dry-run",  is_flag=True,  help="Print config and exit without fetching")
@click.option("--tail",     default=5,     show_default=True, help="Rows to preview")
@click.option("--tz",       default=None,  help="Convert timestamps to this timezone (e.g. America/Sao_Paulo)")
def fetch_observed(city, station, output, dry_run, tail, tz):
    """
    Fetch 24h hourly observed data from CGESP (with ↑/↓ temperature trend).

    \b
    Examples:
      python cli.py fetch-observed sao_paulo
      python cli.py fetch-observed sao_paulo --station butanta
      python cli.py fetch-observed sao_paulo --output observed.parquet
    """
    loc = _load_city(city)
    channel_map, _ = _station_maps(loc, city)

    click.echo(f"▶ fetch-observed  city={city}")

    if dry_run:
        click.echo(f"  stations (channel_id): {json.dumps(channel_map, indent=4)}")
        return

    try:
        if station:
            if station not in channel_map:
                click.echo(f"❌ Unknown station '{station}'. Available: {list(channel_map)}")
                sys.exit(1)
            df = fetch_station_history(channel_map[station], station)
        else:
            df = fetch_all_stations_history(channel_map)

    except CGESPError as exc:
        click.echo(f"❌ Fetch failed: {exc}")
        sys.exit(1)

    # Optional timezone conversion
    if tz and "timestamp" in df.columns:
        df["timestamp"] = (
            pd.to_datetime(df["timestamp"])
              .dt.tz_localize("UTC", ambiguous="infer", nonexistent="shift_forward")
              .dt.tz_convert(tz)
        )

    _print_summary(df, tail)

    if output:
        df.to_parquet(output, index=False)
        click.echo(f"✅ Saved history → {output}  ({len(df)} rows)")


# ──────────────────────────────────────────────────────────────────────────────
# Command: fetch-summary
# ──────────────────────────────────────────────────────────────────────────────

@click.command("fetch-summary")
@click.argument("city")
@click.option("--station",  default=None,  help="Single station name")
@click.option("--output",   default=None,  help="Save summary DataFrame to this .parquet path")
@click.option("--dry-run",  is_flag=True)
def fetch_summary(city, station, output, dry_run):
    """
    Fetch current-conditions summary from CGESP (temp max/min/actual, etc.).

    \b
    Examples:
      python cli.py fetch-summary sao_paulo
      python cli.py fetch-summary sao_paulo --station santana
    """
    loc = _load_city(city)
    _, posto_map = _station_maps(loc, city)

    if not posto_map:
        click.echo("⚠ No 'posto_id' values configured — cannot fetch summaries.")
        click.echo("  Add posto_id to each station in cli_locations.json.")
        sys.exit(1)

    click.echo(f"▶ fetch-summary  city={city}")

    if dry_run:
        click.echo(f"  stations (posto_id): {json.dumps(posto_map, indent=4)}")
        return

    try:
        if station:
            if station not in posto_map:
                click.echo(f"❌ Unknown station '{station}'. Available: {list(posto_map)}")
                sys.exit(1)
            row  = fetch_station_summary(posto_map[station], station)
            df   = pd.DataFrame([row])
        else:
            df = fetch_all_stations_summary(posto_map)

    except CGESPError as exc:
        click.echo(f"❌ Fetch failed: {exc}")
        sys.exit(1)

    click.echo(df.T.to_string(header=False))

    if output:
        df.to_parquet(output, index=False)
        click.echo(f"✅ Saved summary → {output}")


# ──────────────────────────────────────────────────────────────────────────────
# Command: observed-latest  (convenience: latest reading per station)
# ──────────────────────────────────────────────────────────────────────────────

@click.command("observed-latest")
@click.argument("city")
@click.option("--output", default=None, help="Optional .parquet output")
def observed_latest(city, output):
    """
    Show only the most-recent observation per station.

    \b
    Example:
      python cli.py observed-latest sao_paulo
    """
    loc = _load_city(city)
    channel_map, _ = _station_maps(loc, city)

    click.echo(f"▶ observed-latest  city={city}")

    try:
        df = fetch_all_stations_history(channel_map)
    except CGESPError as exc:
        click.echo(f"❌ {exc}")
        sys.exit(1)

    latest = (
        df.sort_values("timestamp")
          .groupby("station", as_index=False)
          .last()
    )

    click.echo(latest.to_string(index=False))

    if output:
        latest.to_parquet(output, index=False)
        click.echo(f"✅ Saved → {output}")


# ──────────────────────────────────────────────────────────────────────────────
# Registration helper  (call from cli.py)
# ──────────────────────────────────────────────────────────────────────────────

def register_cgesp_commands(cli_group: click.Group) -> None:
    """Register all CGESP commands on an existing Click group."""
    cli_group.add_command(fetch_observed)
    cli_group.add_command(fetch_summary)
    cli_group.add_command(observed_latest)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(df: pd.DataFrame, tail: int) -> None:
    """Pretty-print a tail of the DataFrame with trend arrows."""
    sub = df.tail(tail).copy()

    if "temp_trend" in sub.columns:
        sub["temp_trend"] = sub["temp_trend"].map(
            {"up": "↑", "down": "↓", None: " "}
        ).fillna(" ")

    click.echo(sub.to_string(index=False))
    click.echo(f"\n  {len(df)} total rows  |  stations: {df['station'].unique().tolist()}")