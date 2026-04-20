import subprocess
import json
import click
from daily_max_temp.live_stream.sao_paulo.cgesp_fetcher import (
    fetch_station_history,
    fetch_all_stations_history,
    fetch_station_summary,
    fetch_all_stations_summary,
    CGESPError,
)

CONFIG_PATH = "cli_locations.json"

def load_city(city):
    """Load city config, exit cleanly if not found."""
    with open(CONFIG_PATH) as f:
        locations = json.load(f)
    if city not in locations:
        available = ", ".join(locations.keys())
        click.echo(f"❌ City '{city}' not found. Available: {available}")
        raise SystemExit(1)
    return locations[city]


def _station_maps(loc: dict, city: str) -> tuple[dict, dict]:
    """
    Extract (channel_map, posto_map) from a city config entry.

    Supports two formats:
      New → "butanta": { "channel_id": "1000887", "posto_id": "591" }
      Old → "butanta": "1000887"   (channel_id only, no summary)
    """
    raw = loc.get("cgesp_stations")
    if not raw:
        click.echo(f"❌ No 'cgesp_stations' configured for '{city}'")
        raise SystemExit(1)

    channel_map, posto_map = {}, {}
    for name, val in raw.items():
        if isinstance(val, dict):
            if cid := val.get("channel_id"):
                channel_map[name] = cid
            if pid := val.get("posto_id"):
                posto_map[name] = pid
        else:
            channel_map[name] = str(val)

    return channel_map, posto_map


# --- CLI group ---
@click.group()
def cli():
    """Weather Market pipeline."""
    pass


# --- fetch-ensemble ---
@cli.command("fetch-ensemble")
@click.argument("city")
@click.option("--dry-run", is_flag=True)
def fetch_ensemble(city, dry_run):
    """Fetch ensemble forecast for a city."""
    loc = load_city(city)
    cmd = [
        "python", "./daily_max_temp/forecast/forecast_ensemble.py",
        "--lat",      str(loc["lat"]),
        "--lon",      str(loc["lon"]),
        "--timezone", loc["timezone"],
        "--city",     city,
        "--root-dir", loc["ensemble_dir"],
    ]
    click.echo(f"▶ fetch-ensemble {city}")
    if dry_run:
        click.echo("  [dry-run] " + " ".join(cmd))
    else:
        subprocess.run(cmd, check=True)


# --- fetch-determ ---
@cli.command("fetch-determ")
@click.argument("city")
@click.option("--dry-run", is_flag=True)
def fetch_determ(city, dry_run):
    """Fetch deterministic forecast for a city."""
    loc = load_city(city)
    cmd = [
        "python", "./daily_max_temp/forecast/forecast_deterministic.py",
        "--lat",  str(loc["lat"]),
        "--lon",  str(loc["lon"]),
        "--time", loc["timezone"],
        "--city", city,
        "--root-dir", loc["deterministic_dir"],
    ]
    click.echo(f"▶ fetch-determ {city}")
    if dry_run:
        click.echo("  [dry-run] " + " ".join(cmd))
    else:
        subprocess.run(cmd, check=True)


# --- report ---
@cli.command("report")
@click.argument("city")
@click.argument("day")
@click.option("--temprange", default=None, help="Override temprange, e.g. 10-18")
@click.option("--dry-run", is_flag=True)
def report(city, day, temprange, dry_run):
    """Generate a forecast report for a city and date."""
    loc = load_city(city)
    cmd = [
        "python", "daily_max_temp/forecast/report_forecast.py",
        "--day",       day,
        "--temprange", temprange or loc["temprange"],
        "--city", city
    ]
    click.echo(f"▶ report {city} {day}")
    if dry_run:
        click.echo("  [dry-run] " + " ".join(cmd))
    else:
        subprocess.run(cmd, check=True)


# --- live-stream ---
@cli.command("live-stream")
@click.argument("city")
@click.option("--json", "as_json", is_flag=True, help="Export output as JSON")
@click.option("--dry-run", is_flag=True)
def live_stream(city, as_json, dry_run):
    """Print current METAR for a city's associated ICAO station."""
    loc = load_city(city)

    if "icao" not in loc:
        click.echo(f"❌ No ICAO code defined for '{city}' in locations.json")
        raise SystemExit(1)

    icao = loc["icao"]
    tz = loc["timezone"]
    cmd = ["python", "./daily_max_temp/live_stream/metar_fetcher_live.py", "-s", icao, "-tz", tz]

    if as_json:
        cmd.append("--json")

    click.echo(f"▶ live-stream {city} → {icao}")
    if dry_run:
        click.echo("  [dry-run] " + " ".join(cmd))
    else:
        subprocess.run(cmd, check=True)


# --- fetch-both ---
@cli.command("fetch-both")
@click.argument("city")
@click.option("--dry-run", is_flag=True)
@click.pass_context
def fetch_all(ctx, city, dry_run):
    """Run both fetch-ensemble and fetch-determ for a city."""
    ctx.invoke(fetch_ensemble, city=city, dry_run=dry_run)
    ctx.invoke(fetch_determ,   city=city, dry_run=dry_run)


# --- add-city ---
@cli.command("add-city")
@click.argument("city")
@click.option("--lat",       required=True, type=float)
@click.option("--lon",       required=True, type=float)
@click.option("--timezone",  required=True)
@click.option("--temprange", default="10-20")
@click.option("--icao",      default=None, help="ICAO station code e.g. EGLC")
def add_city(city, lat, lon, timezone, temprange, icao):
    """Add a new city to locations.json."""
    with open(CONFIG_PATH) as f:
        locations = json.load(f)
    if city in locations:
        click.echo(f"⚠️  '{city}' already exists.")
        return
    locations[city] = {
        "lat": lat, "lon": lon,
        "timezone": timezone,
        "temprange": temprange,
        "icao": icao,
        "ensemble_dir": "./apiresult/ensemble",
        "deterministic_dir": "./apiresult/deterministic",
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(locations, f, indent=2)
    click.echo(f"✅ Added {city} to locations.json" + (f" (ICAO: {icao})" if icao else " (no ICAO set)"))


# ──────────────────────────────────────────────────────────────────────────────
# CGESP COMMANDS
# ──────────────────────────────────────────────────────────────────────────────

@cli.command("fetch-observed")
@click.argument("city")
@click.option("--station", default=None, help="Single station name to fetch")
@click.option("--output",  default=None, help="Save to this .parquet path")
@click.option("--tail",    default=5, show_default=True, help="Rows to preview")
@click.option("--dry-run", is_flag=True)
def fetch_observed(city, station, output, tail, dry_run):
    """Fetch 24h hourly observed data from CGESP (with temp ↑/↓ trend)."""
    loc = load_city(city)
    channel_map, _ = _station_maps(loc, city)

    click.echo(f"▶ fetch-observed {city}")

    if dry_run:
        click.echo(f"  stations: {json.dumps(channel_map, indent=4)}")
        return

    try:
        if station:
            if station not in channel_map:
                click.echo(f"❌ Unknown station '{station}'. Available: {list(channel_map)}")
                raise SystemExit(1)
            df = fetch_station_history(channel_map[station], station)
        else:
            df = fetch_all_stations_history(channel_map)
    except CGESPError as exc:
        click.echo(f"❌ Fetch failed: {exc}")
        raise SystemExit(1)

    # Pretty-print trend arrows in preview
    preview = df.tail(tail).copy()
    preview["temp_trend"] = preview["temp_trend"].map(
        lambda v: "↑" if v == "up" else ("↓" if v == "down" else " ")
    )
    click.echo(preview.to_string(index=False))
    click.echo(f"\n  {len(df)} total rows  |  stations: {df['station'].unique().tolist()}")

    if output:
        df.to_parquet(output, index=False)
        click.echo(f"✅ Saved → {output}")


@cli.command("fetch-summary")
@click.argument("city")
@click.option("--station", default=None, help="Single station name")
@click.option("--output",  default=None, help="Save to this .parquet path")
@click.option("--dry-run", is_flag=True)
def fetch_summary(city, station, output, dry_run):
    """Fetch current conditions from CGESP (temp max/min/atual, wind, pressure)."""
    loc = load_city(city)
    _, posto_map = _station_maps(loc, city)

    if not posto_map:
        click.echo("⚠  No 'posto_id' values configured — cannot fetch summaries.")
        click.echo("   Add posto_id to each station in cli_locations.json.")
        raise SystemExit(1)

    click.echo(f"▶ fetch-summary {city}")

    if dry_run:
        click.echo(f"  stations: {json.dumps(posto_map, indent=4)}")
        return

    try:
        if station:
            if station not in posto_map:
                click.echo(f"❌ Unknown station '{station}'. Available: {list(posto_map)}")
                raise SystemExit(1)
            import pandas as pd
            df = pd.DataFrame([fetch_station_summary(posto_map[station], station)])
        else:
            df = fetch_all_stations_summary(posto_map)
    except CGESPError as exc:
        click.echo(f"❌ Fetch failed: {exc}")
        raise SystemExit(1)

    click.echo(df.T.to_string(header=False))

    if output:
        df.to_parquet(output, index=False)
        click.echo(f"✅ Saved → {output}")


@cli.command("observed-latest")
@click.argument("city")
@click.option("--output", default=None, help="Optional .parquet output")
def observed_latest(city, output):
    """Show the most-recent observation per CGESP station."""
    loc = load_city(city)
    channel_map, _ = _station_maps(loc, city)

    click.echo(f"▶ observed-latest {city}")

    try:
        df = fetch_all_stations_history(channel_map)
    except CGESPError as exc:
        click.echo(f"❌ {exc}")
        raise SystemExit(1)

    latest = (
        df.sort_values("timestamp")
          .groupby("station", as_index=False)
          .last()
    )

    latest["temp_trend"] = latest["temp_trend"].map(
        lambda v: "↑" if v == "up" else ("↓" if v == "down" else " ")
    )
    click.echo(latest.to_string(index=False))

    if output:
        latest.to_parquet(output, index=False)
        click.echo(f"✅ Saved → {output}")


if __name__ == "__main__":
    cli()