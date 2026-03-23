import subprocess
import json
import click

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

# --- CLI group ---
@click.group()
def cli():
    """Forecast pipeline."""
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


## LIVE STREAM
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
    cmd = ["python", "scripts/metar_fetcher.py", "-s", icao]

    if as_json:
        cmd.append("--json")

    click.echo(f"▶ live-stream {city} → {icao}")
    if dry_run:
        click.echo("  [dry-run] " + " ".join(cmd))
    else:
        subprocess.run(cmd, check=True)


# --- fetch-all (bonus: run both fetches at once) ---
@cli.command("fetch-all")
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
@click.option("--icao",      default=None, help="ICAO station code e.g. EGLC")  # ← new
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

if __name__ == "__main__":
    cli()