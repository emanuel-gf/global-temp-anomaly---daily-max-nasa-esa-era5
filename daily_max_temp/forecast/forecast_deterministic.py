"""
UTC implemented. Should be passed as a flag.

---------------
Fetches hourly temperature_2m forecasts for the next 1-3 days.

deterministic NWP models via the Open-Meteo Forecast API.

Models used
-----------
Each model is queried from its dedicated Open-Meteo endpoint.

| Model                        | API string                      | Endpoint base                        |
|------------------------------|---------------------------------|--------------------------------------|
| ECMWF IFS HRES 9km           | ecmwf_ifs_analysis_long_window  | https://api.open-meteo.com/v1/ecmwf  |
| UK Met Office UK 2km         | ukmo_uk_deterministic_2km       | https://api.open-meteo.com/v1/forecast|
| UK Met Office Global 10km    | ukmo_global_deterministic_10km  | https://api.open-meteo.com/v1/forecast|
| Météo-France ARPEGE Europe   | meteofrance_arpege_europe       | https://api.open-meteo.com/v1/forecast|
| GErman DWD - D2   2km        | dwd_icon_d2                     | https://api.open-meteo.com/v1/forecast|
| DWD ICON EU                  | icon_eu                         | https://api.open-meteo.com/v1/forecast|

Usage
-----
    python forecast_deterministic.py --lat 51.505 --lon 0.055 --time 'Europe/London' --city London --root-dir ./apiresult/deterministic
    python forecast_deterministic.py --lat 48.8566 --lon 2.3522 --timezone 'Europe/Paris' --city Paris --root-dir ./apiresult/deterministic \\
        --models "ECMWF IFS HRES 9km" "UKMO UK 2km" "DWD ICON EU"

"""
from typing import List
import argparse
import sys
from datetime import date, timedelta, datetime
import os

import numpy as np
import pandas as pd
import requests_cache
import openmeteo_requests
from retry_requests import retry
from pathlib import Path


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# Each entry maps a human-readable name to its API model string and the
# endpoint that serves it. Models on the /v1/ecmwf endpoint cannot be
# batched with models on /v1/forecast, so they are kept in a separate group.

MODELS = {
    "ECMWF IFS HRES 9km": {
        "base_url": "https://api.open-meteo.com/v1/ecmwf",
        "model_param": "ecmwf_ifs_analysis_long_window",
        "forecast_days": 3,
        "color": "#e41a1c",
    },
    "UKMO UK 2km": {
        "base_url": "https://api.open-meteo.com/v1/forecast",
        "model_param": "ukmo_uk_deterministic_2km",
        "forecast_days": 3,
        "color": "#377eb8",
    },
    "UKMO Global 10km": {
        "base_url": "https://api.open-meteo.com/v1/forecast",
        "model_param": "ukmo_global_deterministic_10km",
        "forecast_days": 3,
        "color": "#4daf4a",
    },
    "MeteoFrance ARPEGE Europe": {
        "base_url": "https://api.open-meteo.com/v1/forecast",
        "model_param": "arpege_europe",
        "forecast_days": 3,
        "color": "#984ea3",
    },
    "DWD Germany": {
        "base_url": "https://api.open-meteo.com/v1/forecast",
        "model_param": "icon_d2",
        "forecast_days": 2,
        "color": "#ff7f00",
    },
    "DWD ICON EU": {
        "base_url": "https://api.open-meteo.com/v1/forecast",
        "model_param": "icon_eu",
        "forecast_days": 3,
        "color": "#a65628",
    },
    "MeteoFrance AROME_HD": {
        "base_url": "https://api.open-meteo.com/v1/forecast",
        "model_param": "arome_france",
        "forecast_days": 3,
        "color": "#984ea3",
    },
    "Italia_Meteo_AROME_HD": {
        "base_url": "https://api.open-meteo.com/v1/forecast",
        "model_param": "italia_meteo_arpae_icon_2i",
        "forecast_days": 3,
        "color": "#984ea3",
    },
}


# ---------------------------------------------------------------------------
# Open-Meteo client (shared, with cache + retry)
# ---------------------------------------------------------------------------

def build_openmeteo_client(cache_ttl: int = 3600) -> openmeteo_requests.Client:
    """Return a Client backed by a disk-cached, auto-retrying session."""
    cache_session = requests_cache.CachedSession(".cache", expire_after=cache_ttl)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def _group_models_by_endpoint(model_names: List[str]) -> dict[str, List[str]]:
    """
    Partition the requested model names by their base_url so we can issue
    one batched request per endpoint.

    Returns
    -------
    {base_url: [model_name, ...]}
    """
    groups: dict[str, List[str]] = {}
    for name in model_names:
        url = MODELS[name]["base_url"]
        groups.setdefault(url, []).append(name)
    return groups


def fetch_all_models(
    model_names: List[str],
    lat: float,
    lon: float,
    hourly_vars: str | List[str],
    timezone: str,
    client: openmeteo_requests.Client,
) -> dict[str, pd.Series]:
    """
    Fetch all requested models, batching by endpoint.

    The Open-Meteo API returns one response object per model when multiple
    model strings are passed in the ``models`` parameter.  Models that live
    on different base URLs (e.g. /v1/ecmwf vs /v1/forecast) are queried in
    separate calls and the results are merged.

    Returns
    -------
    {model_name: pd.Series(temperature_2m, index=DatetimeTZAware)}
    """
    if isinstance(hourly_vars, str):
        hourly_vars = [hourly_vars]

    all_series: dict[str, pd.Series] = {}
    groups = _group_models_by_endpoint(model_names)

    for base_url, names in groups.items():
        # The forecast_days cap for a batch is the minimum across the group
        # so that every model can satisfy the request.
        forecast_days = min(MODELS[n]["forecast_days"] for n in names)
        model_params = [MODELS[n]["model_param"] for n in names]

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": hourly_vars,
            "models": model_params,
            "forecast_days": forecast_days,
            "timezone": timezone,
            "temperature_unit": "celsius",
        }

        print(f"  Querying {base_url} for: {names}")
        try:
            responses = client.weather_api(base_url, params=params)
        except Exception as exc:
            print(f"  [WARN] batch request to {base_url} failed — {exc}")
            continue

        # The API returns one response per model, in the same order as
        # the model_params list.
        for name, response in zip(names, responses):
            try:
                hourly = response.Hourly()
                times = pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left",
                )
                # temperature_2m is always the first (and only) variable
                values = hourly.Variables(0).ValuesAsNumpy()
                series = pd.Series(values, index=times, name=name)
                all_series[name] = series
                print(f"    ✓ {name}: {len(series)} hourly records")
            except Exception as exc:
                print(f"  [WARN] {name}: could not parse response — {exc}")

    return all_series


# ---------------------------------------------------------------------------
# Daily maximum
# ---------------------------------------------------------------------------

def compute_daily_max(series: pd.Series, target_dates: list[date]) -> dict[date, float]:
    """Extract the daily maximum for each target date."""
    # Normalise index to naive-date comparison regardless of tz
    index_dates = series.index.date
    result = {}
    for d in target_dates:
        day_data = series[index_dates == d].dropna()
        if len(day_data) > 0:
            result[d] = {
                "temperature": float(day_data.max()),
                "time": day_data.idxmax(),
            }
        else:
            result[d] = {"temperature": np.nan, "time": np.nan}
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-model deterministic temperature forecast — daily max extractor"
    )
    parser.add_argument("--lat", type=float, default=48.8566, help="Latitude")
    parser.add_argument("--lon", type=float, default=2.3522, help="Longitude")
    parser.add_argument("--forecast-period", type=int, default=3)
    parser.add_argument("--timezone", type=str, default="UTC",
                        help="Timezone string (e.g. 'Europe/Paris')")
    parser.add_argument("--root-dir", type=str, default=None,
                        help="Root directory to save the parquet file")
    parser.add_argument("--city", type=str, default="City",
                        help="City name used in output filename and log")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
        metavar="MODEL",
        help=(
            "One or more model names to query (default: all). "
            f"Available: {list(MODELS.keys())}"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    HOURLY_VARS = "temperature_2m"

    if args.root_dir is None:
        raise ValueError("Please input the root directory with --root-dir.")
    root_dir = Path(args.root_dir)
    os.makedirs(root_dir, exist_ok=True)

    today = date.today()
    target_dates = [today + timedelta(days=i) for i in range(0, args.forecast_period + 1)]

    print(f"\nFetching forecasts for {args.city} ({args.lat:.4f}, {args.lon:.4f})")
    print(f"Target dates : {[str(d) for d in target_dates]}")
    print(f"Models       : {args.models}\n")

    # Build shared client (cache + retry)
    client = build_openmeteo_client()

    # Fetch — one batched request per endpoint
    all_series = fetch_all_models(
        model_names=args.models,
        lat=args.lat,
        lon=args.lon,
        hourly_vars=HOURLY_VARS,
        timezone=args.timezone,
        client=client,
    )

    if not all_series:
        print("No model data retrieved. Exiting.")
        sys.exit(1)

    # --- Compute daily max per model ---
    daily_maxima: dict[date, dict[str, float]] = {d: {} for d in target_dates}
    for name, series in all_series.items():
        maxes = compute_daily_max(series, target_dates)
        for d, val in maxes.items():
            daily_maxima[d][name] = val

    # --- Build summary DataFrame (unchanged output contract) ---
    rows = []
    for dat, models in daily_maxima.items():
        for model, data in models.items():
            rows.append({
                "date": dat,
                "model": model,
                "time": data["time"],
                "temperature": data["temperature"],
            })
    df_maxima = pd.DataFrame(rows)

    print("\n=== Daily Maximum Temperature (°C) per model ===")
    print(df_maxima.round(2).to_string())

    filename_out = datetime.now().strftime("%Y-%m-%d-%H:%M")
    out_path = root_dir / f"{args.city}-Deterministic_{filename_out}.parquet"
    df_maxima.to_parquet(out_path)
    print(f"\nParquet saved at: {out_path}")


if __name__ == "__main__":
    main()