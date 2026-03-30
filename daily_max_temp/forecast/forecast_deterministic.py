"""
UTC implemented. Should be passed as as flag. 

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
| GErman DWD - D2   2km        | dwd_icon_d2                         | https://api.open-meteo.com/v1/forecast|
| DWD ICON EU                  | icon_eu                         | https://api.open-meteo.com/v1/forecast|

Usage
-----
    python forecast_deterministic.py --lat 51.505 --lon 0.055 --time 'Europe/London' --city London --root-dir ./apiresult/deterministic


"""
from typing import List
import argparse
import sys
from datetime import date, timedelta,datetime
import os 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from pathlib import Path

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = [
    'ukmo_uk_deterministic_2km',
    'ukmo_global_deterministic_10km',
    'arome_france',
    'arpege_world',
    'arpege_europe',
]
MODELS = {
    "ECMWF IFS HRES 9km": {
        "base_url": "https://api.open-meteo.com/v1/ecmwf",
        "model_param": None,          # ECMWF endpoint serves IFS HRES by default
        "forecast_days": 3,
        "color": "#e41a1c",
    },
    "UKMO UK 2km": {
        "base_url": "https://api.open-meteo.com/v1/forecast",
        "model_param": "ukmo_uk_deterministic_2km",
        "forecast_days": 3,           # 5-day max, but UK domain only
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
        "forecast_days": 3,           # 4-day max
        "color": "#984ea3",
    },
    "DWD Germany": {
        "base_url": "https://api.open-meteo.com/v1/forecast",
        "model_param": "icon_d2",
        "forecast_days": 2,           # 2-day max
        "color": "#ff7f00",
    },
    "DWD ICON EU": {
        "base_url": "https://api.open-meteo.com/v1/forecast",
        "model_param": "icon_eu",
        "forecast_days": 3,           # 5-day max
        "color": "#a65628",
    },
    "MeteoFrance AROME_HD": {
        "base_url": "https://api.open-meteo.com/v1/forecast",
        "model_param": "arome_france",
        "forecast_days": 3,           # 4-day max
        "color": "#984ea3",
    },
    "Italia_Meteo_AROME_HD": {
        "base_url": "https://api.open-meteo.com/v1/forecast",
        "model_param": "italia_meteo_arpae_icon_2i",
        "forecast_days": 3,           # 4-day max
        "color": "#984ea3",
    }
}


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def build_params(lat: float,
                 lon: float, 
                 model_cfg: dict,
                 hourly_vars: List[str], 
                 timezone: str = None,
                 ) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": hourly_vars,
        "forecast_days": model_cfg["forecast_days"],
        "timezone": "UTC",
        "temperature_unit": "celsius",
    }
    if model_cfg["model_param"] is not None:
        params["models"] = model_cfg["model_param"]
    if timezone is not None:
        params['timezone'] = timezone
    return params


def fetch_model(name: str, lat: float, lon: float,
                hourly_vars:List[str],timezone: str) -> pd.Series | None:
    """
    Returns a Series indexed by UTC datetime with hourly temperature_2m,
    or None on failure.
    """
    cfg = MODELS[name]
    params = build_params(lat,
                            lon,
                            model_cfg = cfg,
                            hourly_vars=hourly_vars,
                            timezone=timezone)

    try:
        resp = requests.get(cfg["base_url"], params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"  [WARN] {name}: request failed — {exc}")
        return None

    try:
        hourly = data["hourly"]
        times = pd.to_datetime(hourly["time"])
        temps = pd.Series(hourly["temperature_2m"], index=times, name=name)
        return temps
    except KeyError as exc:
        print(f"  [WARN] {name}: unexpected response structure — {exc}")
        return None

## ---- DAILY MAX
def compute_daily_max(series: pd.Series, target_dates: list[date]) -> dict[date, float]:
    """Extract the daily maximum for each target date."""
    result = {}
    for d in target_dates:
        day_data = series[series.index.date == d].dropna()
        if len(day_data) > 0:
            result[d]= {'temperature':float(day_data.max()),
                        'time':day_data.idxmax()
                        }
        else:
            result[d]= {'temperature':np.nan,
                            'time':np.nan
                        }
    return result
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-model - Deterministic temperature CDF builder")
    parser.add_argument("--lat", type=float, default=48.8566, help="Latitude")
    parser.add_argument("--lon", type=float, default=2.3522, help="Longitude")
    parser.add_argument("--forecast-period", type=int, default=3)
    parser.add_argument("--timezone", type=str, default = None, help='Timezone of the point')
    parser.add_argument("--root-dir", type=str,default=None, help="Root directory to save the parquet file")
    parser.add_argument("--city", type=str, default="City", help="City name for plot title")

    ##TODO: ad a verbose parser argument.
    ## IF VERBOSE is true, then print. Instead of, work silent.
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # set up var
    HOURLY_VARS = "temperature_2m"

    ## make dir case doesnt exist
    if args.root_dir is None:
        raise("Please input the root directory to save the query.")
    root_dir = Path(str(args.root_dir))
    os.makedirs(root_dir, exist_ok=True)

    ## get today's
    today = date.today()
    target_dates = [today + timedelta(days=i) for i in range(0, args.forecast_period+1)]  # D+0, D+1, D+2
    print(f"\nFetching forecasts for {args.city} ({args.lat:.4f}, {args.lon:.4f})")
    print(f"Target dates: {[str(d) for d in target_dates]}\n")

    # --- Fetch all models ---
    all_series: dict[str, pd.Series] = {}
    for name in MODELS:
        print(f"  Fetching {name}...")
        s = fetch_model(name,
                        args.lat,
                        args.lon,
                        hourly_vars=HOURLY_VARS,
                        timezone = args.timezone)
        if s is not None:
            all_series[name] = s

    if not all_series:
        print("No model data retrieved. Exiting.")
        sys.exit(1)

    # --- Compute daily max per model ---
    daily_maxima: dict[date, dict[str, float]] = {d: {} for d in target_dates}
    for name, series in all_series.items():
        maxes = compute_daily_max(series, target_dates)
        for d, val in maxes.items():
            daily_maxima[d][name] = val

    # --- Summary table ---
    rows = []
    for dat, models in daily_maxima.items():
        for model, data in models.items():
            rows.append({
                "date": dat,
                "model": model,
                "time": data["time"],
                "temperature": data["temperature"]
            })
    df_maxima = pd.DataFrame(rows)
    print("\n=== Daily Maximum Temperature (°C) per model ===")
    print(df_maxima.round(2).to_string())

    ## save df as a parquet 
    filename_out = datetime.now().strftime("%Y-%m-%d-%H:%m")
    df_maxima.to_parquet(f"{root_dir}/{args.city}-Deterministic_{filename_out}.parquet")
    print(f"Parquet saved at:{root_dir}/{args.city}-Deterministic_{filename_out}.parquet")



if __name__ == "__main__":
    main()