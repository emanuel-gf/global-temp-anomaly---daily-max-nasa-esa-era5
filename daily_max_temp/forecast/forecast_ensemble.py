"""
python forecast_ensemble.py --lat 51.505 --lon 0.055 --timezone 'Europe/London' --city London --root-dir ./apiresult/ensemble


PERHAPS THIS SHOULD BE ADAPT TO POINTS OUTSIDE EUROPE. ADAPT THE MODELS.

forecast_cdf_ensemble.py
------------------------
For each ensemble model, ALL members are fetched and used as individual draw.

Architecture
------------
Layer 2 — Ensemble models (up to 193 members across 4 models)
    Used for the core probability distribution. Each member is an
    independent perturbation of initial conditions, explicitly sampling
    atmospheric uncertainty — which is the dominant source of T_max
    forecast error beyond day 1.


Ensemble models avalable
--------------------
| Model              | API string            | Members | Horizon |
|--------------------|-----------------------|---------|---------|
| ECMWF IFS 0.25°    | ecmwf_ifs05           | 51      | 15d     |
| ECMWF AIFS 0.25°   | ecmwf_aifs025         | 51      | 15d     |
| DWD ICON EPS       | icon_seamless         | 40      | 7.5d    |
| DWD ICON EPS D2    | 
| GFS Ensemble Seamless | gfs_seamless
| UK MetOffice UK 2km | 
| GEM Global Ensemble

Current use:
"models": ["icon_seamless_eps", "ukmo_uk_ensemble_2km", "ncep_gefs_seamless", "ecmwf_ifs025_ensemble", "gem_global_ensemble", "icon_d2_eps"],
"forecast_days": 3,
Notes
-----
- Ensemble member columns are named member01, member02, ... by each model.
- The ECMWF AIFS model uses AI-based physics, giving genuinely different
  error correlations from the IFS physics run — valuable CDF diversity.
- UKMO 2km and ItaliaMeteo ICON 2I are regional; queries outside their
  domain will warn and skip gracefully.
"""
import argparse
import sys
from datetime import date, timedelta
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime

# --- Configuration ---
AVAILABLE_MODELS = [
    "ecmwf_ifs025_ensemble",    # Global, 51 members
    "ecmwf_aifs025_ensemble",   # Global, AI-based, 51 members
    "ncep_gefs_seamless",       # Global (USA), 31 members
    "gem_global_ensemble",      # Global (Canada), 21 members
    "icon_seamless_eps",        # Europe/Global mix, 40 members
    "ukmo_uk_ensemble_2km",     # UK Regional (will skip if out of bounds)
    "icon_d2_eps"               # Germany Regional (will skip if out of bounds)
]

def fetch_all_ensembles(lat: float,
                         lon: float,
                        models: list,
                        forecast_days: int=3, 
                        timezone: str = "UTC") -> pd.DataFrame:
    """
    Fetches all ensemble models in a single request and returns a combined DataFrame.
    """
    url = "https://ensemble-api.open-meteo.com/v1/ensemble"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "models": ",".join(models), 
        "forecast_days": forecast_days,
        "timezone": timezone,
        "temperature_unit": "celsius",
    }

    print(f"  Requesting models: {len(models)} ensembles...")
    try:
        r = requests.get(url, params=params, timeout=45)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        print(f"    [ERROR] API request failed: {exc}")
        return pd.DataFrame()

    if "hourly" not in data:
        print("    [WARN] No hourly data in response.")
        return pd.DataFrame()

    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # Note: Open-Meteo appends model names to columns if multiple models are requested
    # e.g., 'temperature_2m_ecmwf_ifs025_ensemble_member01'
    return df

def process_daily_maxima(df: pd.DataFrame, target_dates: list) -> pd.DataFrame:
    """
    Calculates the max temperature per day for every single member column.
    """
    # Filter only temperature columns (ignore 'time' if it was a column)
    temp_cols = [c for c in df.columns if "temperature_2m" in c]
    
    # Group by date and get max
    daily_max = df[temp_cols].groupby(df.index.date).max()
    
    # Filter only the dates we care about
    daily_max = daily_max.loc[daily_max.index.isin(target_dates)]
    return daily_max

def main():
    parser = argparse.ArgumentParser(description="Ensemble Weather Fetcher")
    parser.add_argument("--lat", type=float, default=51.505)
    parser.add_argument("--lon", type=float, default=0.055)
    parser.add_argument("--forecast-period", type=int, default=3)
    parser.add_argument("--timezone", type=str, default="auto")
    parser.add_argument("--root-dir", type=str, default="./data")
    parser.add_argument("--city",type=str, help='Alias for saving it.')
    args = parser.parse_args()

    # Setup Directory
    save_path = Path(args.root_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    target_dates = [date.today() + timedelta(days=i) for i in range(0, args.forecast_period + 1)]

    print(f"\n{'='*60}")
    print("  Fetching Ensemble CDF Data")
    print(f"  Loc: {args.lat}, {args.lon} | Days: {args.days}")
    print(f"{'='*60}\n")

    # 1. Fetch
    full_df = fetch_all_ensembles(args.lat, args.lon, AVAILABLE_MODELS, args.days, args.timezone)

    if full_df.empty:
        sys.exit("No data retrieved.")

    # 2. Process
    daily_max_df = process_daily_maxima(full_df,
                                        target_dates)

    # 3. Summary Statistics (Aggregating all members across all models for a true CDF)
    print("── Daily Max Temperature Summary (°C) ───────────────────────")
    summary_rows = []
    for d in target_dates:
        if d in daily_max_df.index:
            vals = daily_max_df.loc[d].dropna().values
            summary_rows.append({
                "Date": d,
                "Mean": f"{np.mean(vals):.2f}",
                "StdDev": f"{np.std(vals):.2f}",
                "P10 (Cold)": f"{np.percentile(vals, 10):.2f}",
                "P90 (Hot)": f"{np.percentile(vals, 90):.2f}",
                "Members": len(vals)
            })
    
    print(pd.DataFrame(summary_rows).to_string(index=False))

    # 4. Save raw data
    filename = save_path / f"ensemble-{args.city}_{datetime.now().strftime("%Y-%m-%d-%H:%m")}.parquet"
    daily_max_df.to_parquet(filename)
    print(f"\n[INFO] Raw hourly data saved to: {filename}")

if __name__ == "__main__":
    main()