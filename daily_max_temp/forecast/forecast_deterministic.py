"""
WARNING: UTC NOT IMPLEMENTED. SHOULD BE DONE FOR NEXT POINTS

forecast_cdf.py
---------------
Fetches hourly temperature_2m forecasts for the next 3 days from 6
deterministic NWP models via the Open-Meteo Forecast API, computes
the daily maximum per model + time! 
Builds a full empirical + smoothed CDF suitable for pricing prediction markets / computing betting edge.

Models used
-----------
Each model is queried from its dedicated Open-Meteo endpoint, as some
(ECMWF IFS HRES 9km, UKMO) have their own API base URL.

| Model                        | API string                      | Endpoint base                        |
|------------------------------|---------------------------------|--------------------------------------|
| ECMWF IFS HRES 9km           | ecmwf_ifs_analysis_long_window  | https://api.open-meteo.com/v1/ecmwf  |
| UK Met Office UK 2km         | ukmo_uk_deterministic_2km       | https://api.open-meteo.com/v1/forecast|
| UK Met Office Global 10km    | ukmo_global_deterministic_10km  | https://api.open-meteo.com/v1/forecast|
| Météo-France ARPEGE Europe   | meteofrance_arpege_europe       | https://api.open-meteo.com/v1/forecast|
| ItaliaMeteo ARPAE ICON 2I    | italia_meteo_arpae_icon_2i      | https://api.open-meteo.com/v1/forecast|
| DWD ICON EU                  | icon_eu                         | https://api.open-meteo.com/v1/forecast|

Usage
-----
    python forecast_cdf.py --lat 48.8566 --lon 2.3522 --city Paris


"""

import argparse
import sys
from datetime import date, timedelta,datetime
import os 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.stats import gaussian_kde, norm


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

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
        "model_param": "meteofrance_arpege_europe",
        "forecast_days": 3,           # 4-day max
        "color": "#984ea3",
    },
    "ItaliaMeteo ICON 2I": {
        "base_url": "https://api.open-meteo.com/v1/forecast",
        "model_param": "italia_meteo_arpae_icon_2i",
        "forecast_days": 3,           # 3-day max
        "color": "#ff7f00",
    },
    "DWD ICON EU": {
        "base_url": "https://api.open-meteo.com/v1/forecast",
        "model_param": "icon_eu",
        "forecast_days": 3,           # 5-day max
        "color": "#a65628",
    },
}

HOURLY_VARS = "temperature_2m"


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def build_params(lat: float, lon: float, model_cfg: dict) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": HOURLY_VARS,
        "forecast_days": model_cfg["forecast_days"],
        "timezone": "UTC",
        "temperature_unit": "celsius",
    }
    if model_cfg["model_param"] is not None:
        params["models"] = model_cfg["model_param"]
    return params


def fetch_model(name: str, lat: float, lon: float) -> pd.Series | None:
    """
    Returns a Series indexed by UTC datetime with hourly temperature_2m,
    or None on failure.
    """
    cfg = MODELS[name]
    params = build_params(lat, lon, cfg)

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


# ---------------------------------------------------------------------------
# Daily max & CDF construction
# ---------------------------------------------------------------------------

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


def build_cdf(values: np.ndarray, x_grid: np.ndarray):
    """
    Build both an empirical CDF and a KDE-smoothed CDF over x_grid.
    Returns (empirical_cdf, kde_cdf).
    """
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return None, None

    # Empirical CDF
    sorted_vals = np.sort(values)
    empirical_p = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

    # KDE-smoothed CDF  (bandwidth = Scott's rule, works fine for n=6)
    kde = gaussian_kde(values, bw_method="scott")
    kde_pdf = kde(x_grid)
    kde_cdf = np.cumsum(kde_pdf) * (x_grid[1] - x_grid[0])
    kde_cdf = kde_cdf / kde_cdf[-1]   # normalise to [0, 1]

    return (sorted_vals, empirical_p), (x_grid, kde_cdf)


##---------------------------------------------------------------------------
##Betting edge helper
##---------------------------------------------------------------------------

def compute_edge(kde_cdf_tuple, threshold: float, market_prob: float) -> dict:
    """
    Compute implied probability from the CDF and edge vs a market probability.

    Parameters
    ----------
    kde_cdf_tuple : (x_grid, cdf_values)
    threshold     : temperature threshold, e.g. "will max exceed 25°C?"
    market_prob   : market's implied probability of exceeding the threshold

    Returns a dict with model_prob, edge, and kelly_fraction.
    """
    x_grid, cdf_vals = kde_cdf_tuple
    # P(T_max > threshold) = 1 - CDF(threshold)
    model_prob = float(1.0 - np.interp(threshold, x_grid, cdf_vals))
    edge = model_prob - market_prob

    # Kelly fraction = edge / (1 - market_prob) assuming binary bet paying 1/market_prob
    # i.e. bet on "exceeds threshold" at market odds = 1/market_prob
    if market_prob > 0 and market_prob < 1:
        kelly = edge / (1.0 - market_prob)
    else:
        kelly = np.nan

    return {
        "model_prob": round(model_prob, 4),
        "market_prob": market_prob,
        "edge": round(edge, 4),
        "kelly_fraction": round(kelly, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-model - Deterministic temperature CDF builder")
    parser.add_argument("--lat", type=float, default=48.8566, help="Latitude")
    parser.add_argument("--lon", type=float, default=2.3522, help="Longitude")
    parser.add_argument("--root-dir", type=str,default='./query', help="Root directory to save the parquet")
    parser.add_argument("--city", type=str, default="City", help="City name for plot title")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Temperature threshold (°C) for edge calculation, e.g. 25.0"
    )
    parser.add_argument(
        "--market-prob", type=float, default=None,
        help="Market implied probability of exceeding threshold, e.g. 0.40"
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    ## make dir case doesnt exist
    os.makedirs(args.root_dir, exist_ok=True)
    ## get today's
    today = date.today()
    target_dates = [today + timedelta(days=i) for i in range(1, 4)]  # D+1, D+2, D+3

    print(f"\nFetching forecasts for {args.city} ({args.lat:.4f}, {args.lon:.4f})")
    print(f"Target dates: {[str(d) for d in target_dates]}\n")

    # --- Fetch all models ---
    all_series: dict[str, pd.Series] = {}
    for name in MODELS:
        print(f"  Fetching {name}...")
        s = fetch_model(name, args.lat, args.lon)
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
    df_maxima.to_parquet(f"{args.root_dir}/{args.city}-Deterministic_{filename_out}.parquet")
    print(f"Parquet saved at:{dir}/{args.city}-Deterministic_{filename_out}.parquet")

    
    
    # --- CDF per day ---
    fig, axes = plt.subplots(1, len(target_dates), figsize=(5 * len(target_dates), 5), sharey=True)
    if len(target_dates) == 1:
        axes = [axes]

    cdf_results = {}

    for ax, d in zip(axes, target_dates):
        vals_dict = daily_maxima[d]
        values = np.array([v for v in vals_dict.values() if not np.isnan(v)])

        if len(values) < 2:
            ax.set_title(f"{d}\n(insufficient data)")
            continue

        x_min, x_max = values.min() - 4, values.max() + 4
        x_grid = np.linspace(x_min, x_max, 500)

        emp, kde_result = build_cdf(values, x_grid)
        cdf_results[d] = kde_result

        # Plot individual model estimates as vertical lines
        for name, val in vals_dict.items():
            if not np.isnan(val):
                ax.axvline(val, color=MODELS[name]["color"], alpha=0.6,
                           linewidth=1.5, linestyle="--", label=name)

        # Plot empirical CDF as step
        if emp:
            ax.step(emp[0], emp[1], where="post", color="black",
                    linewidth=2, label="Empirical CDF", zorder=5)

        # Plot KDE-smoothed CDF
        ax.plot(kde_result[0], kde_result[1], color="black", linewidth=2.5,
                linestyle="-", label="KDE CDF", zorder=6)

        # Highlight threshold if given
        if args.threshold is not None:
            p_exceed = float(1.0 - np.interp(args.threshold, kde_result[0], kde_result[1]))
            ax.axvline(args.threshold, color="red", linewidth=2, zorder=7,
                       label=f"Threshold {args.threshold}°C\nP(exceed)={p_exceed:.2%}")

        ax.set_title(f"{d}", fontsize=11)
        ax.set_xlabel("T_max (°C)")
        ax.set_ylabel("CDF" if d == target_dates[0] else "")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        f"Daily Max Temperature CDF — {args.city}\n"
        f"Models: {', '.join(all_series.keys())}",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig("forecast_cdf.png", dpi=150, bbox_inches="tight")
    print("\nCDF plot saved → forecast_cdf.png")

    # --- Betting edge ---
    if args.threshold is not None and args.market_prob is not None:
        print(f"\n=== Betting Edge (threshold={args.threshold}°C, market_prob={args.market_prob}) ===")
        for d in target_dates:
            if d in cdf_results and cdf_results[d] is not None:
                edge_result = compute_edge(cdf_results[d], args.threshold, args.market_prob)
                print(
                    f"  {d} | model_prob={edge_result['model_prob']:.2%} | "
                    f"edge={edge_result['edge']:+.2%} | "
                    f"kelly={edge_result['kelly_fraction']:+.4f}"
                )

    print("\nDone.\n")


if __name__ == "__main__":
    main()