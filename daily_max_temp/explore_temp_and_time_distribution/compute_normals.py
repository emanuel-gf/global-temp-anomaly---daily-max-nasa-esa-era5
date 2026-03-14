"""
Compute ERA5-Land climatological normals (Stream 3 prior).
 
Reads yearly ERA5 parquets produced by download_era5.py and outputs a
365-row climate_normals.parquet with one row per day-of-year (DOY).
 
Output columns
--------------
doy           : int, 1–365
tmax_mean     : mean of daily Tmax across all years
tmax_std      : std  of daily Tmax
tmax_p10/p25  : lower percentiles of daily Tmax distribution
tmax_p75/p90  : upper percentiles
tmax_trend    : linear warming trend (°C / year, from scipy linregress)
t2m_mean      : mean of daily-mean t2m
ssrd_mean     : mean of daily-total ssrd  (J m⁻²)
sshf_mean     : mean of daily-total sensible heat flux
slhf_mean     : mean of daily-total latent heat flux
skt_mean      : mean of daily-mean skin temperature (°C)
stl1_mean     : mean of daily-mean soil layer-1 temperature (°C)
sp_mean       : mean of daily-mean surface pressure (Pa)
u10_mean      : mean of daily-mean 10m U wind (m s⁻¹)
v10_mean      : mean of daily-mean 10m V wind (m s⁻¹)
wind_speed_mean: mean of daily-mean wind speed magnitude (m s⁻¹)
 
Usage
-----
    python compute_normals.py \
        [--raw_dir ./era5_data] \
        [--output    ./climate_normals.parquet] \
        [--smooth_window 7]
"""

import polars as pl
import numpy as np
import os
import pandas as pd 
from pathlib import Path
from scipy import stats
import argparse 

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ERA5 climatological normals from yearly parquets."
    )
    parser.add_argument("--raw_dir",      default="./era5_data")
    parser.add_argument("--output",       default="./climate_normals.parquet")
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=7,
        help="Rolling window (days) to smooth per-DOY stats. Set 1 to disable.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
 
    all_files = sorted(raw_dir.glob("era5_*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"No era5_*.parquet files found in {raw_dir}")
 
    print(f"[info] Found {len(all_files)} yearly file(s): "
          f"{all_files[0].stem} → {all_files[-1].stem}")
    
    ## create dir if not exist
    os.makedirs(Path(args.output).parent,exist_ok=True)
    # ------------------------------------------------------------------
    # 1. Lazy scan — no memory spike even with many years
    # ------------------------------------------------------------------
    df = pl.scan_parquet(all_files)
 
    # ------------------------------------------------------------------
    # 2. Hourly → daily aggregation
    #    ssrd/sshf/slhf are per-hour increments → sum for daily total
    #    temperatures, sp, winds → mean
    # ------------------------------------------------------------------
    daily = (
        df
        .with_columns(pl.col("datetime").cast(pl.Date).alias("date"))
        .group_by("date")
        .agg([
            pl.col("t2m").max().alias("tmax"),
            pl.col("t2m").mean().alias("t2m_mean"),
            pl.col("ssrd").sum().alias("ssrd_daily"),
            pl.col("sshf").sum().alias("sshf_daily"),
            pl.col("slhf").sum().alias("slhf_daily"),
            pl.col("skt").mean().alias("skt_mean"),
            pl.col("stl1").mean().alias("stl1_mean"),
            pl.col("sp").mean().alias("sp_mean"),
            pl.col("u10").mean().alias("u10_mean"),
            pl.col("v10").mean().alias("v10_mean"),
            # Wind speed magnitude — compute from components
            (pl.col("u10")**2 + pl.col("v10")**2).sqrt().mean().alias("wind_speed_mean"),
        ])
        .with_columns(pl.col("date").dt.ordinal_day().alias("doy"))
        .collect()
    )
 
    print(f"[info] Daily records: {len(daily):,}  "
          f"({daily['date'].min()} → {daily['date'].max()})")
 
    # ------------------------------------------------------------------
    # 3. Per-DOY statistics
    # ------------------------------------------------------------------
    normals = (
        daily
        .group_by("doy")
        .agg([
            pl.col("tmax").mean().alias("tmax_mean"),
            pl.col("tmax").std().alias("tmax_std"),
            pl.col("tmax").quantile(0.10).alias("tmax_p10"),
            pl.col("tmax").quantile(0.25).alias("tmax_p25"),
            pl.col("tmax").quantile(0.75).alias("tmax_p75"),
            pl.col("tmax").quantile(0.90).alias("tmax_p90"),
            pl.col("t2m_mean").mean().alias("t2m_mean"),
            pl.col("ssrd_daily").mean().alias("ssrd_mean"),
            pl.col("sshf_daily").mean().alias("sshf_mean"),
            pl.col("slhf_daily").mean().alias("slhf_mean"),
            pl.col("skt_mean").mean().alias("skt_mean"),
            pl.col("stl1_mean").mean().alias("stl1_mean"),
            pl.col("sp_mean").mean().alias("sp_mean"),
            pl.col("u10_mean").mean().alias("u10_mean"),
            pl.col("v10_mean").mean().alias("v10_mean"),
            pl.col("wind_speed_mean").mean().alias("wind_speed_mean"),
        ])
        .sort("doy")
    )
 
    # ------------------------------------------------------------------
    # 4. tmax_trend: linear warming slope (°C/year) per DOY
    #    Polars doesn't have linregress, so we use scipy via pandas
    # ------------------------------------------------------------------
    daily_pd = daily.to_pandas()
    daily_pd["year"] = pd.to_datetime(daily_pd["date"]).dt.year
 
    trends = []
    for doy in range(1, 366):
        subset = daily_pd[daily_pd["doy"] == doy]
        if len(subset) >= 5:
            slope, *_ = stats.linregress(subset["year"], subset["tmax"])
        else:
            slope = 0.0
        trends.append({"doy": doy, "tmax_trend": float(slope)})
 
    normals = normals.join(pl.DataFrame(trends), on="doy")
 
    # ------------------------------------------------------------------
    # 5. Optional rolling smooth — reduces single-year DOY noise
    #    e.g. one anomalous year can spike tmax_p90 for DOY 74 only
    # ------------------------------------------------------------------
    smooth_cols = [
        "tmax_mean", "tmax_std",
        "tmax_p10", "tmax_p25", "tmax_p75", "tmax_p90",
        "ssrd_mean", "sshf_mean", "slhf_mean",
        "skt_mean", "stl1_mean",
    ]
 
    if args.smooth_window > 1:
        normals_pd = normals.to_pandas().set_index("doy")
        for col in smooth_cols:
            normals_pd[col] = (
                normals_pd[col]
                .rolling(window=args.smooth_window, center=True, min_periods=1)
                .mean()
            )
        normals = pl.from_pandas(normals_pd.reset_index())
        print(f"[info] Applied {args.smooth_window}-day rolling smooth to: {smooth_cols}")
 
    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    normals = normals.sort("doy")
    normals.write_parquet(args.output)
 
    print(f"[done] Saved {len(normals)} DOY rows → {args.output}")
    print(normals)
 
 
if __name__ == "__main__":
    main()