import polars as pl
import numpy as np
from pathlib import Path
from scipy import stats

RAW_DIR = Path("era5_raw")
ALL_YEARS = sorted(RAW_DIR.glob("era5_*.parquet"))

# 1. Load all years at once — Polars handles this efficiently
df = pl.scan_parquet(ALL_YEARS)    # lazy scan, no memory spike

# 2. Aggregate hourly → daily
daily = (
    df
    .with_columns(pl.col("datetime").cast(pl.Date).alias("date"))
    .group_by("date")
    .agg([
        pl.col("t2m").max().alias("tmax"),
        pl.col("t2m").mean().alias("t2m_mean"),
        pl.col("ssrd_hourly").sum().alias("ssrd_daily"),   # Wh/m²
        pl.col("blh").mean().alias("blh_mean"),
    ])
    .with_columns(pl.col("date").dt.ordinal_day().alias("doy"))
    .collect()
)

# 3. Compute per-DOY statistics
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
        pl.col("blh_mean").mean().alias("blh_mean"),
    ])
    .sort("doy")
)

# 4. tmax_trend: linear regression slope per DOY (°C/year)
# This needs the yearly dimension preserved — do it outside Polars
daily_pd = daily.to_pandas()
trends = []
for doy in range(1, 366):
    subset = daily_pd[daily_pd["doy"] == doy].copy()
    subset["year"] = pd.to_datetime(subset["date"]).dt.year
    if len(subset) >= 5:   # need enough years
        slope, _, _, _, _ = stats.linregress(subset["year"], subset["tmax"])
    else:
        slope = 0.0
    trends.append({"doy": doy, "tmax_trend": slope})

trends_df = pl.DataFrame(trends)
normals = normals.join(trends_df, on="doy")

# 5. Optional: 7-day rolling smooth to reduce DOY noise
# (a single unusual year can spike one DOY — smoothing helps)
normals_pd = normals.to_pandas().set_index("doy")
smooth_cols = ["tmax_mean", "tmax_std", "tmax_p10", "tmax_p25", "tmax_p75", "tmax_p90"]
for col in smooth_cols:
    normals_pd[col] = (
        normals_pd[col]
        .rolling(window=7, center=True, min_periods=1)
        .mean()
    )

normals_final = pl.from_pandas(normals_pd.reset_index())
normals_final.write_parquet("climate_normals.parquet")
print(normals_final)