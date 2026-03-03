import os
import math
import argparse
import pandas as pd
import numpy as np
import xarray as xr
import sys
import calendar
import time
from functools import wraps
from dotenv import load_dotenv
load_dotenv()

"""
This script computes zonal statistics for ERA5 data. It aims to calculate weighted averages of temperatures.
To store it, it creates cache coarsed resolution of 2 degree. This cache represents a daily output 
The cache allows to re-run the experiment by changing zone boundaries or weigthing schemes.

Once the structure is properly defined, the cache can be deleted easily. 

## Folder Structure

Running `--year 2020 2021 --month 1 3` produces:
```
era5_zonal_features/          ← final outputs (tiny, ~KB each)
├── 2020/
│   ├── 01/zonal.parquet
│   ├── 02/zonal.parquet
│   └── 03/zonal.parquet
└── 2021/
    ├── 01/zonal.parquet
    └── ...

era5_coarsened_cache/         ← intermediate cache (~4MB each)
├── 2020/
│   ├── 01/t2m_coarsened_2.0deg.parquet
│   ├── 02/t2m_coarsened_2.0deg.parquet
│   └── 03/t2m_coarsened_2.0deg.parquet
└── 2021/
    └── ...
"""
# ─────────────────────────────────────────────────────────────────────────────
# NASA/GISS Zone Definitions
# ─────────────────────────────────────────────────────────────────────────────
LATITUDE_ZONES = {
    "ocean_90S_25S": (-90, -25),
    "ocean_25S_0":   (-25,   0),
    "ocean_0_25N":   (  0,  25),
    "ocean_25N_90N": ( 25,  90),
    "land_90S_60S":  (-90, -60),
    "land_60S_30S":  (-60, -30),
    "land_30S_0":    (-30,   0),
    "land_0_30N":    (  0,  30),
    "land_30N_60N":  ( 30,  60),
    "land_60N_90N":  ( 60,  90),
}

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        mins, secs = divmod(elapsed, 60)
        print(f"  [TIME] {func.__name__} took {int(mins)}m {secs:.1f}s")
        return result
    return wrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Extract ERA5 area-weighted zonal features.")
    parser.add_argument('--year',       nargs='+', type=int, required=True,
                        help="Year or range: --year 2020 or --year 2020 2023")
    parser.add_argument('--month',      nargs='+', type=int,
                        help="Month or range: --month 1 or --month 1 12. Defaults to full year.")
    parser.add_argument('--path_output', type=str, default="era5_zonal_features",
                        help="Root directory for final zonal parquet outputs.")
    parser.add_argument('--path_cache', type=str, default="era5_coarsened_cache",
                        help="Root directory for intermediate coarsened cache.")
    parser.add_argument('--resolution', type=float, default=2.0,
                        help="Target resolution in degrees for coarsening (default: 2.0).")
    return parser.parse_args()

def get_range(input_list):
    if not input_list:
        return None, None
    start = input_list[0]
    end   = input_list[1] if len(input_list) > 1 else start
    return start, end

def month_folder(root, year, month):
    """
    Returns path: <root>/<year>/<month:02d>/
    Creates it if it doesn't exist.
    
    Example: era5_zonal_features/2020/01/
    """
    path = os.path.join(root, str(year), f"{month:02d}")
    os.makedirs(path, exist_ok=True)
    return path

def build_cos_lat_weights(da):
    """cos(lat) area weights broadcast to (lat, lon)."""
    lat_rad  = np.deg2rad(da.latitude)
    weights  = np.cos(lat_rad)
    weights_2d = weights * xr.ones_like(da.isel(valid_time=0, drop=True))
    return weights_2d.fillna(0)

def load_land_sea_mask(ds, resolution):
    """Load and coarsen ERA5 land-sea mask to working resolution."""
    if "lsm" not in ds:
        print("  Warning: 'lsm' not found. Falling back to all-ocean mask.")
        lsm = xr.zeros_like(ds.t2m.isel(valid_time=0, drop=True))
    else:
        lsm = ds.lsm.isel(valid_time=0, drop=True)

    factor = max(1, int(resolution / 0.25))
    if factor > 1:
        lsm = lsm.coarsen(latitude=factor, longitude=factor, boundary="trim").mean()
    return lsm

def coarsen_to_resolution(da, resolution):
    """Subsample from ERA5 native 0.25° to target resolution."""
    factor = max(1, int(resolution / 0.25))
    if factor > 1:
        return da.coarsen(latitude=factor, longitude=factor, boundary="trim").mean()
    return da

def compute_zonal_features(daily_t2m, lsm, zones):
    """
    Compute area-weighted daily mean temperature per zone.

    Parameters
    ----------
    daily_t2m : xr.DataArray (valid_time, lat, lon) — Celsius, daily means, coarsened
    lsm       : xr.DataArray (lat, lon)             — land-sea mask at same resolution
    zones     : dict zone_name -> (lat_min, lat_max)

    Returns
    -------
    pd.DataFrame columns: valid_time, zone, temp_c
    """
    records     = []
    cos_weights = build_cos_lat_weights(daily_t2m)

    for zone_name, (lat_min, lat_max) in zones.items():
        is_ocean = zone_name.startswith("ocean_")

        # ERA5 latitude is descending → slice(max, min)
        zone_t2m = daily_t2m.sel(latitude=slice(lat_max, lat_min))
        zone_lsm = lsm.sel(latitude=slice(lat_max, lat_min))
        zone_w   = cos_weights.sel(latitude=slice(lat_max, lat_min))

        mask       = (zone_lsm < 0.5).astype(float) if is_ocean else (zone_lsm >= 0.5).astype(float)
        combined_w = zone_w * mask

        if float(combined_w.sum()) == 0:
            print(f"  Warning: zone '{zone_name}' has zero weight — skipping.")
            continue

        zone_mean = (
            (zone_t2m * combined_w).sum(dim=["latitude", "longitude"])
            / combined_w.sum()
        ).compute()

        df = zone_mean.to_dataframe(name="temp_c").reset_index()
        df["zone"] = zone_name
        records.append(df)

    return pd.concat(records, ignore_index=True)

@timeit
def process_month(ds, lsm, year, month, resolution, path_output, path_cache):
    """
    Full pipeline for a single (year, month):
      1. Check if final output already exists → skip
      2. Check if coarsened cache exists → load it, else download + coarsen + cache
      3. Compute zonal features
      4. Save to output/<year>/<month>/zonal.parquet
    """
    # ── Output path ──────────────────────────────────────────────────────────
    out_dir  = month_folder(path_output, year, month)
    out_file = os.path.join(out_dir, "zonal.parquet")

    if os.path.exists(out_file):
        print(f"  [SKIP] Output exists: {out_file}")
        return

    # ── Coarsened cache path ──────────────────────────────────────────────────
    cache_dir  = month_folder(path_cache, year, month)
    cache_file = os.path.join(cache_dir, f"t2m_coarsened_{resolution}deg.parquet")

    start_date = f"{year}-{month:02d}-01"
    # Use next month to define end cleanly, then trim with the slice
    # With this:
    last_day = calendar.monthrange(year, month)[1]
    end_date = f"{year}-{month:02d}-{last_day:02d}"

    # ── Step 1: Load or build coarsened cache ─────────────────────────────────
    if os.path.exists(cache_file):
        print(f"  [CACHE HIT] Loading coarsened data: {cache_file}")
        df_cache   = pd.read_parquet(cache_file)
        # Reconstruct xarray DataArray from cached parquet
        daily_t2m  = (
            df_cache
            .set_index(["valid_time", "latitude", "longitude"])["temp_c"]
            .to_xarray()
        )
    else:
        print(f"  [DOWNLOAD] Fetching ERA5: {start_date} → {end_date}")
        t2m_raw = ds.t2m.sel(valid_time=slice(start_date, end_date))

        print(f"  [COARSEN] → {resolution}° resolution")
        t2m_coarse = coarsen_to_resolution(t2m_raw, resolution)

        print("  [RESAMPLE] → daily means")
        daily_t2m  = (
            (t2m_coarse.astype("float32") - 273.15)
            .resample(valid_time="1D").mean()
            .compute()                         # materialise before caching
        )

        # Save coarsened cache as parquet
        df_cache = (
            daily_t2m
            .to_dataframe(name="temp_c")
            .reset_index()
        )
        df_cache.to_parquet(cache_file, index=False)
        print(f"  [CACHE SAVED] {cache_file}  "
              f"({os.path.getsize(cache_file)/1024**2:.1f} MB)")

    # ── Step 2: Compute zonal features ───────────────────────────────────────
    print("  [ZONES] Computing area-weighted zonal means...")
    df_zonal = compute_zonal_features(daily_t2m, lsm, LATITUDE_ZONES)
    df_zonal["year"]  = year
    df_zonal["month"] = month

    df_zonal.to_parquet(out_file, index=False)
    print(f"  [SAVED] {out_file}")


def main():
    args = parse_args()

    PAT = os.getenv("earth_data_hub")
    if not PAT:
        print("Error: 'earth_data_hub' env variable not found.")
        sys.exit(1)

    os.makedirs(args.path_output, exist_ok=True)
    os.makedirs(args.path_cache,  exist_ok=True)

    year_start,  year_end  = get_range(args.year)
    month_start, month_end = get_range(args.month)
    if month_start is None:
        month_start, month_end = 1, 12

    print("Connecting to Earth Data Hub...")
    ds = xr.open_dataset(
        f"https://edh:{PAT}@data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr",
        chunks={},
        engine="zarr",
    )

    print(f"Loading land-sea mask at {args.resolution}° resolution...")
    lsm = load_land_sea_mask(ds, args.resolution)

    # ── Main loop: year → month ───────────────────────────────────────────────
    for year in range(year_start, year_end + 1):
        for month in range(month_start, month_end + 1):
            print(f"\n{'─'*60}")
            print(f"  Year: {year}  |  Month: {month:02d}")
            print(f"{'─'*60}")
            try:
                process_month(
                    ds, lsm, year, month,
                    args.resolution,
                    args.path_output,
                    args.path_cache,
                )
            except Exception as e:
                print(f"  [ERROR] {year}-{month:02d}: {e}")
                raise

    print("\nDone.")

if __name__ == "__main__":
    main()
