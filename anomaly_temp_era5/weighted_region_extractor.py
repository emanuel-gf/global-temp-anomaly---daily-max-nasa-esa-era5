import os
import argparse
import pandas as pd
import numpy as np
import xarray as xr
import sys
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────
# NASA/GISS Zone Definitions
# Mirrors exactly how NASA aggregates:
#   - 4 ocean zones
#   - 6 latitude bands for land
# ─────────────────────────────────────────────
LATITUDE_ZONES = {
    # NASA ocean zones (used for SST weighting)
    "ocean_90S_25S": (-90, -25),
    "ocean_25S_0":   (-25,   0),
    "ocean_0_25N":   (  0,  25),
    "ocean_25N_90N": ( 25,  90),
    
    # Land zones (broader bands)
    "land_90S_60S":  (-90, -60),
    "land_60S_30S":  (-60, -30),
    "land_30S_0":    (-30,   0),
    "land_0_30N":    (  0,  30),
    "land_30N_60N":  ( 30,  60),
    "land_60N_90N":  ( 60,  90),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Extract ERA5 area-weighted zonal features.")
    parser.add_argument('--year',  nargs='+', type=int, required=True)
    parser.add_argument('--month', nargs='+', type=int)
    parser.add_argument('--path_output', type=str, default="era5_zonal_features")
    parser.add_argument('--resolution', type=float, default=2.0,
                        help="Subsample ERA5 to this degree resolution before averaging (default: 2.0). "
                             "Reduces memory and compute. ERA5 native is 0.25°.")
    return parser.parse_args()

def get_range(input_list):
    if not input_list:
        return None, None
    start = input_list[0]
    end = input_list[1] if len(input_list) > 1 else start
    return start, end

def build_cos_lat_weights(da):
    """
    Build a 2D DataArray of cos(lat) weights matching da's lat/lon grid.
    This is the standard area-weighting for spherical grids.
    """
    lat_rad = np.deg2rad(da.latitude)
    weights = np.cos(lat_rad)
    # Broadcast to 2D (lat, lon) — every lon gets the same lat weight
    weights_2d = weights.expand_dims({"longitude": da.longitude}, axis=-1)
    weights_2d = weights_2d * xr.ones_like(da.isel(valid_time=0, drop=True))
    return weights_2d.fillna(0)

def load_land_sea_mask(ds, resolution):
    """
    ERA5 single-levels contains 'lsm' (land-sea mask): 
      1.0 = land, 0.0 = sea
    We coarsen it to match our working resolution.
    Returns a DataArray of shape (lat, lon) with values in [0, 1].
    """
    if "lsm" not in ds:
        print("Warning: 'lsm' not found in dataset. Using all-ocean mask fallback.")
        lsm = xr.zeros_like(ds.t2m.isel(valid_time=0, drop=True))
    else:
        lsm = ds.lsm.isel(valid_time=0, drop=True)  # static field
    
    # Coarsen to working resolution
    factor = max(1, int(resolution / 0.25))
    if factor > 1:
        lsm = lsm.coarsen(latitude=factor, longitude=factor, boundary="trim").mean()
    return lsm

def coarsen_to_resolution(da, resolution):
    """Subsample ERA5 from 0.25° to target resolution to reduce memory."""
    factor = max(1, int(resolution / 0.25))
    if factor > 1:
        return da.coarsen(latitude=factor, longitude=factor, boundary="trim").mean()
    return da

def compute_zonal_features(daily_t2m, lsm, zones):
    """
    For each zone definition, compute the area-weighted daily mean temperature,
    separately for land and ocean cells.

    Parameters
    ----------
    daily_t2m : xr.DataArray (valid_time, lat, lon) in Celsius, daily means
    lsm       : xr.DataArray (lat, lon), values in [0, 1]
    zones     : dict of zone_name -> (lat_min, lat_max)

    Returns
    -------
    pd.DataFrame with columns: valid_time, zone_name, temp_c
    """
    records = []
    cos_weights = build_cos_lat_weights(daily_t2m)

    for zone_name, (lat_min, lat_max) in zones.items():
        is_ocean_zone = zone_name.startswith("ocean_")
        
        # 1. Spatial slice
        zone_t2m = daily_t2m.sel(latitude=slice(lat_max, lat_min))  # ERA5 lat is descending
        zone_lsm = lsm.sel(latitude=slice(lat_max, lat_min))
        zone_w   = cos_weights.sel(latitude=slice(lat_max, lat_min))

        # 2. Build land/ocean mask
        if is_ocean_zone:
            # Ocean: cells where lsm < 0.5 (majority ocean)
            mask = (zone_lsm < 0.5).astype(float)
        else:
            # Land: cells where lsm >= 0.5
            mask = (zone_lsm >= 0.5).astype(float)

        # 3. Combined weights: cos(lat) × land_or_ocean_mask
        combined_w = zone_w * mask

        # 4. Weighted mean over (lat, lon) for each time step
        # sum(T * w) / sum(w)
        weighted_sum = (zone_t2m * combined_w).sum(dim=["latitude", "longitude"])
        weight_total = combined_w.sum()

        if float(weight_total) == 0:
            print(f"  Warning: Zone '{zone_name}' has zero weight — skipping.")
            continue

        zone_mean = (weighted_sum / weight_total).compute()

        # 5. To dataframe
        df_zone = zone_mean.to_dataframe(name="temp_c").reset_index()
        df_zone["zone"] = zone_name
        records.append(df_zone)

    return pd.concat(records, ignore_index=True)


def main():
    args = parse_args()
    PAT = os.getenv("earth_data_hub")
    if not PAT:
        print("Error: environment variable 'earth_data_hub' not found.")
        sys.exit(1)

    os.makedirs(args.path_output, exist_ok=True)
    year_start, year_end = get_range(args.year)
    month_start, month_end = get_range(args.month)
    if month_start is None:
        month_start, month_end = 1, 12

    print("Connecting to Earth Data Hub...")
    ds = xr.open_dataset(
        f"https://edh:{PAT}@data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr",
        chunks={},
        engine="zarr",
    )

    # Load static land-sea mask once
    print(f"Loading land-sea mask at {args.resolution}° resolution...")
    lsm = load_land_sea_mask(ds, args.resolution)

    for year in range(year_start, year_end + 1):
        file_path = os.path.join(args.path_output, f"zonal_{year}_{month_start:02d}-{month_end:02d}.parquet")

        if os.path.exists(file_path):
            print(f"Skipping {year} (exists: {file_path})")
            continue

        print(f"\n--- Processing {year} (Months {month_start}–{month_end}) ---")

        try:
            start_date = f"{year}-{month_start:02d}-01"
            end_date   = f"{year}-{month_end:02d}-31"

            print(f"  Selecting time slice: {start_date} → {end_date}")
            t2m_raw = ds.t2m.sel(valid_time=slice(start_date, end_date))

            # Coarsen to working resolution (saves huge memory)
            print(f"  Coarsening to {args.resolution}° resolution...")
            t2m_coarse = coarsen_to_resolution(t2m_raw, args.resolution)

            # Convert K → °C and resample to daily mean
            print("  Resampling to daily means...")
            t2m_daily = (t2m_coarse.astype("float32") - 273.15).resample(valid_time="1D").mean()

            # Compute zonal features
            print("  Computing area-weighted zonal features...")
            df_zonal = compute_zonal_features(t2m_daily, lsm, LATITUDE_ZONES)

            df_zonal.to_parquet(file_path, index=False)
            print(f"  Saved: {file_path}")

        except Exception as e:
            print(f"  Failed for {year}: {e}")
            raise


if __name__ == "__main__":
    main()