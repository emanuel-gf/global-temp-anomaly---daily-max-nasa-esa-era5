import os
import math
import calendar
import argparse
import pandas as pd
import numpy as np
import xarray as xr
import sys
import time
import signal
import threading
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

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

# ─────────────────────────────────────────────────────────────────────────────
# Progress Tracker — survives Ctrl+C
# ─────────────────────────────────────────────────────────────────────────────
class ProgressTracker:
    def __init__(self, all_tasks, args):
        self.all_tasks     = all_tasks
        self.args          = args
        self.completed     = []
        self.failed        = []
        self._lock         = threading.Lock()
        self._stop_event   = threading.Event()   # ← replaces sys.exit

        signal.signal(signal.SIGINT,  self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def should_stop(self):
        return self._stop_event.is_set()

    def _handle_interrupt(self, signum, frame):
        # Only set the flag — never call sys.exit() from here
        self._stop_event.set()
    def mark_done(self, year, month):
        with self._lock:
            self.completed.append((year, month))

    def mark_failed(self, year, month, error):
        with self._lock:
            self.failed.append((year, month, str(error)))

    def pending(self):
        done = set(self.completed)
        return [(y, m) for y, m in self.all_tasks if (y, m) not in done]

    def print_summary(self):
        print(f"\n{'═'*60}")
        print(f"  PROGRESS SUMMARY")
        print(f"{'═'*60}")
        print(f"  Completed : {len(self.completed)} / {len(self.all_tasks)}")
        print(f"  Failed    : {len(self.failed)}")

        if self.failed:
            print(f"\n  Failed months:")
            for y, m, err in self.failed:
                print(f"    {y}-{m:02d}  →  {err}")

        remaining = self.pending()
        if remaining:
            first_y, first_m = remaining[0]
            last_y,  last_m  = remaining[-1]
            args = self.args
            print(f"\n  To resume from where you stopped, run:")
            print(f"    python {sys.argv[0]} \\")
            print(f"      --year {first_y} {last_y} \\")
            print(f"      --month {first_m} {last_m} \\")
            print(f"      --path_output {args.path_output} \\")
            print(f"      --path_cache {args.path_cache} \\")
            print(f"      --resolution {args.resolution} \\")
            print(f"      --workers {args.workers}")
        print(f"{'═'*60}\n")

    def _handle_interrupt(self, signum, frame):
        print("\n\n  [INTERRUPTED] Caught signal — finishing in-flight tasks...")
        self.print_summary()
        sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Decorators
# ─────────────────────────────────────────────────────────────────────────────
def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start  = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        mins, secs = divmod(elapsed, 60)
        print(f"  [TIME] {func.__name__} took {int(mins)}m {secs:.1f}s")
        return result
    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Extract ERA5 area-weighted zonal features.")
    parser.add_argument('--year',        nargs='+', type=int, required=True,
                        help="Year or range: --year 2020 or --year 2020 2023")
    parser.add_argument('--month',       nargs='+', type=int,
                        help="Month or range: --month 1 or --month 1 12. Defaults to full year.")
    parser.add_argument('--path_output', type=str, default="era5_zonal_features")
    parser.add_argument('--path_cache',  type=str, default="era5_coarsened_cache")
    parser.add_argument('--resolution',  type=float, default=2.0,
                        help="Target resolution in degrees (default: 2.0).")
    parser.add_argument('--workers',     type=int, default=2,
                        help="Parallel download workers (default: 2). Increase carefully.")
    return parser.parse_args()

def get_range(input_list):
    if not input_list:
        return None, None
    start = input_list[0]
    end   = input_list[1] if len(input_list) > 1 else start
    return start, end

def month_folder(root, year, month):
    path = os.path.join(root, str(year), f"{month:02d}")
    os.makedirs(path, exist_ok=True)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# ERA5 Helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_cos_lat_weights(da):
    lat_rad    = np.deg2rad(da.latitude)
    weights_2d = np.cos(lat_rad) * xr.ones_like(da.isel(valid_time=0, drop=True))
    return weights_2d.fillna(0)

def load_land_sea_mask(ds, resolution):
    if "lsm" not in ds:
        print("  Warning: 'lsm' not found. Falling back to all-ocean mask.")
        lsm = xr.zeros_like(ds.t2m.isel(valid_time=0, drop=True))
    else:
        lsm = ds.lsm.isel(valid_time=0, drop=True)

    # Use same stride logic as download to keep grids aligned
    stride = max(1, int(resolution / 0.25))
    if stride > 1:
        lsm = lsm.isel(latitude=slice(None, None, stride),
                        longitude=slice(None, None, stride))
    return lsm

def stride_select(ds_var, resolution, start_date, end_date):
    """
    Option 2 core: select coarser resolution BEFORE downloading.
    Strides lat/lon index so only 1-in-N points are requested from the server,
    reducing chunk count by stride² (e.g. 64× for 2° from 0.25°).
    """
    stride = max(1, int(resolution / 0.25))

    lat_idx = ds_var.latitude[::stride]
    lon_idx = ds_var.longitude[::stride]

    return ds_var.sel(
        valid_time=slice(start_date, end_date),
        latitude=lat_idx,
        longitude=lon_idx,
    )

def compute_zonal_features(daily_t2m, lsm, zones):
    records     = []
    cos_weights = build_cos_lat_weights(daily_t2m)

    for zone_name, (lat_min, lat_max) in zones.items():
        is_ocean = zone_name.startswith("ocean_")

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


# ─────────────────────────────────────────────────────────────────────────────
# Core Month Processor
# ─────────────────────────────────────────────────────────────────────────────
MAX_RETRIES = 3
RETRY_BACKOFF = [5, 15, 30]   # seconds to wait between retries

@timeit
def process_month(ds, lsm, year, month, resolution, path_output, path_cache):
    out_file   = os.path.join(month_folder(path_output, year, month), "zonal.parquet")
    cache_file = os.path.join(month_folder(path_cache,  year, month),
                              f"t2m_coarsened_{resolution}deg.parquet")

    if os.path.exists(out_file):
        print(f"  [SKIP] Output exists: {out_file}")
        return

    last_day   = calendar.monthrange(year, month)[1]
    start_date = f"{year}-{month:02d}-01"
    end_date   = f"{year}-{month:02d}-{last_day:02d}"

    if os.path.exists(cache_file):
        print(f"  [CACHE HIT] {cache_file}")
        df_cache  = pd.read_parquet(cache_file)
        daily_t2m = (
            df_cache
            .set_index(["valid_time", "latitude", "longitude"])["temp_c"]
            .to_xarray()
        )
    else:
        # ── Retry loop for network errors ─────────────────────────────────
        daily_t2m = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"  [DOWNLOAD] {start_date} → {end_date}"
                      f"  (stride {int(resolution/0.25)}×)"
                      f"  attempt {attempt}/{MAX_RETRIES}")

                t2m_strided = stride_select(ds.t2m, resolution, start_date, end_date)

                print(f"  [RESAMPLE] → daily means")
                daily_t2m = (
                    (t2m_strided.astype("float32") - 273.15)
                    .resample(valid_time="1D").mean()
                    .compute()
                )
                break   # success — exit retry loop

            except Exception as e:
                print(f"  [RETRY {attempt}/{MAX_RETRIES}] {year}-{month:02d} failed: {e}")
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF[attempt - 1]
                    print(f"  Waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    raise   # re-raise after final attempt

        df_cache = daily_t2m.to_dataframe(name="temp_c").reset_index()
        df_cache.to_parquet(cache_file, index=False)
        size_mb = os.path.getsize(cache_file) / 1024**2
        print(f"  [CACHE SAVED] {cache_file}  ({size_mb:.1f} MB)")

    print(f"  [ZONES] Computing area-weighted zonal means...")
    df_zonal          = compute_zonal_features(daily_t2m, lsm, LATITUDE_ZONES)
    df_zonal["year"]  = year
    df_zonal["month"] = month
    df_zonal.to_parquet(out_file, index=False)
    print(f"  [SAVED] {out_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
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

    # ── Build full task list ──────────────────────────────────────────────────
    all_tasks = [
        (year, month)
        for year  in range(year_start,  year_end  + 1)
        for month in range(month_start, month_end + 1)
    ]

    tracker = ProgressTracker(all_tasks, args)

    print(f"\nTotal tasks : {len(all_tasks)}")
    print(f"Workers     : {args.workers}")
    print(f"Resolution  : {args.resolution}°  (stride {int(args.resolution/0.25)}×)\n")

    # ── Option 1: parallel execution ──────────────────────────────────────────
    def _run(task):
        year, month = task

        if tracker.should_stop():          # ← check flag before starting
            return

        print(f"\n{'─'*60}")
        print(f"  Year: {year}  |  Month: {month:02d}")
        print(f"{'─'*60}")
        try:
            process_month(ds, lsm, year, month, args.resolution, args.path_output, args.path_cache)
            tracker.mark_done(year, month)
        except Exception as e:
            tracker.mark_failed(year, month, e)
            print(f"  [ERROR] {year}-{month:02d}: {e}")
    
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run, task): task for task in all_tasks}
        for future in as_completed(futures):
            year, month = futures[future]
            try:
                future.result()
            except Exception as e:
                tracker.mark_failed(year, month, e)

    # Main thread handles exit cleanly
    if tracker.should_stop():
        print("\n  [INTERRUPTED] Stopped by user.")

    tracker.print_summary()


if __name__ == "__main__":
    main()