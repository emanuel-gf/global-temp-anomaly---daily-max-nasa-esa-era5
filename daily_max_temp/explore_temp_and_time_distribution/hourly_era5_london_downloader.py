"""
ERA5 Land hourly 2m temperature downloader.

Downloads data for a single lat/lon point from the DestinE Earth Data Hub
and saves monthly Parquet files under the structure:

You should have Earth_Data_Hub as a variable in your .env file. 
    output_dir/
    └── <YEAR>/
        └── <MONTH>/
            └── t2m_<YEAR>_<MONTH>.parquet

Usage
-----
    python download_era5.py \
        --lat 51.505 \
        --lon 0.055 \
        --date_start 2020-01-01 \
        --date_end 2020-06-30 \
        [--output_dir ./era5_data]
"""

import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import xarray as xr
import time 

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ERA5 Land 2m temperature for a point and save monthly Parquet files."
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=51.505,
        help="Latitude of the target point (default: 51.505).",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=0.055,
        help="Longitude of the target point (default: 0.055).",
    )
    parser.add_argument(
        "--date_start",
        required=True,
        help="Start date, inclusive, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--date_end",
        required=True,
        help="End date, inclusive, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--output_dir",
        default="./era5_data",
        help="Root output directory (default: ./era5_data).",
    )
    return parser.parse_args()


def build_yearly_windows(
    date_start: str, date_end: str
) -> list[tuple[str, str, int]]:
    """
    Split [date_start, date_end] into (year_start, year_end, year) triples,
    one per calendar year contained in the range.
 
    Example
    -------
    2019-06-01 → 2021-03-31  yields:
        ("2019-06-01", "2019-12-31", 2019)
        ("2020-01-01", "2020-12-31", 2020)
        ("2021-01-01", "2021-03-31", 2021)
    """
    start = pd.Timestamp(date_start)
    end = pd.Timestamp(date_end)
 
    if start > end:
        raise ValueError("--date_start must be before or equal to --date_end.")
 
    windows: list[tuple[str, str, int]] = []
    cursor = start
 
    while cursor <= end:
        year = cursor.year
        year_end = pd.Timestamp(f"{year}-12-31")
        window_end = min(year_end, end)
        windows.append((cursor.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d"), year))
        cursor = pd.Timestamp(f"{year + 1}-01-01")
 
    return windows


def open_zarr_point(
    PAT,
    lat: float,
    lon: float,
    variables:list = ["t2m", "ssrd"]
) -> pd.DataFrame:
    """Pre loading zarr operation."""
    
    ds_point = xr.open_dataset(
        f"https://edh:{PAT}@data.earthdatahub.destine.eu/era5/reanalysis-era5-land-no-antartica-v0.zarr",   # chunk by ~1 month of hours
        engine="zarr",
        chunks = {},
    )[variables]    

    return ds_point.sel(latitude=lat, longitude=lon, method="nearest")

def extract_xarray(
        ds: xr.Dataset,
        year:int,
        date_start: str,
        date_end: str,
        lat:float,
        lon: float,
    ):
    annual = (
                ds
                .sel(valid_time=slice(date_start, date_end))
                .compute()  # triggers the actual remote fetch
            )
    
    ## convert to df
    # Trigger the actual download / computation
    df = annual.to_dataframe().reset_index()

    # Keep only the columns we care about
    df = df[["valid_time", "t2m","ssrd"]].copy()
    df.rename(columns={"valid_time": "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Drop spatial index columns that xarray adds after .sel()
    df = df.drop(columns=["latitude", "longitude"], errors="ignore")
    
    ## convert to degree
    df["t2m"] = df["t2m"].astype("float32") - 273.15

    # ssrd: accumulated → hourly flux (W/m²)
    # ERA5 accumulates from 00:00 UTC, resets at midnight
    # diff within each day, clip negatives (reset artifact)
    df = df.sort_values("datetime")
    df["ssrd"] = df.groupby(df["datetime"].dt.date)["ssrd"].diff().clip(lower=0).fillna(df["ssrd"])

    ## enforce order 
    df = df[["datetime", "t2m", "ssrd"]].reset_index(drop=True)
 
    return df


def save_monthly_parquet(df: pd.DataFrame, output_dir: str) -> None:
    """Split a DataFrame by year/month and save each chunk as a Parquet file."""
    root = Path(output_dir)

    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month

    groups = df.groupby(["year", "month"])
    total = len(groups)

    for idx, ((year, month), chunk) in enumerate(groups, start=1):
        month_str = f"{month:02d}"
        folder = root / str(year) / month_str
        folder.mkdir(parents=True, exist_ok=True)

        filepath = folder / f"t2m_{year}_{month_str}.parquet"

        # Drop helper columns before saving
        chunk = chunk.drop(columns=["year", "month"])
        chunk.to_parquet(filepath, index=False)

        print(f"[{idx}/{total}] Saved → {filepath}  ({len(chunk)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    EARTH_HUB_KEY = os.environ.get("earth_data_hub")

    args = parse_args()

    # Validate dates
    try:
        start = pd.Timestamp(args.date_start)
        end = pd.Timestamp(args.date_end)
    except Exception as exc:
        raise ValueError(f"Invalid date format: {exc}") from exc

    if start > end:
        raise ValueError("--date_start must be before --date_end.")

    print(f"[info] Period  : {start.date()} → {end.date()}")
    print(f"[info] Location: lat={args.lat}, lon={args.lon}")
    print(f"[info] Output  : {args.output_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    windows = build_yearly_windows(args.date_start, args.date_end)

     # Open the remote store once and reuse across years
    ds_point = open_zarr_point(EARTH_HUB_KEY, args.lat, args.lon)

    for date_start, date_end, year in windows:
        start_time_monitor = time.perf_counter()
        outpath = output_dir / f"era5_{year}.parquet"

        if Path(outpath).exists():
            print(f"[skip] {outpath} already exists — delete to re-download.")
            continue
            end_time_monitor= time.perf_counter()
        else:
            df = extract_xarray(ds_point,
                                year,
                                date_start,
                                date_end,
                                args.lat,
                                args.lon)

            print(f"[info] Downloaded {len(df):,} hourly records.")
            end_time_monitor = time.perf_counter()
            print(f"TIME EXECUTION: {start_time_monitor - end_time_monitor}")
            df.to_parquet(outpath, index=False)
            print(f"[saved] {outpath}  ({len(df):,} rows, {df['datetime'].min().date()} → {df['datetime'].max().date()})")

    print("[done] All files written successfully.")


if __name__ == "__main__":
    main()