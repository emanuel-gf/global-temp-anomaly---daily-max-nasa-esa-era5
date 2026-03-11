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


def open_dataset(pat: str) -> xr.Dataset:
    """Open the ERA5 Zarr store (lazy, chunked)."""
    url = f"https://edh:{pat}@data.earthdatahub.destine.eu/era5/reanalysis-era5-land-no-antartica-v0.zarr"
    print("[info] Opening Zarr store …")
    ds = xr.open_dataset(url, chunks={}, engine="zarr")
    return ds


def extract_point_timeseries(
    ds: xr.Dataset,
    date_start: str,
    date_end: str,
    lat: float,
    lon: float,
) -> pd.DataFrame:
    """Slice the dataset in time and space, convert K→°C, return a DataFrame."""
    print(f"[info] Selecting point lat={lat}, lon={lon} …")
    ds_point = ds.sel(
        valid_time=slice(date_start, date_end),
        latitude=lat,
        longitude=lon,
        method="nearest",
    )

    print("[info] Converting to °C and computing …")
    t2m: xr.DataArray = ds_point["t2m"].astype("float32") - 273.15
    t2m.attrs["units"] = "°C"

    # Trigger the actual download / computation
    df = t2m.to_dataframe(name="t2m").reset_index()

    # Keep only the columns we care about
    df = df[["valid_time", "t2m"]].copy()
    df.rename(columns={"valid_time": "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["lat"] = lat
    df["lon"] = lon

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

    ds = open_dataset(EARTH_HUB_KEY)
    df = extract_point_timeseries(ds, args.date_start, args.date_end, args.lat, args.lon)

    print(f"[info] Downloaded {len(df):,} hourly records.")
    save_monthly_parquet(df, args.output_dir)

    print("[done] All files written successfully.")


if __name__ == "__main__":
    main()