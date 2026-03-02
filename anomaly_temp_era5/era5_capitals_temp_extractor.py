import os
import argparse
import pandas as pd
import xarray as xr
import sys
from dotenv import load_dotenv
load_dotenv()
def parse_args():
    parser = argparse.ArgumentParser(description="Extract ERA5 data for capitals.")
    
    # Flags for year and month (accepts 1 or 2 values)
    parser.add_argument('--year', nargs='+', type=int, required=True,
                        help="Year or range: --year 2023 or --year 2020 2023")
    parser.add_argument('--month', nargs='+', type=int,
                        help="Month or range: --month 1 or --month 1 12. Defaults to full year.")
    
    # Other flags
    parser.add_argument('--csvcapitals', type=str, default="/home/camarada/Documents/projects/temp-grss-nasa/data_/geolocated_capitals/geolocated_capitals.csv",
                        help="Path to the capitals CSV file.")
    parser.add_argument('--path_output', type=str, default="era5_yearly_chunks",
                        help="Directory to save output files.")
    
    return parser.parse_args()

def get_range(input_list):
    """Converts a list of 1 or 2 elements into a start/end range."""
    if not input_list:
        return None, None
    start = input_list[0]
    end = input_list[1] if len(input_list) > 1 else start
    return start, end

def main():
    args = parse_args()
    
    # 1. Setup Environment
    PAT = os.getenv("earth_data_hub")
    if not PAT:
        print("Error: environment variable 'earth_data_hub' not found.")
        sys.exit(1)
        
    os.makedirs(args.path_output, exist_ok=True)
    
    # 2. Load Metadata
    df_capitals = pd.read_csv(args.csvcapitals)
    lats = xr.DataArray(df_capitals['lat'].values, dims="point")
    lons = xr.DataArray(df_capitals['lon'].values, dims="point")
    
    # 3. Handle Time Ranges
    year_start, year_end = get_range(args.year)
    month_start, month_end = get_range(args.month)
    
    # Default to full year if month is not provided
    if month_start is None:
        month_start, month_end = 1, 12

    print("Connecting to Earth Data Hub...")
    ds = xr.open_dataset(
        f"https://edh:{PAT}@data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr",
        chunks={},
        engine="zarr",
    )

    # 4. Processing Loop
    for year in range(year_start, year_end + 1):
        file_path = os.path.join(args.path_output, f"era5_{year}_{month_start}-{month_end}.parquet")
        
        if os.path.exists(file_path):
            print(f"Skipping {year} (File exists: {file_path})")
            continue
            
        print(f"--- Processing {year} (Months {month_start} to {month_end}) ---")
        
        try:
            # Define time slice strings
            start_date = f"{year}-{month_start:02d}-01"
            # Logic to handle month end (simplified for the slice)
            end_date = f"{year}-{month_end:02d}-31" 

            # Selection and computation
            print(f"Selecting time range:{start_date} to {end_date }")
            year_ds = ds.t2m.sel(valid_time=slice(start_date, end_date))
            
            points_in_year = year_ds.sel(
                latitude=lats, 
                longitude=lons, 
                method="nearest"
            )
            
            # Unit conversion and daily resampling
            daily_data = (points_in_year.astype("float32") - 273.15).resample(valid_time="1D").mean().compute()
            
            # Prepare final DataFrame
            df_year = daily_data.to_dataframe(name="temp_c").reset_index()
            df_final = df_year.merge(df_capitals, left_on="point", right_index=True)
            
            df_final.to_parquet(file_path, index=False)
            print(f"Successfully saved: {file_path}")
            
        except Exception as e:
            print(f"Failed processing {year}: {e}")

if __name__ == "__main__":
    main()