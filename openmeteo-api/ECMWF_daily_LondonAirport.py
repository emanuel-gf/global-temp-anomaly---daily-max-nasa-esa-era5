import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import os 
import datetime

def fetch_ecmwf_data(start_date: str, end_date: str, latitude: float, longitude: float):
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Use the historical-forecast-api for dates in the past
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"

    metrics = [
        "temperature_2m", "cloud_cover", "wind_speed_10m", 
        "cape", "rain", "pressure_msl", 
        "shortwave_radiation", "direct_radiation"
    ]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": metrics,
        "models": "ecmwf_ifs025", # High-res ECMWF
    }
    
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}

    # Map variables dynamically based on the metrics list
    for i, metric_name in enumerate(metrics):
        hourly_data[metric_name] = hourly.Variables(i).ValuesAsNumpy()

    return pd.DataFrame(data = hourly_data)

def main():
    # Configuration for long-term ECMWF pull
    total_start = "2017-01-01"
    total_end = "2026-03-01" # Assuming current date is early 2026
    latitude = 51.505  # London City Airport
    longitude = 0.055
    
    dir_save = "/home/camarada/Documents/projects/temp-grss-nasa/data_/ecmwf_hres_hourly/"
    os.makedirs(dir_save, exist_ok=True)

    start_dt = datetime.datetime.strptime(total_start, "%Y-%m-%d").date()
    end_dt = datetime.datetime.strptime(total_end, "%Y-%m-%d").date()

    for year in range(start_dt.year, end_dt.year + 1):
        year_start = max(start_dt, datetime.date(year, 1, 1))
        year_end = min(end_dt, datetime.date(year, 12, 31))
        
        str_start = year_start.strftime("%Y-%m-%d")
        str_end = year_end.strftime("%Y-%m-%d")

        print(f"--- Fetching ECMWF for {year}: {str_start} to {str_end} ---")
        
        try:
            df = fetch_ecmwf_data(str_start, str_end, latitude, longitude)
            
            file_name = f"londocity_ECMWF_{year}.parquet"
            save_path = os.path.join(dir_save, file_name)

            df.to_parquet(save_path, index=False)
            print(f"Saved {len(df)} rows to: {save_path}")
            
        except Exception as e:
            print(f"Error for year {year}: {e}")

if __name__ == "__main__":
    main()