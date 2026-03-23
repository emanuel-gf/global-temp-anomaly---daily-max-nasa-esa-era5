import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import os 
import datetime
import json
from timezonefinder import TimezoneFinder

def fetch_weather_with_metadata(start_date, end_date, lat, lon, model_name, folder_path):
    # 1. Setup API client
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # 2. Determine Local Timezone automatically
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lng=lon, lat=lat) or "UTC"

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    metrics = ["temperature_2m", "cloud_cover", "wind_speed_10m", "rain", "pressure_msl"]

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": metrics,
        "models": model_name,
    }
    
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # 3. Process Hourly Data
    hourly = response.Hourly()
    # Create the UTC index first
    utc_index = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )

    hourly_data = {"datetime_utc": utc_index}
    for i, metric in enumerate(metrics):
        hourly_data[metric] = hourly.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data=hourly_data)

    # 4. Convert to Local Time and remove TZ-awareness for easy concatenation
    df['datetime_local'] = df['datetime_utc'].dt.tz_convert(tz_name).dt.tz_localize(None)
    
    # 5. Save Metadata
    metadata = {
        "location": {"lat": lat, "lon": lon},
        "period": {"start": start_date, "end": end_date},
        "model": model_name,
        "timezone": tz_name,
        "utc_offset_at_request": response.UtcOffsetSeconds(),
        "generated_at": datetime.datetime.now().isoformat()
    }
    
    # Filename logic
    base_name = f"weather_{lat}_{lon}_{start_date[:4]}"
    df.to_parquet(os.path.join(folder_path, f"{base_name}.parquet"), index=False)
    
    with open(os.path.join(folder_path, f"{base_name}_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)

    return df, metadata

# Example Usage
if __name__ == "__main__":
    dir_path = "data_/weather_exports"
    os.makedirs(dir_path, exist_ok=True)
    
    df, meta = fetch_weather_with_metadata(
        "2023-01-01", "2023-12-31", 
        51.505, 0.055, 
        "ecmwf_ifs025", 
        dir_path
    )
    print(f"Data saved for {meta['timezone']}. Local time head:")
    print(df[['datetime_utc', 'datetime_local']].head())