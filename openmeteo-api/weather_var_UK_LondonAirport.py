import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import os 
import datetime

"""
Fetch data from open-meteo using the UK meteo 2km model for London City Airpot (EGLC) and save it as parquet files. 

"""

def fetch_weather_data(start_date: str, 
                        end_date:str,
                        latitude: float,
                        longitude: float):
        """
        Fetches ECMWF IFS for the given parameters.
        It looks for a single point, which is also passed inside the params dict

        Args:
            openmeteo = Object API
            params: dict 
                Dict of the API parameters.
        """
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)
        
        ## url = "https://archive-api.open-meteo.com/v1/archive"

        ## This retrives the UK UKmeteo 2km model, which the latest date is 2022-03-01.

        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ["temperature_2m", "cloud_cover_low", "cloud_cover_mid",
                        "cloud_cover_high", "dew_point_2m", "pressure_msl", "surface_pressure",
                          "wind_speed_10m", "vapour_pressure_deficit"
                          ],
            "models": "ukmo_uk_deterministic_2km",
        }
        responses = openmeteo.weather_api(url, params=params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]
        print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
        print(f"Elevation: {response.Elevation()} m asl")
        print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_cloud_cover_low = hourly.Variables(1).ValuesAsNumpy()
        hourly_cloud_cover_mid = hourly.Variables(2).ValuesAsNumpy()
        hourly_cloud_cover_high = hourly.Variables(3).ValuesAsNumpy()
        hourly_dew_point_2m = hourly.Variables(4).ValuesAsNumpy()
        hourly_pressure_msl = hourly.Variables(5).ValuesAsNumpy()
        hourly_surface_pressure = hourly.Variables(6).ValuesAsNumpy()
        hourly_wind_speed_10m = hourly.Variables(7).ValuesAsNumpy()
        hourly_vapour_pressure_deficit = hourly.Variables(8).ValuesAsNumpy()

        hourly_data = {"date": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
            end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )}

        hourly_data["temperature_2m"] = hourly_temperature_2m
        hourly_data["cloud_cover_low"] = hourly_cloud_cover_low
        hourly_data["cloud_cover_mid"] = hourly_cloud_cover_mid
        hourly_data["cloud_cover_high"] = hourly_cloud_cover_high
        hourly_data["dew_point_2m"] = hourly_dew_point_2m
        hourly_data["pressure_msl"] = hourly_pressure_msl
        hourly_data["surface_pressure"] = hourly_surface_pressure
        hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
        hourly_data["vapour_pressure_deficit"] = hourly_vapour_pressure_deficit

        hourly_dataframe = pd.DataFrame(data = hourly_data)
        
        return hourly_dataframe


def main():
    # Configuration
    total_start = "2022-03-01"
    # Setting end date to just a few days ago to ensure data availability
    total_end = "2026-02-15" 
    latitude = 51.505 
    longitude = 0.055
    
    dir_save = "data_/london_hourly/"
    os.makedirs(dir_save, exist_ok=True)

    # Convert strings to date objects for logic
    start_dt = datetime.datetime.strptime(total_start, "%Y-%m-%d").date()
    end_dt = datetime.datetime.strptime(total_end, "%Y-%m-%d").date()

    # Loop through years
    for year in range(start_dt.year, end_dt.year + 1):
        # Determine the start date for this specific year chunk
        # Use total_start if it's the first year, otherwise Jan 1st
        year_start = max(start_dt, datetime.date(year, 1, 1))
        
        # Determine the end date for this specific year chunk
        # Use total_end if it's the final year, otherwise Dec 31st
        year_end = min(end_dt, datetime.date(year, 12, 31))
        
        # Format dates back to strings for the API
        str_start = year_start.strftime("%Y-%m-%d")
        str_end = year_end.strftime("%Y-%m-%d")

        print(f"--- Fetching data for {year}: {str_start} to {str_end} ---")
        
        try:
            weather_history_df = fetch_weather_data(
                start_date=str_start, 
                end_date=str_end,
                latitude=latitude, 
                longitude=longitude
            )

            # Create specific filename for this year
            file_name = f"londocityairport_UKV_{year}.parquet"
            save_path = os.path.join(dir_save, file_name)

            # Save results
            weather_history_df.to_parquet(save_path, index=False)
            print(f"Successfully saved: {save_path}")
            
        except Exception as e:
            print(f"Error fetching data for year {year}: {e}")

if __name__ == "__main__":
    main()