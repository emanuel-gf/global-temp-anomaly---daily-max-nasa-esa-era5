import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import date

def fetch_weather_data(openmeteo, df_input, start_idx, end_idx,
                        start_date="2023-01-01" , end_date = "2026-01-31" ):
        """
        Fetches ECMWF IFS data for a slice of the dataframe.
        """
        df_chunk = df_input.iloc[start_idx:end_idx]
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": df_chunk['lat'].tolist(),
            "longitude": df_chunk['lon'].tolist(),
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_mean",
            "models": "ecmwf_ifs",
            "timezone": "auto"
        }

        # API call returns a list of responses (one per location)
        responses = openmeteo.weather_api(url, params=params)
        
        all_city_frames = []

        for i, response in enumerate(responses):
            # Accessing attributes via function calls as per FlatBuffers requirement
            daily = response.Daily()
            
            # Documentation method for time range construction
            start_ts = pd.to_datetime(daily.Time(), unit="s", utc=True)
            end_ts = pd.to_datetime(daily.TimeEnd(), unit="s", utc=True)
            
            # Variables(0) is temperature_2m_mean
            temp_mean = daily.Variables(0).ValuesAsNumpy()
            
            # Get city name from the original dataframe slice
            city_name = df_chunk.iloc[i]['capital']
            
            # Create DataFrame for this specific location
            city_df = pd.DataFrame({
                "date": pd.date_range(
                    start=start_ts,
                    end=end_ts,
                    freq=pd.Timedelta(seconds=daily.Interval()),
                    inclusive="left"
                ),
                "capital": city_name,
                "avg_temp_c": temp_mean,
                "latitude": response.Latitude(),
                "longitude": response.Longitude()
            })
            
            all_city_frames.append(city_df)
            print(f"Processed: {city_name}")

        return pd.concat(all_city_frames, ignore_index=True)


def main():
    ## imported geolocated dataset
    df_geolocated = pd.read_csv("data_/geolocated_capitals.csv")

    # 1. Configuration & Session Setup
    # Indefinite caching is recommended for historical data to avoid re-requesting the same years
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # --- EXECUTION ---
    # Change these indices based on which batch you are running
    # Today: 0 to 60 | Tomorrow: 60 to 117
    START = 0
    END = 117
    date_start = "2017-01-01"
    date_end = "2019-12-31"
    print(f"Requesting data for capitals {START} to {END}...")
    weather_history_df = fetch_weather_data(openmeteo,
                                            df_geolocated,
                                            START,
                                            END,
                                            date_start,
                                            date_end
                                            )

    # Save results
    weather_history_df.to_csv(f"ecmwf_weather_batch_{START}_{END}_{date_start}_{date_end}.csv", index=False)
    print("Done! Data saved to CSV.")

if __name__== "__main__":
     main()