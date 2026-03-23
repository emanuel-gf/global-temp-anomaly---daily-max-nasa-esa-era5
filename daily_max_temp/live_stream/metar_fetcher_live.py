import requests
import pandas as pd
import argparse
from datetime import datetime, timezone

def get_metar_data(station_id):
    """Fetches METAR data and returns a cleaned DataFrame."""
    params = {'ids': station_id, 'format': 'json'}
    response = requests.get('https://aviationweather.gov/api/data/metar', params=params)
    
    # Check if request was successful
    response.raise_for_status()
    data = response.json()
    
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Convert observation time to UTC datetime
    if 'obsTime' in df.columns:
        df['obsTime'] = df['obsTime'].apply(
            lambda x: datetime.fromtimestamp(int(x), tz=timezone.utc)
        )

    # Select and order specific columns
    cols = ['icaoId', 'obsTime', 'reportTime', 'temp', 'dewp', 'wdir', 'wspd', 'qcField']
    # Only select columns that actually exist in the response
    existing_cols = [c for c in cols if c in df.columns]
    
    return df[existing_cols]

def main():
    parser = argparse.ArgumentParser(description="Fetch METAR data for a specific station.")
    parser.add_argument("-s", "--station_id", type=str, default="EGLC", help="ICAO Station ID (e.g., EGLC)")
    parser.add_argument("--json", action="store_true", help="Output as JSON string instead of a table")

    args = parser.parse_args()

    try:
        df = get_metar_data(args.station_id)

        if df.empty:
            print(f"No data found for station: {args.station_id}")
            return

        if args.json:
            # orient='records' is usually best for dashboards/web APIs
            print(df.to_json(orient='records', date_format='iso'))
        else:
            print(df.to_string(index=False))

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()