import requests
import pandas as pd
import argparse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

def get_metar_data(station_id, tz_name="Europe/Warsaw"):
    """Fetches METAR data and returns a cleaned DataFrame.
    
    Args:
        station_id: ICAO station identifier (e.g. 'EPWA')
        tz_name: Timezone name for local time conversion (default: 'Europe/Warsaw')
    """
    params = {'ids': station_id, 'format': 'json'}
    response = requests.get('https://aviationweather.gov/api/data/metar', params=params)
    
    response.raise_for_status()
    data = response.json()
    
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    local_tz = ZoneInfo(tz_name)

    # Convert observation time to local datetime
    if 'obsTime' in df.columns:
        df['obsTime'] = df['obsTime'].apply(
            lambda x: datetime.fromtimestamp(int(x), tz=local_tz)
        )

    # Convert reportTime string to local datetime
    if 'reportTime' in df.columns:
        df['reportTime'] = pd.to_datetime(df['reportTime'], utc=True).dt.tz_convert(local_tz)
    
    if 'receiptTime' in df.columns:
        df['receiptTime'] = pd.to_datetime(df['receiptTime'], utc=True).dt.tz_convert(local_tz)

    # Select and order specific columns
    cols = ['icaoId', 'obsTime', 'reportTime','receiptTime', 'temp', 'dewp', 'wdir', 'wspd', 'qcField']
    existing_cols = [c for c in cols if c in df.columns]
    
    return df[existing_cols]

def main():
    parser = argparse.ArgumentParser(description="Fetch METAR data for a specific station.")
    parser.add_argument("-s", "--station_id", type=str, default="EGLC", help="ICAO Station ID (e.g., EGLC)")
    parser.add_argument("--json", action="store_true", help="Output as JSON string instead of a table")
    parser.add_argument("-tz", type=str, help="Timezone name e.g Europe/Warsaw")
    args = parser.parse_args()

    try:
        df = get_metar_data(args.station_id, tz_name = args.tz)

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