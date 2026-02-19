import sqlite3
import requests
from datetime import datetime, timezone
from pathlib import Path

class WeatherDB:
    def __init__(self, db_path='weather_data.db'):
        self.db_path = db_path
        self.create_table()
    
    def create_table(self):
        """Create the weather table if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weather_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                icaoId TEXT NOT NULL,
                obsTime TEXT NOT NULL,
                reportTime TEXT,
                temp REAL,
                dewp REAL,
                wdir INTEGER,
                wspd INTEGER,
                qcField INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(icaoId, obsTime)
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_icao_obstime 
            ON weather_observations(icaoId, obsTime)
        ''')
        
        conn.commit()
        conn.close()
    
    def insert_observation(self, data):
        """Insert a single observation, ignore if duplicate"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO weather_observations 
                (icaoId, obsTime, reportTime, temp, dewp, wdir, wspd, qcField)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['icaoId'],
                data['obsTime'],
                data['reportTime'],
                data['temp'],
                data['dewp'],
                data['wdir'],
                data['wspd'],
                data['qcField']
            ))
            conn.commit()
            return cursor.rowcount > 0  # True if inserted, False if duplicate
        except Exception as e:
            print(f"Error inserting data: {e}")
            return False
        finally:
            conn.close()
    
    def get_latest(self, icao_id, limit=10):
        """Get latest observations for a station"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM weather_observations 
            WHERE icaoId = ?
            ORDER BY obsTime DESC
            LIMIT ?
        ''', (icao_id, limit))
        
        results = cursor.fetchall()
        conn.close()
        return results


def fetch_and_store_weather(icao_id):
    """Fetch weather data and store in database"""
    # Initialize database
    db = WeatherDB()
    
    # Fetch data from API
    response = requests.get(
        'https://aviationweather.gov/api/data/metar',
        params={'ids': icao_id, 'format': 'json'}
    )
    
    if response.status_code != 200:
        print(f"API request failed: {response.status_code}")
        return
    
    data = response.json()
    
    if not data:
        print("No data returned from API")
        return
    
    # Process and insert each observation
    for observation in data:
        # Convert Unix timestamp to ISO format string
        obs_time_dt = datetime.fromtimestamp(
            observation['obsTime'], 
            tz=timezone.utc
        )
        observation['obsTime'] = obs_time_dt.isoformat()
        
        # Insert into database
        inserted = db.insert_observation(observation)
        if inserted:
            print(f"Inserted observation for {icao_id} at {observation['obsTime']}")
        else:
            print(f"Duplicate observation skipped for {icao_id}")


# Usage
if __name__ == "__main__":
    icao_id = "EGLC"
    fetch_and_store_weather(icao_id)
    
    # View latest records
    db = WeatherDB()
    latest = db.get_latest(icao_id, limit=5)
    for record in latest:
        print(record)