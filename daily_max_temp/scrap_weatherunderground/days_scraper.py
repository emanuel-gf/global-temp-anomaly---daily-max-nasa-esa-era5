import os
import time
import random
import pandas as pd
import undetected_chromedriver as uc
from datetime import datetime, timedelta
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import argparse
from webdriver_manager.chrome import ChromeDriverManager
import sys

##map to the proper URL 
map_dict_city = {
    'madrid':'es',
    'london':'gb',
    'paris':'fr'
}

def get_driver():
    options = uc.ChromeOptions()
    # Ensure this path is correct for your Chromium installation
    options.binary_location = "/usr/bin/chromium" 

    # 1. Use webdriver-manager to get the path to the correct driver
    # ChromeDriverManager will automatically match your browser version
    driver_path = ChromeDriverManager().install()

    # 2. Initialize undetected-chromedriver
    # We pass the driver_executable_path directly to uc.Chrome
    driver = uc.Chrome(
        options=options,
        driver_executable_path=driver_path
    )
    
    return driver

def scrape_single_day(driver, url):
    try:
        driver.get(url)
        # Wait for the specific observation table to load
        wait = WebDriverWait(driver, 15)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "mat-mdc-table")))
        
        # Give JS a moment to populate the cells
        time.sleep(random.uniform(2, 4))
        
        # Use Pandas to read the HTML directly from the driver source
        # This is faster than manual BeautifulSoup parsing for simple tables
        dfs = pd.read_html(driver.page_source)
        
        # Wunderground usually has multiple tables; the observations one is typically the last
        # or can be identified by its columns
        for df in dfs:
            if 'Time' in df.columns and 'Temperature' in df.columns:
                return df
        return None
    except Exception as e:
        return None

def parse_args():
    parser = argparse.ArgumentParser(description="WeatherUnderground Fetcher")
    parser.add_argument("--start", type=str, help="First day to start fecthing. YYYY-MM-DD e.g: 2026-03-21")
    parser.add_argument("--end", type=str, help="Last day to retrieve data ; 2026-03-21")
    parser.add_argument("--station-id", type=str, default="unknown", help="ID of the METAR estation to be retrieved. It is used to save the file as the given id.")
    parser.add_argument("--city", type=str, default=None, help="Name of the parent folder, should be associated to the ID METAR.")
    parser.add_argument("--root", type=str, default=None, help="ROot folder which saves all the formated files per year than month subfolders.")
    return parser.parse_args()


def main():
    args = parse_args()
    station_id = args.station_id
    date_start = str(args.start)
    date_end = str(args.end)
    start = datetime.strptime(date_start, "%Y-%m-%d")
    end = datetime.strptime(date_end, "%Y-%m-%d")   
    if start > end :
        current_date = end
    else:
        current_date = start

    city  = args.city
    if args.root is None:
        root =  "/home/camarada/Documents/projects/temp-grss-nasa/data_"
    else:
        root = str(args.root)
    driver = get_driver()
    last_processed_date = None

    try:
        while current_date <= end:
            year = current_date.strftime("%Y")
            month = current_date.strftime("%m")
            day = current_date.strftime("%d")
            
            # Create Folder Structure: Year/Month/data_
            folder_path = os.path.join(root, "wunderground",city,year,month)
            os.makedirs(folder_path, exist_ok=True)
            
            file_name = f"{station_id}_{year}_{month}_{day}.parquet"
            file_path = os.path.join(folder_path, file_name)
            
            # Skip if already exists (Resume capability)
            if os.path.exists(file_path):
                print(f"Skipping {current_date.date()}, file exists.")
                current_date += timedelta(days=1)
                continue

            url = f"https://www.wunderground.com/history/daily/{map_dict_city[city]}/{city}/{station_id}/date/{year}-{int(month)}-{int(day)}"
            
            print(f"Scraping: {current_date.date()}...")
            df = scrape_single_day(driver, url)
            
            if df is not None:
                # Save to Parquet
                df.to_parquet(file_path, engine='fastparquet')
                last_processed_date = current_date
                # Random delay to avoid rate limiting
                time.sleep(random.uniform(5, 10))
            else:
                print(f"FAILED to retrieve data for {current_date.date()} — skipping.")
                time.sleep(random.uniform(30, 60))  # longer pause on failure
                current_date += timedelta(days=1)   # skip bad day and continue
                continue
                
            current_date += timedelta(days=1)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        driver.quit()
        if last_processed_date:
            print(f"--- SCRAPER STOPPED. Last successful date: {last_processed_date.date()} ---")
            
        # Count how many days were expected vs how many parquet files exist
        expected_days = (end - start).days + 1
        saved_files = []
        check_date = start
        while check_date <= end:
            year  = check_date.strftime("%Y")
            month = check_date.strftime("%m")
            day   = check_date.strftime("%d")
            file_path = os.path.join(root, "wunderground", year, month,
                                    f"{station_id}_{year}_{month}_{day}.parquet")
            if os.path.exists(file_path):
                saved_files.append(file_path)
            check_date += timedelta(days=1)

        missing = expected_days - len(saved_files)
        print(f"Progress: {len(saved_files)}/{expected_days} days saved. Missing: {missing}")

        if missing == 0:
            print("✅ All days complete!")
            sys.exit(0)   # SUCCESS — tells bash "we're done"
        else:
            print(f"⚠️  Incomplete — {missing} days still missing.")
            sys.exit(1)   # FAILURE — tells bash "please retry"

if __name__ == "__main__":
    main()