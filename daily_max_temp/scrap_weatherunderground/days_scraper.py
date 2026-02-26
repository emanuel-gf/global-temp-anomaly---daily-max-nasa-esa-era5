import os
import time
import random
import pandas as pd
import undetected_chromedriver as uc
from datetime import datetime, timedelta
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def get_driver():
    options = uc.ChromeOptions()
    options.binary_location = "/usr/bin/chromium"
    # options.add_argument('--headless')
    driver = uc.Chrome(options=options)
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

def main(date_start, date_end, station_id="EGLC"):
    start = datetime.strptime(date_start, "%Y-%m-%d")
    end = datetime.strptime(date_end, "%Y-%m-%d")
    current_date = start
    
    driver = get_driver()
    last_processed_date = None

    try:
        while current_date <= end:
            year = current_date.strftime("%Y")
            month = current_date.strftime("%m")
            day = current_date.strftime("%d")
            
            # Create Folder Structure: Year/Month/data_
            root =  "/home/camarada/Documents/projects/temp-grss-nasa/data_"
            folder_path = os.path.join(root, "wunderground", year, month)
            os.makedirs(folder_path, exist_ok=True)
            
            file_name = f"{station_id}_{year}_{month}_{day}.parquet"
            file_path = os.path.join(folder_path, file_name)
            
            # Skip if already exists (Resume capability)
            if os.path.exists(file_path):
                print(f"Skipping {current_date.date()}, file exists.")
                current_date += timedelta(days=1)
                continue

            url = f"https://www.wunderground.com/history/daily/gb/london/{station_id}/date/{year}-{int(month)}-{int(day)}"
            
            print(f"Scraping: {current_date.date()}...")
            df = scrape_single_day(driver, url)
            
            if df is not None:
                # Save to Parquet
                df.to_parquet(file_path, engine='fastparquet')
                last_processed_date = current_date
                # Random delay to avoid rate limiting
                time.sleep(random.uniform(5, 10))
            else:
                print(f"FAILED to retrieve data for {current_date.date()}")
                break # Exit loop to report where we stopped
                
            current_date += timedelta(days=1)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        driver.quit()
        if last_processed_date:
            print(f"--- SCRAPER STOPPED. Last successful date: {last_processed_date.date()} ---")
        else:
            print("--- SCRAPER STOPPED. No data was saved. ---")

if __name__ == "__main__":
    # Format: YYYY-MM-DD
    main("2020-11-27", "2020-12-31")