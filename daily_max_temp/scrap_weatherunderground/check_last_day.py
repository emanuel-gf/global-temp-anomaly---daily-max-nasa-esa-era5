# Count how many days were expected vs how many parquet files exist
def main():
    folder = ## name of the folder to check 
    ## look for the last day observed 


    ## return the last day observed + 1 
    
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
