The folder is related to WEather Underground. 

API-request handles the Request to the API at the right London Station. 
However, it can be adapted to any station, with different times for keeping running
async. 
Here it how to use the function:

```
# Basic usage (defaults to EGLC, 15s interval, targets minute 20 or 50)
python metar_poller.py

# Different station
python metar_poller.py --station KJFK

# Custom interval and reference minutes
python metar_poller.py --station EGLC --interval 10 --ref-minutes 20 50

# Save results to CSV
python metar_poller.py --station EGLC --output metar.csv

python metar_poller.py                          # defaults
python metar_poller.py --station KJFK --db weather_data.db
```

The database is managed at Wunderground_DB, and therefore called inside the python file 