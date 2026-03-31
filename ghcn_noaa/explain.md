
![alt text](image.png)


What GHCN-M v4 is: A global archive of monthly mean temperatures from over 25,000 weather stations going back as far as the 1800s. It's the dataset behind most long-term climate analysis.
The three dataset flavors you can download:

- QCU — raw, quality-checked but not adjusted. Best if you want to study bias correction yourself.

- QCF — adjusted using the Pairwise Homogeneity Algorithm (PHA), which corrects for things like station moves or equipment changes. This is the most commonly used one.

- QFE — estimated/infilled, covering only 1961–2010, useful for calculating climate normals where station coverage is patchy.

Each download contains two files:

A .inv file — one row per station with its ID, lat/lon, elevation, and name.
A .dat file — the actual temperature records. Each row = one station × one year, with 12 monthly values packed in fixed-width columns. Temperatures are stored as integers (e.g. 2350 = 23.50°C) and -9999 means missing data. Each monthly value has three flag characters attached (data measurement, QC, and data source).

## HOW TO

Get the data:
```bash
wget https://www.ncei.noaa.gov/pub/data/ghcn/v4/ghcnm.tavg.latest.qcf.tar.gz

#extract 
tar -zxvf ghcnm.tavg.latest.qcf.tar.gz 

## quick look into the data
head -n 5 *.dat

##same for metadata
head -n 5 *.inv
```