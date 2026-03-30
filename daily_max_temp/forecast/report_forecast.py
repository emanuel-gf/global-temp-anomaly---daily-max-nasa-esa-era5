import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import math
import os
from pathlib import Path
from datetime import datetime
from pandas.api.types import is_datetime64_any_dtype
import argparse

FOLDER_DETERMINISTIC = "/home/camarada/Documents/projects/temp-grss-nasa/apiresult/deterministic"
FOLDER_ENSEMBLE = "/home/camarada/Documents/projects/temp-grss-nasa/apiresult/ensemble"

### ===--------------- FUNCTIONS
def get_most_recent_file(folder: str, city: str):
    folder_path = Path(folder)
    
    # 1. Filter files that start with the city and end with .parquet
    # This prevents the max() function from comparing different cities
    city_files = [
        f for f in folder_path.glob(f"{city}-*.parquet") 
        if f.is_file()
    ]
    
    if not city_files:
        return None

    def extract_datetime(path: Path):
        try:
            # Stem gets 'London-Deterministic_2026-03-23-21:03'
            # Splitting by '_' and taking the second element
            timestamp_str = path.stem.split('_')[1]
            return datetime.strptime(timestamp_str, "%Y-%m-%d-%H:%M")
        except (IndexError, ValueError):
            # Return a very old date so malformed files don't win 'max()'
            return datetime.min

    # 2. Find the max based on the extracted date
    most_recent = max(city_files, key=extract_datetime)
    
    return str(most_recent)


## DETERMINISTIC
def calc_mean_std_deterministic_forecast(temperature:np.ndarray,
                                        weights:np.ndarray):
    """
    compute mean and weighted std
    """
    # weighted mean
    mu_w = np.average(temperature,
                        weights=weights)
    
    # weight variance and std
    variance_w = np.average((temperature - mu_w)**2,
                            weights=weights
                            )
    
    # std
    std_w = np.sqrt(variance_w)

    return mu_w, std_w


def print_deterministic_report(df_print, 
                            analysis_date,
                            mu_w,
                            std_w,
                            weights_dict,
                            verbose_weight=True):
    #print_deterministic
    print("="*30)
    print("Deterministic models: ")
    print(f"Dia:{analysis_date}")
    print("-"*30)

    ## drop nan values
    df_print = df_print.dropna(axis=0)
    
    # Determine a reasonable width for the model column
    model_width = max(df_print['model'].str.len()) + 2  # +2 for spacing

    ## head of the print
    print(f"{'model':<{model_width}} | {'Temperature':>5} | {'Time':>5} | {'rel weight':<10}")
    for _, row in df_print.iterrows():
        if verbose_weight:
            assert weights_dict is not None
            print(f"{row['model']:<{model_width}} | {row['temperature']:>5} °C | at {row['time'].strftime('%H:%M')} | {weights_dict[row['model']]}")
        else:
            print(f"{row['model']:<{model_width}} | {row['temperature']:>5} °C | at {row['time'].strftime('%H:%M')}")

    print("-"*30)
    print("Weighted:")
    print(f"Average:{float(mu_w):.2f}")
    print(f"Standard Desviation:{std_w:.2f}")
    print("."*30)


def report_deterministic(df_det:pd.DataFrame,
                        analysis_day,
                        weights_dict:dict):
    # TODO 
    ## check if date is pd_datetime instead of convert pd_datetime

    ## sort 
    df_det = df_det.sort_values(by=['date','model'])

    #filter for the given day
    ## filter the df 
    df_day = (df_det.loc[df_det['date'] == analysis_day]).sort_values(
                                                                    by='model'
                                                                    ) ## import to sort! 
    ## get the weights
    weights = [weights_dict[i] for i in df_day['model'].values]

    ## calc mean and std
    mu_w, std_w = calc_mean_std_deterministic_forecast(
        temperature = df_day['temperature'].values,
        weights = weights 
    )

    df_print = df_day.head(len(df_day['model'].unique()))[['model','time','temperature']]

    print_deterministic_report(df_print,
                                analysis_day,
                                mu_w,
                                std_w,
                                weights_dict,
                                verbose_weight=True)

    ## TODO
    #how to send as a message in telegram


## ENSEMBLE
def print_ensemble_report(analysis_day,
                          ensemble_row):
    ## TODO:
    ## standardize sep_s
    sep_s = "·" * 42
    values = ensemble_row.values.flatten()
    values = values[~np.isnan(values)]  # Filter out any missing data

    # bins
    bin_min = math.floor(values.min())
    bin_max = math.ceil(values.max())
    num_bins = abs(bin_min - bin_max)

    hist, bin_edges = np.histogram(
        values,
        bins=num_bins,
        range=(bin_min, bin_max)
    )

    # 
    print("="*30)
    print(f"Ensemble Analysis \n for: {analysis_day}")
    print(f"Total number of ensemble members: {len(values)}")
    print("."*30)

    # Use the transposed description for the specific day
    stats = ensemble_row.T.describe(percentiles=[0.1,0.25,0.5,0.75,0.9])
    for i, row in stats.iterrows():
        # Since the column name is the index of the original row, we access by index 0
        print(f"{i:<8} | {row.iloc[0]:.2f}")

    print('-' * 30)
    print('Histogram (Probability Distribution)')

    # Print each range and its percentage
    percentages = (hist / hist.sum()) * 100

    for i in range(len(hist)):
        lower = bin_edges[i]
        upper = bin_edges[i+1]
        percent = percentages[i]
        
        # Create a visual bar using characters for a quick console "plot"
        bar = "█" * int(percent / 2) 
        print(f"{lower:>2.0f} - {upper:>2.0f}°C | {percent:>5.1f}% {bar}")

    print(""*30)


def report_ensemble(df_ens,
                    analysis_day):
    ## TODO
    ## assume analysis_day is proper 2026-03-12 and raise error in case 
    # Ensure we only have numeric temperature columns for the calculation
    ensemble_row = df_ens.loc[df_ens['date'] == analysis_day].drop(columns=['date'])

    
    print_ensemble_report(analysis_day, ensemble_row)


## HISTORICAL REPORT ERROR ANALYSIS
def print_report_historic_uk2m(analysis_day,
                            day_stats, 
                            week_stats, 
                            month_stats):
    sep   = "─" * 42
    sep_s = "·" * 42

    # ── helpers 
    def _row(label, val, unit=""):
        if pd.isna(val):
            return f"  {label:12}  n/a"
        return f"  {label:12}  {val:.2f}{unit}"


    # ── header 
    print(f"\n{'═'*42}")
    print("  UK2km vs WeaUnder — Historic Report")
    print(f" Dia : {analysis_day.strftime('%Y-%m-%d')}")
    print(f"{'═'*42}")

    # day across all years
    print(f"\n  [{analysis_day.strftime('%m-%d')}]  Same day across all years")
    print(sep_s)
    for stat, val in day_stats.items():
        print(_row(stat, val))

    # ── 2. first week of that month ──────────────────
    month_name = analysis_day.strftime("%B")
    print(f"\n Range of +-3 days error")
    print(sep_s)
    for stat, val in week_stats.items():
        print(_row(stat, val))

    # full month ────────────────────────────────
    print(f"\n  [{month_name}]  Full month")
    print(sep_s)
    for stat, val in month_stats.items():
        print(_row(stat, val))


def report_historic_uk2m(df:pd.DataFrame,
                        analysis_day: str):
    """
    Args:
        df            : daily DataFrame with DatetimeIndex and an 'error' column
                    (error = WeaUnder − UK2km, so negative → overestimate)
    analysis_day  : ISO string like '2022-03-01'
    """
    ## convert to datetime
    if not is_datetime64_any_dtype(df['datetime']):
        df['datetime'] = pd.to_datetime(df['datetime'], yearfirst=True)
    
    day = pd.to_datetime(analysis_day)       # safe parse
    md  = day.strftime("%m-%d")             # "03-01"
    agg = ["mean", "std", "max"]

    df = df.copy()
    
    df["month_day"] = df['datetime'].dt.strftime("%m-%d")
    df['day_of_year'] = df['datetime'].dt.day_of_year
    df['error_abs'] = np.abs(df['error'].values)


    # 1. same day of year analysis
    day_stats = (
        df[df["month_day"] == md]["error_abs"]
        .agg(agg)
    )

    # looking to +3 and -3 days from the current day
    week_stats = df.loc[(df['day_of_year'] >= day.dayofyear - 3) & (df['day_of_year']<= day.dayofyear +3)]["error_abs"].agg(agg)

    # # 3. full month
    month_stats = (
        df[df['datetime'].dt.month == day.month]["error_abs"]
        .agg(agg)
    )

    print_report_historic_uk2m(day,
                                day_stats,
                                week_stats,
                                month_stats,
                                )


## ERROR ANALYSIS - UK2M AND WEATHER UNDERGROUND
def analyse_error_by_temp(df:pd.DataFrame,
                            analysis_day:pd.Timestamp,
                            freq_analysis:str,
                            target_col = "WeaUnder"
                            ):
        """
        Analyse the Error bias for a given period. 
        Args:
            df: 
                Df with Max Temp for WeatherUnder and UK2km or other forecast. Column with Error is mandatory.
            analysis_day:
                Timestamp of the day
            freq_analysis: ["day","week"]
                The period can be the day of year over the historical range. 
                If given day, it filter the same day for all over the historical range. If given week
                so it creates a window +3 days and -3 days over the given day to analyse.
            target_col: string
                Name of the column that will be used as reference for histograms.
        
        Returns: df, hist_min, hist_max
            df: aggregate with bins and frequency bias analysise.
            Historical min and max: range for the given period analysis, either day or week

        """
        df["month_day"] = df['datetime'].dt.strftime("%m-%d")
        df['day_of_year'] = df['datetime'].dt.day_of_year
        df['error_abs'] = np.abs(df['error'].values)
        df['error_sign'] = df['error'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Zero'))
        
        match freq_analysis.strip().lower():
            case  'day':
                window = df.loc[
                                (df['day_of_year'] == analysis_day.dayofyear)
                            ].copy()
        
            case 'week':
                window = df.loc[
                                (df['day_of_year'] >= analysis_day.dayofyear - 3) & 
                                (df['day_of_year'] <= analysis_day.dayofyear + 3)
                            ].copy()
            case _:
                  raise ValueError(f"Please select either 'day' or 'week'. Got '{freq_analysis}'")
        
        ## retrieve historical
        hist_min, hist_max = window[target_col].min(), window[target_col].max()
    
        # 2. Create 1-degree temperature bins
        temp_min = np.floor(window[target_col].min())
        temp_max = np.ceil(window[target_col].max())
        bins = np.arange(temp_min, temp_max + 1, 1)  # 1°C bins

        ## add bins to the df 
        window['temp_bin'] = pd.cut(window[target_col], bins=bins, right=False)

        agg = (
                window.groupby(['temp_bin', 'error_sign'], observed=True)
                    .size() ## same as count 
                    .unstack(fill_value=0) ## pass to a proper df 
                    .assign( # create new columns 
                        total=lambda x: x.sum(axis=1),
                        dominant_sign=lambda x: x[['Negative', 'Positive']].idxmax(axis=1).where(x['Negative'] != x['Positive'], 'Tie').where(x['Negative'] + x['Positive'] > 1, 'Inconclusive') , 
                        bias=lambda x: x.get('Positive', 0) - x.get('Negative', 0),  # + = tends to underestimate
                        mean_error=window.groupby('temp_bin', observed=True)['error_abs'].mean(),
                        std_error=window.groupby('temp_bin', observed=True)['error_abs'].std(),
                    )
            )
        return agg, hist_min, hist_max


def flag_new_prediction(new_temp, agg, hist_min, hist_max):
    """
    Given a new temperature (float, or list of 2 or 3 floats) to predict,
    flag if it's out of range and retrieve error behaviour.
    
    Args:
        new_temp: float or list of floats:
                  - 1 value  : single temperature
                  - 2 values : range [low, high] — all bins in between are checked
                  - 3 values : specific temperatures [t1, t2, t3] checked individually
        agg: aggregated df from analyse_error_by_temp
        hist_min: historical min temperature
        hist_max: historical max temperature
    """
    
    # --- Normalise input ---
    if isinstance(new_temp, (int, float)):
        temps = [new_temp]
    elif isinstance(new_temp, (list, tuple)):
        if len(new_temp) == 2:
            temp_low, temp_high = sorted(new_temp)
            temps = list(np.round(np.arange(temp_low, temp_high + 0.1, 0.1), 1))
        elif len(new_temp) == 3:
            temps = sorted(new_temp)  # treat as 3 individual points
        else:
            raise ValueError("List input must have 2 or 3 values.")
    else:
        raise TypeError("new_temp must be a float or a list/tuple of 2 or 3 floats.")

    bin_intervals = agg.index.array

    def get_bin_result(temp):
        matched_bin = None
        extrapolated = False

        for interval in bin_intervals:
            if temp in interval:
                matched_bin = interval
                break

        if matched_bin is None:
            midpoints = np.array([iv.mid for iv in bin_intervals])
            nearest_idx = np.argmin(np.abs(midpoints - temp))
            matched_bin = bin_intervals[nearest_idx]
            extrapolated = True

        return {
            'temp'          : temp,
            'out_of_range'  : not (hist_min <= temp <= hist_max),
            'extrapolated'  : extrapolated,
            'matched_bin'   : matched_bin,
            **agg.loc[matched_bin][['Negative', 'Positive', 'total', 'dominant_sign', 
                                    'bias', 'mean_error', 'std_error']].to_dict()
        }

    # --- Build results, deduplicate by bin ---
    results = []
    seen_bins = set()
    for t in temps:
        res = get_bin_result(t)
        bin_key = str(res['matched_bin'])
        if bin_key not in seen_bins:
            seen_bins.add(bin_key)
            results.append(res)

    results_df = pd.DataFrame(results).set_index('matched_bin')

    return results_df, temps


def print_flag_predictions_bias(results_df, temps, hist_min, hist_max):
        # --- Summary flags ---
    any_out_of_range = results_df['out_of_range'].any()
    any_extrapolated = results_df['extrapolated'].any()
    dominant_signs   = results_df['dominant_sign'].unique()

    print(f"Historical range   : {hist_min:.1f}°C – {hist_max:.1f}°C")
    print(f"Queried temps      : {[round(t, 1) for t in [min(temps), max(temps)]]}")
    print(f"Out of range       : {any_out_of_range} {'ANOMALY' if any_out_of_range else 'No'}")
    print(f"Extrapolated bins  : {any_extrapolated}")
    print(f"Dominant signs     : {dominant_signs}")
    print(f"\nBin breakdown:")
    print(results_df[['temp', 'out_of_range', 'Negative', 'Positive', 
                       'dominant_sign', 'bias', 'mean_error', 'std_error']])


## ----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble Weather Fetcher")
    parser.add_argument("--city", type=str, help=" Name of city belonging the given lat,lon. Same a the saved by the forecast python files.")
    parser.add_argument("--day", type=str, help="YYYY-MM-DD e.g: 2026-03-21")
    parser.add_argument("--temprange", type=str, help="Min and max temperature to analysis errors. e.g; 11-14")
    return parser.parse_args()
    
def main():
    args = parse_args()

    assert args.day and (
        pd.to_datetime(args.day, errors="coerce").strftime("%Y-%m-%d") == args.day
    ), "Invalid date. Expected format YYYY-MM-DD (e.g., 2026-03-21)"
    analysis_day = str(args.day)
    print(analysis_day)
    ## TODO assert args.temprange 
    temp_min_ = str(args.temprange).split('-')[0]
    temp_max_ = str(args.temprange).split('-')[1]
    
    folder_deterministic = FOLDER_DETERMINISTIC
    folder_ensemble = FOLDER_ENSEMBLE
    
    ## get the most recent files from the folder 
    mr_det_file = get_most_recent_file(folder_deterministic, str(args.city))
    mr_ensem_file = get_most_recent_file(folder_ensemble, str(args.city))
    print(f" most recent file:\n deterministic:{mr_det_file}")
    print(f" ensemble :{mr_ensem_file}")


    ## create the df 
    df_det = pd.read_parquet(
        os.path.join(folder_deterministic, mr_det_file)
    )
    df_ens = pd.read_parquet(
        os.path.join(folder_ensemble, mr_ensem_file)
    )

    ## convert to datetime 
    df_det['date'] =pd.to_datetime(df_det['date'], yearfirst=True)
    df_ens = df_ens.reset_index().rename(columns={'index':'date'})
    df_ens['date'] = pd.to_datetime(df_ens['date'],
                                    yearfirst=True)
    
    ## report pipeline
    weights_dict = {'DWD Germany':0.1,
                    'DWD ICON EU':0.1,
                    'ECMWF IFS HRES 9km':0.15,
                    'MeteoFrance ARPEGE Europe': 0.15,
                    'UKMO Global 10km':	0.1,
                    'UKMO UK 2km':0.3,
                    'MeteoFrance AROME_HD':0.1
                }

    report_deterministic(df_det,
                            analysis_day,
                            weights_dict
                        )

    ## ensemble
    report_ensemble(df_ens,
                    analysis_day
                    )

    ## TODO
    ## add a way to pass diferent historical folders linked with cities.
    ## to keep it going it will be an if statement only for london

    if str(args.city).lower().strip() == "london":
        ## historical UK2m ========================
        folder_historical_uk2m = "/home/camarada/Documents/projects/temp-grss-nasa/daily_max_temp/explore_temp_and_time_distribution/historic_uk2m_aggregated"
        file = "daily_agg_2022-2025_uk2m_wunder_error.csv"

        ## create the df
        df_uk2_err = pd.read_csv(os.path.join(folder_historical_uk2m, file))

        report_historic_uk2m(df_uk2_err,
                        analysis_day
                            )


        ## error and bias analysis
        ## calculate bias of the UK2m over Weather Underground
        error_bias_agg, historic_min, historic_max =  analyse_error_by_temp(df_uk2_err,
                                                                    pd.to_datetime(analysis_day, yearfirst=True),
                                                                    'week',
                                                                    target_col = 'WeaUnder'
        )

        ## Compare actual forecast with bias of prediction
        df_flag_prediction_bias, temps = flag_new_prediction([float(temp_min_),float(temp_max_)], error_bias_agg, historic_min, historic_max)

        ## print for report 
        print_flag_predictions_bias(df_flag_prediction_bias, temps, historic_min, historic_max)

 
if __name__ =="__main__":
    main()