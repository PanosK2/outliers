# data_loader.py
import os
import glob
import joblib
import pandas as pd
import config
from nwp_manager import generate_weather_file


def load_park_data(park_folder_path):
    """Φορτώνει και επιστρέφει: (park_name, capacity, df_prod, df_weather)."""
    park_name = os.path.basename(park_folder_path)

    # 1. Φόρτωση Info & Capacity
    pickle_path = os.path.join(park_folder_path, 'park_info.pickle')
    rated_capacity = 10.0
    lat, lon = None, None

    if os.path.exists(pickle_path):
        try:
            info = joblib.load(pickle_path)
            # Capacity
            cap = info.get('new_rated') or info.get('rated') or info.get('capacity')
            if cap is None and 'info' in info and 'CAPACITY [MW]' in info['info'].columns:
                cap = info['info']['CAPACITY [MW]'].iloc[0]
            if cap: rated_capacity = cap

            # Coords (για weather gen)
            if 'info' in info and 'GPS Lat' in info['info'].columns:
                lat = info['info']['GPS Lat'].iloc[0]
                lon = info['info']['GPS Lon'].iloc[0]
            else:
                lat, lon = info.get('lat'), info.get('long')
        except:
            pass

    # 2. Φόρτωση Παραγωγής
    all_csvs = glob.glob(os.path.join(park_folder_path, '*.csv'))
    prod_files = [f for f in all_csvs if 'wind_ts' not in f and 'outliers' not in f
                  and 'CLEANED' not in f and 'timeseries_weather' not in f]

    if not prod_files:
        print("   ❌ Δεν βρέθηκε αρχείο παραγωγής.")
        return None

    df_prod = pd.read_csv(prod_files[0])
    time_col = 'TIME' if 'TIME' in df_prod.columns else df_prod.columns[0]
    df_prod[time_col] = pd.to_datetime(df_prod[time_col])
    df_prod = df_prod.set_index(time_col).sort_index()
    df_prod = df_prod[~df_prod.index.duplicated(keep='first')]

    power_col = next((c for c in df_prod.columns if 'MW' in c or 'Power' in c), None)
    if power_col:
        df_prod = df_prod.rename(columns={power_col: 'MW'})
    else:
        return None

    # 3. Φόρτωση Καιρού
    weather_path = os.path.join(park_folder_path, 'timeseries_weather.csv')
    if not os.path.exists(weather_path) and lat and lon:
        generate_weather_file(park_folder_path, lat, lon)

    df_weather = pd.DataFrame()
    if os.path.exists(weather_path):
        df_weather = pd.read_csv(weather_path, index_col=0, parse_dates=True)
        df_weather = df_weather[~df_weather.index.duplicated(keep='first')]

    return park_name, rated_capacity, df_prod, df_weather, prod_files[0]