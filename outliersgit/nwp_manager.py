# nwp_manager.py
import os
import glob
import joblib
import pandas as pd
import numpy as np
import config


def generate_weather_file(park_folder, lat, lon):
    """Δημιουργεί το timeseries_weather.csv από τα NWP pickles αν δεν υπάρχει."""
    print(f"   🌤️ Δημιουργία αρχείου καιρού...")

    nwp_files = sorted(glob.glob(os.path.join(config.NWP_SOURCE_DIR, 'ecmwf_*.pickle')))
    if not nwp_files:
        print("      ❌ Δεν βρέθηκαν αρχεία ECMWF!")
        return False

    all_weather_data = []

    for nwp_file in nwp_files:
        try:
            daily_data = joblib.load(nwp_file)
            for hour_key, inner_data in daily_data.items():
                timestamp = pd.to_datetime(hour_key, format='%d%m%y%H%M')

                flattened = {k: v.flatten() for k, v in inner_data.items()}
                df_grid = pd.DataFrame(flattened)

                # Υπολογισμός WS/WD
                if 'WS' not in df_grid.columns:
                    df_grid['WS'] = np.sqrt(df_grid['Uwind'] ** 2 + df_grid['Vwind'] ** 2)
                if 'WD' not in df_grid.columns:
                    df_grid['WD'] = (270 - np.rad2deg(np.arctan2(df_grid['Vwind'], df_grid['Uwind']))) % 360

                # Nearest Neighbor
                dist = np.sqrt((df_grid['lat'] - lat) ** 2 + (df_grid['long'] - lon) ** 2)
                nearest_idx = dist.idxmin()

                row = df_grid.loc[nearest_idx].to_dict()
                row['timestamp'] = timestamp
                all_weather_data.append(row)
        except Exception as e:
            print(f"      ⚠️ Σφάλμα σε αρχείο NWP: {e}")

    if all_weather_data:
        df_weather = pd.DataFrame(all_weather_data)
        df_weather = df_weather.set_index('timestamp').sort_index()
        save_path = os.path.join(park_folder, 'timeseries_weather.csv')
        df_weather.to_csv(save_path)
        print(f"      ✅ Αποθηκεύτηκε: timeseries_weather.csv ({len(df_weather)} εγγραφές)")
        return True
    return False