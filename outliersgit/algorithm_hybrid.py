import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import config


def run_hybrid(df_prod, df_weather, rated_capacity):
    print("      🔄 Εκτέλεση: HYBRID Method (Με Καιρό)...")

    # --- 1. MACRO ANALYSIS (Hourly Mean + Isolation Forest) ---
    df_hourly = df_prod.resample('h').mean()

    # Ένωση (Join)
    df_macro = df_weather.join(df_hourly, how='inner', lsuffix='_w', rsuffix='_p')
    if 'MW_p' in df_macro.columns: df_macro = df_macro.rename(columns={'MW_p': 'MW'})

    # Καθαρισμός για να τρέξει το μοντέλο
    df_macro = df_macro.dropna(subset=['MW', 'WS', 'WD'])

    if len(df_macro) < 100:
        print("      ⚠️ Ανεπαρκή κοινά δεδομένα για Hybrid.")
        return None, "ERROR_NO_DATA"

    # Isolation Forest
    iso = IsolationForest(contamination=config.ISO_CONTAMINATION, random_state=42, n_estimators=200)
    df_macro['iso_score'] = iso.fit_predict(df_macro[['MW', 'WS', 'WD']])

    # Κανόνες & Φίλτρα
    mask_neg = df_macro['MW'] < -0.1
    mask_stag = (df_macro['WS'] < config.CUT_IN_SPEED) & (df_macro['MW'] > 0.5)

    # Smart Filter: Αν παράγει πολύ (>30%), δεν είναι βλάβη, είναι ριπή ανέμου
    high_prod = df_macro['MW'] > (rated_capacity * 0.3)
    real_iso = (df_macro['iso_score'] == -1) & (~high_prod)

    df_macro['is_macro_outlier'] = real_iso | mask_neg | mask_stag

    # Μεταφορά αποτελεσμάτων πίσω στα λεπτά
    df_prod['hour_idx'] = df_prod.index.floor('h')
    df_prod = df_prod.join(df_macro[['is_macro_outlier']], on='hour_idx')
    df_prod['is_macro_outlier'] = df_prod['is_macro_outlier'].fillna(False).infer_objects(copy=False).astype(bool)
    if 'hour_idx' in df_prod.columns: df_prod.drop(columns=['hour_idx'], inplace=True)

    # --- 2. MICRO ANALYSIS (Rolling Stats) ---
    mask_clean = ~df_prod['is_macro_outlier']

    # Χρησιμοποιούμε τις παραμέτρους του Config
    roll_mean = df_prod.loc[mask_clean, 'MW'].rolling(config.WINDOW_SIZE, center=True).mean()
    roll_std = df_prod.loc[mask_clean, 'MW'].rolling(config.WINDOW_SIZE, center=True).std()

    upper = roll_mean + config.SIGMA * (roll_std + config.BUFFER)
    lower = roll_mean - config.SIGMA * (roll_std + config.BUFFER)

    frozen = (df_prod.loc[mask_clean, 'MW'].rolling(config.FROZEN_WINDOW).std() < 0.0001) & \
             (df_prod.loc[mask_clean, 'MW'] > 0.1)

    df_prod['is_micro_outlier'] = False
    df_prod.loc[mask_clean, 'is_micro_outlier'] = (
            (df_prod.loc[mask_clean, 'MW'] > upper) |
            (df_prod.loc[mask_clean, 'MW'] < lower) |
            frozen.loc[mask_clean]
    )

    df_prod['FINAL_OUTLIER'] = df_prod['is_macro_outlier'] | df_prod['is_micro_outlier']

    return df_prod, "HYBRID"