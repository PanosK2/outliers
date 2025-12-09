import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import config


# --- Βοηθητική: Micro Analysis ---
def apply_micro_analysis(df, sigma):
    mask_clean = ~df['is_macro_outlier']

    roll_mean = df.loc[mask_clean, 'MW'].rolling(config.WINDOW_SIZE, center=True).mean()
    roll_std = df.loc[mask_clean, 'MW'].rolling(config.WINDOW_SIZE, center=True).std()

    upper = roll_mean + sigma * (roll_std + config.BUFFER)
    lower = roll_mean - sigma * (roll_std + config.BUFFER)

    frozen = (df.loc[mask_clean, 'MW'].rolling(config.FROZEN_WINDOW).std() < 0.0001) & \
             (df.loc[mask_clean, 'MW'] > 0.1)

    df['is_micro_outlier'] = False
    df.loc[mask_clean, 'is_micro_outlier'] = (
            (df.loc[mask_clean, 'MW'] > upper) |
            (df.loc[mask_clean, 'MW'] < lower) |
            frozen.loc[mask_clean]
    )

    df['FINAL_OUTLIER'] = df['is_macro_outlier'] | df['is_micro_outlier']
    return df


# --- Κύρια Συνάρτηση Isolated ---
def run_isolated(df_prod, rated_capacity):
    print(f"      🔄 Εκτέλεση: ISOLATED Method ({config.ISOLATED_METHOD_TYPE})...")

    # 1. Macro Analysis
    df_prod['is_macro_outlier'] = False

    # Βασικοί Κανόνες (πάντα ισχύουν)
    mask_neg = df_prod['MW'] < -0.1
    mask_overcap = df_prod['MW'] > (rated_capacity * 1.1)

    if config.ISOLATED_METHOD_TYPE == 'ISO_FOREST':
        # --- Επιλογή A: Isolation Forest ---
        # Φτιάχνουμε features: MW και Ώρα (0-23) για να βρει μοτίβα
        # Η ώρα βοηθάει να βρει αν π.χ. τη νύχτα έχει περίεργα spikes
        df_features = df_prod[['MW']].copy()
        df_features['hour'] = df_prod.index.hour

        # Καθαρίζουμε NaNs
        valid_data = df_features.dropna()

        if len(valid_data) > 100:
            model = IsolationForest(contamination=config.ISO_CONTAMINATION, random_state=50)
            preds = model.fit_predict(valid_data)

            # Αντιστοίχιση αποτελεσμάτων (όπου -1 = True)
            iso_outliers = pd.Series(preds == -1, index=valid_data.index)

            # Ενημέρωση του DataFrame (με .loc για ασφάλεια)
            # Σημείωση: Το update γίνεται μόνο όπου βρέθηκε outlier
            df_prod.loc[iso_outliers[iso_outliers].index, 'is_macro_outlier'] = True

    elif config.ISOLATED_METHOD_TYPE == 'STATISTICAL':
        # --- Επιλογή B: Μόνο Στατιστική (Πιο "άκαμπτη") ---
        # Εδώ δεν κάνουμε κάτι έξτρα στο Macro, βασιζόμαστε μόνο στα Micro αργότερα
        pass

    # Εφαρμογή των κανόνων (πάνω από ό,τι βρήκε το ISO)
    df_prod['is_macro_outlier'] = df_prod['is_macro_outlier'] | mask_neg | mask_overcap

    # 2. Micro Analysis
    # Αν τρέχουμε Statistical, θέλουμε πιο αυστηρό Micro (Sigma=3.5)
    # Αν τρέχουμε Iso Forest, είμαστε πιο χαλαροί (Sigma=4.5 ή 5)
    sigma_val = 3.5 if config.ISOLATED_METHOD_TYPE == 'STATISTICAL' else 4.5

    df_prod = apply_micro_analysis(df_prod, sigma=sigma_val)

    return df_prod, f"ISOLATED_{config.ISOLATED_METHOD_TYPE}"