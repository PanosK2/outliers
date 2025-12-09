import os
import pandas as pd
import matplotlib.pyplot as plt
import config
from data_loader import load_park_data

# Εισαγωγή των δύο αλγορίθμων από τα ξεχωριστά αρχεία
from algorithm_hybrid import run_hybrid
from algorithm_isolated import run_isolated


def process_all_parks():
    print(f"🏁 Έναρξη Batch Processing...")
    results = []

    for folder in os.listdir(config.ROOT_DIR):
        path = os.path.join(config.ROOT_DIR, folder)

        if os.path.isdir(path):
            print(f"\n🚀 Επεξεργασία: {folder}")

            # 1. Φόρτωση
            data = load_park_data(path)
            if not data: continue
            park_name, capacity, df_prod, df_weather, prod_filepath = data

            # 2. Έλεγχος Έτους
            if not df_prod.empty:
                year = df_prod.index.min().year
                print(f"   📅 Έτος Δεδομένων: {year}")
            else:
                continue

            # 3. Επιλογή Αλγορίθμου
            df_clean = None
            mode = "UNKNOWN"

            try:
                # Λογική: Αν 2023 ΚΑΙ έχουμε καιρό -> Hybrid. Αλλιώς -> Isolated.
                if year == 2023 and not df_weather.empty:
                    # Έλεγχος επικάλυψης
                    common = df_prod.index.intersection(df_weather.index)
                    if len(common) > 24:
                        df_clean, mode = run_hybrid(df_prod, df_weather, capacity)
                    else:
                        print("      ⚠️ 2023 χωρίς επικάλυψη -> Isolated.")
                        df_clean, mode = run_isolated(df_prod, capacity)
                else:
                    print(f"      ℹ️ Έτος {year} (ή έλλειψη καιρού) -> Isolated.")
                    df_clean, mode = run_isolated(df_prod, capacity)

            except Exception as e:
                print(f"   ❌ Σφάλμα αλγορίθμου: {e}")
                import traceback
                traceback.print_exc()
                continue

            if df_clean is None: continue

            # 4. Αποθήκευση & Plotting
            out_name = f"{os.path.splitext(os.path.basename(prod_filepath))[0]}_CLEANED.csv"
            df_clean.to_csv(os.path.join(path, out_name))

            pct = df_clean['FINAL_OUTLIER'].mean() * 100
            print(f"   ✅ Ολοκληρώθηκε ({mode}). Outliers: {pct:.2f}%")

            # Plotting (Smart Zoom)
            weekly_out = df_clean['FINAL_OUTLIER'].resample('W').sum()
            if not weekly_out.empty:
                worst = weekly_out.idxmax()
                sub = df_clean[worst - pd.Timedelta(days=7): worst]

                plt.figure(figsize=(15, 7))
                plt.plot(sub.index, sub['MW'], color='gray', alpha=0.5, label='Raw')

                if sub['is_macro_outlier'].any():
                    plt.scatter(sub[sub['is_macro_outlier']].index, sub[sub['is_macro_outlier']]['MW'], c='red', s=15,
                                label='Macro')
                if sub['is_micro_outlier'].any():
                    plt.scatter(sub[sub['is_micro_outlier']].index, sub[sub['is_micro_outlier']]['MW'], c='orange',
                                marker='x', s=30, label='Micro')

                plt.title(f"{park_name} [{mode}]: Outlier Analysis")
                plt.legend()


                plt.savefig(os.path.join(path, f"{park_name}_ANALYSIS.png"))
                plt.close()

            results.append({'park': park_name, 'year': year, 'mode': mode, 'outliers_pct': pct})

    if results:
        print("\n" + "=" * 50)
        print(pd.DataFrame(results))


if __name__ == "__main__":
    process_all_parks()
