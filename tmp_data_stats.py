import pandas as pd
import glob
import os

files = glob.glob('data/*.csv')
for f in files:
    try:
        df = pd.read_csv(f)
        print(f"--- {os.path.basename(f)} ---")
        print(f"Rows: {len(df)}")
        if 'Gmt time' in df.columns:
            df['Gmt time'] = pd.to_datetime(df['Gmt time'], errors='coerce')
            print(f"Start: {df['Gmt time'].min()}")
            print(f"End:   {df['Gmt time'].max()}")
        else:
            print("No 'Gmt time' column.")
        print()
    except Exception as e:
        print(f"Error reading {f}: {e}\n")
