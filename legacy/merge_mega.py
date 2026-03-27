import pandas as pd

def merge_datasets():
    # 1. Load existing combined data (2020 - early 2025)
    df1 = pd.read_csv("data/EURUSD_Combined_2020_2025.csv")
    df1['Gmt time'] = pd.to_datetime(df1['Gmt time'])
    
    # 2. Load Yahoo data (late 2024 - 2026)
    df2 = pd.read_csv("data/EURUSD_Yahoo_730d.csv")
    df2['Gmt time'] = pd.to_datetime(df2['Gmt time']).dt.tz_localize(None) # Remove timezone if any
    
    # 3. Combine and deduplicate
    combined = pd.concat([df1, df2], ignore_index=True)
    combined.sort_values('Gmt time', inplace=True)
    combined.drop_duplicates(subset=['Gmt time'], inplace=True)
    
    # 4. Save
    output_path = "data/EURUSD_MEGA_DATASET.csv"
    combined.to_csv(output_path, index=False)
    print(f"Mega dataset created with {len(combined)} rows.")
    print(f"Time Range: {combined['Gmt time'].min()} to {combined['Gmt time'].max()}")

if __name__ == "__main__":
    merge_datasets()
