import pandas as pd

try:
    print("Loading data...")
    df1 = pd.read_csv(r"data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv")
    df2 = pd.read_csv(r"data/test_EURUSD_Candlestick_1_Hour_BID_20.02.2023-22.02.2025.csv")

    df = pd.concat([df1, df2])
    
    print("Parsing dates and dropping duplicates...")
    df["Gmt time"] = pd.to_datetime(df["Gmt time"], dayfirst=True)
    df = df.drop_duplicates(subset=["Gmt time"])
    df = df.sort_values(by="Gmt time")

    out_path = r"data/EURUSD_Combined_2020_2025.csv"
    df.to_csv(out_path, index=False)
    print(f"Merged successfully. Total rows: {len(df)}")

except Exception as e:
    print(f"Error merging: {e}")
