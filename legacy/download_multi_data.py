"""
download_multi_data.py  –  Phase 7 Production Data Downloader
Downloads H1 OHLCV data for multiple Forex pairs from Yahoo Finance.
Saves in LONG format (one row per bar per symbol) to FOREX_MULTI_SET.csv.
"""

import yfinance as yf
import pandas as pd
import os

SYMBOLS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]

def download_multi(period: str = "730d", interval: str = "1h") -> None:
    print(f"Downloading {SYMBOLS} for {period}...")
    os.makedirs("data", exist_ok=True)
    all_dfs = []

    for sym in SYMBOLS:
        # Download ONE symbol at a time (avoids multi-index wide format)
        raw = yf.download(sym, period=period, interval=interval,
                          auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            print(f"  {sym}: No data returned.")
            continue

        # yfinance sometimes returns a MultiIndex column (ticker, field)
        # Flatten it if needed
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Ensure we have the required columns
        needed = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in raw.columns for c in needed):
            print(f"  {sym}: Missing columns {needed}, got {raw.columns.tolist()}")
            continue

        df = raw[needed].copy()
        df.index.name = "Gmt time"
        df = df.reset_index()

        # Force UTC datetime
        df["Gmt time"] = pd.to_datetime(df["Gmt time"], utc=True)

        # Ensure numeric OHLCV
        for col in needed:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=needed)
        df["Symbol"] = sym
        all_dfs.append(df)
        print(f"  {sym}: {len(df)} rows downloaded")

    if not all_dfs:
        print("No data found. Check your internet connection.")
        return

    # Stack all symbols (long format)
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["Symbol", "Gmt time"]).reset_index(drop=True)

    out_path = "data/FOREX_MULTI_SET.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nMulti-asset dataset saved to {out_path} ({len(combined)} rows)")
    print(f"Columns: {combined.columns.tolist()}")
    print(combined.head(2))


if __name__ == "__main__":
    download_multi()
