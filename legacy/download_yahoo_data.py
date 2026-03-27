import yfinance as yf
import pandas as pd
import os

def download_yahoo_data(symbol="EURUSD=X", interval="1h", period="730d"):
    print(f"Downloading {symbol} data for period: {period}...")
    df = yf.download(symbol, period=period, interval=interval)
    
    if df.empty:
        print("Download failed.")
        return None
        
    # Standardize columns for indicators.py
    df.reset_index(inplace=True)
    df.rename(columns={
        'Datetime': 'Gmt time',
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume'
    }, inplace=True)
    
    output_path = "data/EURUSD_Yahoo_730d.csv"
    os.makedirs("data", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    return output_path

if __name__ == "__main__":
    download_yahoo_data()
