import pandas as pd
import pandas_ta as ta
import numpy as np


def load_and_preprocess_data(csv_path: str):
    """
    Loads EURUSD data from CSV and preprocesses it by adding RELATIVE technical features.

    CSV expected columns: [Time (EET), Open, High, Low, Close, Volume]
    The returned DataFrame still contains OHLCV for env internals,
    but `feature_cols` lists only the RELATIVE columns to feed the agent.
    """
    # Detect time column
    time_col = "Gmt time" if "Gmt time" in pd.read_csv(csv_path, nrows=1).columns else "Time (EET)"

    df = pd.read_csv(
        csv_path,
        parse_dates=[time_col],
        dayfirst=True,
    )

    # Strip any trailing spaces in headers (e.g. 'Volume ')
    df.columns = df.columns.str.strip()

    # Datetime index
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)
    df.index = pd.DatetimeIndex(df.index)
    df.sort_index(inplace=True)

    # Ensure numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- Temporal Features (Cyclical) ----
    # df.index is already the datetime index from line 26
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    
    # Day cycle (5 days for Forex typically, Mon-Fri: 0-4)
    df["day_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 5)
    df["day_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 5)

    # ---- Log Returns (Primary Stationary Feature) ----
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # ---- Technicals ----
    # RSI and ATR (already scale-invariant-ish)
    df["rsi_14"] = ta.rsi(df["Close"], length=14)
    df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # MACD, BBands, ADX
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.adx(length=14, append=True)

    # Price action features
    # Instead of raw pips, use Log Returns or ATR-normalized distance
    df["body_size"] = (df["Close"] - df["Open"]) / df["atr_14"]
    df["candle_range"] = (df["High"] - df["Low"]) / df["atr_14"]

    # Moving averages (Slopes)
    df["ma_20"] = ta.sma(df["Close"], length=20)
    df["ma_50"] = ta.sma(df["Close"], length=50)
    df["ma_20_slope"] = df["ma_20"].diff() / df["atr_14"]
    df["ma_50_slope"] = df["ma_50"].diff() / df["atr_14"]

    # Z-Score Normalization (Stationarity)
    # We normalize indicators by their 100-period rolling stats
    cols_to_zscore = [
        "rsi_14", "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9", 
        "BBB_20_2.0_2.0", "BBP_20_2.0_2.0", "ADX_14"
    ]
    for col in cols_to_zscore:
        if col in df.columns:
            rolling_mean = df[col].rolling(window=100).mean()
            rolling_std = df[col].rolling(window=100).std()
            df[f"{col}_z"] = (df[col] - rolling_mean) / (rolling_std + 1e-8)

    # Drop initial NaNs from indicators
    df.dropna(inplace=True)

    # Columns the AGENT should see (Stationary/Normalized only)
    feature_cols = [
        "log_return",
        "rsi_14_z",
        "MACD_12_26_9_z",
        "MACDh_12_26_9_z",
        "BBB_20_2.0_2.0_z",
        "BBP_20_2.0_2.0_z",
        "ADX_14_z",
        "body_size",
        "candle_range",
        "ma_20_slope",
        "ma_50_slope",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
    ]

    return df, feature_cols
