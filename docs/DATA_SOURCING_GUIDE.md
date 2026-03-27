# Data Sourcing Guide: Dukascopy + Yahoo Finance
# ================================================
# This document explains how to download, combine, and maintain
# the training dataset for the Forex RL trading agent.

## 1. Dataset Overview

| Source | Period | Frequency | Format | Timezone |
|--------|--------|-----------|--------|----------|
| Dukascopy | 2020-2025 | H1 (hourly) | CSV | GMT (UTC) |
| Yahoo Finance | 2024-2026 | H1 (hourly) | API | UTC-aware |

Combined: `data/FOREX_MULTI_SET.csv` — 69,000+ bars across EURUSD, GBPUSD, USDJPY, AUDUSD

## 2. Downloading from Dukascopy (Free, High Quality)

1. Go to https://www.dukascopy.com/swiss/english/marketwatch/historic/
2. Select instrument: **EUR/USD**
3. Select timeframe: **Hourly (H1)**
4. Select date range: **2020-01-01 to 2025-01-01**
5. Download format: **CSV**
6. Repeat for: GBPUSD, USDJPY, AUDUSD

### Column Format Expected
```
Gmt time,Open,High,Low,Close,Volume
01.01.2020 00:00:00.000,1.12151,1.12195,1.12108,1.12171,1234
```

Save files to `data/` folder:
- `data/EURUSD_2020_2025.csv`
- `data/GBPUSD_2020_2025.csv`
- etc.

## 3. Downloading from Yahoo Finance (Automated)

Run the built-in downloader (fetches last 730 days):

```bash
python download_multi_data.py
```

This downloads EURUSD=X, GBPUSD=X, USDJPY=X, AUDUSD=X via `yfinance`
and saves them to `data/FOREX_MULTI_SET.csv`.

### Extending the date range

Edit `download_multi_data.py`:
```python
# Change period="730d" to period="max" for all available data
df = yf.download(sym, period="max", interval="1h", auto_adjust=True)
```

> **Warning:** Yahoo Finance free tier has limits:
> - Max history for 1h bars: ~730 days
> - For longer history, use Dukascopy CSV download

## 4. Combining Sources (merge_mega.py)

```bash
python merge_mega.py
```

This merges Dukascopy CSVs (2020-2025) with Yahoo data (2024-2026),
deduplicates overlapping timestamps, and outputs `data/FOREX_MEGA_MERGED.csv`.

### How the merge works
1. **Parse all timestamps as UTC** — eliminates GMT/EET/EST drift
2. **Concatenate** both sources
3. **Drop duplicates** on the datetime index
4. **Sort** chronologically
5. **Forward-fill** any gaps ≤ 3 hours (weekend gaps are kept)

## 5. Alternative Data Sources

### Alpha Vantage (Free API)
```python
import alpha_vantage
# 5 API calls/minute free tier
# Provides FX_INTRADAY endpoint
# URL: https://www.alphavantage.co/
```

### OANDA Historical (Requires demo account)
```python
import oandapyV20
from oandapyV20.endpoints import instruments
# Provides candlestick data via REST API
# Documentation: https://developer.oanda.com/rest-live-v20/instrument-ep/
```

### Polygon.io (Paid, institutional quality)
```
# Provides tick-level Forex data
# Very reliable for backtesting (no survivorship bias)
# URL: https://polygon.io/forex
```

## 6. Dataset Quality Checklist

Before training, verify your dataset with:

```python
import pandas as pd
df = pd.read_csv("data/FOREX_MULTI_SET.csv", low_memory=False)

# 1. Check for NaN values
print(df.isnull().sum())

# 2. Check timestamp gaps (should be max 3-5 hour gaps, not days)
df['Gmt time'] = pd.to_datetime(df['Gmt time'], utc=True)
for sym in df['Symbol'].unique():
    sdf = df[df['Symbol']==sym].sort_values('Gmt time')
    gaps = sdf['Gmt time'].diff().dt.total_seconds() / 3600
    big_gaps = gaps[gaps > 5]
    print(f"{sym}: {len(big_gaps)} gaps > 5h")

# 3. Check price sanity (no zero or negative prices)
print("Min Close:", df['Close'].min())
print("Max Close:", df['Close'].max())

# 4. Check volume is present
print("Volume zeros:", (df['Volume'] == 0).sum())
```

## 7. Recommended Data Refresh Schedule

| Action | Frequency |
|--------|-----------|
| Re-run `download_multi_data.py` | Weekly |
| Re-run `merge_mega.py` | Weekly |
| Re-run `train_agent.py` (fine-tune) | Monthly |
| Full retrain from scratch | Every 6 months |

> **Note:** Do NOT retrain the StandardScaler (`scaler_features.pkl`) unless you are
> doing a full retrain from scratch. The scaler must be consistent between training
> and live inference or the model will receive out-of-distribution inputs.
