"""
download_data.py  –  Institutional-Grade Forex Data Downloader (Phase 9)
=========================================================================
Data source: HistData.com — free, real broker tick data resampled to H1.
Covers 2003–2024 for all major pairs. No API key required.

Why NOT yfinance:
  - Yahoo Forex has no Bid/Ask, fictional Volume, and gaps at rollover
  - HistData provides actual OHLCV from real FX brokers

Pairs downloaded: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD
Years: configurable (default 2010–2024, ~14 years of H1 data per pair)

Usage:
    .\.venv\Scripts\python download_data.py
    .\.venv\Scripts\python download_data.py --pairs EURUSD GBPUSD --start 2015 --end 2024

Output: data/FOREX_MULTI_SET.csv  (same format as before, drop-in replacement)
"""

from __future__ import annotations

import argparse
import io
import os
import time
import zipfile
from datetime import datetime

import pandas as pd
import requests

# ── Config ─────────────────────────────────────────────────────────────────────

HISTDATA_URL = "https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{pair}/{year}/{month}"
HISTDATA_POST = "https://www.histdata.com/get.php"

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]
START_YEAR = 2010
END_YEAR   = 2024

OUTPUT_FILE = "data/FOREX_MULTI_SET.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.histdata.com/",
}

# ── Download helpers ───────────────────────────────────────────────────────────

def _download_month(session: requests.Session, pair: str, year: int, month: int) -> pd.DataFrame | None:
    """Download one month of M1 data from HistData, return as DataFrame."""
    pair_lower = pair.lower()
    referer = f"https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{pair_lower}/{year}/{month:02d}"

    # Step 1: get the page to find the hidden form token
    try:
        page = session.get(referer, headers=HEADERS, timeout=30)
    except requests.RequestException as e:
        print(f"    [WARN] Page fetch failed for {pair} {year}/{month:02d}: {e}")
        return None

    # Step 2: POST to get.php — HistData requires a form post with tk token
    import re
    token_match = re.search(r'id="tk"\s+value="([^"]+)"', page.text)
    token = token_match.group(1) if token_match else ""

    data = {
        "tk": token,
        "date": str(year),
        "datemonth": f"{year}{month:02d}",
        "platform": "ASCII",
        "timeframe": "M1",
        "fxpair": pair,
    }
    headers = {**HEADERS, "Referer": referer, "Content-Type": "application/x-www-form-urlencoded"}
    try:
        resp = session.post(HISTDATA_POST, data=data, headers=headers, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    [WARN] Download failed {pair} {year}/{month:02d}: {e}")
        return None

    if resp.headers.get("Content-Type", "").startswith("text"):
        print(f"    [WARN] Non-zip response for {pair} {year}/{month:02d}")
        return None

    # Step 3: Unzip and parse
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = next((n for n in zf.namelist() if n.endswith(".csv") or n.endswith(".txt")), None)
            if not csv_name:
                return None
            with zf.open(csv_name) as f:
                # HistData M1 format: DateTime;Open;High;Low;Close;Volume
                df = pd.read_csv(f, sep=";", header=None,
                                 names=["DateTime", "Open", "High", "Low", "Close", "Volume"])
    except (zipfile.BadZipFile, Exception) as e:
        print(f"    [WARN] Zip parse error {pair} {year}/{month:02d}: {e}")
        return None

    if df.empty:
        return None

    # Parse datetime
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%Y%m%d %H%M%S", errors="coerce")
    df = df.dropna(subset=["DateTime"])
    df = df.set_index("DateTime")
    df.index = df.index.tz_localize("UTC")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _resample_to_h1(df_m1: pd.DataFrame) -> pd.DataFrame:
    """Resample M1 OHLCV to H1."""
    df_h1 = df_m1.resample("1h").agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }).dropna(subset=["Open", "Close"])
    return df_h1


def download_pair(
    session: requests.Session,
    pair: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Download all months for a pair, return H1 DataFrame."""
    all_months: list[pd.DataFrame] = []
    now = datetime.utcnow()

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Don't request future months
            if year == now.year and month > now.month:
                break
            print(f"  {pair} {year}/{month:02d} ...", end=" ", flush=True)
            df_m = _download_month(session, pair, year, month)
            if df_m is not None and not df_m.empty:
                all_months.append(df_m)
                print(f"ok ({len(df_m)} M1 bars)")
            else:
                print("skip")
            time.sleep(0.5)   # polite rate limit

    if not all_months:
        print(f"  [ERROR] No data downloaded for {pair}")
        return pd.DataFrame()

    df_all_m1 = pd.concat(all_months).sort_index()
    df_all_m1 = df_all_m1[~df_all_m1.index.duplicated(keep="last")]
    df_h1 = _resample_to_h1(df_all_m1)
    df_h1["Symbol"] = pair
    print(f"  ✅ {pair}: {len(df_h1)} H1 bars ({start_year}–{end_year})")
    return df_h1


# ── Main ───────────────────────────────────────────────────────────────────────

def main(pairs: list[str], start: int, end: int) -> None:
    os.makedirs("data", exist_ok=True)

    session = requests.Session()
    all_dfs: list[pd.DataFrame] = []

    for pair in pairs:
        print(f"\n{'='*50}\nDownloading {pair} ({start}–{end})\n{'='*50}")
        df = download_pair(session, pair, start, end)
        if not df.empty:
            all_dfs.append(df)
            # Save per-pair backup
            per_pair_path = f"data/{pair}_H1.csv"
            df.to_csv(per_pair_path)
            print(f"  Saved → {per_pair_path}")

    if not all_dfs:
        print("\n❌  No data downloaded. Check internet connection.")
        return

    combined = pd.concat(all_dfs).sort_index()
    combined.index.name = "Gmt time"
    combined = combined.reset_index()
    combined["Gmt time"] = combined["Gmt time"].dt.strftime("%Y.%m.%d %H:%M:%S")
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅  Combined dataset saved → {OUTPUT_FILE}")
    print(f"   Total rows: {len(combined):,}")
    print(f"   Pairs: {combined['Symbol'].unique().tolist()}")
    print(f"   Date range: {combined['Gmt time'].min()} → {combined['Gmt time'].max()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download institutional-grade Forex H1 data")
    parser.add_argument("--pairs", nargs="+", default=PAIRS,
                        help=f"Pairs to download (default: {PAIRS})")
    parser.add_argument("--start", type=int, default=START_YEAR,
                        help=f"Start year (default: {START_YEAR})")
    parser.add_argument("--end", type=int, default=END_YEAR,
                        help=f"End year (default: {END_YEAR})")
    args = parser.parse_args()

    print(f"Downloading {len(args.pairs)} pairs × {args.end - args.start + 1} years of H1 data")
    print(f"Source: HistData.com (real broker data, Bid-based OHLCV)")
    print(f"Output: {OUTPUT_FILE}\n")

    main(args.pairs, args.start, args.end)
