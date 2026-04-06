"""
download_dukascopy.py  –  Dukascopy Tick Data + Volume Bar Pipeline (Phase 10)
=================================================================================
Downloads Dukascopy tick data and converts to fixed-tick volume bars.

Why Volume Bars > Time Bars (H1)
---------------------------------
Time bars: 1 bar = 1 hour. Quiet periods and frantic news hours look identical.
Volume bars: 1 bar = N ticks. Regime-adaptive, information-dense.
Result: far fewer NaN indicators, better signal/noise ratio for RL training.

Source: Dukascopy public BI5 tick data archive (free, no registration)
URL:    https://datafeed.dukascopy.com/datafeed/{PAIR}/{YEAR}/{MONTH:02}/{DAY:02}/{HOUR:02}h_ticks.bi5

Output:
    data/{PAIR}_ticks.parquet     raw tick data
    data/{PAIR}_volbars_{N}.csv   volume bars for the requested bar spec
    data/{PAIR}_volbars.csv       compatibility alias to the latest requested bar spec

Usage:
    python download_dukascopy.py
    python download_dukascopy.py --pairs EURUSD GBPUSD --days 365 --ticks-per-bar 2000
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import struct
import sys
import time as time_mod
from datetime import datetime, timedelta, timezone


import numpy as np
import pandas as pd
import requests
import concurrent.futures

from trading_config import (
    DEFAULT_BAR_CONSTRUCTION_TICKS_PER_BAR,
    resolve_bar_construction_ticks_per_bar,
)

# ── Config ─────────────────────────────────────────────────────────────────────
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
DAYS_BACK = 1095          # 3 years of history by default
TICKS_PER_BAR = DEFAULT_BAR_CONSTRUCTION_TICKS_PER_BAR
OUTPUT_DIR   = "data"

DUKA_BASE    = "https://datafeed.dukascopy.com/datafeed"
# Point size (price divisor): 5 for JPY pairs, else 5 for all on Dukascopy
POINT_SIZE   = {"USDJPY": 1000, "GBPJPY": 1000, "EURJPY": 1000}

HEADERS = {"User-Agent": "Mozilla/5.0"}
REQUEST_RETRIES = 4
RETRY_BACKOFF_S = 1.5

# ── Dukascopy BI5 parser ──────────────────────────────────────────────────────

def _parse_bi5(data: bytes, pair: str) -> pd.DataFrame:
    """
    Parse Dukascopy BI5 binary tick format.
    Each record: 5 × int32 = ms_offset, ask, bid, ask_vol, bid_vol
    Prices are in points (divide by point_size to get price).
    """
    if not data:
        return pd.DataFrame()

    import lzma
    try:
        raw = lzma.decompress(data)
    except Exception:
        return pd.DataFrame()

    record_size = 20  # 5 × 4 bytes
    n = len(raw) // record_size
    if n == 0:
        return pd.DataFrame()

    records = struct.unpack(f">{n * 5}i", raw[:n * record_size])
    arr = np.array(records, dtype=np.float64).reshape(n, 5)

    point = POINT_SIZE.get(pair, 100_000)
    df = pd.DataFrame({
        "ms":      arr[:, 0],
        "ask":     arr[:, 1] / point,
        "bid":     arr[:, 2] / point,
        "ask_vol": arr[:, 3] / 1_000_000,
        "bid_vol": arr[:, 4] / 1_000_000,
    })
    df["mid"]    = (df["ask"] + df["bid"]) / 2
    df["spread"] = (df["ask"] - df["bid"])
    df["volume"] = 1.0  # Use Tick Volume (standard for FX resampled bars)
    return df


def _download_hour(
    session: requests.Session,
    pair: str,
    dt: datetime,
) -> pd.DataFrame | None:
    """Download one hour of Dukascopy tick data, return parsed DataFrame."""
    url = (
        f"{DUKA_BASE}/{pair}/{dt.year}/{dt.month - 1:02d}/"
        f"{dt.day:02d}/{dt.hour:02d}h_ticks.bi5"
    )
    # Note: Dukascopy month is 0-indexed in the URL
    resp = None
    for attempt in range(REQUEST_RETRIES):
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            break
        except requests.RequestException:
            resp = None
            if attempt == REQUEST_RETRIES - 1:
                return None
            time_mod.sleep(RETRY_BACKOFF_S * (attempt + 1))

    if resp is None:
        return None

    df = _parse_bi5(resp.content, pair)
    if df.empty:
        return None

    # Convert ms offset to absolute UTC timestamps
    base_ts = dt.replace(tzinfo=timezone.utc).timestamp() * 1000
    df["timestamp"] = pd.to_datetime((base_ts + df["ms"]).astype(np.int64), unit="ms", utc=True)
    df = df.set_index("timestamp").drop(columns=["ms"])
    return df


# ── Volume bar builder ────────────────────────────────────────────────────────

def build_volume_bars(ticks: pd.DataFrame, bar_volume: float) -> pd.DataFrame:
    """
    Convert tick data to volume bars (Vectorized for high performance).
    """
    if ticks.empty: return pd.DataFrame()
    
    # 1. Use robust index-based grouping for 'ticks' per bar
    # Since we set volume=1.0 per tick, bar_volume=2000 means standard 2000-tick bars.
    n = len(ticks)
    group_ids = (np.arange(n) // int(bar_volume)).astype(np.int64)
    
    # 2. Add as-column to DataFrame for grouping
    t = ticks.reset_index()
    t["group_id"] = group_ids
    
    # 3. Aggregate
    # Note: price_col for Dukascopy is usually 'mid'
    res = t.groupby("group_id").agg({
        "timestamp": "first",
        "mid":       ["first", "max", "min", "last"],
        "spread":    "mean",
        "volume":    "sum"
    })
    
    # 4. Flatten the MultiIndex resulting from agg
    res.columns = ["Gmt time", "Open", "High", "Low", "Close", "avg_spread", "Volume"]
    res = res.set_index("Gmt time")
    
    return res


# ── Main download logic ───────────────────────────────────────────────────────

def download_pair_ticks(
    session: requests.Session,
    pair: str,
    days: int,
    ticks_per_bar: int,
    *,
    force_refresh: bool = False,
    max_workers: int = 20,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download 'days' of tick data for pair using multiple threads.
    Returns (tick_df, volume_bar_df).
    """
    effective_end = (end_dt or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0)
    effective_start = start_dt or (effective_end - timedelta(days=days))

    # 1. Skip if parquet already exists
    parquet_path = os.path.join(OUTPUT_DIR, f"{pair}_ticks.parquet")
    existing_ticks: pd.DataFrame | None = None
    existing_hours: set[datetime] = set()
    if os.path.exists(parquet_path) and not force_refresh:
        print(f"  [TURBO-RESUME] {pair}: Loading existing ticks from {parquet_path}")
        try:
            existing_ticks = pd.read_parquet(parquet_path)
            if not existing_ticks.empty:
                if existing_ticks.index.tz is None:
                    existing_ticks.index = existing_ticks.index.tz_localize("UTC")
                else:
                    existing_ticks.index = existing_ticks.index.tz_convert("UTC")
                existing_ticks = existing_ticks.sort_index()
                print(f"  {pair}: {len(existing_ticks):,} ticks loaded.")
                hours = existing_ticks.index.floor("h").to_pydatetime()
                existing_hours = set(hours)
        except Exception as e:
            print(f"  [WARN] Could not load parquet: {e}. Re-downloading...")
            existing_ticks = None

    # 1. Prepare target hours
    target_hours: list[datetime] = []
    current = effective_start
    while current < effective_end:
        if current.weekday() < 5:  # Mon-Fri
            if current not in existing_hours:
                target_hours.append(current)
        current += timedelta(hours=1)
    
    total_hours = len(target_hours)
    print(f"\n  Turbo Download {pair}: {effective_start.date()} -> {effective_end.date()} ({total_hours} hours)")

    # 2. Parallel Download
    all_ticks_dfs: list[pd.DataFrame] = []
    downloaded_count = 0
    
    # Use 20 workers for ~20 concurrent HTTP requests
    print(f"  Starting ThreadPool (Workers={max_workers}) ...", flush=True)

    def _worker(dt: datetime) -> pd.DataFrame | None:
        return _download_hour(session, pair, dt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_dt = {executor.submit(_worker, dt): dt for dt in target_hours}
        
        for future in concurrent.futures.as_completed(future_to_dt):
            df_h = future.result()
            downloaded_count += 1
            
            if df_h is not None and not df_h.empty:
                all_ticks_dfs.append(df_h)
            
            if downloaded_count % 100 == 0:
                prog = (downloaded_count / total_hours) * 100
                print(f"    - {pair} Progress: {downloaded_count}/{total_hours} hours ({prog:.1f}%)", flush=True)

    if not all_ticks_dfs:
        if existing_ticks is None or existing_ticks.empty:
            print(f"  [WARN] No tick data for {pair}")
            return pd.DataFrame(), pd.DataFrame()
        print(f"  {pair}: no missing hours; reusing existing ticks.")
        ticks = existing_ticks
        vbars = build_volume_bars(ticks, ticks_per_bar)
        return ticks, vbars

    # 3. Combine and Sort (CRITICAL for parallel results)
    print(f"  Combining and sorting {len(all_ticks_dfs)} hourly blocks ...")
    downloaded = pd.concat(all_ticks_dfs).sort_index()
    if existing_ticks is not None and not existing_ticks.empty:
        ticks = pd.concat([existing_ticks, downloaded]).sort_index()
    else:
        ticks = downloaded
    ticks = ticks[~ticks.index.duplicated(keep="last")]
    print(f"  {pair}: {len(ticks):,} ticks downloaded")

    # 4. Build volume bars
    print(f"  Generating Volume Bars (ticks_per_bar={ticks_per_bar}) ...")
    vbars = build_volume_bars(ticks, int(ticks_per_bar))
    print(f"  {pair}: {len(vbars):,} volume bars created")

    return ticks, vbars


def main(
    pairs: list[str],
    days: int,
    ticks_per_bar: int,
    *,
    force_pairs: set[str] | None = None,
    max_workers: int = 20,
    start: str | None = None,
    end: str | None = None,
) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session = requests.Session()
    force_pairs = {pair.upper() for pair in (force_pairs or set())}
    pairs = [pair.upper() for pair in pairs]

    all_vbars: list[pd.DataFrame] = []
    parsed_start = pd.Timestamp(start, tz="UTC").to_pydatetime() if start else None
    parsed_end = pd.Timestamp(end, tz="UTC").to_pydatetime() if end else None

    for pair in pairs:
        print(f"\n{'='*55}\n{pair}\n{'='*55}")
        ticks, vbars = download_pair_ticks(
            session,
            pair,
            days,
            int(ticks_per_bar),
            force_refresh=pair.upper() in force_pairs,
            max_workers=max_workers,
            start_dt=parsed_start,
            end_dt=parsed_end,
        )

        if ticks.empty:
            continue

        # Save raw ticks
        tick_path = os.path.join(OUTPUT_DIR, f"{pair}_ticks.parquet")
        try:
            ticks.to_parquet(tick_path)
            print(f"  Ticks saved -> {tick_path}")
        except Exception as e:
            tick_csv = tick_path.replace(".parquet", ".csv")
            ticks.to_csv(tick_csv)
            print(f"  Ticks saved -> {tick_csv} (parquet failed: {e})")

        if not vbars.empty:
            vbars["Symbol"] = pair
            all_vbars.append(vbars)

            # Per-pair CSV
            vbar_path = os.path.join(OUTPUT_DIR, f"{pair}_volbars_{int(ticks_per_bar)}.csv")
            vbars_out = vbars.reset_index()
            vbars_out["Gmt time"] = vbars_out["Gmt time"].dt.strftime("%Y.%m.%d %H:%M:%S")
            vbars_out.to_csv(vbar_path, index=False)
            print(f"  Volume bars saved -> {vbar_path}")
            legacy_vbar_path = os.path.join(OUTPUT_DIR, f"{pair}_volbars.csv")
            if legacy_vbar_path != vbar_path:
                shutil.copyfile(vbar_path, legacy_vbar_path)
                print(f"  Volume bars alias -> {legacy_vbar_path}")

    if all_vbars:
        combined = pd.concat(all_vbars)
        combined = combined.reset_index()
        combined["Gmt time"] = combined["Gmt time"].dt.strftime("%Y.%m.%d %H:%M:%S")
        out = os.path.join(OUTPUT_DIR, "FOREX_MULTI_SET.csv")
        combined.to_csv(out, index=False)
        print(f"\n[OK]  Combined volume-bar dataset: {out}")
        print(f"   Rows: {len(combined):,}  |  Pairs: {combined['Symbol'].unique().tolist()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dukascopy tick + Volume Bar downloader")
    parser.add_argument("--pairs",      nargs="+", default=PAIRS)
    parser.add_argument("--days",       type=int,  default=DAYS_BACK,
                        help="Days of history (default 1095 = 3 years)")
    parser.add_argument(
        "--ticks-per-bar",
        type=int,
        default=resolve_bar_construction_ticks_per_bar("BAR_SPEC_TICKS_PER_BAR", "TRADING_TICKS_PER_BAR"),
        help="Ticks per bar (defaults to BAR_SPEC_TICKS_PER_BAR / TRADING_TICKS_PER_BAR / 2000)",
    )
    parser.add_argument("--bar-volume", type=float, default=None,
                        help="Deprecated alias for --ticks-per-bar (kept for compatibility)")
    parser.add_argument("--start", type=str, default=None, help="Optional UTC start timestamp (ISO-like)")
    parser.add_argument("--end", type=str, default=None, help="Optional UTC end timestamp (ISO-like)")
    parser.add_argument(
        "--force-refresh-pairs",
        nargs="*",
        default=[],
        help="Pairs to re-download even if parquet already exists",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help="Concurrent HTTP workers per pair",
    )
    args = parser.parse_args()
    ticks_per_bar = int(args.ticks_per_bar)
    if args.bar_volume is not None:
        ticks_per_bar = int(args.bar_volume)

    print(f"Dukascopy tick download + Volume Bar resampling")
    print(f"Pairs: {args.pairs}  |  Days: {args.days}  |  Ticks/bar: {ticks_per_bar}")
    if args.force_refresh_pairs:
        print(f"Force refresh: {args.force_refresh_pairs}")
    main(
        args.pairs,
        args.days,
        ticks_per_bar,
        force_pairs=set(args.force_refresh_pairs),
        max_workers=max(1, args.max_workers),
        start=args.start,
        end=args.end,
    )
