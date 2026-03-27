"""
Build consolidated volume-bar datasets from per-pair tick files.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


TICKS_PER_BAR = 2_000
DATA_DIR = "data"
MODELS_DIR = "models"
OUTPUT_FILE = "data/DATA_CLEAN_VOLUME.csv"
COMPAT_OUTPUT = "data/FOREX_MULTI_SET.csv"
QC_REPORT = "data/volume_bars_qc_report.json"


def purge_old_data(keep_data: bool = False) -> None:
    """Delete generated datasets and scaler files without touching raw inputs."""
    if not keep_data:
        patterns = [
            "DATA_CLEAN_VOLUME.csv",
            "FOREX_MULTI_SET.csv",
            "*_volbars.csv",
            "*_volbars_*.csv",
        ]
        csv_files: list[str] = []
        for pattern in patterns:
            csv_files.extend(glob.glob(os.path.join(DATA_DIR, pattern)))
        csv_files = sorted(set(csv_files))

        for file_path in csv_files:
            os.remove(file_path)
            print(f"  Deleted: {file_path}")

        if csv_files:
            print(f"  Purged {len(csv_files)} generated CSV files from {DATA_DIR}/")
        else:
            print(f"  No generated CSV files to purge in {DATA_DIR}/")

    pkl_files = glob.glob(os.path.join(MODELS_DIR, "scaler_*.pkl"))
    combined_pkl = os.path.join(MODELS_DIR, "scaler_features.pkl")
    all_pkl = sorted(set(pkl_files + ([combined_pkl] if os.path.exists(combined_pkl) else [])))
    for file_path in all_pkl:
        os.remove(file_path)
        print(f"  Deleted: {file_path}")

    if all_pkl:
        print(f"  Purged {len(all_pkl)} scaler files from {MODELS_DIR}/")
    else:
        print(f"  No scaler files to purge in {MODELS_DIR}/")


def _load_ticks_for_pair(pair: str) -> pd.DataFrame | None:
    parquet_path = os.path.join(DATA_DIR, f"{pair}_ticks.parquet")
    csv_path = os.path.join(DATA_DIR, f"{pair}_ticks.csv")

    if os.path.exists(parquet_path):
        print(f"  Loading {parquet_path} ...")
        try:
            return pd.read_parquet(parquet_path)
        except Exception as exc:
            print(f"  [WARN] parquet load failed: {exc}")

    if os.path.exists(csv_path):
        print(f"  Loading {csv_path} ...")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df

    vbar_candidates = sorted(glob.glob(os.path.join(DATA_DIR, f"{pair}_volbars*.csv")))
    if vbar_candidates:
        vbar_path = vbar_candidates[0]
        print(f"  [INFO] No tick data found - loading pre-built volume bars: {vbar_path}")
        df = pd.read_csv(vbar_path, parse_dates=["Gmt time"])
        df = df.set_index("Gmt time")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df["Symbol"] = pair
        return df

    return None


def build_volume_bars(ticks: pd.DataFrame, ticks_per_bar: int) -> pd.DataFrame:
    if "Close" in ticks.columns and "Open" in ticks.columns:
        return _resample_ohlcv_to_vbars(ticks, ticks_per_bar)
    return build_volume_bars_from_ticks(ticks, ticks_per_bar)


def build_volume_bars_from_ticks(
    ticks: pd.DataFrame, ticks_per_bar: int, price_col: str = "mid"
) -> pd.DataFrame:
    if ticks.empty:
        return pd.DataFrame()

    group_id = (np.arange(len(ticks)) // ticks_per_bar).astype(np.int64)
    frame = ticks.reset_index()
    frame["group_id"] = group_id
    frame["spread"] = (
        (frame["ask"] - frame["bid"]).abs()
        if "ask" in frame.columns and "bid" in frame.columns
        else 0.0
    )

    result = frame.groupby("group_id").agg(
        {
            "timestamp": "first",
            price_col: ["first", "max", "min", "last"],
            "spread": "mean",
        }
    )
    result.columns = ["Gmt time", "Open", "High", "Low", "Close", "avg_spread"]
    result["Volume"] = float(ticks_per_bar)
    result = result.set_index("Gmt time")
    result.index = pd.DatetimeIndex(result.index, tz="UTC")
    result["time_delta_s"] = result.index.to_series().diff().dt.total_seconds().fillna(0.0)
    return result


def _resample_ohlcv_to_vbars(ohlcv: pd.DataFrame, rows_per_bar: int) -> pd.DataFrame:
    groups = np.arange(len(ohlcv)) // rows_per_bar
    result = ohlcv.groupby(groups).agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
    )
    first_timestamps = ohlcv.index.to_series().groupby(groups).first()
    result.index = pd.DatetimeIndex(first_timestamps.values, tz="UTC")
    result["time_delta_s"] = result.index.to_series().diff().dt.total_seconds().fillna(0.0)
    return result


def discover_pairs() -> list[str]:
    paths = (
        glob.glob(os.path.join(DATA_DIR, "*_ticks.parquet"))
        + glob.glob(os.path.join(DATA_DIR, "*_ticks.csv"))
        + glob.glob(os.path.join(DATA_DIR, "*_volbars*.csv"))
    )
    pairs: list[str] = []
    for path in paths:
        pair = os.path.basename(path).split("_")[0].upper()
        if len(pair) == 6 and pair not in pairs:
            pairs.append(pair)
    return pairs


def main(ticks_per_bar: int, keep_data: bool) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("-" * 60)
    print(f"Volume Bar Pipeline ({ticks_per_bar} ticks/bar)")
    print("-" * 60)

    print("\n[1] Purging generated data ...")
    purge_old_data(keep_data=keep_data)

    pairs = discover_pairs()
    if not pairs:
        print("\n[ERROR] No tick data found in data/")
        print("Run first: .\\.venv\\Scripts\\python download_dukascopy.py")
        return

    print(f"\n[2] Found pairs: {pairs}")

    all_bars: list[pd.DataFrame] = []
    qc: dict[str, dict[str, float | int | str]] = {}
    print(f"\n[3] Building volume bars ({ticks_per_bar} ticks/bar) ...")
    for pair in pairs:
        ticks = _load_ticks_for_pair(pair)
        if ticks is None:
            print(f"  [SKIP] {pair}: no tick data")
            continue

        if "Symbol" in ticks.columns and "Close" in ticks.columns:
            print(f"  {pair}: using pre-built volume bars ({len(ticks)} bars)")
            all_bars.append(ticks)
            continue

        bars = build_volume_bars(ticks, ticks_per_bar)
        if bars.empty:
            print(f"  [SKIP] {pair}: volume bar construction returned empty")
            continue

        bars["Symbol"] = pair
        all_bars.append(bars)
        print(f"  {pair}: {len(bars):,} volume bars (from {len(ticks):,} ticks)")
        qc[pair] = {
            "bars": int(len(bars)),
            "start_utc": str(bars.index.min()) if len(bars) else "",
            "end_utc": str(bars.index.max()) if len(bars) else "",
            "avg_spread_mean": float(np.nanmean(bars.get("avg_spread", pd.Series([0.0])).values)) if len(bars) else 0.0,
            "time_delta_s_mean": float(np.nanmean(bars.get("time_delta_s", pd.Series([0.0])).values)) if len(bars) else 0.0,
        }

        per_pair_path = os.path.join(DATA_DIR, f"{pair}_volbars_{ticks_per_bar}.csv")
        out = bars.reset_index()
        out["Gmt time"] = out["Gmt time"].dt.strftime("%Y.%m.%d %H:%M:%S")
        out.to_csv(per_pair_path, index=False)
        print(f"  Saved -> {per_pair_path}")

        legacy_pair_path = os.path.join(DATA_DIR, f"{pair}_volbars.csv")
        if legacy_pair_path != per_pair_path:
            shutil.copyfile(per_pair_path, legacy_pair_path)
            print(f"  Alias -> {legacy_pair_path}")

    if not all_bars:
        print("\n[ERROR] No volume bars built. Check tick data availability.")
        return

    combined = pd.concat(all_bars).sort_index().reset_index()
    combined = combined.dropna(subset=["Open", "High", "Low", "Close", "Volume", "Symbol"])
    if "Gmt time" in combined.columns and "Symbol" in combined.columns:
        combined = combined.drop_duplicates(subset=["Gmt time", "Symbol"], keep="last")
    if "Gmt time" in combined.columns and hasattr(combined["Gmt time"], "dt"):
        combined["Gmt time"] = combined["Gmt time"].dt.strftime("%Y.%m.%d %H:%M:%S")

    combined.to_csv(OUTPUT_FILE, index=False)
    combined.to_csv(COMPAT_OUTPUT, index=False)

    print(f"\nOK: {OUTPUT_FILE}")
    print(f"Compatibility alias: {COMPAT_OUTPUT}")
    print(f"   Total bars : {len(combined):,}")
    print(f"   Symbols    : {combined['Symbol'].unique().tolist()}")
    if "Gmt time" in combined.columns:
        print(f"   Date range : {combined['Gmt time'].min()} -> {combined['Gmt time'].max()}")
    Path(QC_REPORT).write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(f"QC report -> {QC_REPORT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build consolidated volume-bar datasets")
    parser.add_argument("--ticks-per-bar", type=int, default=TICKS_PER_BAR)
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Keep existing generated CSV files and only purge scalers",
    )
    args = parser.parse_args()
    main(args.ticks_per_bar, args.keep_data)
