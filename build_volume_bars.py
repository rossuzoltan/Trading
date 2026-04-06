"""
Build consolidated volume-bar datasets from per-pair tick files.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_validation import validate_symbol_bar_spec
from project_paths import DATASET_BUILD_INFO_PATH, LEGACY_DATASET_QC_REPORT_PATH, parquet_row_count
from trading_config import (
    DEFAULT_BAR_CONSTRUCTION_TICKS_PER_BAR,
    resolve_bar_construction_ticks_per_bar,
)


TICKS_PER_BAR = DEFAULT_BAR_CONSTRUCTION_TICKS_PER_BAR
DATA_DIR = "data"
MODELS_DIR = "models"
OUTPUT_FILE = "data/DATA_CLEAN_VOLUME.csv"
COMPAT_OUTPUT = "data/FOREX_MULTI_SET.csv"


def purge_old_data(keep_data: bool = False, *, purge_scalers: bool = False) -> None:
    """Delete generated datasets without touching raw tick inputs by default."""
    if not keep_data:
        patterns = [
            "DATA_CLEAN_VOLUME.csv",
            "FOREX_MULTI_SET.csv",
            "*_volbars.csv",
            "*_volbars_*.csv",
            DATASET_BUILD_INFO_PATH.name,
            LEGACY_DATASET_QC_REPORT_PATH.name,
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

    if purge_scalers:
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
    else:
        print("  Keeping model scaler files in models/ (use --purge-scalers to remove them)")


def _load_ticks_for_pair(
    pair: str,
    ticks_per_bar: int,
    *,
    allow_prebuilt_bars: bool,
) -> pd.DataFrame | None:
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

    exact_vbar_path = os.path.join(DATA_DIR, f"{pair}_volbars_{ticks_per_bar}.csv")
    alias_vbar_path = os.path.join(DATA_DIR, f"{pair}_volbars.csv")
    ordered_candidates = [path for path in (exact_vbar_path, alias_vbar_path) if os.path.exists(path)]
    if ordered_candidates:
        if not allow_prebuilt_bars:
            print(
                f"  [WARN] No raw tick data found for {pair}; refusing to use pre-built volume bars "
                "without --allow-prebuilt-bars."
            )
            return None
        vbar_path = ordered_candidates[0]
        print(f"  [INFO] No tick data found - loading pre-built volume bars: {vbar_path}")
        df = pd.read_csv(vbar_path, parse_dates=["Gmt time"])
        df = df.set_index("Gmt time")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df["Symbol"] = pair
        return df

    other_vbar_candidates = sorted(glob.glob(os.path.join(DATA_DIR, f"{pair}_volbars*.csv")))
    if other_vbar_candidates:
        print(
            f"  [WARN] Found pre-built volume bars for {pair}, but none match requested "
            f"ticks_per_bar={ticks_per_bar}. Rebuild volume bars or restore raw tick data."
        )

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


def _summarize_bars(
    pair: str,
    bars: pd.DataFrame,
    *,
    source: str,
    ticks_per_bar: int,
    source_path: str | Path | None = None,
) -> dict[str, float | int | str]:
    ordered = bars.sort_index()
    delta = ordered.index.to_series().diff().dt.total_seconds().dropna()
    raw_tick_path = Path(DATA_DIR) / f"{pair}_ticks.parquet"
    raw_tick_rows: int | None = None
    if raw_tick_path.exists():
        try:
            raw_tick_rows = parquet_row_count(raw_tick_path)
        except Exception:
            raw_tick_rows = None
    expected_bars = None
    if raw_tick_rows is not None:
        expected_bars = (int(raw_tick_rows) + int(ticks_per_bar) - 1) // int(ticks_per_bar)
    return {
        "source": source,
        "bars": int(len(ordered)),
        "start_utc": str(ordered.index.min()) if len(ordered) else "",
        "end_utc": str(ordered.index.max()) if len(ordered) else "",
        "avg_spread_mean": float(np.nanmean(ordered.get("avg_spread", pd.Series([0.0])).values)) if len(ordered) else 0.0,
        "time_delta_s_mean": float(np.nanmean(ordered.get("time_delta_s", pd.Series([0.0])).values)) if len(ordered) else 0.0,
        "duplicate_timestamp_count": int(ordered.index.duplicated().sum()) if len(ordered) else 0,
        "gap_count_gt_6h": int((delta > 21600).sum()) if not delta.empty else 0,
        "gap_count_gt_24h": int((delta > 86400).sum()) if not delta.empty else 0,
        "max_gap_hours": float(delta.max() / 3600.0) if not delta.empty else 0.0,
        "median_gap_minutes": float(delta.median() / 60.0) if not delta.empty else 0.0,
        "source_path": str(source_path) if source_path is not None else "",
        "raw_tick_path": str(raw_tick_path) if raw_tick_path.exists() else "",
        "raw_tick_rows": int(raw_tick_rows) if raw_tick_rows is not None else None,
        "expected_bars_from_ticks": int(expected_bars) if expected_bars is not None else None,
        "bar_count_matches_raw_ticks": bool(expected_bars == len(ordered)) if expected_bars is not None else None,
    }


def _build_dataset_metadata(
    *,
    ticks_per_bar: int,
    combined: pd.DataFrame,
    symbol_stats: dict[str, dict[str, float | int | str]],
) -> dict[str, object]:
    symbol_counts = (
        combined["Symbol"].astype(str).str.upper().value_counts().sort_index().to_dict()
        if "Symbol" in combined.columns
        else {}
    )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": OUTPUT_FILE,
        "compatibility_dataset_path": COMPAT_OUTPUT,
        "bar_construction_ticks_per_bar": int(ticks_per_bar),
        "ticks_per_bar": int(ticks_per_bar),
        "combined_rows": int(len(combined)),
        "symbols": sorted(symbol_counts.keys()),
        "symbol_counts": {symbol: int(count) for symbol, count in symbol_counts.items()},
        "symbol_stats": symbol_stats,
    }


def main(
    ticks_per_bar: int,
    keep_data: bool,
    *,
    allow_prebuilt_bars: bool = False,
    purge_scalers: bool = False,
) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("-" * 60)
    print(f"Volume Bar Pipeline ({ticks_per_bar} ticks/bar)")
    print("-" * 60)

    print("\n[1] Purging generated data ...")
    purge_old_data(
        keep_data=keep_data,
        purge_scalers=purge_scalers,
    )

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
        ticks = _load_ticks_for_pair(
            pair,
            ticks_per_bar,
            allow_prebuilt_bars=allow_prebuilt_bars,
        )
        if ticks is None:
            print(f"  [SKIP] {pair}: no tick data")
            continue

        if "Symbol" in ticks.columns and "Close" in ticks.columns:
            print(f"  {pair}: using pre-built volume bars ({len(ticks)} bars)")
            validate_symbol_bar_spec(ticks, expected_ticks_per_bar=ticks_per_bar, symbol=pair)
            all_bars.append(ticks)
            exact_vbar_path = os.path.join(DATA_DIR, f"{pair}_volbars_{ticks_per_bar}.csv")
            source_path = exact_vbar_path if os.path.exists(exact_vbar_path) else os.path.join(DATA_DIR, f"{pair}_volbars.csv")
            qc[pair] = _summarize_bars(
                pair,
                ticks,
                source="prebuilt_volume_bars",
                ticks_per_bar=ticks_per_bar,
                source_path=source_path,
            )
            continue

        bars = build_volume_bars(ticks, ticks_per_bar)
        if bars.empty:
            print(f"  [SKIP] {pair}: volume bar construction returned empty")
            continue

        bars["Symbol"] = pair
        validate_symbol_bar_spec(bars, expected_ticks_per_bar=ticks_per_bar, symbol=pair)
        all_bars.append(bars)
        print(f"  {pair}: {len(bars):,} volume bars (from {len(ticks):,} ticks)")
        qc[pair] = _summarize_bars(
            pair,
            bars,
            source="raw_ticks",
            ticks_per_bar=ticks_per_bar,
            source_path=os.path.join(DATA_DIR, f"{pair}_ticks.parquet"),
        )

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
    metadata = _build_dataset_metadata(
        ticks_per_bar=ticks_per_bar,
        combined=combined,
        symbol_stats=qc,
    )
    DATASET_BUILD_INFO_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LEGACY_DATASET_QC_REPORT_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Dataset build info -> {DATASET_BUILD_INFO_PATH}")
    print(f"QC report alias -> {LEGACY_DATASET_QC_REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build consolidated volume-bar datasets")
    parser.add_argument(
        "--ticks-per-bar",
        type=int,
        default=resolve_bar_construction_ticks_per_bar("BAR_SPEC_TICKS_PER_BAR", "TRADING_TICKS_PER_BAR"),
        help="Ticks per bar (defaults to BAR_SPEC_TICKS_PER_BAR / TRADING_TICKS_PER_BAR / 2000)",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Keep existing generated CSV files and skip dataset purge.",
    )
    parser.add_argument(
        "--allow-prebuilt-bars",
        action="store_true",
        help="Allow using pre-built *_volbars*.csv files when raw tick data is missing.",
    )
    parser.add_argument(
        "--purge-scalers",
        action="store_true",
        help="Also delete generated scaler_*.pkl files from models/ before rebuilding the dataset.",
    )
    args = parser.parse_args()
    main(
        args.ticks_per_bar,
        args.keep_data,
        allow_prebuilt_bars=args.allow_prebuilt_bars,
        purge_scalers=args.purge_scalers,
    )
