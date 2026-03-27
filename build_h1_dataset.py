from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from project_paths import DATA_DIR


DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
DEFAULT_OUTPUT = DATA_DIR / "FOREX_H1_MULTI_SET.csv"
DEFAULT_MANIFEST = DATA_DIR / "FOREX_H1_MULTI_SET.manifest.json"
DEFAULT_TICKS_PER_BAR = 2_000
DEFAULT_MAX_GAP_HOURS = 72


def _to_utc_timestamp(value: str | None, *, fallback: pd.Timestamp | None = None) -> pd.Timestamp:
    if value is None:
        if fallback is None:
            raise ValueError("A timestamp value is required.")
        return fallback

    timestamp = pd.Timestamp(value)
    if timestamp.tz is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _normalize_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if isinstance(output.index, pd.DatetimeIndex):
        index = output.index
    elif "timestamp" in output.columns:
        index = pd.to_datetime(output["timestamp"], utc=True, errors="coerce")
        output = output.drop(columns=["timestamp"])
    elif "Gmt time" in output.columns:
        index = pd.to_datetime(output["Gmt time"], utc=True, errors="coerce")
        output = output.drop(columns=["Gmt time"])
    else:
        index = pd.to_datetime(output.index, utc=True, errors="coerce")

    index = pd.to_datetime(index, utc=True, errors="coerce")
    output.index = pd.DatetimeIndex(index)
    output = output[~output.index.isna()]
    output = output[~output.index.duplicated(keep="last")]
    output = output.sort_index()
    output.index.name = "Gmt time"
    return output


def _load_local_ticks(pair: str) -> pd.DataFrame | None:
    parquet_path = DATA_DIR / f"{pair}_ticks.parquet"
    csv_path = DATA_DIR / f"{pair}_ticks.csv"

    if parquet_path.exists():
        frame = pd.read_parquet(parquet_path)
        return _normalize_tick_frame(frame)

    if csv_path.exists():
        frame = pd.read_csv(csv_path)
        return _normalize_tick_frame(frame)

    return None


def _normalize_tick_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = _normalize_datetime_index(frame)
    if "mid" not in output.columns:
        if {"ask", "bid"}.issubset(output.columns):
            output["mid"] = (pd.to_numeric(output["ask"], errors="coerce") + pd.to_numeric(output["bid"], errors="coerce")) / 2.0
        elif "Close" in output.columns:
            output["mid"] = pd.to_numeric(output["Close"], errors="coerce")
        else:
            raise ValueError("Tick data must contain either 'mid', 'ask'+'bid', or 'Close'.")

    if "spread" not in output.columns:
        if {"ask", "bid"}.issubset(output.columns):
            output["spread"] = (pd.to_numeric(output["ask"], errors="coerce") - pd.to_numeric(output["bid"], errors="coerce")).abs()
        else:
            output["spread"] = np.nan

    if "volume" not in output.columns:
        if "tick_volume" in output.columns:
            output["volume"] = pd.to_numeric(output["tick_volume"], errors="coerce")
        else:
            output["volume"] = 1.0
    else:
        output["volume"] = pd.to_numeric(output["volume"], errors="coerce")

    for column_name in ("mid", "spread"):
        output[column_name] = pd.to_numeric(output[column_name], errors="coerce")

    output = output.dropna(subset=["mid"])
    output["volume"] = output["volume"].fillna(1.0)
    output["spread"] = output["spread"].fillna(0.0)
    return output


def _local_tick_coverage(frame: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if frame.empty:
        return None, None
    return frame.index.min(), frame.index.max()


def _build_h1_from_ticks(ticks: pd.DataFrame, pair: str, source_label: str) -> pd.DataFrame:
    if ticks.empty:
        return pd.DataFrame()

    frame = ticks.sort_index().copy()
    hourly = frame.resample("1h", label="left", closed="left")

    bars = hourly.agg(
        Open=("mid", "first"),
        High=("mid", "max"),
        Low=("mid", "min"),
        Close=("mid", "last"),
        Volume=("volume", "sum"),
        avg_spread=("spread", "mean"),
    )
    bars["tick_count"] = hourly.size()
    bars = bars.dropna(subset=["Open", "High", "Low", "Close"])
    bars["time_delta_s"] = bars.index.to_series().diff().dt.total_seconds().fillna(0.0)
    bars["Symbol"] = pair
    bars["source"] = source_label
    bars["avg_spread"] = bars["avg_spread"].fillna(0.0)
    bars["Volume"] = pd.to_numeric(bars["Volume"], errors="coerce").fillna(0.0)
    bars["tick_count"] = pd.to_numeric(bars["tick_count"], errors="coerce").fillna(0).astype(np.int64)
    bars.index.name = "Gmt time"
    return bars


def _load_mt5_h1(pair: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError("MetaTrader5 is not installed.") from exc

    if not mt5.initialize():
        raise RuntimeError("MetaTrader5 initialize() failed.")

    try:
        symbol_info = mt5.symbol_info(pair)
        point_size = float(symbol_info.point) if symbol_info and symbol_info.point else (0.01 if pair.endswith("JPY") else 0.0001)
        rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_H1, start.to_pydatetime(), end.to_pydatetime())
        if rates is None or len(rates) == 0:
            return pd.DataFrame()

        bars = pd.DataFrame(rates)
        bars["Gmt time"] = pd.to_datetime(bars["time"], unit="s", utc=True)
        bars = bars.set_index("Gmt time")
        bars = bars.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "tick_volume": "Volume",
                "real_volume": "real_volume",
            }
        )
        bars["avg_spread"] = pd.to_numeric(bars.get("spread", 0.0), errors="coerce").fillna(0.0) * point_size
        bars["tick_count"] = pd.to_numeric(bars.get("Volume", 0.0), errors="coerce").fillna(0).astype(np.int64)
        bars["time_delta_s"] = bars.index.to_series().diff().dt.total_seconds().fillna(0.0)
        bars["Symbol"] = pair
        bars["source"] = "mt5_h1"
        bars.index.name = "Gmt time"
        return bars[["Open", "High", "Low", "Close", "Volume", "avg_spread", "tick_count", "time_delta_s", "Symbol", "source"]]
    finally:
        mt5.shutdown()


def _load_yfinance_h1(pair: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed.") from exc

    ticker = f"{pair}=X"
    raw = yf.download(ticker, start=start.tz_convert("UTC").to_pydatetime(), end=end.tz_convert("UTC").to_pydatetime(), interval="1h", auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(-1)

    needed = ["Open", "High", "Low", "Close"]
    if not all(column in raw.columns for column in needed):
        raise RuntimeError(f"yfinance returned unexpected columns: {raw.columns.tolist()}")

    bars = raw.copy()
    bars.index = pd.to_datetime(bars.index, utc=True, errors="coerce")
    bars = bars.dropna(subset=needed)
    bars["Volume"] = pd.to_numeric(bars.get("Volume", 0.0), errors="coerce").fillna(0.0)
    bars["avg_spread"] = 0.0
    bars["tick_count"] = bars["Volume"].round().astype(np.int64)
    bars["time_delta_s"] = bars.index.to_series().diff().dt.total_seconds().fillna(0.0)
    bars["Symbol"] = pair
    bars["source"] = "yfinance_h1"
    bars.index.name = "Gmt time"
    return bars[["Open", "High", "Low", "Close", "Volume", "avg_spread", "tick_count", "time_delta_s", "Symbol", "source"]]


def _load_dukascopy_h1(
    pair: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_workers: int,
    ticks_per_bar: int,
    force_refresh: bool,
) -> pd.DataFrame:
    try:
        import requests
        from download_dukascopy import download_pair_ticks
    except ImportError as exc:
        raise RuntimeError("Dukascopy downloader dependencies are not available.") from exc

    session = requests.Session()
    ticks, _ = download_pair_ticks(
        session,
        pair,
        days=max(1, int((end - start).days)),
        ticks_per_bar=ticks_per_bar,
        force_refresh=force_refresh,
        max_workers=max_workers,
        start_dt=start.to_pydatetime(),
        end_dt=end.to_pydatetime(),
    )
    if ticks.empty:
        return pd.DataFrame()

    normalized = _normalize_tick_frame(ticks)
    normalized = normalized.loc[(normalized.index >= start) & (normalized.index < end)].copy()
    if normalized.empty:
        return pd.DataFrame()

    parquet_path = DATA_DIR / f"{pair}_ticks.parquet"
    normalized.to_parquet(parquet_path)
    return _build_h1_from_ticks(normalized, pair, "dukascopy_ticks")


def _ensure_closed_window(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_floor = start.floor("h")
    end_floor = end.floor("h")
    if end_floor <= start_floor:
        raise ValueError("The selected window is too short to form closed H1 bars.")
    return start_floor, end_floor


def _build_for_pair(
    pair: str,
    *,
    source: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_workers: int,
    ticks_per_bar: int,
    force_refresh: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    local_ticks = _load_local_ticks(pair)
    start_floor, end_floor = _ensure_closed_window(start, end)
    local_coverage_start, local_coverage_end = _local_tick_coverage(local_ticks) if local_ticks is not None else (None, None)

    if source == "local":
        if local_ticks is None:
            raise FileNotFoundError(f"No local tick cache found for {pair}.")
        filtered = local_ticks.loc[(local_ticks.index >= start_floor) & (local_ticks.index < end_floor)].copy()
        bars = _build_h1_from_ticks(filtered, pair, "local_ticks")
        return bars, {
            "source": "local_ticks",
            "local_tick_coverage_start": str(local_coverage_start) if local_coverage_start is not None else None,
            "local_tick_coverage_end": str(local_coverage_end) if local_coverage_end is not None else None,
        }

    if source == "auto" and local_ticks is not None and local_coverage_start is not None and local_coverage_end is not None:
        if local_coverage_start <= start_floor and local_coverage_end >= end_floor:
            filtered = local_ticks.loc[(local_ticks.index >= start_floor) & (local_ticks.index < end_floor)].copy()
            bars = _build_h1_from_ticks(filtered, pair, "local_ticks")
            return bars, {
                "source": "local_ticks",
                "local_tick_coverage_start": str(local_coverage_start),
                "local_tick_coverage_end": str(local_coverage_end),
            }

    if source in {"auto", "dukascopy"}:
        bars = _load_dukascopy_h1(
            pair,
            start=start_floor,
            end=end_floor,
            max_workers=max_workers,
            ticks_per_bar=ticks_per_bar,
            force_refresh=force_refresh,
        )
        if not bars.empty:
            return bars, {
                "source": "dukascopy_ticks",
                "local_tick_coverage_start": str(local_coverage_start) if local_coverage_start is not None else None,
                "local_tick_coverage_end": str(local_coverage_end) if local_coverage_end is not None else None,
            }
        if source == "dukascopy":
            raise RuntimeError(f"Dukascopy download returned no data for {pair}.")

    if source in {"auto", "mt5"}:
        bars = _load_mt5_h1(pair, start_floor, end_floor)
        if not bars.empty:
            return bars, {"source": "mt5_h1"}
        if source == "mt5":
            raise RuntimeError(f"MT5 returned no data for {pair}.")

    if source in {"auto", "yfinance"}:
        bars = _load_yfinance_h1(pair, start_floor, end_floor)
        if not bars.empty:
            return bars, {"source": "yfinance_h1"}
        if source == "yfinance":
            raise RuntimeError(f"yfinance returned no data for {pair}.")

    raise RuntimeError(f"Unable to source H1 data for {pair}.")


def _write_pair_csv(pair: str, bars: pd.DataFrame) -> Path:
    out_path = DATA_DIR / f"{pair}_h1.csv"
    export = bars.reset_index()
    export["Gmt time"] = pd.to_datetime(export["Gmt time"], utc=True).dt.strftime("%Y.%m.%d %H:%M:%S")
    export.to_csv(out_path, index=False)
    return out_path


def _write_manifest(
    *,
    output_path: Path,
    manifest_path: Path,
    rows_by_symbol: dict[str, int],
    range_by_symbol: dict[str, dict[str, str | None]],
    source_by_symbol: dict[str, str],
    coverage_by_symbol: dict[str, dict[str, Any]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    payload = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "window_start_utc": start.isoformat(),
        "window_end_utc": end.isoformat(),
        "output_path": str(output_path),
        "rows_by_symbol": rows_by_symbol,
        "range_by_symbol": range_by_symbol,
        "source_by_symbol": source_by_symbol,
        "coverage_by_symbol": coverage_by_symbol,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _expected_h1_rows(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return max(int((end - start).total_seconds() // 3600), 0)


def _coverage_summary(
    bars: pd.DataFrame,
    *,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
    min_years: float,
    max_gap_hours: int,
) -> dict[str, Any]:
    expected_rows = _expected_h1_rows(requested_start, requested_end)
    if bars.empty:
        return {
            "rows": 0,
            "expected_rows": expected_rows,
            "coverage_ratio": 0.0,
            "coverage_years": 0.0,
            "largest_gap_hours": None,
            "gap_count_over_threshold": 0,
            "meets_min_years": False,
            "meets_gap_threshold": False,
        }

    index = pd.DatetimeIndex(bars.index).sort_values()
    gap_hours = index.to_series().diff().dt.total_seconds().div(3600.0).fillna(1.0)
    largest_gap_hours = float(gap_hours.max()) if len(gap_hours) else 1.0
    gap_count_over_threshold = int((gap_hours > float(max_gap_hours)).sum())
    actual_start = pd.Timestamp(index.min())
    actual_end = pd.Timestamp(index.max())
    coverage_years = max((actual_end - actual_start).total_seconds() / (365.25 * 24 * 3600.0), 0.0)
    coverage_ratio = float(len(index) / expected_rows) if expected_rows > 0 else 0.0
    return {
        "rows": int(len(index)),
        "expected_rows": int(expected_rows),
        "coverage_ratio": coverage_ratio,
        "coverage_years": coverage_years,
        "largest_gap_hours": largest_gap_hours,
        "gap_count_over_threshold": gap_count_over_threshold,
        "meets_min_years": bool(coverage_years >= float(min_years)),
        "meets_gap_threshold": bool(gap_count_over_threshold == 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build 5-10 years of closed H1 Forex candles from tick caches or live sources."
    )
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    parser.add_argument("--years", type=int, default=7, help="Historical window length in years.")
    parser.add_argument("--start", type=str, default=None, help="Optional UTC window start timestamp.")
    parser.add_argument("--end", type=str, default=None, help="Optional UTC window end timestamp.")
    parser.add_argument(
        "--source",
        choices=["auto", "local", "dukascopy", "mt5", "yfinance"],
        default="auto",
        help="Source preference order. auto prefers local tick caches, then Dukascopy, then MT5, then yfinance.",
    )
    parser.add_argument("--max-workers", type=int, default=20, help="Concurrency for Dukascopy downloads.")
    parser.add_argument(
        "--ticks-per-bar",
        type=int,
        default=DEFAULT_TICKS_PER_BAR,
        help="Cache bar size passed through to the existing Dukascopy helper.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cached Dukascopy hours when downloading.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Combined H1 dataset path.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(DEFAULT_MANIFEST),
        help="JSON manifest path.",
    )
    parser.add_argument("--min-years", type=float, default=5.0, help="Minimum required covered history per pair.")
    parser.add_argument("--max-gap-hours", type=int, default=DEFAULT_MAX_GAP_HOURS, help="Largest tolerated hole in the H1 series.")
    parser.add_argument(
        "--strict-coverage",
        action="store_true",
        help="Fail the build if a pair does not meet min-years or has large coverage gaps.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    end_timestamp = _to_utc_timestamp(args.end, fallback=pd.Timestamp.now(tz="UTC"))
    start_timestamp = _to_utc_timestamp(
        args.start,
        fallback=end_timestamp - pd.Timedelta(days=int(args.years * 365.25)),
    )
    start_timestamp, end_timestamp = _ensure_closed_window(start_timestamp, end_timestamp)

    print("-" * 72)
    print("H1 Forex Dataset Builder")
    print("-" * 72)
    print(f"Source      : {args.source}")
    print(f"Window      : {start_timestamp.isoformat()} -> {end_timestamp.isoformat()}")
    print(f"Pairs       : {args.pairs}")
    print(f"Output      : {output_path}")
    print(f"Manifest    : {manifest_path}")

    all_frames: list[pd.DataFrame] = []
    rows_by_symbol: dict[str, int] = {}
    range_by_symbol: dict[str, dict[str, str | None]] = {}
    source_by_symbol: dict[str, str] = {}
    coverage_by_symbol: dict[str, dict[str, Any]] = {}
    coverage_failures: list[str] = []

    for pair in args.pairs:
        print(f"\n[{pair}] collecting...")
        bars, metadata = _build_for_pair(
            pair,
            source=args.source,
            start=start_timestamp,
            end=end_timestamp,
            max_workers=max(1, int(args.max_workers)),
            ticks_per_bar=int(args.ticks_per_bar),
            force_refresh=bool(args.force_refresh),
        )
        if bars.empty:
            print(f"  {pair}: no closed H1 bars found.")
            continue

        bars = bars.sort_index()
        rows_by_symbol[pair] = int(len(bars))
        source_by_symbol[pair] = str(metadata.get("source", args.source))
        range_by_symbol[pair] = {
            "start_utc": str(bars.index.min()),
            "end_utc": str(bars.index.max()),
            "coverage_years": f"{(bars.index.max() - bars.index.min()).days / 365.25:.2f}",
        }
        coverage = _coverage_summary(
            bars,
            requested_start=start_timestamp,
            requested_end=end_timestamp,
            min_years=float(args.min_years),
            max_gap_hours=int(args.max_gap_hours),
        )
        coverage_by_symbol[pair] = coverage
        pair_output = _write_pair_csv(pair, bars)
        print(
            f"  bars={len(bars):,}  source={source_by_symbol[pair]}  "
            f"range={range_by_symbol[pair]['start_utc']} -> {range_by_symbol[pair]['end_utc']}"
        )
        print(
            f"  coverage={coverage['coverage_years']:.2f}y  ratio={coverage['coverage_ratio']:.3f}  "
            f"largest_gap_h={coverage['largest_gap_hours']}"
        )
        if not coverage["meets_min_years"]:
            coverage_failures.append(
                f"{pair}: covered history {coverage['coverage_years']:.2f}y < required {float(args.min_years):.2f}y"
            )
        if not coverage["meets_gap_threshold"]:
            coverage_failures.append(
                f"{pair}: found {coverage['gap_count_over_threshold']} gaps > {int(args.max_gap_hours)}h"
            )
        print(f"  Saved -> {pair_output}")
        export = bars.reset_index()
        export["Symbol"] = pair
        all_frames.append(export)

    if not all_frames:
        raise RuntimeError("No H1 data collected for any symbol.")

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["Symbol", "Gmt time"]).reset_index(drop=True)
    combined = combined.dropna(subset=["Gmt time"])
    combined["Gmt time"] = pd.to_datetime(combined["Gmt time"], utc=True, errors="coerce").dt.strftime("%Y.%m.%d %H:%M:%S")
    preferred_columns = ["Gmt time", "Open", "High", "Low", "Close", "Volume", "avg_spread", "tick_count", "time_delta_s", "Symbol", "source"]
    combined = combined[[column for column in preferred_columns if column in combined.columns]]
    combined.to_csv(output_path, index=False)

    _write_manifest(
        output_path=output_path,
        manifest_path=manifest_path,
        rows_by_symbol=rows_by_symbol,
        range_by_symbol=range_by_symbol,
        source_by_symbol=source_by_symbol,
        coverage_by_symbol=coverage_by_symbol,
        start=start_timestamp,
        end=end_timestamp,
    )

    if coverage_failures:
        print("\nCoverage warnings:")
        for failure in coverage_failures:
            print(f"  - {failure}")
        if args.strict_coverage:
            raise RuntimeError("Strict coverage check failed.")

    print("\nOK")
    print(f"Combined bars : {len(combined):,}")
    print(f"Combined file  : {output_path}")
    print(f"Manifest file  : {manifest_path}")


if __name__ == "__main__":
    main()
