from __future__ import annotations

from typing import Any

import pandas as pd


def summarize_symbol_bar_spec(
    frame: pd.DataFrame,
    *,
    expected_ticks_per_bar: int,
    symbol: str,
    volume_column: str = "Volume",
    timestamp_column: str = "Gmt time",
) -> dict[str, Any]:
    if volume_column not in frame.columns:
        raise RuntimeError(
            f"Dataset rows for {symbol} are missing required volume column {volume_column!r}."
        )

    working = frame.copy()
    if timestamp_column in working.columns:
        timestamps = pd.to_datetime(working[timestamp_column], utc=True, errors="coerce")
        if timestamps.notna().all():
            working = working.assign(_timestamp=timestamps).sort_values("_timestamp", kind="stable")
    working = working.reset_index(drop=True)

    volumes = pd.to_numeric(working[volume_column], errors="coerce")
    if volumes.isna().any():
        raise RuntimeError(f"Dataset rows for {symbol} contain non-numeric {volume_column} values.")

    expected = int(expected_ticks_per_bar)
    rounded = volumes.round().astype(int)
    too_large_mask = rounded > expected
    nonpositive_mask = rounded <= 0
    partial_mask = ~(too_large_mask | nonpositive_mask) & (rounded != expected)

    partial_positions = [int(i) for i in partial_mask[partial_mask].index.tolist()]
    distinct_volume_values = sorted({int(v) for v in rounded.unique().tolist()})

    return {
        "symbol": str(symbol).upper(),
        "expected_ticks_per_bar": expected,
        "total_rows": int(len(working)),
        "exact_match_rows": int((rounded == expected).sum()),
        "partial_rows": int(partial_mask.sum()),
        "too_large_rows": int(too_large_mask.sum()),
        "nonpositive_rows": int(nonpositive_mask.sum()),
        "distinct_volume_values": distinct_volume_values[:10],
        "last_volume_value": int(rounded.iloc[-1]) if len(rounded) else None,
        "partial_positions": partial_positions[:10],
        "partial_only_at_tail": bool(
            partial_positions and partial_positions == [int(len(working) - 1)]
        ),
    }


def validate_symbol_bar_spec(
    frame: pd.DataFrame,
    *,
    expected_ticks_per_bar: int,
    symbol: str,
    volume_column: str = "Volume",
    timestamp_column: str = "Gmt time",
) -> dict[str, Any]:
    summary = summarize_symbol_bar_spec(
        frame,
        expected_ticks_per_bar=expected_ticks_per_bar,
        symbol=symbol,
        volume_column=volume_column,
        timestamp_column=timestamp_column,
    )

    if summary["too_large_rows"] or summary["nonpositive_rows"]:
        raise RuntimeError(
            f"Dataset rows for {summary['symbol']} contain invalid volume-bar sizes for "
            f"expected_ticks_per_bar={summary['expected_ticks_per_bar']}. "
            f"distinct_volume_values={summary['distinct_volume_values']}"
        )

    partial_rows = int(summary["partial_rows"])
    if partial_rows == 0:
        return summary

    if not summary["partial_only_at_tail"]:
        raise RuntimeError(
            f"Dataset rows for {summary['symbol']} mix bar sizes for expected_ticks_per_bar="
            f"{summary['expected_ticks_per_bar']}. partial_rows={partial_rows}, "
            f"partial_positions={summary['partial_positions']}, "
            f"distinct_volume_values={summary['distinct_volume_values']}"
        )

    return summary
