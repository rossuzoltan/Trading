from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluate_oos
from runtime_common import build_replay_evaluation_metrics
from validation_metrics import save_json_report


def _load_holdout_feature_frame(symbol: str) -> pd.DataFrame:
    holdout = evaluate_oos.load_replay_context(symbol).holdout_feature_frame.copy()
    if holdout.empty:
        raise RuntimeError("Holdout feature frame is empty.")
    return holdout


def _bucket_holding_period(value: int) -> str:
    if value <= 2:
        return "01_02_bars"
    if value <= 5:
        return "03_05_bars"
    if value <= 10:
        return "06_10_bars"
    return "11plus_bars"


def _group_trade_stats(rows: list[dict[str, Any]], key_fn) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(key_fn(row)), []).append(row)
    summary: dict[str, Any] = {}
    for key, items in grouped.items():
        summary[key] = {
            "trade_count": int(len(items)),
            "gross_pnl_usd": float(sum(float(item.get("gross_pnl_usd", 0.0)) for item in items)),
            "net_pnl_usd": float(sum(float(item.get("net_pnl_usd", 0.0)) for item in items)),
            "transaction_cost_usd": float(sum(float(item.get("transaction_cost_usd", 0.0)) for item in items)),
            "avg_holding_bars": float(np.mean([float(item.get("holding_bars", 0.0)) for item in items])) if items else 0.0,
        }
    return summary


def _top_loss_drivers(rows: list[dict[str, Any]], metrics: dict[str, Any]) -> list[dict[str, Any]]:
    long_net = float(sum(float(row.get("net_pnl_usd", 0.0)) for row in rows if int(row.get("direction", 0) or 0) > 0))
    short_net = float(sum(float(row.get("net_pnl_usd", 0.0)) for row in rows if int(row.get("direction", 0) or 0) < 0))
    forced_net = float(sum(float(row.get("net_pnl_usd", 0.0)) for row in rows if bool(row.get("forced_close", False))))
    quick_trade_net = float(
        sum(float(row.get("net_pnl_usd", 0.0)) for row in rows if int(row.get("holding_bars", 0) or 0) <= 2)
    )
    candidates = [
        {
            "driver": "transaction_cost_drag",
            "magnitude_usd": float(metrics.get("total_transaction_cost_usd", 0.0) or 0.0),
            "detail": "Total commissions, spread, and slippage paid on closed trades.",
        },
        {
            "driver": "weak_gross_signal",
            "magnitude_usd": abs(float(min(float(metrics.get("gross_pnl_usd", 0.0) or 0.0), 0.0))),
            "detail": "Gross PnL before costs is already negative, so the strategy lacks raw edge.",
        },
        {
            "driver": "forced_close_losses",
            "magnitude_usd": abs(float(min(forced_net, 0.0))),
            "detail": "Trades closed by end-of-path flattening or other forced exits lost money.",
        },
        {
            "driver": "long_side_losses",
            "magnitude_usd": abs(float(min(long_net, 0.0))),
            "detail": "Net losses attributable to long trades.",
        },
        {
            "driver": "short_side_losses",
            "magnitude_usd": abs(float(min(short_net, 0.0))),
            "detail": "Net losses attributable to short trades.",
        },
        {
            "driver": "ultra_short_holds",
            "magnitude_usd": abs(float(min(quick_trade_net, 0.0))),
            "detail": "Trades closed within two bars are losing after costs.",
        },
    ]
    candidates = [item for item in candidates if float(item["magnitude_usd"]) > 0.0]
    candidates.sort(key=lambda item: float(item["magnitude_usd"]), reverse=True)
    return candidates[:3]


def build_replay_diagnostics(symbol: str) -> dict[str, Any]:
    replay_context = evaluate_oos.load_replay_context(symbol)
    equity_curve, timestamps, trade_log, execution_log, diagnostics = evaluate_oos.run_replay(
        replay_context=replay_context
    )
    metrics = build_replay_evaluation_metrics(
        equity_curve=equity_curve,
        timestamps=timestamps,
        trade_log=trade_log,
        execution_diagnostics=diagnostics,
        execution_log_count=len(execution_log),
        initial_equity=1_000.0,
    )
    holdout_features = _load_holdout_feature_frame(symbol.upper())
    close_events = [event for event in execution_log if str(event.get("side", "")).lower() == "close"]
    vol_quantiles = holdout_features["vol_norm_atr"].quantile([0.33, 0.66]).to_dict()
    rows: list[dict[str, Any]] = []
    for idx, trade in enumerate(trade_log):
        row = dict(trade)
        close_event = close_events[idx] if idx < len(close_events) else {}
        close_time_msc = close_event.get("time_msc")
        close_timestamp = pd.to_datetime(close_time_msc, unit="ms", utc=True) if close_time_msc is not None else None
        regime = "unknown"
        if close_timestamp is not None:
            location = holdout_features.index.get_indexer([close_timestamp], method="pad")
            if int(location[0]) >= 0:
                vol_value = float(holdout_features.iloc[int(location[0])]["vol_norm_atr"])
                if vol_value <= float(vol_quantiles.get(0.33, vol_value)):
                    regime = "low_vol"
                elif vol_value <= float(vol_quantiles.get(0.66, vol_value)):
                    regime = "mid_vol"
                else:
                    regime = "high_vol"
        row["close_timestamp_utc"] = close_timestamp.isoformat() if close_timestamp is not None else None
        row["regime"] = regime
        row["direction_label"] = "long" if int(row.get("direction", 0) or 0) > 0 else "short"
        row["holding_bucket"] = _bucket_holding_period(int(row.get("holding_bars", 0) or 0))
        row["transition_bucket"] = (
            f"{row['direction_label']}_{'forced' if bool(row.get('forced_close', False)) else 'regular'}_{str(row.get('reason', 'unknown')).lower()}"
        )
        rows.append(row)

    summary = {
        "gross_alpha_before_costs_usd": float(metrics.get("gross_pnl_usd", 0.0) or 0.0),
        "net_pnl_usd": float(metrics.get("net_pnl_usd", 0.0) or 0.0),
        "total_transaction_cost_usd": float(metrics.get("total_transaction_cost_usd", 0.0) or 0.0),
        "commission_usd": float(metrics.get("total_commission_usd", 0.0) or 0.0),
        "spread_slippage_cost_usd": float(metrics.get("total_spread_slippage_cost_usd", 0.0) or 0.0),
        "cost_per_trade_usd": float(metrics.get("total_transaction_cost_usd", 0.0) or 0.0) / max(int(metrics.get("trade_count", 0) or 0), 1),
        "cost_per_1000_steps_usd": (1000.0 * float(metrics.get("total_transaction_cost_usd", 0.0) or 0.0))
        / max(int(metrics.get("steps", 0) or 0), 1),
        "gross_minus_spread_slippage_usd": float(metrics.get("gross_pnl_usd", 0.0) or 0.0)
        - float(metrics.get("total_spread_slippage_cost_usd", 0.0) or 0.0),
        "gross_minus_commission_usd": float(metrics.get("gross_pnl_usd", 0.0) or 0.0)
        - float(metrics.get("total_commission_usd", 0.0) or 0.0),
    }
    payload = {
        "symbol": symbol.upper(),
        "metrics": metrics,
        "summary": summary,
        "hold_duration_buckets": _group_trade_stats(rows, lambda row: row["holding_bucket"]),
        "transition_buckets": _group_trade_stats(rows, lambda row: row["transition_bucket"]),
        "forced_close_contribution": _group_trade_stats(rows, lambda row: "forced_close" if bool(row.get("forced_close", False)) else "regular_close"),
        "long_short_contribution": _group_trade_stats(rows, lambda row: row["direction_label"]),
        "regime_performance": _group_trade_stats(rows, lambda row: row["regime"]),
        "top_loss_drivers": _top_loss_drivers(rows, metrics),
        "trade_rows": rows,
    }
    return payload


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = dict(payload.get("summary", {}) or {})
    lines = [
        f"# Replay Diagnostics - {payload.get('symbol')}",
        "",
        f"Gross alpha before costs: ${float(summary.get('gross_alpha_before_costs_usd', 0.0)):.2f}",
        f"Net PnL after costs: ${float(summary.get('net_pnl_usd', 0.0)):.2f}",
        f"Total transaction cost: ${float(summary.get('total_transaction_cost_usd', 0.0)):.2f}",
        f"Cost per trade: ${float(summary.get('cost_per_trade_usd', 0.0)):.2f}",
        f"Cost per 1000 steps: ${float(summary.get('cost_per_1000_steps_usd', 0.0)):.2f}",
        "",
        "## Top Loss Drivers",
    ]
    for item in list(payload.get("top_loss_drivers", []) or []):
        lines.append(f"- {item['driver']}: ${float(item['magnitude_usd']):.2f} - {item['detail']}")
    return "\n".join(lines + [""])


def main() -> int:
    parser = argparse.ArgumentParser(description="Produce replay diagnostics for the current symbol OOS evaluation.")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--json-path", default=None)
    parser.add_argument("--markdown-path", default=None)
    args = parser.parse_args()
    payload = build_replay_diagnostics(str(args.symbol).upper())
    json_path = Path(args.json_path) if args.json_path else Path("models") / f"replay_diagnostics_{str(args.symbol).lower()}.json"
    md_path = Path(args.markdown_path) if args.markdown_path else Path("models") / f"replay_diagnostics_{str(args.symbol).lower()}.md"
    save_json_report(payload, json_path)
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({"json_path": str(json_path), "markdown_path": str(md_path), "top_loss_drivers": payload["top_loss_drivers"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
