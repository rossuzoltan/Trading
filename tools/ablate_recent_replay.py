from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from evaluate_oos import (
    _evaluate_policy,
    _load_promoted_manifest_context,
    _rule_action_provider,
)
from feature_engine import FeatureEngine, WARMUP_BARS
from mt5_historical_replay import _session_bucket
from selector_manifest import load_selector_manifest
from strategies.rule_logic import compute_rule_direction


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _load_replay_bars(bars_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with bars_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _recent_replay_metrics(
    *,
    bars: list[dict[str, Any]],
    rule_family: str,
    rule_params: dict[str, Any],
    runtime_constraints: dict[str, Any],
    event_log_path: Path | None = None,
) -> dict[str, Any]:
    feature_engine = FeatureEngine()
    if len(bars) < WARMUP_BARS + 10:
        raise RuntimeError(f"Need at least {WARMUP_BARS + 10} bars; only got {len(bars)}")

    warmup_rows: list[dict[str, Any]] = []
    for index, bar in enumerate(bars[:WARMUP_BARS]):
        previous = bars[index - 1]["timestamp"] if index > 0 else bar["timestamp"]
        delta_s = max(
            (
                pd.Timestamp(bar["timestamp"], tz="UTC").to_pydatetime()
                - pd.Timestamp(previous, tz="UTC").to_pydatetime()
            ).total_seconds(),
            1.0,
        )
        warmup_rows.append(
            {
                "Open": bar["open"],
                "High": bar["high"],
                "Low": bar["low"],
                "Close": bar["close"],
                "Volume": float(bar["tick_count"]),
                "avg_spread": float(bar["avg_spread"]),
                "time_delta_s": float(delta_s),
            }
        )
    warmup_df = pd.DataFrame(warmup_rows)
    warmup_df.index = pd.DatetimeIndex([pd.Timestamp(bar["timestamp"]) for bar in bars[:WARMUP_BARS]])
    warmup_df.index.name = "Gmt time"
    feature_engine.warm_up(warmup_df)

    allowed_sessions = {
        str(item).strip()
        for item in list(runtime_constraints.get("allowed_sessions", []) or [])
        if str(item).strip()
    }
    spread_limit = float(runtime_constraints.get("spread_sanity_max_pips", 999.0) or 999.0)
    rollover_hours = {int(item) for item in list(runtime_constraints.get("rollover_block_utc_hours", []) or [])}

    records: list[dict[str, Any]] = []
    position_direction = 0
    previous_ts: pd.Timestamp | None = None
    if event_log_path is not None:
        event_log_path.parent.mkdir(parents=True, exist_ok=True)
        event_handle = event_log_path.open("w", encoding="utf-8")
    else:
        event_handle = None

    try:
        for bar in bars[WARMUP_BARS:]:
            current_ts = pd.Timestamp(bar["timestamp"])
            if previous_ts is None:
                time_delta_s = 300.0
            else:
                time_delta_s = max((current_ts - previous_ts).total_seconds(), 1.0)
            series = pd.Series(
                {
                    "Open": bar["open"],
                    "High": bar["high"],
                    "Low": bar["low"],
                    "Close": bar["close"],
                    "Volume": float(bar["tick_count"]),
                    "avg_spread": float(bar["avg_spread"]),
                    "time_delta_s": float(time_delta_s),
                },
                name=current_ts,
            )
            feature_engine.push(series)
            if feature_engine._buffer is None or feature_engine._buffer.empty:
                previous_ts = current_ts
                continue

            features = feature_engine._buffer.iloc[-1].to_dict()
            signal = int(compute_rule_direction(rule_family, features, rule_params) or 0)
            hour = int(current_ts.hour)
            session_name = _session_bucket(hour)
            session_ok = bool(session_name in allowed_sessions and hour not in rollover_hours)
            spread_pips = float(bar["avg_spread_pips"])
            spread_ok = bool(spread_pips <= spread_limit)
            price_z = float(features.get("price_z", 0.0) or 0.0)
            spread_z = float(features.get("spread_z", 0.0) or 0.0)
            time_delta_z = float(features.get("time_delta_z", 0.0) or 0.0)
            ma20_slope = float(features.get("ma20_slope", 0.0) or 0.0)
            ma50_slope = float(features.get("ma50_slope", 0.0) or 0.0)
            long_threshold = float(rule_params.get("long_threshold", -rule_params.get("threshold", 1.5)))
            short_threshold = float(rule_params.get("short_threshold", rule_params.get("threshold", 1.5)))
            raw_price_signal = 0
            if price_z <= long_threshold:
                raw_price_signal = 1
            elif price_z >= short_threshold:
                raw_price_signal = -1
            guard_failures = {
                "spread": bool(spread_z > float(rule_params.get("max_spread_z", 999.0))),
                "time_delta": bool(abs(time_delta_z) > float(rule_params.get("max_time_delta_z", 999.0))),
                "ma20": bool(abs(ma20_slope) > float(rule_params.get("max_abs_ma20_slope", 999.0))),
                "ma50": bool(abs(ma50_slope) > float(rule_params.get("max_abs_ma50_slope", 999.0))),
            }
            # Match runtime/shadow semantics:
            # - Session/spread gates can block both opens and closes.
            # - A "no signal" (signal==0) can still authorize a close when already in a position.
            allow_execution = bool(session_ok and spread_ok)
            reason = "authorized" if (allow_execution and signal != 0) else "no signal"
            if signal != 0 and not session_ok:
                reason = "session blocked"
            elif signal != 0 and session_ok and not spread_ok:
                reason = "spread blocked"

            prior_position_direction = int(position_direction)
            would_open = bool(allow_execution and signal != 0 and (position_direction == 0 or signal != position_direction))
            would_close = bool(allow_execution and position_direction != 0 and (signal == 0 or signal != position_direction))
            if would_close:
                position_direction = 0
            if would_open and position_direction == 0:
                position_direction = signal
            record = {
                "bar_ts": current_ts.isoformat(),
                "hour_utc": hour,
                "session": session_name,
                "signal": signal,
                "raw_price_signal": int(raw_price_signal),
                "allow_execution": allow_execution,
                "reason": reason,
                "spread_pips": round(spread_pips, 4),
                "price_z": round(price_z, 6),
                "spread_z": round(spread_z, 6),
                "time_delta_z": round(time_delta_z, 6),
                "ma20_slope": round(ma20_slope, 6),
                "ma50_slope": round(ma50_slope, 6),
                "guard_failures": guard_failures,
                "session_ok": session_ok,
                "spread_ok": spread_ok,
                "position_before": prior_position_direction,
                "would_open": would_open,
                "would_close": would_close,
                "active_state": "long" if position_direction > 0 else "short" if position_direction < 0 else "flat",
            }
            if would_close and signal == 0 and allow_execution:
                record["reason"] = "authorized_exit"
            elif would_open and allow_execution:
                record["reason"] = "authorized"
            records.append(record)
            if event_handle is not None:
                event_handle.write(json.dumps(record, default=_json_default) + "\n")
            previous_ts = current_ts
    finally:
        if event_handle is not None:
            event_handle.close()

    frame = pd.DataFrame(records)
    opens = int(frame["would_open"].sum()) if not frame.empty else 0
    closes = int(frame["would_close"].sum()) if not frame.empty else 0
    longs = int(frame[(frame["would_open"]) & (frame["signal"] > 0)].shape[0]) if not frame.empty else 0
    shorts = int(frame[(frame["would_open"]) & (frame["signal"] < 0)].shape[0]) if not frame.empty else 0
    guard_counts = {
        "spread": int(frame["guard_failures"].map(lambda item: bool(item.get("spread", False))).sum()) if not frame.empty else 0,
        "time_delta": int(frame["guard_failures"].map(lambda item: bool(item.get("time_delta", False))).sum()) if not frame.empty else 0,
        "ma20": int(frame["guard_failures"].map(lambda item: bool(item.get("ma20", False))).sum()) if not frame.empty else 0,
        "ma50": int(frame["guard_failures"].map(lambda item: bool(item.get("ma50", False))).sum()) if not frame.empty else 0,
    }
    signal_breakdown = {
        "raw_long": int(frame["raw_price_signal"].eq(1).sum()) if not frame.empty else 0,
        "raw_short": int(frame["raw_price_signal"].eq(-1).sum()) if not frame.empty else 0,
        "guarded_long": int(frame["signal"].eq(1).sum()) if not frame.empty else 0,
        "guarded_short": int(frame["signal"].eq(-1).sum()) if not frame.empty else 0,
    }
    return {
        "bars_processed": int(len(frame)),
        "would_open_count": opens,
        "would_close_count": closes,
        "long_open_count": longs,
        "short_open_count": shorts,
        "signal_count": int(frame["signal"].ne(0).sum()) if not frame.empty else 0,
        "live_trades_per_bar": float(opens / max(len(frame), 1)),
        "long_short_ratio": float(longs / max(shorts, 1)),
        "guard_failure_counts": guard_counts,
        "signal_breakdown": signal_breakdown,
        "opens_by_session": frame.loc[frame["would_open"], "session"].value_counts().to_dict() if not frame.empty else {},
        "signals_by_session": frame.loc[frame["signal"].ne(0), "session"].value_counts().to_dict() if not frame.empty else {},
        "reason_counts": {
            "session": int(frame["reason"].eq("session blocked").sum()) if not frame.empty else 0,
            "spread": int(frame["reason"].eq("spread blocked").sum()) if not frame.empty else 0,
            "no_signal": int(frame["reason"].eq("no signal").sum()) if not frame.empty else 0,
            "authorized": int(frame["reason"].eq("authorized").sum()) if not frame.empty else 0,
        },
        "examples": {
            "authorized_opens": frame.loc[frame["would_open"], ["bar_ts", "session", "signal", "price_z", "spread_z", "ma20_slope", "ma50_slope"]].head(10).to_dict("records") if not frame.empty else [],
            "blocked_signals": frame.loc[(frame["raw_price_signal"] != 0) & (~frame["would_open"]), ["bar_ts", "session", "raw_price_signal", "signal", "reason", "guard_failures", "price_z", "spread_z", "ma20_slope", "ma50_slope"]].head(10).to_dict("records") if not frame.empty else [],
        },
    }


def _variant_catalog(base_params: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = dict(base_params)
    base_threshold = float(base_params.get("threshold", 1.5))
    base_long = float(base_params.get("long_threshold", -base_threshold))
    base_short = float(base_params.get("short_threshold", base_threshold))
    return [
        {
            "name": "rc1_baseline",
            "rule_family": "mean_reversion",
            "rule_params": baseline,
        },
        {
            # Make shorts harder to trigger (reduce short concentration) while preserving existing long trigger.
            "name": "asym_short_1.80",
            "rule_family": "mean_reversion",
            "rule_params": {**baseline, "long_threshold": base_long, "short_threshold": 1.80},
        },
        {
            # Make longs easier to trigger (increase long participation) while preserving existing short trigger.
            "name": "asym_long_-1.20",
            "rule_family": "mean_reversion",
            "rule_params": {**baseline, "long_threshold": -1.20, "short_threshold": base_short},
        },
        {
            # Combined asymmetry: easier longs + harder shorts.
            "name": "asym_long_-1.20_short_1.80",
            "rule_family": "mean_reversion",
            "rule_params": {**baseline, "long_threshold": -1.20, "short_threshold": 1.80},
        },
        {
            "name": "relax_ma50_0.10",
            "rule_family": "mean_reversion",
            "rule_params": {**baseline, "max_abs_ma50_slope": 0.10},
        },
        {
            "name": "relax_ma50_0.15",
            "rule_family": "mean_reversion",
            "rule_params": {**baseline, "max_abs_ma50_slope": 0.15},
        },
        {
            "name": "relax_ma20_0.20_ma50_0.10",
            "rule_family": "mean_reversion",
            "rule_params": {**baseline, "max_abs_ma20_slope": 0.20, "max_abs_ma50_slope": 0.10},
        },
        {
            "name": "relax_ma20_0.25_ma50_0.15",
            "rule_family": "mean_reversion",
            "rule_params": {**baseline, "max_abs_ma20_slope": 0.25, "max_abs_ma50_slope": 0.15},
        },
        {
            "name": "relax_spread_0.75_ma20_0.20_ma50_0.10",
            "rule_family": "mean_reversion",
            "rule_params": {
                **baseline,
                "max_spread_z": 0.75,
                "max_abs_ma20_slope": 0.20,
                "max_abs_ma50_slope": 0.10,
            },
        },
        {
            "name": "price_only_no_guards",
            "rule_family": "price_mean_reversion",
            "rule_params": {
                "threshold": float(base_params.get("threshold", 1.5)),
                "long_threshold": float(base_params.get("long_threshold", -base_params.get("threshold", 1.5))),
                "short_threshold": float(base_params.get("short_threshold", base_params.get("threshold", 1.5))),
            },
        },
    ]


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Recent Replay Ablation - {payload['symbol']}",
        "",
        f"* Manifest: `{payload['manifest_path']}`",
        f"* Bars source: `{payload['bars_path']}`",
        "",
        "| Variant | Holdout Net | Holdout PF | Holdout Trades | Replay Opens | Replay Long | Replay Short | Replay L/S |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["variants"]:
        holdout = item["holdout_metrics"]
        recent = item["recent_replay_metrics"]
        lines.append(
            f"| {item['name']} | {holdout['net_pnl_usd']:.2f} | {holdout['profit_factor']:.3f} | "
            f"{int(holdout['trade_count'])} | {int(recent['would_open_count'])} | "
            f"{int(recent['long_open_count'])} | {int(recent['short_open_count'])} | "
            f"{recent['long_short_ratio']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Detail",
        ]
    )
    for item in payload["variants"]:
        recent = item["recent_replay_metrics"]
        holdout = item["holdout_metrics"]
        lines.extend(
            [
                f"### {item['name']}",
                f"* Holdout net: `{holdout['net_pnl_usd']:.2f}` | PF: `{holdout['profit_factor']:.3f}` | Trades: `{int(holdout['trade_count'])}`",
                f"* Replay opens: `{int(recent['would_open_count'])}` | long `{int(recent['long_open_count'])}` | short `{int(recent['short_open_count'])}`",
                f"* Guard failures: `spread={recent['guard_failure_counts']['spread']}`, `time_delta={recent['guard_failure_counts']['time_delta']}`, `ma20={recent['guard_failure_counts']['ma20']}`, `ma50={recent['guard_failure_counts']['ma50']}`",
                f"* Signal breakdown: `raw_long={recent['signal_breakdown']['raw_long']}`, `raw_short={recent['signal_breakdown']['raw_short']}`, `guarded_long={recent['signal_breakdown']['guarded_long']}`, `guarded_short={recent['signal_breakdown']['guarded_short']}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare RC1 guard variants on holdout OOS and recent MT5 replay bars.")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--bars-path", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path).resolve()
    manifest = load_selector_manifest(
        manifest_path,
        verify_manifest_hash=True,
        strict_manifest_hash=True,
        require_component_hashes=True,
    )
    bars_path = Path(args.bars_path).resolve() if args.bars_path else manifest_path.parent / "mt5_historical_replay_report.bars.jsonl"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else ROOT / "artifacts" / "research" / "recent_replay_ablation" / manifest.strategy_symbol / manifest.manifest_hash
    output_dir.mkdir(parents=True, exist_ok=True)

    bars = _load_replay_bars(bars_path)
    os.environ["EVAL_MANIFEST_PATH"] = str(manifest_path)
    context = _load_promoted_manifest_context(manifest.strategy_symbol)
    if context is None:
        raise RuntimeError(f"Failed to load promoted manifest context from {manifest_path}")

    variants = []
    for variant in _variant_catalog(dict(manifest.rule_params or {})):
        variant_context = replace(
            context,
            engine_type="RULE",
            rule_family=str(variant["rule_family"]),
            rule_params=dict(variant["rule_params"]),
        )
        provider = partial(
            _rule_action_provider,
            rule_family=str(variant["rule_family"]),
            rule_params=dict(variant["rule_params"]),
        )
        holdout_payload = _evaluate_policy(
            replay_context=variant_context,
            action_index_provider=provider,
            disable_alpha_gate=True,
        )
        recent_metrics = _recent_replay_metrics(
            bars=bars,
            rule_family=str(variant["rule_family"]),
            rule_params=dict(variant["rule_params"]),
            runtime_constraints=dict(manifest.runtime_constraints or {}),
            event_log_path=output_dir / f"{variant['name']}.recent_replay_events.jsonl",
        )
        variants.append(
            {
                "name": variant["name"],
                "rule_family": variant["rule_family"],
                "rule_params": dict(variant["rule_params"]),
                "holdout_metrics": dict(holdout_payload["metrics"]),
                "recent_replay_metrics": recent_metrics,
            }
        )

    payload = {
        "symbol": manifest.strategy_symbol,
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest.manifest_hash,
        "bars_path": str(bars_path),
        "variants": variants,
    }
    (output_dir / "recent_replay_ablation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output_dir / "recent_replay_ablation.md").write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "variant_count": len(variants)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
