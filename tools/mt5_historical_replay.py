"""
mt5_historical_replay.py - Phase P7 pre-shadow accelerator

Fetches recent real MT5 ticks, builds volume bars on the fly, runs the
rule-first selector through them, and writes both human-readable and
machine-readable evidence.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

log = logging.getLogger("mt5_historical_replay")


def _session_bucket(hour_utc: int) -> str:
    if 0 <= hour_utc < 7:
        return "Asia"
    if 7 <= hour_utc < 12:
        return "London"
    if 12 <= hour_utc < 17:
        return "London/NY"
    if 17 <= hour_utc < 21:
        return "NY"
    return "Rollover"


def _verdict(ratio: float, ok: float = 0.10, warn: float = 0.30, crit: float = 0.60) -> str:
    delta = abs(ratio - 1.0)
    if delta <= ok:
        return "OK"
    if delta <= warn:
        return "WATCH"
    if delta <= crit:
        return "DRIFT_WARNING"
    return "DRIFT_CRITICAL"

def _direction_verdict(delta_pp: float, *, ok_pp: float = 5.0, watch_pp: float = 15.0, warn_pp: float = 20.0) -> str:
    """
    Direction drift verdict based on deviation from the expected long-share.

    Do not assume long/short balance should be ~1.0; compare to an exact-runtime
    RC reference distribution (from the RC1 scoreboard) and measure absolute
    deviation in percentage points.
    """
    delta_pp = abs(float(delta_pp))
    if delta_pp <= ok_pp:
        return "OK"
    if delta_pp <= watch_pp:
        return "WATCH"
    if delta_pp <= warn_pp:
        return "DRIFT_WARNING"
    return "DRIFT_CRITICAL"


def _worst_verdict(verdicts: list[str]) -> str:
    order = {
        "OK": 0,
        "WATCH": 1,
        "DRIFT_WARNING": 2,
        "DRIFT_CRITICAL": 3,
        "NO_REFERENCE": 4,
        "NO_DATA": 5,
    }
    worst = "OK"
    for verdict in verdicts:
        if order.get(verdict, 0) > order.get(worst, 0):
            worst = verdict
    return worst


def fetch_ticks_from_mt5(mt5, symbol: str, start_utc: datetime, end_utc: datetime) -> list[dict[str, Any]]:
    log.info("Fetching ticks %s %s -> %s", symbol, start_utc.date(), end_utc.date())
    all_ticks: list[dict[str, Any]] = []
    cursor = start_utc
    batch = 100_000

    while cursor < end_utc:
        raw = mt5.copy_ticks_from(symbol, cursor, batch, mt5.COPY_TICKS_ALL)
        if raw is None or len(raw) == 0:
            break
        for row in raw:
            time_msc = int(row["time_msc"] if "time_msc" in row.dtype.names else row["time"] * 1000)
            if time_msc / 1000 > end_utc.timestamp():
                break
            all_ticks.append(
                {
                    "time_msc": time_msc,
                    "bid": float(row["bid"]),
                    "ask": float(row["ask"]),
                }
            )
        last_ms = all_ticks[-1]["time_msc"]
        cursor = datetime.fromtimestamp(last_ms / 1000, tz=timezone.utc) + timedelta(milliseconds=1)
        log.info("Fetched %d ticks so far; cursor=%s", len(all_ticks), cursor.strftime("%Y-%m-%d %H:%M"))
        if len(raw) < batch:
            break

    log.info("Total ticks fetched: %d", len(all_ticks))
    return all_ticks


def build_volume_bars(ticks: list[dict[str, Any]], ticks_per_bar: int) -> list[dict[str, Any]]:
    bars: list[dict[str, Any]] = []
    buffer: list[dict[str, Any]] = []

    for tick in ticks:
        buffer.append(tick)
        if len(buffer) < ticks_per_bar:
            continue
        mid_prices = [(row["bid"] + row["ask"]) / 2.0 for row in buffer]
        spreads = [row["ask"] - row["bid"] for row in buffer]
        bars.append(
            {
                "timestamp": datetime.fromtimestamp(buffer[0]["time_msc"] / 1000, tz=timezone.utc),
                "open": mid_prices[0],
                "high": max(mid_prices),
                "low": min(mid_prices),
                "close": mid_prices[-1],
                "avg_spread": sum(spreads) / len(spreads),
                "avg_spread_pips": (sum(spreads) / len(spreads)) * 10_000,
                "tick_count": len(buffer),
            }
        )
        buffer = []

    log.info("Built %d volume bars from %d ticks", len(bars), len(ticks))
    return bars


def run_rule_on_bars(bars: list[dict[str, Any]], manifest_path: Path) -> list[dict[str, Any]]:
    from feature_engine import FeatureEngine, WARMUP_BARS
    from rule_selector import RuleSelector

    selector = RuleSelector(manifest_path)
    feature_engine = FeatureEngine()
    rule_params = dict(selector.manifest.rule_params or {})

    if len(bars) < WARMUP_BARS + 10:
        raise RuntimeError(f"Need at least {WARMUP_BARS + 10} bars; only got {len(bars)}")

    warmup_rows: list[dict[str, Any]] = []
    prev_ts: datetime | None = None
    for bar in bars[:WARMUP_BARS]:
        current_ts = bar["timestamp"]
        if prev_ts is None:
            time_delta_s = 300.0
        else:
            time_delta_s = max((current_ts - prev_ts).total_seconds(), 1.0)
        warmup_rows.append(
            {
                "Open": bar["open"],
                "High": bar["high"],
                "Low": bar["low"],
                "Close": bar["close"],
                "Volume": float(bar["tick_count"]),
                "avg_spread": bar["avg_spread"],
                "time_delta_s": float(time_delta_s),
            }
        )
        prev_ts = current_ts
    warmup_df = pd.DataFrame(warmup_rows)
    warmup_df.index = pd.DatetimeIndex([bar["timestamp"] for bar in bars[:WARMUP_BARS]])
    warmup_df.index.name = "Gmt time"
    feature_engine.warm_up(warmup_df)

    records: list[dict[str, Any]] = []
    position_direction = 0

    for bar in bars[WARMUP_BARS:]:
        current_ts = bar["timestamp"]
        if prev_ts is None:
            time_delta_s = 300.0
        else:
            time_delta_s = max((current_ts - prev_ts).total_seconds(), 1.0)
        series = pd.Series(
            {
                "Open": bar["open"],
                "High": bar["high"],
                "Low": bar["low"],
                "Close": bar["close"],
                "Volume": float(bar["tick_count"]),
                "avg_spread": bar["avg_spread"],
                "time_delta_s": float(time_delta_s),
            },
            name=current_ts,
        )
        feature_engine.push(series)
        if feature_engine._buffer is None or feature_engine._buffer.empty:
            continue

        features = feature_engine._buffer.iloc[-1].to_dict()
        spread_pips = float(bar["avg_spread_pips"])
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
            "spread": bool(spread_z > float(rule_params.get("max_spread_z", 0.5))),
            "time_delta": bool(abs(time_delta_z) > float(rule_params.get("max_time_delta_z", 2.0))),
            "ma20": bool(abs(ma20_slope) > float(rule_params.get("max_abs_ma20_slope", 0.15))),
            "ma50": bool(abs(ma50_slope) > float(rule_params.get("max_abs_ma50_slope", 0.08))),
        }
        portfolio_state = {
            "current_positions": 1 if position_direction != 0 else 0,
            "current_direction": position_direction,
            "position_direction": position_direction,
            "daily_pnl_usd": 0.0,
        }
        hour = int(current_ts.hour)
        is_session = not (
            current_ts.weekday() == 5
            or (current_ts.weekday() == 6 and hour < 22)
            or (current_ts.weekday() == 4 and hour >= 22)
        )
        decision = selector.decide(
            features=features,
            current_spread_pips=spread_pips,
            portfolio_state=portfolio_state,
            is_session_open=is_session,
            current_hour_utc=hour,
            bar_ts_utc=current_ts,
        )
        context_daily = dict((decision.context or {}).get("daily", {}) or {}) if isinstance(decision.context, dict) else {}
        context_slice = dict((decision.context or {}).get("slice", {}) or {}) if isinstance(decision.context, dict) else {}

        signal = 1 if decision.signal > 0 else -1 if decision.signal < 0 else 0
        would_open = bool(decision.allow_execution and signal != 0 and (position_direction == 0 or signal != position_direction))
        would_close = bool(decision.allow_execution and position_direction != 0 and (signal == 0 or signal != position_direction))
        would_hold = bool(position_direction != 0 and not would_close)
        would_flat = bool(position_direction == 0 and not would_open)

        if would_close:
            position_direction = 0
        if would_open and position_direction == 0:
            position_direction = signal

        active_state = "long" if position_direction == 1 else "short" if position_direction == -1 else "flat"
        records.append(
            {
                "bar_ts": bar["timestamp"].isoformat(),
                "hour_utc": hour,
                "session": _session_bucket(hour),
                "signal": signal,
                "allow_execution": bool(decision.allow_execution),
                "reason": decision.reason or "",
                "spread_pips": round(spread_pips, 4),
                "price_z": round(price_z, 6),
                "spread_z": round(spread_z, 6),
                "time_delta_z": round(time_delta_z, 6),
                "ma20_slope": round(ma20_slope, 6),
                "ma50_slope": round(ma50_slope, 6),
                "raw_price_signal": int(raw_price_signal),
                "guard_failures": guard_failures,
                "would_open": would_open,
                "would_close": would_close,
                "would_hold": would_hold,
                "would_flat": would_flat,
                "active_state": active_state,
                "context_day_type": context_daily.get("day_type"),
                "context_event_risk": context_daily.get("event_risk"),
                "context_in_blackout": context_slice.get("in_blackout"),
                "context_blackout_kind": context_slice.get("blackout_kind"),
                "context_active_event_id": context_slice.get("active_event_id"),
                "context_aggressiveness_mode": context_slice.get("effective_aggressiveness_mode"),
                "context_block_policy": context_slice.get("effective_block_policy"),
                "context_reason_codes": list(context_slice.get("reason_codes", []) or []),
            }
        )
        prev_ts = current_ts

    return records


def _resolve_replay_reference_metrics(
    *,
    manifest_path: Path,
    symbol: str,
) -> tuple[float, int, int, float | None]:
    scoreboard_candidates = [
        manifest_path.parent / "baseline_scoreboard_rc1.json",
    ]
    for candidate in scoreboard_candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            rc_candidate = dict(payload.get("rc_candidate") or {})
            replay_bars = int(rc_candidate.get("replay_bars", 0) or 0)
            replay_trade_count = int(rc_candidate.get("trade_count", 0) or 0)
            long_count = float(rc_candidate.get("long_count", 0) or 0)
            short_count = float(rc_candidate.get("short_count", 0) or 0)
            expected_long_share = None
            total = long_count + short_count
            if total > 0:
                expected_long_share = float(long_count / total)
            if replay_bars > 0:
                return float(replay_trade_count / replay_bars), replay_trade_count, replay_bars, expected_long_share
        except Exception:
            continue

    # Fallback: compute exact replay reference on the current evaluator/runtime path.
    try:
        from functools import partial
        from evaluate_oos import _load_promoted_manifest_context, _selector_action_provider, run_replay
        from rule_selector import RuleSelector

        os.environ["EVAL_MANIFEST_PATH"] = str(manifest_path)
        context = _load_promoted_manifest_context(symbol.upper())
        if context is None:
            return 0.0, 0, 0, None
        selector = RuleSelector(manifest_path)
        provider = partial(_selector_action_provider, selector=selector)
        _, _, trade_log, _, _ = run_replay(
            replay_context=context,
            action_index_provider=provider,
            disable_alpha_gate=False,
        )
        replay_bars = int(len(context.replay_frame))
        replay_trade_count = int(len(trade_log))
        if replay_bars <= 0:
            return 0.0, replay_trade_count, replay_bars, None
        return float(replay_trade_count / replay_bars), replay_trade_count, replay_bars, None
    except Exception:
        return 0.0, 0, 0, None


def build_summary(
    records: list[dict[str, Any]],
    *,
    symbol: str,
    days: int,
    spread_backtest_pips: float,
    replay_trades_per_bar: float,
    replay_trade_count: int,
    replay_bars: int,
    expected_long_share: float | None,
) -> dict[str, Any]:
    if not records:
        return {
            "symbol": symbol.upper(),
            "days": int(days),
            "bars_processed": 0,
            "overall_verdict": "NO_DATA",
            "overall_verdict_label": "NO_DATA",
        }

    frame = pd.DataFrame(records)
    total = len(frame)
    opens = int(frame["would_open"].sum())
    closes = int(frame["would_close"].sum())
    holds = int(frame["would_hold"].sum())
    flats = int(frame["would_flat"].sum())
    longs = int(frame[(frame["would_open"]) & (frame["signal"] > 0)].shape[0])
    shorts = int(frame[(frame["would_open"]) & (frame["signal"] < 0)].shape[0])
    live_trades_per_bar = opens / max(total, 1)
    signal_density_ratio = (
        live_trades_per_bar / max(float(replay_trades_per_bar), 1e-6)
        if replay_trades_per_bar > 0.0
        else 0.0
    )
    live_avg_spread_pips = float(frame["spread_pips"].mean())
    spread_ratio = live_avg_spread_pips / max(spread_backtest_pips, 1e-6)

    live_long_share = None
    total_opens = longs + shorts
    if total_opens > 0:
        live_long_share = float(longs / total_opens)
    directional_delta_pp = None
    if expected_long_share is not None and live_long_share is not None:
        directional_delta_pp = abs(float(live_long_share) - float(expected_long_share)) * 100.0

    if replay_trades_per_bar <= 0.0:
        overall_verdict = "NO_REFERENCE"
    else:
        density_verdict = _verdict(float(signal_density_ratio))
        spread_verdict = _verdict(float(spread_ratio))
        direction_verdict = _direction_verdict(float(directional_delta_pp)) if directional_delta_pp is not None else None
        verdicts = [density_verdict, spread_verdict]
        if direction_verdict is not None:
            verdicts.append(direction_verdict)
        overall_verdict = _worst_verdict(verdicts)
    labels = {
        "OK": "OK - Live behaviour matches backtest expectations",
        "WATCH": "WATCH - Minor deviations, monitor closely",
        "DRIFT_WARNING": "DRIFT_WARNING - Notable deviation from backtest",
        "DRIFT_CRITICAL": "DRIFT_CRITICAL - Significant live vs backtest divergence",
        "NO_REFERENCE": "NO_REFERENCE - Replay reference metrics unavailable",
    }
    return {
        "symbol": symbol.upper(),
        "days": int(days),
        "bars_processed": int(total),
        "window_start_utc": str(frame["bar_ts"].iloc[0]),
        "window_end_utc": str(frame["bar_ts"].iloc[-1]),
        "would_open_count": opens,
        "would_close_count": closes,
        "would_hold_count": holds,
        "would_flat_count": flats,
        "would_open_pct": float(opens * 100.0 / total),
        "would_close_pct": float(closes * 100.0 / total),
        "would_hold_pct": float(holds * 100.0 / total),
        "would_flat_pct": float(flats * 100.0 / total),
        "long_open_count": longs,
        "short_open_count": shorts,
        "live_trades_per_bar": float(live_trades_per_bar),
        "replay_trades_per_bar": float(replay_trades_per_bar),
        "replay_trade_count": int(replay_trade_count),
        "replay_bars": int(replay_bars),
        "signal_density_ratio": float(signal_density_ratio),
        "live_avg_spread_pips": float(live_avg_spread_pips),
        "backtest_spread_pips": float(spread_backtest_pips),
        "spread_ratio": float(spread_ratio),
        "expected_long_share": float(expected_long_share) if expected_long_share is not None else None,
        "live_long_share": float(live_long_share) if live_long_share is not None else None,
        "directional_delta_pp": float(directional_delta_pp) if directional_delta_pp is not None else None,
        "reason_counts": {
            "spread": int(frame[frame["reason"].str.contains("spread", case=False, na=False)].shape[0]),
            "session": int(frame[frame["reason"].str.contains("session", case=False, na=False)].shape[0]),
            "no_signal": int(frame[frame["reason"].str.contains("signal", case=False, na=False)].shape[0]),
        },
        "session_opens": {key: int(value) for key, value in frame[frame["would_open"]].groupby("session").size().to_dict().items()},
        "overall_verdict": overall_verdict,
        "overall_verdict_label": labels[overall_verdict],
    }


def render_report(
    records: list[dict[str, Any]],
    symbol: str,
    days: int,
    spread_backtest_pips: float,
    replay_trades_per_bar: float,
    replay_trade_count: int,
    replay_bars: int,
    expected_long_share: float | None = None,
) -> str:
    summary = build_summary(
        records,
        symbol=symbol,
        days=days,
        spread_backtest_pips=spread_backtest_pips,
        replay_trades_per_bar=replay_trades_per_bar,
        replay_trade_count=replay_trade_count,
        replay_bars=replay_bars,
        expected_long_share=expected_long_share,
    )
    if not summary["bars_processed"]:
        return "# MT5 Historical Replay\nNo bars processed."

    lines = [
        f"# MT5 Historical Replay Report - {symbol}",
        f"**Period**: Last {days} days of real MT5 tick data",
        f"**Bars processed**: {summary['bars_processed']}  |  **Ticks/bar**: built from live tick stream",
        f"**Overall verdict**: {summary['overall_verdict_label']}",
        "",
        "---",
        "",
        "## A. Signal Density",
        "| Metric | Value |",
        "|---|---|",
        f"| Would-Open count | {summary['would_open_count']} |",
        f"| Bars processed | {summary['bars_processed']} |",
        f"| Live trades/bar | {summary['live_trades_per_bar']:.5f} |",
        f"| Replay trades/bar | {summary['replay_trades_per_bar']:.5f} ({summary['replay_trade_count']} / {summary['replay_bars']}) |",
        (
            f"| Ratio | {summary['signal_density_ratio']:.2f}x  {_verdict(summary['signal_density_ratio'])} |"
            if summary["replay_trades_per_bar"] > 0.0
            else "| Ratio | n/a (no replay reference) |"
        ),
        "",
        "## B. Spread Reality Check",
        "| Metric | Value |",
        "|---|---|",
        f"| Live avg spread | {summary['live_avg_spread_pips']:.4f} pips |",
        f"| Backtest assumed spread | {summary['backtest_spread_pips']:.4f} pips |",
        f"| Spread ratio | {summary['spread_ratio']:.2f}x  {_verdict(summary['spread_ratio'])} |",
        "",
        "## C. Direction Balance",
        "| Metric | Value |",
        "|---|---|",
        f"| Long opens | {summary['long_open_count']} |",
        f"| Short opens | {summary['short_open_count']} |",
        (
            f"| Live long share | {float(summary['live_long_share']):.2%} |"
            if summary.get("live_long_share") is not None
            else "| Live long share | n/a |"
        ),
        (
            f"| Expected long share (RC1) | {float(summary['expected_long_share']):.2%} |"
            if summary.get("expected_long_share") is not None
            else "| Expected long share (RC1) | n/a |"
        ),
        (
            f"| Delta | {float(summary['directional_delta_pp']):.2f} pp  {_direction_verdict(float(summary['directional_delta_pp']))} |"
            if summary.get("directional_delta_pp") is not None
            else "| Delta | n/a (no RC1 direction reference) |"
        ),
        "",
        "## D. State Occupancy",
        "| State | Bars | % |",
        "|---|---|---|",
        f"| Would-Open | {summary['would_open_count']} | {summary['would_open_pct']:.1f}% |",
        f"| Would-Close | {summary['would_close_count']} | {summary['would_close_pct']:.1f}% |",
        f"| Holding | {summary['would_hold_count']} | {summary['would_hold_pct']:.1f}% |",
        f"| Flat/No-Trade | {summary['would_flat_count']} | {summary['would_flat_pct']:.1f}% |",
        "",
        "## E. Gate Reject Reasons",
        "| Reason | Count |",
        "|---|---|",
        f"| Spread reject | {summary['reason_counts'].get('spread', 0)} |",
        f"| Session reject | {summary['reason_counts'].get('session', 0)} |",
        f"| No signal | {summary['reason_counts'].get('no_signal', 0)} |",
        "",
        "## F. Session Breakdown (Would-Open by session)",
        "| Session | Opens |",
        "|---|---|",
    ]
    for session in ["Asia", "London", "London/NY", "NY", "Rollover"]:
        lines.append(f"| {session} | {summary['session_opens'].get(session, 0)} |")
    lines.extend(
        [
            "",
            "---",
            f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from real MT5 historical ticks.*",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MT5 Historical Tick Replay pre-shadow accelerator")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--days", type=int, default=30, help="How many calendar days back to fetch")
    parser.add_argument("--manifest", default="", help="Path to RC manifest.json (auto-detected if blank)")
    parser.add_argument("--output", default="", help="Output .md path (auto-detected if blank)")
    parser.add_argument("--spread-backtest-pips", type=float, default=0.5, help="Spread assumed in backtest for comparison")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    symbol = args.symbol.upper()
    manifest_path = Path(args.manifest) if args.manifest else None
    if manifest_path is None:
        candidates = sorted(Path(ROOT / "models" / "rc1").glob(f"*{symbol.lower()}*/manifest.json"))
        if not candidates:
            raise SystemExit(f"No RC1 manifest found for {symbol}. Pass --manifest explicitly.")
        manifest_path = candidates[0]
    log.info("Using manifest: %s", manifest_path)

    out_path = Path(args.output) if args.output else manifest_path.parent / "mt5_historical_replay_report.md"
    json_out_path = out_path.with_suffix(".json")

    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise SystemExit("MetaTrader5 not installed. Cannot fetch live tick data.")

    from live_bridge import _connect_mt5
    from selector_manifest import load_selector_manifest

    _connect_mt5(mt5)
    end_utc = datetime.now(timezone.utc)
    start_utc = end_utc - timedelta(days=args.days)
    ticks = fetch_ticks_from_mt5(mt5, symbol, start_utc, end_utc)
    mt5.shutdown()
    if not ticks:
        raise SystemExit("No ticks returned from MT5. Is the terminal running and the symbol selected?")

    manifest = load_selector_manifest(
        manifest_path,
        verify_manifest_hash=True,
        strict_manifest_hash=True,
        require_component_hashes=True,
    )
    ticks_per_bar = int(manifest.ticks_per_bar or manifest.bar_construction_ticks_per_bar or 5000)
    bars = build_volume_bars(ticks, ticks_per_bar)

    raw_bars_path = out_path.with_suffix(".bars.jsonl")
    with raw_bars_path.open("w", encoding="utf-8") as handle:
        for bar in bars:
            handle.write(json.dumps({key: (value.isoformat() if hasattr(value, "isoformat") else value) for key, value in bar.items()}) + "\n")
    log.info("Raw bars saved -> %s", raw_bars_path)

    log.info("Running RC1 rule selector over %d bars...", len(bars))
    records = run_rule_on_bars(bars, manifest_path)
    audit_path = out_path.with_suffix(".audit.jsonl")
    with audit_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    log.info("Audit trace saved -> %s", audit_path)

    replay_trades_per_bar, replay_trade_count, replay_bars, expected_long_share = _resolve_replay_reference_metrics(
        manifest_path=manifest_path,
        symbol=symbol,
    )
    summary = build_summary(
        records,
        symbol=symbol,
        days=args.days,
        spread_backtest_pips=args.spread_backtest_pips,
        replay_trades_per_bar=replay_trades_per_bar,
        replay_trade_count=replay_trade_count,
        replay_bars=replay_bars,
        expected_long_share=expected_long_share,
    )
    summary.update(
        {
            "manifest_path": str(manifest_path),
            "manifest_hash": manifest.manifest_hash,
            "logic_hash": manifest.logic_hash,
            "evaluator_hash": manifest.evaluator_hash,
            "ticks_per_bar": ticks_per_bar,
            "audit_trace_path": str(audit_path),
            "raw_bars_path": str(raw_bars_path),
        }
    )
    report = render_report(
        records,
        symbol,
        args.days,
        args.spread_backtest_pips,
        replay_trades_per_bar,
        replay_trade_count,
        replay_bars,
        expected_long_share=expected_long_share,
    )
    out_path.write_text(report, encoding="utf-8")
    json_out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    print(f"\nFull report: {out_path}")
    print(f"JSON summary: {json_out_path}")
    print(f"Audit trace: {audit_path}")


if __name__ == "__main__":
    main()
