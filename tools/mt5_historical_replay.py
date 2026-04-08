"""
mt5_historical_replay.py  —  Phase P7 Pre-Shadow Accelerator

Fetches real historical ticks from MT5 (last N days), builds volume bars
on the fly, runs the Rule-First selector through them, and writes a full
drift report — giving you weeks of "near-live" evidence in minutes.

Usage:
    python tools/mt5_historical_replay.py --symbol EURUSD --days 30
    python tools/mt5_historical_replay.py --symbol EURUSD --days 7 --output replay_recent.md
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

log = logging.getLogger("mt5_historical_replay")


# ── helpers ────────────────────────────────────────────────────────────────

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
        return "✅ OK"
    if delta <= warn:
        return "👀 WATCH"
    if delta <= crit:
        return "⚠️  DRIFT_WARNING"
    return "🚨 DRIFT_CRITICAL"


# ── MT5 tick batch fetcher ──────────────────────────────────────────────────

def fetch_ticks_from_mt5(mt5, symbol: str, start_utc: datetime, end_utc: datetime) -> list[dict]:
    """Pull all ticks for a symbol between two UTC datetimes."""
    log.info("Fetching ticks %s  %s → %s", symbol, start_utc.date(), end_utc.date())
    all_ticks: list[dict] = []
    cursor = start_utc
    batch = 100_000

    while cursor < end_utc:
        raw = mt5.copy_ticks_from(symbol, cursor, batch, mt5.COPY_TICKS_ALL)
        if raw is None or len(raw) == 0:
            break
        for r in raw:
            t = int(r["time_msc"] if "time_msc" in r.dtype.names else r["time"] * 1000)
            if t / 1000 > end_utc.timestamp():
                break
            all_ticks.append({
                "time_msc": t,
                "bid": float(r["bid"]),
                "ask": float(r["ask"]),
            })
        last_ms = all_ticks[-1]["time_msc"]
        cursor = datetime.fromtimestamp(last_ms / 1000, tz=timezone.utc) + timedelta(milliseconds=1)
        log.info("  … fetched %d ticks so far (last: %s)", len(all_ticks), cursor.strftime("%Y-%m-%d %H:%M"))
        if len(raw) < batch:
            break

    log.info("Total ticks fetched: %d", len(all_ticks))
    return all_ticks


# ── Volume bar builder ──────────────────────────────────────────────────────

def build_volume_bars(ticks: list[dict], ticks_per_bar: int) -> list[dict]:
    bars: list[dict] = []
    buf: list[dict] = []

    for tick in ticks:
        buf.append(tick)
        if len(buf) >= ticks_per_bar:
            mid_prices = [(t["bid"] + t["ask"]) / 2.0 for t in buf]
            spreads = [t["ask"] - t["bid"] for t in buf]
            bar = {
                "timestamp": datetime.fromtimestamp(buf[0]["time_msc"] / 1000, tz=timezone.utc),
                "open": mid_prices[0],
                "high": max(mid_prices),
                "low": min(mid_prices),
                "close": mid_prices[-1],
                "avg_spread": sum(spreads) / len(spreads),
                "avg_spread_pips": (sum(spreads) / len(spreads)) * 10_000,  # approx for major pairs
                "tick_count": len(buf),
            }
            bars.append(bar)
            buf = []

    log.info("Built %d volume bars from %d ticks", len(bars), len(ticks))
    return bars


# ── Feature engine warm-up & rule evaluation ────────────────────────────────

def run_rule_on_bars(bars: list[dict], manifest_path: Path) -> list[dict]:
    """Warm up the FeatureEngine then pump bars through the RuleSelector."""
    from feature_engine import FeatureEngine, WARMUP_BARS
    from rule_selector import RuleSelector

    selector = RuleSelector(manifest_path)
    feature_engine = FeatureEngine()

    # Warm up on the first WARMUP_BARS bars
    if len(bars) < WARMUP_BARS + 10:
        raise RuntimeError(f"Need at least {WARMUP_BARS + 10} bars; only got {len(bars)}")

    warmup_rows = []
    for bar in bars[:WARMUP_BARS]:
        warmup_rows.append({
            "Open": bar["open"], "High": bar["high"], "Low": bar["low"],
            "Close": bar["close"], "Volume": float(bar["tick_count"]),
            "avg_spread": bar["avg_spread"], "time_delta_s": 300.0,
        })
    warmup_df = pd.DataFrame(warmup_rows)
    warmup_df.index = pd.DatetimeIndex([b["timestamp"] for b in bars[:WARMUP_BARS]])
    warmup_df.index.name = "Gmt time"
    feature_engine.warm_up(warmup_df)

    records: list[dict] = []
    position_direction = 0

    for bar in bars[WARMUP_BARS:]:
        series = pd.Series({
            "Open": bar["open"], "High": bar["high"], "Low": bar["low"],
            "Close": bar["close"], "Volume": float(bar["tick_count"]),
            "avg_spread": bar["avg_spread"], "time_delta_s": 300.0,
        }, name=bar["timestamp"])
        feature_engine.push(series)

        if feature_engine._buffer is None or feature_engine._buffer.empty:
            continue

        features = feature_engine._buffer.iloc[-1].to_dict()
        spread_pips = bar["avg_spread_pips"]

        portfolio_state = {
            "current_positions": 1 if position_direction != 0 else 0,
            "current_direction": position_direction,
            "position_direction": position_direction,
            "daily_pnl_usd": 0.0,
        }

        hour = bar["timestamp"].hour
        is_session = not (bar["timestamp"].weekday() == 5 or
                         (bar["timestamp"].weekday() == 6 and hour < 22) or
                         (bar["timestamp"].weekday() == 4 and hour >= 22))

        decision = selector.decide(
            features=features,
            current_spread_pips=spread_pips,
            portfolio_state=portfolio_state,
            is_session_open=is_session,
        )

        sig = 1 if decision.signal > 0 else -1 if decision.signal < 0 else 0
        would_open = bool(decision.allow_execution and sig != 0 and
                          (position_direction == 0 or sig != position_direction))
        would_close = bool(decision.allow_execution and position_direction != 0 and
                           (sig == 0 or sig != position_direction))
        would_hold = bool(position_direction != 0 and not would_close)
        would_flat = bool(position_direction == 0 and not would_open)

        if would_close:
            position_direction = 0
        if would_open and position_direction == 0:
            position_direction = sig

        state = "long" if position_direction == 1 else "short" if position_direction == -1 else "flat"

        records.append({
            "bar_ts": bar["timestamp"].isoformat(),
            "hour_utc": bar["timestamp"].hour,
            "session": _session_bucket(bar["timestamp"].hour),
            "signal": sig,
            "allow_execution": bool(decision.allow_execution),
            "reason": decision.reason or "",
            "spread_pips": round(spread_pips, 4),
            "would_open": would_open,
            "would_close": would_close,
            "would_hold": would_hold,
            "would_flat": would_flat,
            "active_state": state,
        })

    return records


# ── Report rendering ────────────────────────────────────────────────────────

def render_report(records: list[dict], symbol: str, days: int,
                  spread_backtest_pips: float) -> str:
    if not records:
        return "# MT5 Historical Replay\nNo bars processed."

    df = pd.DataFrame(records)
    total = len(df)
    opens = int(df["would_open"].sum())
    closes = int(df["would_close"].sum())
    holds = int(df["would_hold"].sum())
    flats = int(df["would_flat"].sum())
    longs = int(df[(df["would_open"]) & (df["signal"] > 0)].shape[0])
    shorts = int(df[(df["would_open"]) & (df["signal"] < 0)].shape[0])

    tpb = opens / max(total, 1)
    # backtest reference: ~0.013 trades/bar for EURUSD RC1 (27 trades / 2024 bars)
    replay_tpb = 27 / 2024 if "EURUSD" in symbol.upper() else 21 / 2123
    density_ratio = tpb / max(replay_tpb, 1e-6)

    avg_spread = float(df["spread_pips"].mean())
    spread_ratio = avg_spread / max(spread_backtest_pips, 1e-6)

    ls_ratio = longs / max(shorts, 1)

    # Reject reasons
    spread_rej = int(df[df["reason"].str.contains("spread", case=False, na=False)].shape[0])
    session_rej = int(df[df["reason"].str.contains("session", case=False, na=False)].shape[0])
    no_signal = int(df[df["reason"].str.contains("signal", case=False, na=False)].shape[0])

    # Session breakdown
    session_opens = df[df["would_open"]].groupby("session").size().to_dict()

    # Overall verdict
    verdicts = [density_ratio, spread_ratio, ls_ratio]
    worst = max(abs(v - 1.0) for v in verdicts)
    if worst <= 0.10:
        overall = "✅ OK — Live behaviour matches backtest expectations"
    elif worst <= 0.30:
        overall = "👀 WATCH — Minor deviations, monitor closely"
    elif worst <= 0.60:
        overall = "⚠️  DRIFT_WARNING — Notable deviation from backtest"
    else:
        overall = "🚨 DRIFT_CRITICAL — Significant live vs backtest divergence"

    lines = [
        f"# MT5 Historical Replay Report — {symbol}",
        f"**Period**: Last {days} days of real MT5 tick data",
        f"**Bars processed**: {total}  |  **Ticks/bar**: built from live tick stream",
        f"**Overall verdict**: {overall}",
        "",
        "---",
        "",
        "## A. Signal Density",
        "| Metric | Value |",
        "|---|---|",
        f"| Would-Open count | {opens} |",
        f"| Bars processed | {total} |",
        f"| Live trades/bar | {tpb:.5f} |",
        f"| Replay trades/bar | {replay_tpb:.5f} |",
        f"| Ratio | {density_ratio:.2f}x \u00a0 {_verdict(density_ratio)} |",
        "",
        "## B. Spread Reality Check",
        "| Metric | Value |",
        "|---|---|",
        f"| Live avg spread | {avg_spread:.4f} pips |",
        f"| Backtest assumed spread | {spread_backtest_pips:.4f} pips |",
        f"| Spread ratio | {spread_ratio:.2f}x \u00a0 {_verdict(spread_ratio)} |",
        "",
        "## C. Direction Balance",
        "| Metric | Value |",
        "|---|---|",
        f"| Long opens | {longs} |",
        f"| Short opens | {shorts} |",
        f"| L/S ratio | {ls_ratio:.2f}x \u00a0 {_verdict(ls_ratio)} |",
        "",
        "## D. State Occupancy",
        "| State | Bars | % |",
        "|---|---|---|",
        f"| Would-Open | {opens} | {opens/total*100:.1f}% |",
        f"| Would-Close | {closes} | {closes/total*100:.1f}% |",
        f"| Holding | {holds} | {holds/total*100:.1f}% |",
        f"| Flat/No-Trade | {flats} | {flats/total*100:.1f}% |",
        "",
        "## E. Gate Reject Reasons",
        "| Reason | Count |",
        "|---|---|",
        f"| Spread reject | {spread_rej} |",
        f"| Session reject | {session_rej} |",
        f"| No signal | {no_signal} |",
        "",
        "## F. Session Breakdown (Would-Open by session)",
        "| Session | Opens |",
        "|---|---|",
    ]
    for sess in ["Asia", "London", "London/NY", "NY", "Rollover"]:
        lines.append(f"| {sess} | {session_opens.get(sess, 0)} |")

    lines += [
        "",
        "---",
        f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        f"from real MT5 historical ticks.*",
    ]
    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MT5 Historical Tick Replay — Phase P7 pre-shadow accelerator")
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--days", type=int, default=30, help="How many calendar days back to fetch")
    p.add_argument("--manifest", default="", help="Path to RC manifest.json (auto-detected if blank)")
    p.add_argument("--output", default="", help="Output .md path (auto-detected if blank)")
    p.add_argument("--spread-backtest-pips", type=float, default=0.5,
                   help="The spread assumed in the backtest (for comparison)")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    symbol = args.symbol.upper()

    # Auto-resolve manifest
    manifest_path = Path(args.manifest) if args.manifest else None
    if manifest_path is None:
        candidates = sorted(Path(ROOT / "models" / "rc1").glob(f"*{symbol.lower()}*/manifest.json"))
        if not candidates:
            raise SystemExit(f"No RC1 manifest found for {symbol}. Pass --manifest explicitly.")
        manifest_path = candidates[0]
    log.info("Using manifest: %s", manifest_path)

    # Auto-resolve output
    out_path = Path(args.output) if args.output else manifest_path.parent / "mt5_historical_replay_report.md"

    # Connect MT5
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise SystemExit("MetaTrader5 not installed. Cannot fetch live tick data.")

    from live_bridge import _connect_mt5
    _connect_mt5(mt5)

    end_utc = datetime.now(timezone.utc)
    start_utc = end_utc - timedelta(days=args.days)

    # 1. Fetch ticks
    ticks = fetch_ticks_from_mt5(mt5, symbol, start_utc, end_utc)
    mt5.shutdown()

    if not ticks:
        raise SystemExit("No ticks returned from MT5. Is the terminal running and the symbol selected?")

    # 2. Build volume bars
    from selector_manifest import load_selector_manifest
    mf = load_selector_manifest(manifest_path)
    tpb = int(mf.ticks_per_bar or mf.bar_construction_ticks_per_bar or 5000)
    bars = build_volume_bars(ticks, tpb)

    # 3. Save raw bars for audit (optional jsonl)
    raw_bars_path = out_path.with_suffix(".bars.jsonl")
    with raw_bars_path.open("w", encoding="utf-8") as f:
        for b in bars:
            f.write(json.dumps({k: (v.isoformat() if hasattr(v, "isoformat") else v)
                                 for k, v in b.items()}) + "\n")
    log.info("Raw bars saved → %s", raw_bars_path)

    # 4. Run rule
    log.info("Running RC1 rule selector over %d bars…", len(bars))
    records = run_rule_on_bars(bars, manifest_path)
    log.info("Evaluation complete: %d bars evaluated", len(records))

    # 5. Save audit JSONL
    audit_path = out_path.with_suffix(".audit.jsonl")
    with audit_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    log.info("Audit trace saved → %s", audit_path)

    # 6. Render report
    report = render_report(records, symbol, args.days, args.spread_backtest_pips)
    out_path.write_text(report, encoding="utf-8")

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    print(f"\nFull report: {out_path}")
    print(f"Audit trace: {audit_path}")


if __name__ == "__main__":
    main()
