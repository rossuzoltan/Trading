import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import MetaTrader5 as mt5

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shadow_trade_accounting import summarize_shadow_trade_accounting


@dataclass(frozen=True)
class Profile:
    profile_id: str
    manifest_hash: str


@dataclass(frozen=True)
class BarTick:
    time: int
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


def _parse_utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def _read_ladder_profiles(ladder_json_path: str) -> List[Profile]:
    with open(ladder_json_path, "r", encoding="utf-8-sig") as f:
        ladder = json.load(f)
    profiles: List[Profile] = []
    for g in ladder.get("generated", []):
        profiles.append(
            Profile(
                profile_id=str(g["profile_id"]),
                manifest_hash=str(g["manifest_hash"]).lower(),
            )
        )
    profiles.sort(key=lambda p: p.profile_id)
    return profiles


def _read_events_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rows.sort(key=lambda r: str(r.get("timestamp_utc", "")))
    return rows


def _pip_size(symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"MT5 symbol_info({symbol}) returned None")
    point = float(info.point)
    digits = int(info.digits)
    return point * 10.0 if digits in (3, 5) else point


def _tick_at_or_before(symbol: str, bar_ts_utc: datetime, window_minutes: int) -> Optional[BarTick]:
    frm = bar_ts_utc - timedelta(minutes=window_minutes)
    to = bar_ts_utc + timedelta(minutes=1)
    ticks = mt5.copy_ticks_range(symbol, frm, to, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        return None
    target = int(bar_ts_utc.timestamp())
    chosen = None
    for tick in ticks[::-1]:
        if int(tick["time"]) <= target:
            chosen = tick
            break
    if chosen is None:
        return None
    return BarTick(time=int(chosen["time"]), bid=float(chosen["bid"]), ask=float(chosen["ask"]))


def _ensure_events_share_bars(profile_events: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    all_bars: Optional[List[str]] = None
    for pid, evs in profile_events.items():
        bars = [str(e["timestamp_utc"]) for e in evs]
        if all_bars is None:
            all_bars = bars
            continue
        if bars != all_bars:
            raise RuntimeError(f"Bar timestamps differ for {pid}")
    return all_bars or []


def _calc_realized_trades_pips(
    *,
    events: List[Dict[str, Any]],
    bar_ticks: Dict[str, BarTick],
    pip: float,
) -> List[float]:
    pos = 0
    entry_price: Optional[float] = None

    realized: List[float] = []
    for ev in events:
        ts = str(ev["timestamp_utc"])
        tick = bar_ticks.get(ts)
        if tick is None:
            raise RuntimeError(f"Missing at-or-before MT5 tick for event timestamp {ts}")

        if bool(ev.get("would_close", False)) and pos != 0 and entry_price is not None:
            exit_price = tick.bid if pos > 0 else tick.ask
            pnl_pips = (exit_price - entry_price) / pip if pos > 0 else (entry_price - exit_price) / pip
            realized.append(float(pnl_pips))
            entry_price = None
            pos = 0

        if bool(ev.get("would_open", False)):
            new_direction = int(ev.get("signal_direction", ev.get("signal", 0)) or 0)
            if new_direction == 0:
                new_direction = int(ev.get("position_after", 0) or 0)
            if new_direction == 0:
                raise RuntimeError(f"Cannot infer open direction for event timestamp {ts}")
            entry_price = tick.ask if new_direction > 0 else tick.bid
            pos = int(new_direction)

    return realized


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-root", required=True)
    ap.add_argument("--ladder-json", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--window-minutes", type=int, default=10)
    args = ap.parse_args()

    profiles = _read_ladder_profiles(args.ladder_json)
    if not profiles:
        raise SystemExit("No profiles found in ladder JSON.")

    profile_events: Dict[str, List[Dict[str, Any]]] = {}
    for p in profiles:
        path = os.path.join(args.audit_root, args.symbol, p.manifest_hash, "events.jsonl")
        profile_events[p.profile_id] = _read_events_jsonl(path)

    bars = _ensure_events_share_bars(profile_events)
    if not bars:
        raise SystemExit("No bars found in events.jsonl files.")

    if not mt5.initialize():
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        mt5.symbol_select(args.symbol, True)
        pip = _pip_size(args.symbol)

        bar_ticks: Dict[str, BarTick] = {}
        missing = 0
        for ts in bars:
            tick = _tick_at_or_before(args.symbol, _parse_utc(ts), args.window_minutes)
            if tick is None:
                missing += 1
            else:
                bar_ticks[ts] = tick

        if missing:
            raise SystemExit(f"Missing MT5 ticks for {missing}/{len(bars)} bar timestamps.")

        stats: List[Tuple[str, int, float, float, int, int]] = []
        for p in profiles:
            events = profile_events[p.profile_id]
            has_snapshots = any(
                isinstance(ev.get("entry_snapshot"), dict) or isinstance(ev.get("exit_snapshot"), dict)
                for ev in events
            )
            if has_snapshots:
                accounting = summarize_shadow_trade_accounting(
                    events=events,
                    symbol=args.symbol,
                    commission_per_lot=0.0,
                    slippage_pips=0.0,
                )
                if float(accounting.get("realized_trade_coverage", 0.0) or 0.0) < 1.0:
                    raise RuntimeError(
                        f"Incomplete snapshot trade coverage for {p.profile_id}: "
                        f"{float(accounting.get('realized_trade_coverage', 0.0) or 0.0):.2%}"
                    )
                realized = [float(trade["gross_pips"]) for trade in accounting["trades"]]
            else:
                realized = _calc_realized_trades_pips(events=events, bar_ticks=bar_ticks, pip=pip)
            net = sum(realized)
            trades = len(realized)
            wins = sum(1 for x in realized if x > 0)
            losses = sum(1 for x in realized if x < 0)
            avg = net / trades if trades else 0.0
            stats.append((p.profile_id, trades, net, avg, wins, losses))

        stats.sort(key=lambda r: (r[2], r[3]), reverse=True)

        print("profile_id\ttrades\tnet_pips\tavg_pips\twins\tlosses")
        for pid, trades, net, avg, wins, losses in stats:
            print(f"{pid}\t{trades}\t{net:.2f}\t{avg:.2f}\t{wins}\t{losses}")
    finally:
        mt5.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
