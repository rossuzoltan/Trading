from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from feature_engine import FeatureEngine
from rule_selector import RuleSelector
from runtime.runtime_engine import Mt5CursorTickSource, TickCursor, VolumeBarBuilder
from selector_manifest import load_selector_manifest
from symbol_utils import price_to_pips

log = logging.getLogger("shadow_broker")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _is_forex_session_open(timestamp: datetime) -> bool:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    weekday = int(ts.weekday())
    hour_fraction = float(ts.hour) + (float(ts.minute) / 60.0)
    if weekday == 5:
        return False
    if weekday == 6 and hour_fraction < 22.0:
        return False
    if weekday == 4 and hour_fraction >= 22.0:
        return False
    return True


def _iso_utc(timestamp: Any) -> str:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


@dataclass
class ShadowAuditRecord:
    bar_ts: str
    signal: int
    reason: str
    spread: float
    session_ok: bool
    risk_ok: bool
    would_open: bool
    would_close: bool
    would_hold_position: bool
    would_remain_flat: bool
    allow_execution: bool
    spread_ok: bool
    position_ok: bool
    daily_loss_ok: bool
    active_position_state: str  # flat, long, short
    position_side: int          # 0, 1, -1
    position_after: int
    manifest_fingerprint: str
    release_stage: str
    symbol: str
    ticks_per_bar: int


class ShadowBroker:
    """
    Draft shadow bridge for manifest-driven rule selectors.
    It never submits broker orders; it only translates selector output into
    "would open" / "would close" audit events.
    """

    def __init__(
        self,
        selector: RuleSelector | str | Path,
        *,
        audit_path: str | Path,
        manifest_fingerprint: str = "",
        release_stage: str = "unknown",
        symbol: str = "UNKNOWN",
        ticks_per_bar: int = 0,
    ) -> None:
        self.selector = selector if isinstance(selector, RuleSelector) else RuleSelector(selector)
        self.audit_path = Path(audit_path)
        self.manifest_fingerprint = manifest_fingerprint
        self.release_stage = release_stage
        self.symbol = symbol
        self.ticks_per_bar = ticks_per_bar
        
        self.position_direction = 0
        self.daily_pnl_usd = 0.0
        self.records_written = 0

    def _current_portfolio_state(self) -> dict[str, Any]:
        return {
            "current_positions": 1 if self.position_direction != 0 else 0,
            "current_direction": self.position_direction,
            "position_direction": self.position_direction,
            "daily_pnl_usd": self.daily_pnl_usd,
        }

    def evaluate(
        self,
        *,
        bar_ts: Any,
        features: dict[str, Any],
        current_spread_pips: float,
        is_session_open: bool,
        portfolio_state: dict[str, Any] | None = None,
    ) -> ShadowAuditRecord:
        effective_state = dict(self._current_portfolio_state())
        if portfolio_state:
            effective_state.update(portfolio_state)
        current_direction = int(effective_state.get("position_direction", self.position_direction) or 0)

        decision = self.selector.decide(
            features=features,
            current_spread_pips=current_spread_pips,
            is_session_open=is_session_open,
            portfolio_state=effective_state,
        )
        gate_status = self.selector.gate_status(
            signal=decision.signal,
            current_spread_pips=current_spread_pips,
            is_session_open=is_session_open,
            portfolio_state=effective_state,
        )

        normalized_signal = 1 if decision.signal > 0 else -1 if decision.signal < 0 else 0
        would_open = bool(
            decision.allow_execution
            and normalized_signal != 0
            and (current_direction == 0 or normalized_signal != current_direction)
        )
        would_close = bool(
            decision.allow_execution
            and
            current_direction != 0
            and (normalized_signal == 0 or normalized_signal != current_direction)
        )
        
        would_hold_position = bool(current_direction != 0 and not would_close)
        would_remain_flat = bool(current_direction == 0 and not would_open)

        if would_close:
            self.position_direction = 0
        if would_open and self.position_direction == 0:
            self.position_direction = normalized_signal
            
        active_state = "long" if current_direction == 1 else "short" if current_direction == -1 else "flat"

        record = ShadowAuditRecord(
            bar_ts=_iso_utc(bar_ts),
            signal=int(normalized_signal),
            reason=decision.reason,
            spread=float(current_spread_pips),
            session_ok=bool(gate_status["session_ok"]),
            risk_ok=bool(gate_status["risk_ok"]),
            would_open=would_open,
            would_close=would_close,
            would_hold_position=would_hold_position,
            would_remain_flat=would_remain_flat,
            allow_execution=bool(decision.allow_execution),
            spread_ok=bool(gate_status["spread_ok"]),
            position_ok=bool(gate_status["position_ok"]),
            daily_loss_ok=bool(gate_status["daily_loss_ok"]),
            active_position_state=active_state,
            position_side=int(current_direction),
            position_after=int(self.position_direction),
            manifest_fingerprint=self.manifest_fingerprint or decision.manifest_id,
            release_stage=self.release_stage,
            symbol=self.symbol,
            ticks_per_bar=self.ticks_per_bar,
        )
        _append_jsonl(self.audit_path, asdict(record))
        self.records_written += 1
        return record


def run_mt5_shadow_loop(
    *,
    manifest_path: str | Path,
    symbol: str | None = None,
    ticks_per_bar: int | None = None,
    audit_dir: str | Path | None = None,
    poll_interval_ms: int = 250,
    max_bars: int | None = None,
) -> int:
    manifest = load_selector_manifest(manifest_path, verify_manifest_hash=True)
    
    # 1. Startup Safety Assertions
    if manifest.release_stage != "paper_live_candidate":
        raise RuntimeError(f"Shadow loop FATAL: release_stage must be 'paper_live_candidate', got {manifest.release_stage}")
    if manifest.live_trading_approved:
        raise RuntimeError("Shadow loop FATAL: live_trading_approved must be False for shadow execution.")

    resolved_symbol = (symbol or manifest.strategy_symbol).upper()
    resolved_ticks_per_bar = int(ticks_per_bar or manifest.ticks_per_bar or manifest.bar_construction_ticks_per_bar or 0)
    if resolved_ticks_per_bar <= 0:
        raise RuntimeError("Shadow broker requires a positive ticks_per_bar value.")

    from live_bridge import _connect_mt5, _load_warmup_bars
    import random

    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError("MetaTrader5 is required for the shadow simulator.") from exc

    # Default to directory rather than single file
    base_audit_dir = Path(audit_dir) if audit_dir is not None else Path(manifest_path).parent / "shadow_audits"
    base_audit_dir.mkdir(parents=True, exist_ok=True)
    
    def get_daily_audit_path(date_str: str) -> Path:
        return base_audit_dir / f"shadow_audit_{date_str}.jsonl"

    feature_engine = FeatureEngine()
    warmup_frame = _load_warmup_bars(resolved_symbol, resolved_ticks_per_bar)
    feature_engine.warm_up(warmup_frame)

    current_date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    current_audit_path = get_daily_audit_path(current_date_str)

    broker = ShadowBroker(
        manifest_path, 
        audit_path=current_audit_path,
        manifest_fingerprint=manifest.manifest_hash,
        release_stage=manifest.release_stage,
        symbol=resolved_symbol,
        ticks_per_bar=resolved_ticks_per_bar
    )
    
    bar_builder = VolumeBarBuilder(resolved_ticks_per_bar)
    tick_source = Mt5CursorTickSource(mt5)
    cursor = TickCursor()
    processed_bars = 0
    
    # Daily summary tracking
    def new_daily_stats():
        return {
            "total_bars_processed": 0,
            "total_would_open": 0,
            "total_would_close": 0,
            "total_would_hold_position": 0,
            "total_would_remain_flat": 0,
            "total_no_trade": 0,
            "long_open_count": 0,
            "short_open_count": 0,
            "reason_counts": {
                "spread": 0,
                "session": 0,
                "risk": 0,
                "no_signal": 0,
                "max_position": 0
            },
            "first_bar_ts": None,
            "last_bar_ts": None
        }

    daily_stats = new_daily_stats()

    _connect_mt5(mt5)
    log.info(
        "Starting shadow simulator symbol=%s ticks_per_bar=%s audit_dir=%s",
        resolved_symbol,
        resolved_ticks_per_bar,
        base_audit_dir,
    )
    
    consecutive_errors = 0
    max_backoff_s = 60.0

    try:
        while True:
            try:
                if not tick_source.mt5.terminal_info():
                    log.warning("MT5 disconnected. Attempting reconnect...")
                    _connect_mt5(mt5)
                
                ticks, cursor = tick_source.fetch(resolved_symbol, cursor)
                consecutive_errors = 0  # Reset on success
            except Exception as e:
                consecutive_errors += 1
                backoff_time = min(max_backoff_s, (2 ** consecutive_errors)) + random.uniform(0.1, 1.0)
                log.error("MT5 Fetch Error: %s. Backing off for %.2fs", e, backoff_time)
                time.sleep(backoff_time)
                continue
                
            if not ticks:
                time.sleep(max(float(poll_interval_ms), 1.0) / 1000.0)
                continue
                
            for tick in ticks:
                bar = bar_builder.push_tick(tick)
                if bar is None:
                    continue
                
                # Check for UTC day rollover
                bar_date_str = pd.Timestamp(bar.timestamp).tz_convert("UTC").strftime("%Y%m%d")
                if bar_date_str != current_date_str:
                    # Write summary
                    summary_path = base_audit_dir / f"shadow_summary_{current_date_str}.json"
                    with summary_path.open("w", encoding="utf-8") as f:
                        json.dump(daily_stats, f, indent=2)
                    log.info("Daily Rollover Info: %s", json.dumps(daily_stats))
                    
                    # Reset for new day
                    current_date_str = bar_date_str
                    broker.audit_path = get_daily_audit_path(current_date_str)
                    daily_stats = new_daily_stats()
                
                feature_engine.push(bar.to_series())
                if feature_engine._buffer is None or feature_engine._buffer.empty:
                    continue
                    
                latest_features = feature_engine._buffer.iloc[-1].to_dict()
                spread_pips = abs(float(price_to_pips(resolved_symbol, float(bar.avg_spread))))
                
                record = broker.evaluate(
                    bar_ts=bar.timestamp,
                    features=latest_features,
                    current_spread_pips=spread_pips,
                    is_session_open=_is_forex_session_open(bar.timestamp),
                )
                processed_bars += 1
                
                # Update stats
                daily_stats["total_bars_processed"] += 1
                if not daily_stats["first_bar_ts"]:
                    daily_stats["first_bar_ts"] = record.bar_ts
                daily_stats["last_bar_ts"] = record.bar_ts
                
                if record.would_open:
                    daily_stats["total_would_open"] += 1
                    if record.signal > 0:
                        daily_stats["long_open_count"] += 1
                    elif record.signal < 0:
                        daily_stats["short_open_count"] += 1
                elif record.would_close:
                    daily_stats["total_would_close"] += 1
                elif record.would_hold_position:
                    daily_stats["total_would_hold_position"] += 1
                elif record.would_remain_flat:
                    daily_stats["total_would_remain_flat"] += 1
                    daily_stats["total_no_trade"] += 1
                    
                if record.reason:
                    r_mapped = record.reason.lower()
                    for r_key in daily_stats["reason_counts"].keys():
                        if r_key in r_mapped:
                            daily_stats["reason_counts"][r_key] += 1
                            break
                    else:
                        if "signal" in r_mapped:
                            daily_stats["reason_counts"]["no_signal"] += 1

                log.info(
                    "shadow bar=%s signal=%s allow=%s open=%s close=%s flat=%s hold=%s reason=%s",
                    record.bar_ts,
                    record.signal,
                    record.allow_execution,
                    record.would_open,
                    record.would_close,
                    record.would_remain_flat,
                    record.would_hold_position,
                    record.reason,
                )
                if max_bars is not None and processed_bars >= max_bars:
                    # Flush final summary if exiting abruptly
                    summary_path = base_audit_dir / f"shadow_summary_{current_date_str}.json"
                    with summary_path.open("w", encoding="utf-8") as f:
                        json.dump(daily_stats, f, indent=2)
                    return processed_bars
            time.sleep(max(float(poll_interval_ms), 1.0) / 1000.0)
    finally:
        mt5.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the draft MT5 shadow simulator for a rule-selector manifest.")
    parser.add_argument("--manifest", "--manifest-path", dest="manifest", required=True, help="Path to the RC manifest.json file.")
    parser.add_argument("--symbol", help="Override manifest symbol.")
    parser.add_argument("--ticks-per-bar", type=int, help="Override manifest ticks_per_bar.")
    parser.add_argument("--audit-dir", help="Where to write the shadow audit JSONL daily files.")
    parser.add_argument("--poll-interval-ms", type=int, default=250)
    parser.add_argument("--max-bars", type=int, help="Optional max emitted bars before exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_mt5_shadow_loop(
        manifest_path=args.manifest,
        symbol=args.symbol,
        ticks_per_bar=args.ticks_per_bar,
        audit_dir=args.audit_dir,
        poll_interval_ms=args.poll_interval_ms,
        max_bars=args.max_bars,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
