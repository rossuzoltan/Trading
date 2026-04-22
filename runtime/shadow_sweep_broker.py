from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interpreter_guard import ensure_project_venv

ensure_project_venv(project_root=ROOT, script_path=__file__)

from feature_engine import FeatureEngine
from paper_live_metrics import resolve_shadow_evidence_paths
from runtime.runtime_engine import Mt5CursorTickSource, TickCursor, VolumeBarBuilder
from selector_manifest import SelectorManifest, load_selector_manifest
from runtime.shadow_broker import ShadowBroker, _acquire_shadow_lock, _is_forex_session_open
from symbol_utils import price_to_pips

log = logging.getLogger("shadow_sweep_broker")


@dataclass(frozen=True)
class SweepTarget:
    profile_name: str
    manifest_path: Path
    manifest: SelectorManifest
    broker: ShadowBroker
    lock_handle: Any


def _profile_name_for_manifest(path: Path) -> str:
    stem = path.stem
    if stem.startswith("manifest."):
        return stem[len("manifest.") :]
    return stem


def _load_targets(
    *,
    manifest_paths: list[str | Path],
    symbol: str | None,
    ticks_per_bar: int | None,
    audit_dir: str | Path | None,
) -> tuple[list[SweepTarget], str, int]:
    loaded: list[tuple[Path, SelectorManifest]] = []
    for manifest_path in manifest_paths:
        resolved_manifest_path = Path(manifest_path).resolve()
        manifest = load_selector_manifest(
            resolved_manifest_path,
            verify_manifest_hash=True,
            strict_manifest_hash=True,
            require_component_hashes=True,
        )
        if manifest.release_stage != "paper_live_candidate":
            raise RuntimeError(
                f"Shadow sweep FATAL: release_stage must be 'paper_live_candidate', got {manifest.release_stage}"
            )
        if manifest.live_trading_approved:
            raise RuntimeError("Shadow sweep FATAL: live_trading_approved must be False for shadow execution.")
        loaded.append((resolved_manifest_path, manifest))

    if not loaded:
        raise RuntimeError("Shadow sweep requires at least one manifest.")

    resolved_symbol = (symbol or loaded[0][1].strategy_symbol).upper()
    resolved_ticks_per_bar = int(
        ticks_per_bar or loaded[0][1].ticks_per_bar or loaded[0][1].bar_construction_ticks_per_bar or 0
    )
    if resolved_ticks_per_bar <= 0:
        raise RuntimeError("Shadow sweep requires a positive ticks_per_bar value.")

    base_feature_schema_hash = loaded[0][1].feature_schema_hash
    targets: list[SweepTarget] = []
    for resolved_manifest_path, manifest in loaded:
        manifest_symbol = str(manifest.strategy_symbol).upper()
        manifest_ticks = int(manifest.ticks_per_bar or manifest.bar_construction_ticks_per_bar or 0)
        if manifest_symbol != resolved_symbol:
            raise RuntimeError(
                f"Shadow sweep requires a shared symbol. Expected {resolved_symbol}, got {manifest_symbol} "
                f"for {resolved_manifest_path}"
            )
        if manifest_ticks != resolved_ticks_per_bar:
            raise RuntimeError(
                f"Shadow sweep requires shared ticks_per_bar. Expected {resolved_ticks_per_bar}, got {manifest_ticks} "
                f"for {resolved_manifest_path}"
            )
        if manifest.feature_schema_hash != base_feature_schema_hash:
            raise RuntimeError(
                f"Shadow sweep requires matching feature_schema_hash values. "
                f"Expected {base_feature_schema_hash}, got {manifest.feature_schema_hash} for {resolved_manifest_path}"
            )

        shadow_paths = resolve_shadow_evidence_paths(
            symbol=resolved_symbol,
            manifest_hash=manifest.manifest_hash,
            base_dir=audit_dir,
        )
        shadow_paths.root_dir.mkdir(parents=True, exist_ok=True)
        lock_handle = _acquire_shadow_lock(shadow_paths.root_dir / "shadow.lock")
        broker = ShadowBroker(
            resolved_manifest_path,
            audit_path=shadow_paths.events_path,
            manifest_fingerprint=manifest.manifest_hash,
            release_stage=manifest.release_stage,
            symbol=resolved_symbol,
            ticks_per_bar=resolved_ticks_per_bar,
        )
        targets.append(
            SweepTarget(
                profile_name=_profile_name_for_manifest(resolved_manifest_path),
                manifest_path=resolved_manifest_path,
                manifest=manifest,
                broker=broker,
                lock_handle=lock_handle,
            )
        )

    return targets, resolved_symbol, resolved_ticks_per_bar


def run_mt5_shadow_sweep_loop(
    *,
    manifest_paths: list[str | Path],
    symbol: str | None = None,
    ticks_per_bar: int | None = None,
    audit_dir: str | Path | None = None,
    poll_interval_ms: int = 250,
    max_bars: int | None = None,
) -> int:
    from live_bridge import _connect_mt5, _load_warmup_bars

    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError("MetaTrader5 is required for the shadow sweep simulator.") from exc

    targets, resolved_symbol, resolved_ticks_per_bar = _load_targets(
        manifest_paths=manifest_paths,
        symbol=symbol,
        ticks_per_bar=ticks_per_bar,
        audit_dir=audit_dir,
    )

    feature_engine = FeatureEngine()
    warmup_frame = _load_warmup_bars(resolved_symbol, resolved_ticks_per_bar)
    feature_engine.warm_up(warmup_frame)

    bar_builder = VolumeBarBuilder(resolved_ticks_per_bar)
    cursor = TickCursor()
    processed_bars = 0

    _connect_mt5(mt5)
    tick_source = Mt5CursorTickSource(mt5)
    log.info("MT5 server UTC offset hours=%s", getattr(tick_source, "server_utc_offset_hours", None))
    log.info(
        "Starting shadow sweep symbol=%s ticks_per_bar=%s target_count=%s manifests=%s",
        resolved_symbol,
        resolved_ticks_per_bar,
        len(targets),
        [str(item.manifest_path) for item in targets],
    )

    consecutive_errors = 0
    max_backoff_s = 60.0
    heartbeat_seconds = 60.0
    last_heartbeat = time.monotonic()
    last_tick_time_msc: int | None = None

    try:
        while True:
            try:
                if not tick_source.mt5.terminal_info():
                    log.warning("MT5 disconnected. Attempting reconnect...")
                    _connect_mt5(mt5)

                ticks, cursor = tick_source.fetch(resolved_symbol, cursor)
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                backoff_time = min(max_backoff_s, (2**consecutive_errors)) + random.uniform(0.1, 1.0)
                log.error("MT5 Fetch Error: %s. Backing off for %.2fs", exc, backoff_time)
                time.sleep(backoff_time)
                continue

            if not ticks:
                now_mono = time.monotonic()
                if now_mono - last_heartbeat >= heartbeat_seconds:
                    ticks_in_bar = int(getattr(bar_builder.state, "tick_count", 0) or 0)
                    log.info(
                        "shadow sweep heartbeat ticks_fetched=0 ticks_in_bar=%s/%s last_tick_utc=%s targets=%s",
                        ticks_in_bar,
                        resolved_ticks_per_bar,
                        (
                            datetime.fromtimestamp(last_tick_time_msc / 1000.0, tz=timezone.utc).isoformat()
                            if last_tick_time_msc
                            else None
                        ),
                        [target.profile_name for target in targets],
                    )
                    last_heartbeat = now_mono
                time.sleep(max(float(poll_interval_ms), 1.0) / 1000.0)
                continue

            try:
                last_tick_time_msc = int(getattr(ticks[-1], "time_msc", 0) or 0) or last_tick_time_msc
            except Exception:
                last_tick_time_msc = last_tick_time_msc

            for tick in ticks:
                bar = bar_builder.push_tick(tick)
                if bar is None:
                    continue

                feature_engine.push(bar.to_series())
                if feature_engine._buffer is None or feature_engine._buffer.empty:
                    continue

                latest_features = feature_engine._buffer.iloc[-1].to_dict()
                spread_pips = abs(float(price_to_pips(resolved_symbol, float(bar.avg_spread))))
                is_session_open = _is_forex_session_open(bar.timestamp)

                for target in targets:
                    record = target.broker.evaluate(
                        bar_ts=bar.timestamp,
                        features=latest_features,
                        current_spread_pips=spread_pips,
                        is_session_open=is_session_open,
                    )
                    log.info(
                        "shadow sweep profile=%s bar=%s signal=%s allow=%s open=%s close=%s flat=%s hold=%s reason=%s",
                        target.profile_name,
                        record.bar_ts,
                        record.signal,
                        record.allow_execution,
                        record.would_open,
                        record.would_close,
                        record.would_remain_flat,
                        record.would_hold_position,
                        record.reason,
                    )

                processed_bars += 1
                if max_bars is not None and processed_bars >= max_bars:
                    return processed_bars

            now_mono = time.monotonic()
            if now_mono - last_heartbeat >= heartbeat_seconds:
                ticks_in_bar = int(getattr(bar_builder.state, "tick_count", 0) or 0)
                log.info(
                    "shadow sweep heartbeat ticks_fetched=%s ticks_in_bar=%s/%s last_tick_utc=%s targets=%s",
                    len(ticks),
                    ticks_in_bar,
                    resolved_ticks_per_bar,
                    (
                        datetime.fromtimestamp(last_tick_time_msc / 1000.0, tz=timezone.utc).isoformat()
                        if last_tick_time_msc
                        else None
                    ),
                    [target.profile_name for target in targets],
                )
                last_heartbeat = now_mono
            time.sleep(max(float(poll_interval_ms), 1.0) / 1000.0)
    finally:
        mt5.shutdown()
        for target in targets:
            try:
                target.lock_handle.close()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-manifest MT5 shadow sweep in a single process.")
    parser.add_argument(
        "--manifest",
        "--manifest-path",
        dest="manifest_paths",
        action="append",
        required=True,
        help="Path to a manifest.json file. Repeat for each sweep profile.",
    )
    parser.add_argument("--symbol", help="Override manifest symbol.")
    parser.add_argument("--ticks-per-bar", type=int, help="Override manifest ticks_per_bar.")
    parser.add_argument("--audit-dir", help="Where to write the shadow audit JSONL daily files.")
    parser.add_argument("--poll-interval-ms", type=int, default=250)
    parser.add_argument("--max-bars", type=int, help="Optional max emitted shared bars before exit.")
    parser.add_argument(
        "--log-full-features",
        action="store_true",
        help="Include full feature snapshots in each shadow JSONL record (large; best for early RC debugging).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if bool(args.log_full_features):
        os.environ["SHADOW_LOG_FULL_FEATURES"] = "1"
    run_mt5_shadow_sweep_loop(
        manifest_paths=args.manifest_paths,
        symbol=args.symbol,
        ticks_per_bar=args.ticks_per_bar,
        audit_dir=args.audit_dir,
        poll_interval_ms=args.poll_interval_ms,
        max_bars=args.max_bars,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
