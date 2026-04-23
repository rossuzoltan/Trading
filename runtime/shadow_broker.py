from __future__ import annotations

import sys
import argparse
import json
import logging
import math
import os
import platform
import socket
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interpreter_guard import ensure_project_venv

ensure_project_venv(project_root=ROOT, script_path=__file__)

import pandas as pd

from domain.models import VolumeBar
from feature_engine import FeatureEngine
from paper_live_metrics import resolve_shadow_evidence_paths, write_shadow_summary
from rule_selector import RuleSelector
from runtime.runtime_engine import Mt5CursorTickSource, TickCursor, VolumeBarBuilder
from selector_manifest import (
    compute_execution_cost_profile_hash,
    describe_execution_cost_profile,
    load_selector_manifest,
    resolve_execution_cost_profile,
)
from strategies.rule_logic import diagnose_rule_decision
from symbol_utils import price_to_pips

log = logging.getLogger("shadow_broker")


def _acquire_shadow_lock(lock_path: Path) -> Any:
    """
    Best-effort single-instance lock to prevent multiple shadow brokers writing
    into the same evidence directory.

    Windows-first: uses msvcrt byte-range locking. If the platform does not
    support it, we still proceed (operators must ensure single-instance).
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        import msvcrt  # type: ignore

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except ImportError:  # pragma: no cover
        return handle
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"Shadow lock already held: {lock_path}") from exc
    return handle


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return _iso_utc(value)
    if isinstance(value, pd.Timestamp):
        return _iso_utc(value)
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "_asdict") and callable(getattr(value, "_asdict")):
        try:
            return _json_safe(value._asdict())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(payload), sort_keys=True) + "\n")


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


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    return numeric if math.isfinite(numeric) else None


def _position_state_label(direction: int) -> str:
    if int(direction) > 0:
        return "long"
    if int(direction) < 0:
        return "short"
    return "flat"


def _bar_diagnostics(
    *,
    bar_ts: Any,
    bar: VolumeBar | None,
    features: dict[str, Any],
    spread_pips: float,
) -> dict[str, Any]:
    bar_open = _safe_float(bar.open if bar is not None else features.get("Open"))
    bar_high = _safe_float(bar.high if bar is not None else features.get("High"))
    bar_low = _safe_float(bar.low if bar is not None else features.get("Low"))
    bar_close = _safe_float(bar.close if bar is not None else features.get("Close"))
    bar_volume = _safe_float(bar.volume if bar is not None else features.get("Volume"))
    bar_avg_spread_price = _safe_float(bar.avg_spread if bar is not None else features.get("avg_spread"))
    bar_time_delta_s = _safe_float(bar.time_delta_s if bar is not None else features.get("time_delta_s"))
    bar_start_ts_utc = _iso_utc(bar.timestamp if bar is not None else bar_ts)
    bar_end_ts_utc = None
    if bar is not None and getattr(bar, "end_time_msc", None):
        bar_end_ts_utc = _iso_utc(datetime.fromtimestamp(int(bar.end_time_msc) / 1000.0, tz=timezone.utc))

    quote_snapshot = None
    if bar_close is not None:
        half_spread = (bar_avg_spread_price / 2.0) if bar_avg_spread_price is not None else None
        quote_snapshot = {
            "price_source": "bar_close_mid_avg_spread_proxy",
            "decision_ts_utc": bar_end_ts_utc or bar_start_ts_utc,
            "bar_ts_utc": bar_start_ts_utc,
            "bar_end_ts_utc": bar_end_ts_utc,
            "mid_price": bar_close,
            "bid_proxy": (bar_close - half_spread) if half_spread is not None else None,
            "ask_proxy": (bar_close + half_spread) if half_spread is not None else None,
            "spread_price": bar_avg_spread_price,
            "spread_pips": _safe_float(spread_pips),
        }

    return {
        "bar_open": bar_open,
        "bar_high": bar_high,
        "bar_low": bar_low,
        "bar_close": bar_close,
        "bar_volume": bar_volume,
        "bar_avg_spread_price": bar_avg_spread_price,
        "bar_avg_spread_pips": _safe_float(spread_pips),
        "bar_time_delta_s": bar_time_delta_s,
        "bar_start_ts_utc": bar_start_ts_utc,
        "bar_end_ts_utc": bar_end_ts_utc,
        "quote_snapshot": quote_snapshot,
    }


def _fill_side(direction: int, *, phase: str) -> str:
    if phase == "entry":
        return "buy" if int(direction) > 0 else "sell"
    return "sell" if int(direction) > 0 else "buy"


def _fill_price(snapshot: dict[str, Any], direction: int, *, phase: str) -> float | None:
    if phase == "entry":
        return _safe_float(snapshot.get("ask_proxy")) if int(direction) > 0 else _safe_float(snapshot.get("bid_proxy"))
    return _safe_float(snapshot.get("bid_proxy")) if int(direction) > 0 else _safe_float(snapshot.get("ask_proxy"))


def _trade_fill_snapshot(snapshot: dict[str, Any], direction: int, *, phase: str) -> dict[str, Any]:
    payload = dict(snapshot)
    payload["phase"] = phase
    payload["position_direction"] = int(direction)
    payload["fill_side"] = _fill_side(direction, phase=phase)
    payload["fill_price_source"] = "bar_close_proxy_with_spread_side"
    payload["fill_price"] = _fill_price(snapshot, direction, phase=phase)
    payload["execution_latency_ms"] = None
    return payload


def _runtime_gate_block_reason(gate_status: dict[str, Any], decision_reason: str) -> str | None:
    normalized_reason = str(decision_reason or "").strip().lower()
    if normalized_reason.startswith("alpha gate veto") or normalized_reason.startswith("context blocked"):
        return None
    if bool(gate_status.get("allow_execution", False)):
        return None
    gate_reason = str(gate_status.get("reason", "") or "").strip().lower()
    if gate_reason in {"", "authorized", "authorized_exit", "no signal"}:
        return None
    if gate_reason == "session blocked":
        return "session"
    if gate_reason == "spread too high":
        return "spread_pips_limit"
    if gate_reason == "max position blocked":
        return "max_position_limit"
    if gate_reason == "loss stop blocked":
        return "daily_loss_stop"
    return gate_reason or None


def _decision_block(
    decision_reason: str,
    gate_status: dict[str, Any],
    rule_block_reason: str | None,
) -> tuple[str | None, str | None]:
    normalized_reason = str(decision_reason or "").strip()
    if not normalized_reason:
        return None, None
    lowered = normalized_reason.lower()
    if lowered in {"authorized", "authorized_exit"}:
        return None, None
    if lowered.startswith("alpha gate veto"):
        return "alpha_gate", normalized_reason
    if lowered.startswith("context blocked"):
        return "context_gate", normalized_reason
    runtime_gate_reason = _runtime_gate_block_reason(gate_status, decision_reason)
    if runtime_gate_reason is not None:
        return "runtime_gate", runtime_gate_reason
    if lowered == "no signal":
        return "rule", rule_block_reason or "no_signal"
    return "decision", normalized_reason


def collect_shadow_runtime_context(
    *,
    mt5_module: Any,
    tick_source: Any | None,
    mode: str,
    poll_interval_ms: int,
    audit_dir: str | Path | None,
    max_bars: int | None,
    manifest_paths: list[str] | None = None,
    target_count: int | None = None,
) -> dict[str, Any]:
    terminal_info = None
    account_info = None
    version_info = None
    try:
        if hasattr(mt5_module, "terminal_info"):
            terminal_info = mt5_module.terminal_info()
    except Exception:
        terminal_info = None
    try:
        if hasattr(mt5_module, "account_info"):
            account_info = mt5_module.account_info()
    except Exception:
        account_info = None
    try:
        if hasattr(mt5_module, "version"):
            version_info = mt5_module.version()
    except Exception:
        version_info = None

    context: dict[str, Any] = {
        "mode": mode,
        "poll_interval_ms": int(poll_interval_ms),
        "audit_dir": str(audit_dir) if audit_dir is not None else None,
        "max_bars": int(max_bars) if max_bars is not None else None,
        "mt5": {
            "version": _json_safe(version_info),
            "terminal_info": _json_safe(terminal_info),
            "account_info": _json_safe(account_info),
            "server_utc_offset_hours": _json_safe(getattr(tick_source, "server_utc_offset_hours", None)),
        },
    }
    if manifest_paths is not None:
        context["manifest_paths"] = [str(Path(item)) for item in manifest_paths]
    if target_count is not None:
        context["target_count"] = int(target_count)
    return context


@dataclass
class ShadowAuditRecord:
    timestamp_utc: str
    symbol: str
    ticks_per_bar: int
    manifest_hash: str
    logic_hash: str
    evaluator_hash: str
    signal_direction: int
    action_state: str
    no_trade_reason: str
    spread_pips: float
    session_filter_pass: bool
    risk_filter_pass: bool
    position_state: str
    would_hold: bool
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
    active_position_state: str
    position_side: int
    current_position_direction: int
    position_after: int
    manifest_fingerprint: str
    release_stage: str
    core_features: dict[str, Any]
    full_features: dict[str, Any] | None
    event_index: int = 0
    run_id: str | None = None
    profile_name: str | None = None
    execution_cost_profile_hash: str | None = None
    rule_family: str | None = None
    rule_candidate_signal: int = 0
    block_reason: str | None = None
    block_stage: str | None = None
    rule_block_reason: str | None = None
    runtime_gate_block_reason: str | None = None
    rule_diagnostics: dict[str, Any] | None = None
    gate_details: dict[str, Any] | None = None
    bar_open: float | None = None
    bar_high: float | None = None
    bar_low: float | None = None
    bar_close: float | None = None
    bar_volume: float | None = None
    bar_avg_spread_price: float | None = None
    bar_avg_spread_pips: float | None = None
    bar_time_delta_s: float | None = None
    bar_start_ts_utc: str | None = None
    bar_end_ts_utc: str | None = None
    ma20: float | None = None
    ma50: float | None = None
    price_roll_std: float | None = None
    position_before_state: str | None = None
    position_after_state: str | None = None
    entry_snapshot: dict[str, Any] | None = None
    exit_snapshot: dict[str, Any] | None = None
    context_day_type: str | None = None
    context_event_risk: str | None = None
    context_in_blackout: bool | None = None
    context_blackout_kind: str | None = None
    context_active_event_id: str | None = None
    context_aggressiveness_mode: str | None = None
    context_block_policy: str | None = None
    context_reason_codes: list[str] | None = None


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
        run_meta_path: str | Path | None = None,
        manifest_fingerprint: str = "",
        release_stage: str = "unknown",
        symbol: str = "UNKNOWN",
        ticks_per_bar: int = 0,
        run_id: str | None = None,
        profile_name: str | None = None,
    ) -> None:
        selector_manifest_path = None
        if isinstance(selector, RuleSelector):
            self.selector = selector
            selector_manifest_path = getattr(selector, "manifest_path", None)
        else:
            selector_manifest_path = Path(selector)
            self.selector = RuleSelector(selector)

        self.audit_path = Path(audit_path)
        self.run_meta_path = Path(run_meta_path) if run_meta_path is not None else self.audit_path.parent / "run_meta.json"
        selector_manifest = self.selector.manifest
        self.manifest_path = Path(selector_manifest_path).resolve() if selector_manifest_path else None
        self.manifest_fingerprint = manifest_fingerprint or selector_manifest.manifest_hash
        self.release_stage = release_stage or selector_manifest.release_stage
        resolved_symbol = str(symbol or "").strip().upper()
        if not resolved_symbol or resolved_symbol == "UNKNOWN":
            resolved_symbol = str(selector_manifest.strategy_symbol or "UNKNOWN").strip().upper()
        self.symbol = resolved_symbol
        self.ticks_per_bar = int(
            ticks_per_bar or selector_manifest.ticks_per_bar or selector_manifest.bar_construction_ticks_per_bar or 0
        )
        self.logic_hash = selector_manifest.logic_hash
        self.evaluator_hash = selector_manifest.evaluator_hash
        self.execution_cost_profile = resolve_execution_cost_profile(selector_manifest)
        self.execution_cost_sources = describe_execution_cost_profile(selector_manifest)[1]
        self.execution_cost_profile_hash = compute_execution_cost_profile_hash(self.execution_cost_profile)
        self.summary_json_path = self.audit_path.parent / "shadow_summary.json"
        self.summary_markdown_path = self.audit_path.parent / "shadow_summary.md"
        self.profile_name = profile_name or (self.manifest_path.stem if self.manifest_path is not None else None)
        self.run_id = str(run_id or f"shadow-{uuid4().hex}")
        self.run_started_at_utc = datetime.now(timezone.utc).isoformat()
        self.run_context: dict[str, Any] = {}

        self.position_direction = 0
        self.position_entry_snapshot: dict[str, Any] | None = None
        self.position_entry_event_index: int | None = None
        self.daily_pnl_usd = 0.0
        self.records_written = 0
        self._write_run_meta()

    def update_run_context(self, **payload: Any) -> None:
        for key, value in payload.items():
            if value is None:
                continue
            self.run_context[key] = value
        self._write_run_meta()

    def _write_run_meta(
        self,
        *,
        last_event_ts_utc: str | None = None,
        last_action_state: str | None = None,
        last_block_reason: str | None = None,
    ) -> None:
        manifest = self.selector.manifest
        payload = {
            "audit_schema_version": 3,
            "run_id": self.run_id,
            "run_started_at_utc": self.run_started_at_utc,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "symbol": self.symbol,
            "ticks_per_bar": self.ticks_per_bar,
            "profile_name": self.profile_name,
            "manifest_hash": self.manifest_fingerprint,
            "manifest_path": str(self.manifest_path) if self.manifest_path is not None else None,
            "release_stage": self.release_stage,
            "engine_type": manifest.engine_type,
            "rule_family": manifest.rule_family,
            "entry_rule_version": manifest.entry_rule_version,
            "exit_rule_version": manifest.exit_rule_version,
            "logic_hash": self.logic_hash,
            "evaluator_hash": self.evaluator_hash,
            "feature_schema": list(manifest.feature_schema or []),
            "feature_schema_hash": manifest.feature_schema_hash,
            "dataset_id": manifest.dataset_id,
            "dataset_fingerprint": manifest.dataset_fingerprint,
            "created_from_git_commit": manifest.created_from_git_commit,
            "rule_params": dict(manifest.rule_params or {}),
            "runtime_constraints": dict(manifest.runtime_constraints or {}),
            "cost_model": dict(manifest.cost_model or {}),
            "resolved_execution_cost_profile": dict(self.execution_cost_profile),
            "resolved_execution_cost_profile_hash": self.execution_cost_profile_hash,
            "resolved_execution_cost_sources": dict(self.execution_cost_sources),
            "threshold_policy": dict(manifest.threshold_policy or {}),
            "startup_truth_snapshot": dict(manifest.startup_truth_snapshot or {}),
            "replay_parity_reference": manifest.replay_parity_reference,
            "paths": {
                "events_path": str(self.audit_path),
                "summary_json_path": str(self.summary_json_path),
                "summary_markdown_path": str(self.summary_markdown_path),
                "run_meta_path": str(self.run_meta_path),
            },
            "process": {
                "pid": os.getpid(),
                "ppid": os.getppid(),
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
                "argv": list(sys.argv),
                "cwd": os.getcwd(),
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
            },
            "records_written": int(self.records_written),
            "current_position_direction": int(self.position_direction),
            "current_position_state": _position_state_label(self.position_direction),
            "last_event_ts_utc": last_event_ts_utc,
            "last_action_state": last_action_state,
            "last_block_reason": last_block_reason,
            "runtime_context": dict(self.run_context),
        }
        _write_json(self.run_meta_path, payload)

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
        bar: VolumeBar | None = None,
    ) -> ShadowAuditRecord:
        bar_ts_utc = pd.Timestamp(bar_ts)
        if bar_ts_utc.tzinfo is None:
            bar_ts_utc = bar_ts_utc.tz_localize("UTC")
        else:
            bar_ts_utc = bar_ts_utc.tz_convert("UTC")
        current_hour_utc = int(bar_ts_utc.hour)

        effective_state = dict(self._current_portfolio_state())
        if portfolio_state:
            effective_state.update(portfolio_state)
        current_direction = int(effective_state.get("position_direction", self.position_direction) or 0)
        rule_family = str(self.selector.manifest.rule_family or "")
        rule_diagnostics = diagnose_rule_decision(rule_family, features, dict(self.selector.manifest.rule_params or {}))

        decision = self.selector.decide(
            features=features,
            current_spread_pips=current_spread_pips,
            is_session_open=is_session_open,
            portfolio_state=effective_state,
            current_hour_utc=current_hour_utc,
            bar_ts_utc=bar_ts_utc,
        )
        core_feature_keys = (
            "price_z",
            "price_roll_std",
            "spread_z",
            "time_delta_z",
            "ma20",
            "ma50",
            "ma20_slope",
            "ma50_slope",
            "vol_norm_atr",
            "Open",
            "High",
            "Low",
            "Close",
            "avg_spread",
            "time_delta_s",
        )
        core_features: dict[str, Any] = {key: features.get(key) for key in core_feature_keys if key in features}
        log_full_features = os.environ.get("SHADOW_LOG_FULL_FEATURES", "0").strip() == "1"
        full_features = dict(features) if log_full_features else None
        gate_status = self.selector.gate_status(
            signal=decision.signal,
            current_spread_pips=current_spread_pips,
            is_session_open=is_session_open,
            portfolio_state=effective_state,
            current_hour_utc=current_hour_utc,
        )
        context_daily = dict((decision.context or {}).get("daily", {}) or {}) if isinstance(decision.context, dict) else {}
        context_slice = dict((decision.context or {}).get("slice", {}) or {}) if isinstance(decision.context, dict) else {}

        normalized_signal = 1 if decision.signal > 0 else -1 if decision.signal < 0 else 0
        would_open = bool(
            decision.allow_execution
            and normalized_signal != 0
            and (current_direction == 0 or normalized_signal != current_direction)
        )
        would_close = bool(
            decision.allow_execution
            and current_direction != 0
            and (normalized_signal == 0 or normalized_signal != current_direction)
        )
        would_hold_position = bool(current_direction != 0 and not would_close)
        would_remain_flat = bool(current_direction == 0 and not would_open)

        event_index = self.records_written + 1
        active_state = _position_state_label(current_direction)
        bar_info = _bar_diagnostics(
            bar_ts=bar_ts,
            bar=bar,
            features=features,
            spread_pips=current_spread_pips,
        )
        prior_entry_snapshot = dict(self.position_entry_snapshot) if self.position_entry_snapshot is not None else None
        entry_snapshot = None
        exit_snapshot = None

        if would_close:
            if bar_info["quote_snapshot"] is not None:
                exit_snapshot = _trade_fill_snapshot(bar_info["quote_snapshot"], current_direction, phase="exit")
                exit_snapshot["event_index"] = event_index
                exit_snapshot["direction_closing"] = int(current_direction)
                exit_snapshot["position_before_state"] = active_state
                exit_snapshot["transition_sequence"] = 1 if would_open else 0
                if prior_entry_snapshot is not None:
                    exit_snapshot["opened_at_event_index"] = prior_entry_snapshot.get("event_index")
                    exit_snapshot["opened_at_bar_ts_utc"] = prior_entry_snapshot.get("bar_ts_utc")
                    exit_snapshot["entry_mid_price"] = prior_entry_snapshot.get("mid_price")
                    exit_snapshot["entry_bid_proxy"] = prior_entry_snapshot.get("bid_proxy")
                    exit_snapshot["entry_ask_proxy"] = prior_entry_snapshot.get("ask_proxy")
                    opened_index = prior_entry_snapshot.get("event_index")
                    if isinstance(opened_index, int):
                        exit_snapshot["bars_held"] = int(max(event_index - opened_index, 0))
            self.position_direction = 0
            self.position_entry_snapshot = None
            self.position_entry_event_index = None

        if would_open and self.position_direction == 0:
            self.position_direction = normalized_signal
            if bar_info["quote_snapshot"] is not None:
                entry_snapshot = _trade_fill_snapshot(bar_info["quote_snapshot"], normalized_signal, phase="entry")
                entry_snapshot["event_index"] = event_index
                entry_snapshot["bar_ts_utc"] = bar_info["bar_start_ts_utc"]
                entry_snapshot["direction_opening"] = int(normalized_signal)
                entry_snapshot["transition_sequence"] = 2 if would_close else 1
                self.position_entry_snapshot = dict(entry_snapshot)
                self.position_entry_event_index = event_index

        position_after_state = _position_state_label(self.position_direction)
        action_state = "hold"
        if would_open and would_close:
            action_state = "reverse"
        elif would_open:
            action_state = "open"
        elif would_close:
            action_state = "close"
        elif would_hold_position:
            action_state = "hold"
        elif would_remain_flat:
            action_state = "flat"

        rule_block_reason = str(rule_diagnostics.get("block_reason") or "") or None
        runtime_gate_block_reason = _runtime_gate_block_reason(gate_status, decision.reason)
        block_stage, block_reason = _decision_block(decision.reason, gate_status, rule_block_reason)

        record = ShadowAuditRecord(
            timestamp_utc=_iso_utc(bar_ts),
            symbol=self.symbol,
            ticks_per_bar=self.ticks_per_bar,
            manifest_hash=self.manifest_fingerprint or decision.manifest_id,
            logic_hash=self.logic_hash,
            evaluator_hash=self.evaluator_hash,
            signal_direction=int(normalized_signal),
            action_state=action_state,
            no_trade_reason=decision.reason,
            spread_pips=float(current_spread_pips),
            session_filter_pass=bool(gate_status["session_ok"]),
            risk_filter_pass=bool(gate_status["risk_ok"]),
            position_state=active_state,
            would_hold=would_hold_position,
            bar_ts=bar_info["bar_start_ts_utc"],
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
            current_position_direction=int(current_direction),
            position_after=int(self.position_direction),
            manifest_fingerprint=self.manifest_fingerprint or decision.manifest_id,
            release_stage=self.release_stage,
            core_features=core_features,
            full_features=full_features,
            event_index=event_index,
            run_id=self.run_id,
            profile_name=self.profile_name,
            execution_cost_profile_hash=self.execution_cost_profile_hash,
            rule_family=rule_family or None,
            rule_candidate_signal=int(rule_diagnostics.get("candidate_signal", 0) or 0),
            block_reason=block_reason,
            block_stage=block_stage,
            rule_block_reason=rule_block_reason,
            runtime_gate_block_reason=runtime_gate_block_reason,
            rule_diagnostics=rule_diagnostics,
            gate_details=gate_status,
            bar_open=bar_info["bar_open"],
            bar_high=bar_info["bar_high"],
            bar_low=bar_info["bar_low"],
            bar_close=bar_info["bar_close"],
            bar_volume=bar_info["bar_volume"],
            bar_avg_spread_price=bar_info["bar_avg_spread_price"],
            bar_avg_spread_pips=bar_info["bar_avg_spread_pips"],
            bar_time_delta_s=bar_info["bar_time_delta_s"],
            bar_start_ts_utc=bar_info["bar_start_ts_utc"],
            bar_end_ts_utc=bar_info["bar_end_ts_utc"],
            ma20=_safe_float(features.get("ma20")),
            ma50=_safe_float(features.get("ma50")),
            price_roll_std=_safe_float(features.get("price_roll_std")),
            position_before_state=active_state,
            position_after_state=position_after_state,
            entry_snapshot=entry_snapshot,
            exit_snapshot=exit_snapshot,
            context_day_type=context_daily.get("day_type"),
            context_event_risk=context_daily.get("event_risk"),
            context_in_blackout=context_slice.get("in_blackout"),
            context_blackout_kind=context_slice.get("blackout_kind"),
            context_active_event_id=context_slice.get("active_event_id"),
            context_aggressiveness_mode=context_slice.get("effective_aggressiveness_mode"),
            context_block_policy=context_slice.get("effective_block_policy"),
            context_reason_codes=list(context_slice.get("reason_codes", []) or []),
        )
        _append_jsonl(self.audit_path, asdict(record))
        write_shadow_summary(
            events_path=self.audit_path,
            summary_json_path=self.summary_json_path,
            summary_markdown_path=self.summary_markdown_path,
        )
        self.records_written += 1
        self._write_run_meta(
            last_event_ts_utc=record.timestamp_utc,
            last_action_state=record.action_state,
            last_block_reason=record.block_reason or record.reason,
        )
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
    manifest = load_selector_manifest(
        manifest_path,
        verify_manifest_hash=True,
        strict_manifest_hash=True,
        require_component_hashes=True,
    )

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

    shadow_paths = resolve_shadow_evidence_paths(
        symbol=resolved_symbol,
        manifest_hash=manifest.manifest_hash,
        base_dir=audit_dir,
    )
    shadow_paths.root_dir.mkdir(parents=True, exist_ok=True)
    shadow_lock_handle = _acquire_shadow_lock(shadow_paths.root_dir / "shadow.lock")

    feature_engine = FeatureEngine()
    warmup_frame = _load_warmup_bars(resolved_symbol, resolved_ticks_per_bar)
    feature_engine.warm_up(warmup_frame)

    broker = ShadowBroker(
        manifest_path,
        audit_path=shadow_paths.events_path,
        run_meta_path=shadow_paths.run_meta_path,
        manifest_fingerprint=manifest.manifest_hash,
        release_stage=manifest.release_stage,
        symbol=resolved_symbol,
        ticks_per_bar=resolved_ticks_per_bar,
    )

    bar_builder = VolumeBarBuilder(resolved_ticks_per_bar)
    cursor = TickCursor()
    processed_bars = 0

    _connect_mt5(mt5)
    tick_source = Mt5CursorTickSource(mt5)
    broker.update_run_context(
        **collect_shadow_runtime_context(
            mt5_module=mt5,
            tick_source=tick_source,
            mode="shadow_single",
            poll_interval_ms=poll_interval_ms,
            audit_dir=shadow_paths.root_dir,
            max_bars=max_bars,
            manifest_paths=[str(Path(manifest_path).resolve())],
            target_count=1,
        )
    )
    log.info("MT5 server UTC offset hours=%s", getattr(tick_source, "server_utc_offset_hours", None))
    log.info(
        "Starting shadow simulator symbol=%s ticks_per_bar=%s audit_dir=%s",
        resolved_symbol,
        resolved_ticks_per_bar,
        shadow_paths.root_dir,
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
                        "shadow heartbeat ticks_fetched=0 ticks_in_bar=%s/%s last_tick_utc=%s",
                        ticks_in_bar,
                        resolved_ticks_per_bar,
                        (
                            datetime.fromtimestamp(last_tick_time_msc / 1000.0, tz=timezone.utc).isoformat()
                            if last_tick_time_msc
                            else None
                        ),
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

                record = broker.evaluate(
                    bar_ts=bar.timestamp,
                    bar=bar,
                    features=latest_features,
                    current_spread_pips=spread_pips,
                    is_session_open=_is_forex_session_open(bar.timestamp),
                )
                processed_bars += 1

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
                    return processed_bars

            now_mono = time.monotonic()
            if now_mono - last_heartbeat >= heartbeat_seconds:
                ticks_in_bar = int(getattr(bar_builder.state, "tick_count", 0) or 0)
                log.info(
                    "shadow heartbeat ticks_fetched=%s ticks_in_bar=%s/%s last_tick_utc=%s",
                    len(ticks),
                    ticks_in_bar,
                    resolved_ticks_per_bar,
                    (
                        datetime.fromtimestamp(last_tick_time_msc / 1000.0, tz=timezone.utc).isoformat()
                        if last_tick_time_msc
                        else None
                    ),
                )
                last_heartbeat = now_mono
            time.sleep(max(float(poll_interval_ms), 1.0) / 1000.0)
    finally:
        mt5.shutdown()
        try:
            shadow_lock_handle.close()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the draft MT5 shadow simulator for a rule-selector manifest.")
    parser.add_argument("--manifest", "--manifest-path", dest="manifest", required=True, help="Path to the RC manifest.json file.")
    parser.add_argument("--symbol", help="Override manifest symbol.")
    parser.add_argument("--ticks-per-bar", type=int, help="Override manifest ticks_per_bar.")
    parser.add_argument("--audit-dir", help="Where to write the shadow audit JSONL daily files.")
    parser.add_argument("--poll-interval-ms", type=int, default=250)
    parser.add_argument("--max-bars", type=int, help="Optional max emitted bars before exit.")
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
