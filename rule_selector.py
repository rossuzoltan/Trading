from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edge_research import BaselineAlphaGate, load_baseline_alpha_gate
from selector_manifest import _file_sha256, load_selector_manifest, validate_selector_manifest
from strategies.rule_logic import compute_rule_direction

log = logging.getLogger("rule_selector")


def _session_bucket(hour_utc: int | None) -> str | None:
    if hour_utc is None:
        return None
    hour = int(hour_utc)
    if 0 <= hour < 7:
        return "Asia"
    if 7 <= hour < 12:
        return "London"
    if 12 <= hour < 17:
        return "London/NY"
    if 17 <= hour < 21:
        return "NY"
    return "Rollover"


@dataclass
class SelectorDecision:
    signal: int
    allow_execution: bool
    reason: str
    manifest_id: str
    timestamp_utc: str


class RuleSelector:
    """
    Deterministic manifest-driven selector for rule-first execution.
    """

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self.manifest = load_selector_manifest(
            self.manifest_path,
            verify_manifest_hash=True,
            strict_manifest_hash=True,
            require_component_hashes=True,
        )
        self.alpha_gate: BaselineAlphaGate | None = None
        self.alpha_gate_threshold_override: float | None = None
        self.alpha_gate_margin_override: float | None = None
        self._validate_manifest()
        self._load_alpha_gate()

    def _validate_manifest(self) -> None:
        validate_selector_manifest(
            self.manifest,
            verify_manifest_hash=True,
            require_component_hashes=True,
        )
        if self.manifest.engine_type != "RULE":
            raise ValueError(f"RuleSelector only supports RULE manifests, got {self.manifest.engine_type}")
        if not self.manifest.rule_family:
            raise ValueError("RULE manifest must specify rule_family")

    def _load_alpha_gate(self) -> None:
        alpha_gate_cfg = dict(self.manifest.alpha_gate or {})
        if not bool(alpha_gate_cfg.get("enabled", False)):
            return

        model_path = Path(str(alpha_gate_cfg.get("model_path") or "").strip())
        if not model_path.exists():
            raise RuntimeError(f"AlphaGate model_path does not exist: {model_path}")
        expected_sha = str(alpha_gate_cfg.get("model_sha256") or "").strip()
        if expected_sha and _file_sha256(model_path) != expected_sha:
            raise RuntimeError("AlphaGate checksum mismatch for selector manifest.")

        gate = load_baseline_alpha_gate(model_path)
        if str(gate.symbol).upper() != self.manifest.strategy_symbol:
            raise RuntimeError(
                f"AlphaGate symbol mismatch: gate={gate.symbol} manifest={self.manifest.strategy_symbol}"
            )
        self.alpha_gate = gate
        if alpha_gate_cfg.get("probability_threshold") is not None:
            self.alpha_gate_threshold_override = float(alpha_gate_cfg["probability_threshold"])
        if alpha_gate_cfg.get("probability_margin") is not None:
            self.alpha_gate_margin_override = float(alpha_gate_cfg["probability_margin"])
        if alpha_gate_cfg.get("min_edge_pips") is not None:
            gate.min_edge_pips = float(alpha_gate_cfg["min_edge_pips"])

    def gate_status(
        self,
        *,
        signal: int,
        current_spread_pips: float,
        is_session_open: bool,
        portfolio_state: dict[str, Any],
        current_hour_utc: int | None = None,
    ) -> dict[str, Any]:
        session_filter_active = bool(self.manifest.runtime_constraints.get("session_filter_active", False))
        allowed_sessions = {
            str(item).strip().lower()
            for item in list(self.manifest.runtime_constraints.get("allowed_sessions", []) or [])
            if str(item).strip()
        }
        session_name = _session_bucket(current_hour_utc)
        current_positions = int(portfolio_state.get("current_positions", 0) or 0)
        current_direction = int(portfolio_state.get("current_direction", 0) or 0)
        has_open_position = bool(current_positions > 0 or current_direction != 0)
        normalized_signal = int(signal or 0)
        exit_intent = bool(has_open_position and (normalized_signal == 0 or normalized_signal != current_direction))
        entry_intent = bool(normalized_signal != 0 and not has_open_position)

        # Manifest-driven rollover block (shared by shadow, replay, and live paths)
        rollover_hours = list(self.manifest.runtime_constraints.get("rollover_block_utc_hours", [22, 23]))
        rollover_hour_known = current_hour_utc is not None
        in_rollover = bool(rollover_hour_known and int(current_hour_utc) in rollover_hours)
        rollover_guard_ready = bool(not rollover_hours or rollover_hour_known)
        session_name_ok = True
        if entry_intent and allowed_sessions:
            session_name_ok = bool(session_name is not None and session_name.lower() in allowed_sessions)
        session_ok = bool(
            (not session_filter_active)
            or (is_session_open and rollover_guard_ready and not in_rollover and session_name_ok)
        )

        spread_limit = float(self.manifest.runtime_constraints.get("spread_sanity_max_pips", 999.0))
        spread_ok = float(current_spread_pips) <= spread_limit

        max_positions = int(self.manifest.runtime_constraints.get("max_concurrent_positions", 1))
        if not entry_intent:
            position_ok = True
        else:
            position_ok = bool(current_positions < max_positions)

        daily_pnl_usd = float(portfolio_state.get("daily_pnl_usd", 0.0) or 0.0)
        daily_loss_stop_usd = float(self.manifest.runtime_constraints.get("daily_loss_stop_usd", -9999.0))
        daily_loss_ok = bool(daily_pnl_usd > -abs(daily_loss_stop_usd))

        risk_ok = bool(position_ok and daily_loss_ok)
        entry_authorized = bool(session_ok and spread_ok and risk_ok and normalized_signal != 0)
        allow_execution = bool(exit_intent or entry_authorized)

        if exit_intent:
            reason = "authorized_exit"
        elif not session_ok:
            reason = "session blocked"
        elif not spread_ok:
            reason = "spread too high"
        elif not position_ok:
            reason = "max position blocked"
        elif not daily_loss_ok:
            reason = "loss stop blocked"
        elif signal == 0:
            reason = "no signal"
        else:
            reason = "authorized"

        return {
            "allow_execution": allow_execution,
            "reason": reason,
            "session_ok": session_ok,
            "spread_ok": spread_ok,
            "risk_ok": risk_ok,
            "position_ok": position_ok,
            "daily_loss_ok": daily_loss_ok,
            "spread_limit_pips": spread_limit,
            "current_positions": current_positions,
            "current_direction": current_direction,
            "max_concurrent_positions": max_positions,
            "in_rollover": in_rollover,
            "rollover_hour_known": rollover_hour_known,
            "has_open_position": has_open_position,
            "session_name": session_name,
            "session_name_ok": session_name_ok,
            "entry_intent": entry_intent,
            "exit_intent": exit_intent,
        }

    def decide(
        self,
        features: dict[str, Any],
        current_spread_pips: float,
        is_session_open: bool,
        portfolio_state: dict[str, Any],
        current_hour_utc: int | None = None,
    ) -> SelectorDecision:
        ts = datetime.now(timezone.utc).isoformat()
        signal = compute_rule_direction(self.manifest.rule_family, features, self.manifest.rule_params)
        alpha_veto_reason = ""
        if signal != 0 and self.alpha_gate is not None:
            allow_long, allow_short, _ = self.alpha_gate.allowed_directions(
                features,
                threshold_override=self.alpha_gate_threshold_override,
                margin_override=self.alpha_gate_margin_override,
            )
            if signal > 0 and not bool(allow_long):
                alpha_veto_reason = "alpha gate veto long"
                signal = 0
            elif signal < 0 and not bool(allow_short):
                alpha_veto_reason = "alpha gate veto short"
                signal = 0

        gate_status = self.gate_status(
            signal=signal,
            current_spread_pips=current_spread_pips,
            is_session_open=is_session_open,
            portfolio_state=portfolio_state,
            current_hour_utc=current_hour_utc,
        )
        reason = alpha_veto_reason or str(gate_status["reason"])
        return SelectorDecision(
            signal=int(signal),
            allow_execution=bool(gate_status["allow_execution"]),
            reason=reason,
            manifest_id=self.manifest.manifest_hash,
            timestamp_utc=ts,
        )


def load_rule_selector(manifest_path: str | Path) -> RuleSelector:
    return RuleSelector(manifest_path)
