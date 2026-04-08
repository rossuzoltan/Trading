from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from selector_manifest import load_selector_manifest, validate_selector_manifest
from strategies.rule_logic import compute_rule_direction

log = logging.getLogger("rule_selector")


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
        self.manifest = load_selector_manifest(self.manifest_path)
        self._validate_manifest()

    def _validate_manifest(self) -> None:
        validate_selector_manifest(
            self.manifest,
            verify_manifest_hash=bool(str(self.manifest.manifest_hash or "").strip()),
        )
        if self.manifest.engine_type != "RULE":
            raise ValueError(f"RuleSelector only supports RULE manifests, got {self.manifest.engine_type}")
        if not self.manifest.rule_family:
            raise ValueError("RULE manifest must specify rule_family")

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

        # Manifest-driven rollover block (shared by shadow, replay, and live paths)
        rollover_hours = list(self.manifest.runtime_constraints.get("rollover_block_utc_hours", [22, 23]))
        in_rollover = current_hour_utc is not None and int(current_hour_utc) in rollover_hours
        session_ok = bool((is_session_open and not in_rollover) or not session_filter_active)

        spread_limit = float(self.manifest.runtime_constraints.get("spread_sanity_max_pips", 999.0))
        spread_ok = float(current_spread_pips) <= spread_limit

        current_positions = int(portfolio_state.get("current_positions", 0) or 0)
        current_direction = int(portfolio_state.get("current_direction", 0) or 0)
        max_positions = int(self.manifest.runtime_constraints.get("max_concurrent_positions", 1))
        position_ok = bool(signal == 0 or current_positions < max_positions or current_direction != 0)

        daily_pnl_usd = float(portfolio_state.get("daily_pnl_usd", 0.0) or 0.0)
        daily_loss_stop_usd = float(self.manifest.runtime_constraints.get("daily_loss_stop_usd", -9999.0))
        daily_loss_ok = bool(daily_pnl_usd > -abs(daily_loss_stop_usd))

        risk_ok = bool(position_ok and daily_loss_ok)
        allow_execution = bool(session_ok and spread_ok and risk_ok)

        if not session_ok:
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
        gate_status = self.gate_status(
            signal=signal,
            current_spread_pips=current_spread_pips,
            is_session_open=is_session_open,
            portfolio_state=portfolio_state,
            current_hour_utc=current_hour_utc,
        )
        return SelectorDecision(
            signal=int(signal),
            allow_execution=bool(gate_status["allow_execution"]),
            reason=str(gate_status["reason"]),
            manifest_id=self.manifest.manifest_hash,
            timestamp_utc=ts,
        )


def load_rule_selector(manifest_path: str | Path) -> RuleSelector:
    return RuleSelector(manifest_path)
