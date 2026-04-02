from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


LEGACY_ACTION_SL_MULTS: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
LEGACY_ACTION_TP_MULTS: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
SIMPLE_ACTION_SL_MULTS: tuple[float, ...] = (1.0,)
SIMPLE_ACTION_TP_MULTS: tuple[float, ...] = (1.0,)


def resolve_action_space_mode(default: str = "simple") -> str:
    raw_mode = os.environ.get("TRADING_ACTION_SPACE_MODE", default).strip().lower()
    if raw_mode in {"", "simple"}:
        return "simple"
    if raw_mode == "legacy":
        return "legacy"
    raise ValueError(
        f"Unsupported TRADING_ACTION_SPACE_MODE={raw_mode!r}. Expected 'simple' or 'legacy'."
    )


def resolve_action_sl_tp_options() -> tuple[tuple[float, ...], tuple[float, ...]]:
    mode = resolve_action_space_mode()
    if mode == "legacy":
        return LEGACY_ACTION_SL_MULTS, LEGACY_ACTION_TP_MULTS
    return SIMPLE_ACTION_SL_MULTS, SIMPLE_ACTION_TP_MULTS


ACTION_SL_MULTS, ACTION_TP_MULTS = resolve_action_sl_tp_options()

DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_MIN_LEARNING_RATE = 3e-4
DEFAULT_TARGET_KL = 0.015  # Reverted to original as per user request
DEFAULT_CLIP_RANGE = 0.2
DEFAULT_VF_COEF = 0.5
DEFAULT_MAX_GRAD_NORM = 0.5
DEFAULT_SLIPPAGE_START_PIPS = 0.05
DEFAULT_SLIPPAGE_END_PIPS = 0.5  # Reverted to original as per user request

EXPLAINED_VARIANCE_MIN = 0.30
APPROX_KL_MIN = 0.01
APPROX_KL_MAX = 0.05
VALUE_LOSS_STABILITY_MAX = 5.0

DEPLOY_TIMED_SHARPE_MIN = 0.30
DEPLOY_DD_MAX = 0.30
DEPLOY_PROFIT_FACTOR_MIN = 1.10
DEPLOY_EXPECTANCY_MIN = 0.0

DEFAULT_MAX_DRAWDOWN_FRACTION = 0.15
DEFAULT_DAILY_LOSS_FRACTION = 0.05
DEFAULT_STALE_FEED_MS = 30_000
DEFAULT_MAX_BROKER_FAILURES = 3
DEFAULT_RISK_PER_TRADE_FRACTION = 0.01
DEFAULT_LOT_SIZE_MIN = 0.01
DEFAULT_LOT_SIZE_MAX = 0.10
DEFAULT_BAR_CONSTRUCTION_TICKS_PER_BAR = 2_000

# Phase 4: Churn Reduction Config
DEFAULT_CHURN_MIN_HOLD_BARS = 3
DEFAULT_CHURN_ACTION_COOLDOWN = 2
DEFAULT_CHURN_PENALTY_USD = 0.10
DEFAULT_ENTRY_SPREAD_Z_LIMIT = 1.0

# Phase 5: Reward Redesign Coefficients
DEFAULT_REWARD_DOWNSIDE_RISK_COEF = 0.15
DEFAULT_REWARD_TURNOVER_COEF = 0.05
DEFAULT_REWARD_DRAWDOWN_COEF = 2.0
DEFAULT_REWARD_NET_RETURN_COEF = 1.0


@dataclass(frozen=True)
class DeploymentPaths:
    diagnostics_path: Path
    gate_path: Path
    ops_attestation_path: Path
    live_preflight_path: Path
    execution_audit_path: Path


def deployment_paths(symbol: str, *, model_dir: str | Path = "models") -> DeploymentPaths:
    base = Path(model_dir)
    normalized = symbol.strip().upper() or "EURUSD"
    slug = normalized.lower()
    return DeploymentPaths(
        diagnostics_path=base / f"training_diagnostics_{slug}.json",
        gate_path=base / f"deployment_gate_{slug}.json",
        ops_attestation_path=base / f"ops_attestation_{slug}.json",
        live_preflight_path=base / f"live_preflight_{slug}.json",
        execution_audit_path=base / f"execution_audit_{slug}.jsonl",
    )


def live_enforce_deployment_gate() -> bool:
    return os.environ.get("LIVE_ENFORCE_DEPLOYMENT_GATE", "1") != "0"


def resolve_bar_construction_ticks_per_bar(
    *env_var_names: str,
    default: int = DEFAULT_BAR_CONSTRUCTION_TICKS_PER_BAR,
) -> int:
    for name in env_var_names:
        raw = os.environ.get(name, "").strip()
        if raw:
            return int(raw)
    return int(default)
