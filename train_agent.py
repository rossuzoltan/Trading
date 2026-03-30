"""
train_agent.py  –  Phase 11 Production Training Script
=======================================================
Phase 11 additions over Phase 10
---------------------------------
* Curriculum Learning: CurriculumCallback linearly anneals max_slippage_pips
  from 0.0 -> 2.0 over total_timesteps. The agent first learns to be profitable
  in a zero-friction world, then progressively adapts to real execution costs.
* Purged Walk-Forward Validation: a 200-bar 'gap' is inserted between every
  train/val window to prevent indicator leakage. Multiple folds are used to
  detect strategy decay across time.
* Sortino reward (asymmetric downturn penalty) already in trading_env.py.
* 19-feature FeatureEngine (Hurst, Z-spread, vol_norm_atr, log_return_std).
* Per-symbol StandardScaler fitted on training fold only.
"""

from __future__ import annotations

from pathlib import Path
from interpreter_guard import ensure_project_venv, project_venv_python

ensure_project_venv(project_root=Path(__file__).resolve().parent, script_path=__file__)

import os
import copy
import importlib.util
import json
import logging
import multiprocessing as mp
import shutil
import sys
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Callable

import torch
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None
from sb3_contrib import MaskablePPO
from train.maskable_ppo_amp import MaskablePPO_AMP
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from artifact_manifest import DEFAULT_MANIFEST_NAME, create_manifest, save_manifest
from dataset_validation import validate_symbol_bar_spec
from device_utils import configure_training_runtime
from edge_research import run_edge_baseline_research
from feature_engine import FEATURE_COLS, WARMUP_BARS, _compute_raw, SCALER_PATH
from masking_utils import action_mask_fn
from runtime_common import (
    STATE_FEATURE_COUNT,
    build_action_map,
    build_evaluation_accounting,
    build_simple_action_map,
    build_trade_metric_reconciliation,
    compute_trade_metrics,
    validate_evaluation_accounting,
    validate_evaluation_payload,
)
from project_paths import (
    ensure_runtime_dirs,
    resolve_dataset_path,
    validate_dataset_bar_spec,
    validate_dataset_integrity,
)
from run_logging import configure_run_logging, set_log_context
from trading_env import ForexTradingEnv
from trading_config import (
    ACTION_SL_MULTS,
    ACTION_TP_MULTS,
    DEFAULT_BAR_CONSTRUCTION_TICKS_PER_BAR,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MIN_LEARNING_RATE,
    DEFAULT_SLIPPAGE_END_PIPS,
    DEFAULT_SLIPPAGE_START_PIPS,
    DEFAULT_TARGET_KL,
    DEPLOY_EXPECTANCY_MIN,
    DEPLOY_PROFIT_FACTOR_MIN,
    DEPLOY_TIMED_SHARPE_MIN,
    DEFAULT_CHURN_MIN_HOLD_BARS,
    DEFAULT_CHURN_ACTION_COOLDOWN,
    DEFAULT_CHURN_PENALTY_USD,
    DEFAULT_REWARD_DOWNSIDE_RISK_COEF,
    DEFAULT_REWARD_TURNOVER_COEF,
    DEFAULT_REWARD_DRAWDOWN_COEF,
    DEFAULT_REWARD_NET_RETURN_COEF,
    deployment_paths,
    resolve_bar_construction_ticks_per_bar,
)
from validation_metrics import (
    assess_training_data_sufficiency,
    compute_max_drawdown,
    compute_timed_sharpe,
    save_json_report,
    summarize_training_diagnostics,
    training_data_minimums,
)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

if load_dotenv is not None:
    load_dotenv()

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Curriculum config ─────────────────────────────────────────────────────────
SLIPPAGE_START   = DEFAULT_SLIPPAGE_START_PIPS
SLIPPAGE_END     = DEFAULT_SLIPPAGE_END_PIPS
TOTAL_TIMESTEPS  = int(os.environ.get("TRAIN_TOTAL_TIMESTEPS", "3000000"))
TRAIN_SYMBOL     = os.environ.get("TRAIN_SYMBOL", os.environ.get("TRADING_SYMBOL", "EURUSD")).strip().upper()
TRAIN_ENV_MODE = os.environ.get("TRAIN_ENV_MODE", "runtime").strip().lower()
TRAIN_ACTION_SPACE_MODE = os.environ.get("TRAIN_ACTION_SPACE_MODE", "simple").strip().lower() or "simple"
TRAIN_SIMPLE_ACTION_SL_MULT = float(os.environ.get("TRAIN_SIMPLE_ACTION_SL_MULT", "1.0"))
TRAIN_SIMPLE_ACTION_TP_MULT = float(os.environ.get("TRAIN_SIMPLE_ACTION_TP_MULT", "1.0"))
PPO_LEARNING_RATE = float(os.environ.get("TRAIN_PPO_LEARNING_RATE", str(DEFAULT_LEARNING_RATE)))
PPO_MIN_LEARNING_RATE = float(os.environ.get("TRAIN_PPO_MIN_LEARNING_RATE", str(DEFAULT_MIN_LEARNING_RATE)))
PPO_TARGET_KL = float(os.environ.get("TRAIN_PPO_TARGET_KL", str(DEFAULT_TARGET_KL)))
MIN_EXPLAINED_VARIANCE = float(os.environ.get("TRAIN_MIN_EXPLAINED_VARIANCE", "0.30"))
KL_MIN = float(os.environ.get("TRAIN_KL_MIN", "0.01"))
KL_MAX = float(os.environ.get("TRAIN_KL_MAX", "0.05"))
DEPLOY_DD_MAX = float(os.environ.get("TRAIN_DEPLOY_DD_MAX", "0.30"))
MIN_EVAL_TRADE_COUNT = int(os.environ.get("TRAIN_MIN_EVAL_TRADES", os.environ.get("DEPLOY_MIN_TRADE_COUNT", "20")))
HOLDOUT_FRAC = float(os.environ.get("TRAIN_HOLDOUT_FRAC", "0.15"))
POLICY_WIDTH = int(os.environ.get("TRAIN_POLICY_WIDTH", "128"))
BAR_CONSTRUCTION_TICKS_PER_BAR = resolve_bar_construction_ticks_per_bar(
    "TRAIN_BAR_TICKS",
    "BAR_SPEC_TICKS_PER_BAR",
    "TRAIN_TICKS_PER_BAR",
    "TRADING_TICKS_PER_BAR",
)
TRAIN_POINT_IN_TIME_VERIFIED = os.environ.get("TRAIN_POINT_IN_TIME_VERIFIED", "0") == "1"
TRAIN_DATASET_INTEGRITY_VERIFIED = os.environ.get("TRAIN_DATASET_INTEGRITY_VERIFIED", "0") == "1"
PPO_N_STEPS = int(os.environ.get("TRAIN_PPO_N_STEPS", "2048"))
PPO_BATCH_SIZE = int(os.environ.get("TRAIN_PPO_BATCH_SIZE", "4096"))
PPO_N_EPOCHS = int(os.environ.get("TRAIN_PPO_N_EPOCHS", "10"))
TRAIN_EVAL_FREQ = int(os.environ.get("TRAIN_EVAL_FREQ", "20000"))
TRAIN_LOG_INTERVAL = int(os.environ.get("TRAIN_LOG_INTERVAL", "5"))

TRAIN_TORCH_COMPILE = os.environ.get("TRAIN_TORCH_COMPILE", "0") == "1"
TRAIN_TORCH_COMPILE_MODE = os.environ.get("TRAIN_TORCH_COMPILE_MODE", "default").strip()
TRAIN_REDUCE_LOGGING = os.environ.get("TRAIN_REDUCE_LOGGING", "1") == "1"
TRAIN_ASYNC_EVAL = os.environ.get("TRAIN_ASYNC_EVAL", "0") == "1"
TRAIN_SHARED_DATASET = os.environ.get("TRAIN_SHARED_DATASET", "1") == "1"
TRAIN_USE_AMP = os.environ.get("TRAIN_USE_AMP", "0") == "1"
TRAIN_AMP_DTYPE = os.environ.get("TRAIN_AMP_DTYPE", "bf16").strip().lower()

if TRAIN_REDUCE_LOGGING:
    TRAIN_LOG_INTERVAL = max(TRAIN_LOG_INTERVAL, 20)
if TRAIN_ASYNC_EVAL:
    TRAIN_EVAL_FREQ = max(TRAIN_EVAL_FREQ, 100000)


TRAIN_CHURN_MIN_HOLD_BARS = int(os.environ.get("TRAIN_CHURN_MIN_HOLD_BARS", str(DEFAULT_CHURN_MIN_HOLD_BARS)))
TRAIN_CHURN_ACTION_COOLDOWN = int(os.environ.get("TRAIN_CHURN_ACTION_COOLDOWN", str(DEFAULT_CHURN_ACTION_COOLDOWN)))
TRAIN_CHURN_PENALTY_USD = float(os.environ.get("TRAIN_CHURN_PENALTY_USD", str(DEFAULT_CHURN_PENALTY_USD)))

TRAIN_REWARD_DOWNSIDE_RISK_COEF = float(os.environ.get("TRAIN_REWARD_DOWNSIDE_RISK_COEF", str(DEFAULT_REWARD_DOWNSIDE_RISK_COEF)))
TRAIN_REWARD_TURNOVER_COEF = float(os.environ.get("TRAIN_REWARD_TURNOVER_COEF", str(DEFAULT_REWARD_TURNOVER_COEF)))
TRAIN_REWARD_NET_RETURN_COEF = float(os.environ.get("TRAIN_REWARD_NET_RETURN_COEF", str(DEFAULT_REWARD_NET_RETURN_COEF)))
TRAIN_REWARD_DRAWDOWN_PENALTY = float(os.environ.get("TRAIN_REWARD_DRAWDOWN_PENALTY", str(DEFAULT_REWARD_DRAWDOWN_COEF)))
PPO_ENT_COEF = float(os.environ.get("TRAIN_PPO_ENT_COEF", "0.05"))
TRAIN_DEBUG_ALLOW_BASELINE_BYPASS = os.environ.get("TRAIN_DEBUG_ALLOW_BASELINE_BYPASS", "0") == "1"
TRAIN_LEGACY_REQUIRE_BASELINE_GATE = os.environ.get("TRAIN_REQUIRE_BASELINE_GATE", "1") != "0"
TRAIN_STARTUP_SMOKE_ONLY = os.environ.get("TRAIN_STARTUP_SMOKE_ONLY", "0") == "1"
TRAIN_RESUME_LATEST = os.environ.get("TRAIN_RESUME_LATEST", "0") == "1"
BASELINE_TARGET_HORIZON_BARS = int(os.environ.get("TRAIN_BASELINE_TARGET_HORIZON_BARS", "10"))
BASELINE_R2_MIN = float(os.environ.get("TRAIN_BASELINE_R2_MIN", "0.0"))
BASELINE_CORR_MIN = float(os.environ.get("TRAIN_BASELINE_CORR_MIN", "0.05"))
BASELINE_SIGN_ACC_MIN = float(os.environ.get("TRAIN_BASELINE_SIGN_ACC_MIN", "0.52"))
BASELINE_TREE_MAX_DEPTH = int(os.environ.get("TRAIN_BASELINE_TREE_MAX_DEPTH", "3"))
BASELINE_TREE_MAX_ITER = int(os.environ.get("TRAIN_BASELINE_TREE_MAX_ITER", "100"))
BASELINE_MIN_EDGE_PIPS = float(os.environ.get("TRAIN_BASELINE_MIN_EDGE_PIPS", "0.0"))
BASELINE_PROB_THRESHOLD = float(os.environ.get("TRAIN_BASELINE_PROB_THRESHOLD", "0.55"))
BASELINE_PROB_MARGIN = float(os.environ.get("TRAIN_BASELINE_PROB_MARGIN", "0.05"))
TRAIN_COMMISSION_PER_LOT = float(os.environ.get("TRAIN_COMMISSION_PER_LOT", "7.0"))
TRAIN_PARTIAL_FILL_RATIO = float(os.environ.get("TRAIN_PARTIAL_FILL_RATIO", "1.0"))
TRAIN_REWARD_SCALE = float(os.environ.get("TRAIN_REWARD_SCALE", "10000.0"))
TRAIN_REWARD_DRAWDOWN_PENALTY = TRAIN_REWARD_DRAWDOWN_PENALTY  # Consistently using value above
TRAIN_REWARD_TRANSACTION_PENALTY = float(os.environ.get("TRAIN_REWARD_TRANSACTION_PENALTY", "1.0"))
TRAIN_REWARD_CLIP_LOW = float(os.environ.get("TRAIN_REWARD_CLIP_LOW", "-5.0"))
TRAIN_REWARD_CLIP_HIGH = float(os.environ.get("TRAIN_REWARD_CLIP_HIGH", "5.0"))
HEARTBEAT_SCHEMA_VERSION = 2
PROCESS_STARTED_UTC = datetime.now(timezone.utc).isoformat()
TRAINING_STAGE = os.environ.get("TRAINING_STAGE", "stage_a_unlock").strip().lower() or "stage_a_unlock"

TRAINING_RECOVERY_CONFIG: dict[str, Any] = {
    "training_stage": TRAINING_STAGE,
    "slippage_curriculum": {
        "enabled": os.environ.get("TRAIN_RECOVERY_SLIPPAGE_ENABLED", "1") != "0",
        "mode": os.environ.get("TRAIN_RECOVERY_SLIPPAGE_MODE", "staircase").strip().lower() or "staircase",
        "phases": [
            {
                "until_step": int(os.environ.get("TRAIN_RECOVERY_PHASE_1_UNTIL", "750000")),
                "slippage_pips": float(os.environ.get("TRAIN_RECOVERY_PHASE_1_SLIPPAGE_PIPS", "0.05")),
            },
            {
                "until_step": int(os.environ.get("TRAIN_RECOVERY_PHASE_2_UNTIL", "1750000")),
                "slippage_pips": float(os.environ.get("TRAIN_RECOVERY_PHASE_2_SLIPPAGE_PIPS", "0.5")),
            },
            {
                "until_step": int(os.environ.get("TRAIN_RECOVERY_PHASE_3_UNTIL", str(TOTAL_TIMESTEPS))),
                "slippage_pips": float(os.environ.get("TRAIN_RECOVERY_PHASE_3_SLIPPAGE_PIPS", "1.0")),
            },
        ],
        "linear_start_pips": float(os.environ.get("TRAIN_RECOVERY_LINEAR_START_PIPS", "0.1")),
        "linear_end_pips": float(os.environ.get("TRAIN_RECOVERY_LINEAR_END_PIPS", "1.0")),
        "default_slippage_pips": float(
            os.environ.get("TRAIN_RECOVERY_DEFAULT_SLIPPAGE_PIPS", str(DEFAULT_SLIPPAGE_END_PIPS))
        ),
    },
    "participation_bonus": {
        "enabled": os.environ.get("TRAIN_RECOVERY_PARTICIPATION_BONUS_ENABLED", "1") != "0",
        "bonus_value": float(os.environ.get("TRAIN_RECOVERY_PARTICIPATION_BONUS_VALUE", "0.01")),
        "active_until_step": int(os.environ.get("TRAIN_RECOVERY_PARTICIPATION_BONUS_UNTIL", "500000")),
        "cooldown_steps": int(os.environ.get("TRAIN_RECOVERY_PARTICIPATION_BONUS_COOLDOWN", "8")),
        "only_from_flat": os.environ.get("TRAIN_RECOVERY_PARTICIPATION_ONLY_FROM_FLAT", "1") != "0",
        "max_bonus_per_episode": int(os.environ.get("TRAIN_RECOVERY_PARTICIPATION_MAX_PER_EPISODE", "50")),
    },
    "entropy_schedule": {
        "enabled": os.environ.get("TRAIN_RECOVERY_ENTROPY_SCHEDULE_ENABLED", "1") != "0",
        "initial_ent_coef": float(os.environ.get("TRAIN_RECOVERY_ENTROPY_INITIAL", "0.05")),
        "mid_ent_coef": float(os.environ.get("TRAIN_RECOVERY_ENTROPY_MID", "0.01")),
        "final_ent_coef": float(os.environ.get("TRAIN_RECOVERY_ENTROPY_FINAL", "0.001")),
        "phase_1_until": int(os.environ.get("TRAIN_RECOVERY_ENTROPY_PHASE_1_UNTIL", "750000")),
        "phase_2_until": int(os.environ.get("TRAIN_RECOVERY_ENTROPY_PHASE_2_UNTIL", "1750000")),
    },
    "diagnostics": {
        "log_action_distribution": os.environ.get("TRAIN_RECOVERY_LOG_ACTION_DISTRIBUTION", "1") != "0",
        "log_reward_components": os.environ.get("TRAIN_RECOVERY_LOG_REWARD_COMPONENTS", "1") != "0",
        "heartbeat_every_n_updates": int(os.environ.get("TRAIN_RECOVERY_HEARTBEAT_EVERY_N_UPDATES", "10")),
    },
}

# Diagnostic guideposts for the Stage A unlock path. These are intended for
# heartbeat interpretation and post-run review, not as hard stop conditions.
STAGE_A_RECOVERY_TARGETS: dict[str, dict[str, str]] = {
    "200k_steps": {
        "trade_attempt_count": "> 0",
        "entry_count": "> 20",
        "hold_fraction": "< 0.995",
        "participation_bonus_sum": "> 0",
        "explained_variance": "should stop deteriorating",
    },
    "500k_steps": {
        "closed_trade_count": "> 50",
        "hold_fraction": "< 0.98",
        "action_distribution": "should not be fully degenerate",
        "explained_variance": "should show upward trend",
        "participation_bonus_dependency": "reward should not be explained purely by participation bonus",
    },
    "1m_steps": {
        "participation_bonus": "expired or near expiry",
        "activity": "should persist after bonus fades",
        "explained_variance": "should move toward 0.25-0.30",
        "eval_trade_activity": "should be meaningful",
    },
}

# ── Purged walk-forward config ────────────────────────────────────────────────
PURGE_GAP_BARS   = int(os.environ.get("TRAIN_PURGE_GAP_BARS", "200"))
N_FOLDS          = int(os.environ.get("TRAIN_N_FOLDS", "3"))
FOLD_TEST_FRAC   = float(os.environ.get("TRAIN_FOLD_TEST_FRAC", "0.10"))
BEST_VECNORMALIZE_NAME = "best_vecnormalize.pkl"
RESUME_MODEL_NAME = "resume_model.zip"
RESUME_VECNORMALIZE_NAME = "resume_vecnormalize.pkl"
CURRENT_TRAINING_RUN_PATH = Path("checkpoints") / "current_training_run.json"
log = logging.getLogger("train_agent")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def build_execution_cost_profile(*, slippage_pips: float = SLIPPAGE_END) -> dict[str, float | str]:
    return {
        "commission_per_lot": float(TRAIN_COMMISSION_PER_LOT),
        "slippage_pips": float(slippage_pips),
        "partial_fill_ratio": float(TRAIN_PARTIAL_FILL_RATIO),
        "spread_model": "bar_avg_spread_half_side",
        "mark_to_liquidation": True,
    }


def build_reward_profile() -> dict[str, float]:
    return {
        "reward_scale": float(TRAIN_REWARD_SCALE),
        "drawdown_penalty": float(TRAIN_REWARD_DRAWDOWN_PENALTY),
        "transaction_penalty": float(TRAIN_REWARD_TRANSACTION_PENALTY),
        "reward_clip_low": float(TRAIN_REWARD_CLIP_LOW),
        "reward_clip_high": float(TRAIN_REWARD_CLIP_HIGH),
    }


def build_runtime_action_map(
    sl_opts: list[float] | tuple[float, ...] | None = None,
    tp_opts: list[float] | tuple[float, ...] | None = None,
) -> tuple[Any, ...]:
    if TRAIN_ACTION_SPACE_MODE == "simple":
        return build_simple_action_map(
            sl_value=float(TRAIN_SIMPLE_ACTION_SL_MULT),
            tp_value=float(TRAIN_SIMPLE_ACTION_TP_MULT),
        )
    if TRAIN_ACTION_SPACE_MODE == "legacy":
        return build_action_map(list(sl_opts or ACTION_SL_MULTS), list(tp_opts or ACTION_TP_MULTS))
    raise ValueError(f"Unsupported TRAIN_ACTION_SPACE_MODE={TRAIN_ACTION_SPACE_MODE!r}")


def get_current_slippage_pips(global_step: int, cfg: dict[str, Any]) -> float:
    scfg = cfg.get("slippage_curriculum", {}) or {}
    if not bool(scfg.get("enabled", False)):
        return float(scfg.get("default_slippage_pips", DEFAULT_SLIPPAGE_END_PIPS))

    phases = list(scfg.get("phases", []) or [])
    if not phases:
        raise ValueError("TRAINING_RECOVERY_CONFIG.slippage_curriculum.phases must not be empty.")

    mode = str(scfg.get("mode", "staircase")).strip().lower()
    if mode == "staircase":
        for phase in phases:
            if int(global_step) <= int(phase["until_step"]):
                return float(phase["slippage_pips"])
        return float(phases[-1]["slippage_pips"])
    if mode == "linear":
        total_end = max(int(phases[-1]["until_step"]), 1)
        progress = min(max(float(global_step) / float(total_end), 0.0), 1.0)
        start = float(scfg.get("linear_start_pips", DEFAULT_SLIPPAGE_START_PIPS))
        end = float(scfg.get("linear_end_pips", DEFAULT_SLIPPAGE_END_PIPS))
        return float(start + progress * (end - start))
    raise ValueError(f"Unknown slippage curriculum mode: {mode}")


def get_current_phase(global_step: int, cfg: dict[str, Any]) -> int:
    phases = list((cfg.get("slippage_curriculum", {}) or {}).get("phases", []) or [])
    if not phases:
        return 0
    for idx, phase in enumerate(phases, start=1):
        if int(global_step) <= int(phase["until_step"]):
            return idx
    return len(phases)


def get_current_ent_coef(global_step: int, cfg: dict[str, Any]) -> float:
    ecfg = cfg.get("entropy_schedule", {}) or {}
    if not bool(ecfg.get("enabled", False)):
        return float(ecfg.get("final_ent_coef", PPO_ENT_COEF))
    if int(global_step) <= int(ecfg.get("phase_1_until", 0)):
        return float(ecfg.get("initial_ent_coef", PPO_ENT_COEF))
    if int(global_step) <= int(ecfg.get("phase_2_until", 0)):
        return float(ecfg.get("mid_ent_coef", PPO_ENT_COEF))
    return float(ecfg.get("final_ent_coef", PPO_ENT_COEF))


def get_final_slippage_pips(cfg: dict[str, Any]) -> float:
    scfg = cfg.get("slippage_curriculum", {}) or {}
    phases = list(scfg.get("phases", []) or [])
    if phases:
        return float(phases[-1]["slippage_pips"])
    return float(scfg.get("default_slippage_pips", DEFAULT_SLIPPAGE_END_PIPS))


def is_participation_bonus_active(global_step: int, cfg: dict[str, Any]) -> bool:
    pcfg = cfg.get("participation_bonus", {}) or {}
    return bool(pcfg.get("enabled", False)) and int(global_step) <= int(pcfg.get("active_until_step", 0))


def build_train_env_recovery_config(base_cfg: dict[str, Any], *, env_workers: int) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    pcfg = cfg.get("participation_bonus", {}) or {}
    if bool(pcfg.get("enabled", False)) and env_workers > 0:
        pcfg["bonus_value"] = float(pcfg.get("bonus_value", 0.0)) / float(env_workers)
    cfg["participation_bonus"] = pcfg
    return cfg


def aggregate_training_diagnostics(env_snapshots: list[dict[str, Any]] | None) -> dict[str, Any]:
    action_counts = {"hold": 0, "close": 0, "long": 0, "short": 0}
    trade_totals = {
        "action_selected_count": 0,
        "action_accepted_count": 0,
        "accepted_open_count": 0,
        "accepted_close_count": 0,
        "order_executed_count": 0,
        "executed_open_count": 0,
        "executed_close_count": 0,
        "entered_long_count": 0,
        "entered_short_count": 0,
        "entry_signal_long_count": 0,
        "entry_signal_short_count": 0,
        "closed_trade_count": 0,
        "trade_attempt_count": 0,
        "trade_reject_count": 0,
        "forced_close_count": 0,
        "flat_steps": 0,
        "long_steps": 0,
        "short_steps": 0,
        "position_duration_sum": 0.0,
        "position_duration_count": 0,
        "rapid_reversals": 0,
    }
    economic_totals = {
        "gross_pnl_usd": 0.0,
        "net_pnl_usd": 0.0,
        "transaction_cost_usd": 0.0,
        "commission_usd": 0.0,
        "spread_slippage_cost_usd": 0.0,
        "spread_cost_usd": 0.0,
        "slippage_cost_usd": 0.0,
    }
    reward_totals = {
        "pnl_reward_sum": 0.0,
        "slippage_penalty_sum": 0.0,
        "participation_bonus_sum": 0.0,
        "holding_penalty_sum": 0.0,
        "drawdown_penalty_sum": 0.0,
        "net_reward_sum": 0.0,
    }
    total_steps = 0
    duration_samples: list[float] = []

    for snapshot in env_snapshots or []:
        if not isinstance(snapshot, dict):
            continue
        total_steps += int(snapshot.get("total_steps", 0))
        for key in action_counts:
            action_counts[key] += int((snapshot.get("action_counts", {}) or {}).get(key, 0))
        trade_stats = snapshot.get("trade_stats", {}) or {}
        for key in trade_totals:
            trade_totals[key] += float(trade_stats.get(key, 0)) if key == "position_duration_sum" else int(trade_stats.get(key, 0))
        duration_samples.extend(float(value) for value in list(trade_stats.get("position_durations_sample", []) or []))
        for key in economic_totals:
            economic_totals[key] += float((snapshot.get("economics", {}) or {}).get(key, 0.0))
        for key in reward_totals:
            reward_totals[key] += float((snapshot.get("reward_components", {}) or {}).get(key, 0.0))

    total_actions = max(sum(action_counts.values()), 1)
    occupancy_steps = max(
        int(trade_totals["flat_steps"]) + int(trade_totals["long_steps"]) + int(trade_totals["short_steps"]),
        1,
    )
    total_entries = int(trade_totals["entered_long_count"]) + int(trade_totals["entered_short_count"])
    closed_trade_count = int(trade_totals["closed_trade_count"])
    avg_position_duration = (
        float(trade_totals["position_duration_sum"]) / float(trade_totals["position_duration_count"])
        if int(trade_totals["position_duration_count"]) > 0
        else 0.0
    )
    median_position_duration = float(np.median(duration_samples)) if duration_samples else 0.0

    return {
        "total_steps": int(total_steps),
        "action_distribution": {
            "hold": int(action_counts["hold"]),
            "close": int(action_counts["close"]),
            "long": int(action_counts["long"]),
            "short": int(action_counts["short"]),
            "hold_fraction": float(action_counts["hold"]) / float(total_actions),
            "close_fraction": float(action_counts["close"]) / float(total_actions),
            "long_fraction": float(action_counts["long"]) / float(total_actions),
            "short_fraction": float(action_counts["short"]) / float(total_actions),
        },
        "trade_diagnostics": {
            "action_selected_count": int(trade_totals["action_selected_count"]),
            "action_accepted_count": int(trade_totals["action_accepted_count"]),
            "accepted_open_count": int(trade_totals["accepted_open_count"]),
            "accepted_close_count": int(trade_totals["accepted_close_count"]),
            "order_executed_count": int(trade_totals["order_executed_count"]),
            "executed_open_count": int(trade_totals["executed_open_count"]),
            "executed_close_count": int(trade_totals["executed_close_count"]),
            "entered_long_count": int(trade_totals["entered_long_count"]),
            "entered_short_count": int(trade_totals["entered_short_count"]),
            "entry_signal_long_count": int(trade_totals["entry_signal_long_count"]),
            "entry_signal_short_count": int(trade_totals["entry_signal_short_count"]),
            "closed_trade_count": int(trade_totals["closed_trade_count"]),
            "trade_attempt_count": int(trade_totals["trade_attempt_count"]),
            "trade_reject_count": int(trade_totals["trade_reject_count"]),
            "forced_close_count": int(trade_totals["forced_close_count"]),
            "avg_position_duration": float(avg_position_duration),
            "median_position_duration": float(median_position_duration),
            "avg_trades_per_1000_steps": (1000.0 * float(total_entries) / float(total_steps)) if total_steps else 0.0,
            "churn_ratio": (
                float(trade_totals["rapid_reversals"]) / float(max(closed_trade_count, 1))
                if closed_trade_count >= 0
                else 0.0
            ),
            "flat_fraction": float(trade_totals["flat_steps"]) / float(occupancy_steps),
            "long_fraction": float(trade_totals["long_steps"]) / float(occupancy_steps),
            "short_fraction": float(trade_totals["short_steps"]) / float(occupancy_steps),
        },
        "economics": {key: float(value) for key, value in economic_totals.items()},
        "reward_components": {key: float(value) for key, value in reward_totals.items()},
    }


def resolve_train_vec_env_type(
    *,
    requested_envs: int | None,
    effective_envs: int,
    force_dummy: bool,
) -> tuple[str, list[str]]:
    warnings: list[str] = []
    is_windows = sys.platform == "win32"

    if force_dummy:
        warnings.append("TRAIN_FORCE_DUMMY_VEC=1 forces DummyVecEnv for training; use this only for debug or profiling.")
        vec_env_type = "dummy"
    elif is_windows and requested_envs == 1:
        warnings.append("Windows detected: using DummyVecEnv for single-worker training.")
        vec_env_type = "dummy"
    elif is_windows:
        warnings.append("Windows detected: using SubprocVecEnv (experimental stability).")
        vec_env_type = "subproc"
    elif requested_envs == 1:
        warnings.append("TRAIN_NUM_ENVS=1 disables parallel experience collection; use this only for debug or profiling.")
        vec_env_type = "dummy"
    else:
        vec_env_type = "subproc" if effective_envs > 1 else "dummy"

    if vec_env_type == "dummy" and effective_envs > 1:
        warnings.append(f"DummyVecEnv will run {effective_envs} environments sequentially.")

    if vec_env_type == "dummy" and effective_envs <= 1:
        warnings.append("Training is running with a single environment worker; expect lower throughput and noisier PPO updates.")

    return vec_env_type, warnings


# ── Curriculum Learning Callback ──────────────────────────────────────────────

class LegacyCurriculumCallback(BaseCallback):
    """
    Linearly anneals max_slippage_pips in the training environments
    from SLIPPAGE_START to SLIPPAGE_END over total_timesteps.

    Phase 1 (0–33%)  : 0.0 pip slippage — agent learns basic strategy
    Phase 2 (33–66%) : 1.0 pip slippage — adapts to execution friction
    Phase 3 (66–100%): 2.0 pip slippage — fully realistic execution
    """

    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self._last_logged = -1

    def _current_slippage(self) -> float:
        progress = min(self.num_timesteps / self.total_timesteps, 1.0)
        return SLIPPAGE_START + progress * (SLIPPAGE_END - SLIPPAGE_START)

    def _on_step(self) -> bool:
        slip = self._current_slippage()
        try:
            self.training_env.set_attr("max_slippage_pips", slip)
        except Exception:
            pass

        # Log every 50k steps
        milestone = self.num_timesteps // 50_000
        if milestone != self._last_logged:
            self._last_logged = milestone
            if self.verbose:
                print(f"[Curriculum] Step {self.num_timesteps:,}  slippage={slip:.2f} pips")
        return True


class CurriculumCallback(BaseCallback):
    LOG_EVERY_STEPS = 50_000

    def __init__(
        self,
        recovery_cfg: dict[str, Any],
        *,
        train_env=None,
        eval_envs: list[Any] | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.recovery_cfg = copy.deepcopy(recovery_cfg)
        self.train_env = train_env
        self.eval_envs = list(eval_envs or [])
        self.last_slippage: float | None = None
        self.last_ent_coef: float | None = None
        self._last_logged_step = -1

    def _iter_target_envs(self) -> list[Any]:
        envs: list[Any] = []
        if self.train_env is not None:
            envs.append(self.train_env)
        elif getattr(self, "training_env", None) is not None:
            envs.append(self.training_env)
        envs.extend(env for env in self.eval_envs if env is not None)
        unique: list[Any] = []
        seen_ids: set[int] = set()
        for env in envs:
            env_id = id(env)
            if env_id not in seen_ids:
                unique.append(env)
                seen_ids.add(env_id)
        return unique

    def _set_env_method(self, env, method_name: str, value: Any) -> None:
        try:
            env.env_method(method_name, value)
            return
        except Exception:
            pass
        if method_name == "set_slippage_pips":
            try:
                env.set_attr("max_slippage_pips", value)
            except Exception:
                pass

    def _apply_curriculum_state(self) -> None:
        step = int(self.num_timesteps)
        current_slippage = get_current_slippage_pips(step, self.recovery_cfg)
        current_ent_coef = get_current_ent_coef(step, self.recovery_cfg)

        targets = self._iter_target_envs()
        for env in targets:
            self._set_env_method(env, "set_global_step", step)
        if self.last_slippage != current_slippage:
            for env in targets:
                self._set_env_method(env, "set_slippage_pips", current_slippage)
            self.last_slippage = float(current_slippage)

        # MaskablePPO.train() reads self.ent_coef directly on each update in the
        # installed sb3_contrib version, so updating the scalar here affects the
        # subsequent PPO update cycle without changing the optimizer setup.
        if self.last_ent_coef != current_ent_coef:
            self.model.ent_coef = float(current_ent_coef)
            self.last_ent_coef = float(current_ent_coef)

    def snapshot(self) -> dict[str, Any]:
        step = int(self.num_timesteps)
        return {
            "slippage_mode": str((self.recovery_cfg.get("slippage_curriculum", {}) or {}).get("mode", "staircase")),
            "current_slippage_pips": float(
                self.last_slippage if self.last_slippage is not None else get_current_slippage_pips(step, self.recovery_cfg)
            ),
            "current_phase": int(get_current_phase(step, self.recovery_cfg)),
            "entropy_coef": float(
                self.last_ent_coef if self.last_ent_coef is not None else get_current_ent_coef(step, self.recovery_cfg)
            ),
            "participation_bonus_enabled": bool(
                (self.recovery_cfg.get("participation_bonus", {}) or {}).get("enabled", False)
            ),
            "participation_bonus_active": bool(is_participation_bonus_active(step, self.recovery_cfg)),
        }

    def _on_training_start(self) -> None:
        self._apply_curriculum_state()

    def _on_step(self) -> bool:
        self._apply_curriculum_state()
        if self.verbose and (
            self._last_logged_step < 0 or (self.num_timesteps - self._last_logged_step) >= self.LOG_EVERY_STEPS
        ):
            self._last_logged_step = int(self.num_timesteps)
            snapshot = self.snapshot()
            print(
                f"[Curriculum] Step {self.num_timesteps:,} | "
                f"phase={snapshot['current_phase']} | "
                f"slippage={snapshot['current_slippage_pips']:.2f} pips | "
                f"ent_coef={snapshot['entropy_coef']:.4f}"
            )
        return True


class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        vecnormalize = self.model.get_vec_normalize_env()
        if vecnormalize is not None:
            Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
            vecnormalize.save(self.save_path)
            if self.verbose:
                print(f"[VecNormalize] Saved stats -> {self.save_path}")
        return True


class FullPathEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        *,
        train_vecnormalize: VecNormalize,
        eval_vecnormalize: VecNormalize,
        best_model_save_path: str | Path,
        best_vecnormalize_path: str | Path | None = None,
        eval_freq: int = 10_000,
        metric_key: str = "timed_sharpe",
        history_path: str | Path | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self._train_vecnormalize = train_vecnormalize
        self._eval_vecnormalize = eval_vecnormalize
        self.best_model_save_path = Path(best_model_save_path)
        self.best_vecnormalize_path = Path(best_vecnormalize_path) if best_vecnormalize_path else None
        self.eval_freq = max(int(eval_freq), 0)
        self.metric_key = str(metric_key)
        self.history_path = Path(history_path) if history_path else None
        self.best_metric = -float("inf")
        self.latest_metrics: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            sync_vecnormalize_stats(self._train_vecnormalize, self._eval_vecnormalize)
            _, metrics = evaluate_model(self.model, self.eval_env)
            # Add metadata before serialization
            metrics = {
                **metrics,
                "num_timesteps": int(self.num_timesteps),
                "path_runs": 1,
                "deterministic": True,
                "full_path_eval_used": True,
            }

            # Strategy: Only append/save if accounting reconciliation passed
            try:
                validate_evaluation_payload(metrics)
                self.latest_metrics = metrics
                self.history.append(metrics)
            except Exception as e:
                log.warning(f"FullPathEvalCallback: Skipping serialization of failed evaluation at step {self.num_timesteps}: {e}")
                return True

            if self.history_path is not None:
                self.history_path.parent.mkdir(parents=True, exist_ok=True)
                self.history_path.write_text(
                    json.dumps({"evaluations": self.history}, indent=2, default=_json_default),
                    encoding="utf-8",
                )

            current_metric = float(metrics.get(self.metric_key, -float("inf")))
            if np.isfinite(current_metric) and current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_model_save_path.mkdir(parents=True, exist_ok=True)
                self.model.save(str((self.best_model_save_path / "best_model").with_suffix("")))
                if self.best_vecnormalize_path is not None:
                    self.best_vecnormalize_path.parent.mkdir(parents=True, exist_ok=True)
                    self._train_vecnormalize.save(str(self.best_vecnormalize_path))

            if self.verbose:
                print(
                    f"[Eval] Step {self.num_timesteps:,} | "
                    f"timed_sharpe={float(metrics.get('timed_sharpe', 0.0)):.3f} | "
                    f"final_equity={float(metrics.get('final_equity', 0.0)):.2f} | "
                    f"max_drawdown={float(metrics.get('max_drawdown', 0.0)):.1%}"
                )
        return True


class TrainingDiagnosticsCallback(BaseCallback):
    def __init__(self, verbose: int = 0, *, print_every_steps: int | None = None):
        super().__init__(verbose)
        self.metrics: dict[str, list[float]] = {
            "train/approx_kl": [],
            "train/explained_variance": [],
            "train/value_loss": [],
        }
        self._last_update_index: int | None = None
        self._last_metric_snapshot: tuple[tuple[str, float], ...] | None = None
        if print_every_steps is None:
            print_every_steps = int(os.environ.get("TRAIN_PROGRESS_EVERY_STEPS", "50000"))
        self._print_every_steps = max(int(print_every_steps), 0)
        self._next_print = self._print_every_steps

    def _on_step(self) -> bool:
        logger_values = getattr(self.model.logger, "name_to_value", {})
        current_values: dict[str, float] = {}
        for key in self.metrics:
            value = logger_values.get(key)
            if value is None:
                continue
            try:
                current_values[key] = float(value)
            except (TypeError, ValueError):
                continue
        update_index: int | None = None
        raw_update_index = logger_values.get("train/n_updates")
        if raw_update_index is not None:
            try:
                update_index = int(raw_update_index)
            except (TypeError, ValueError):
                update_index = None
        metric_snapshot = tuple(sorted(current_values.items()))
        duplicate_update = update_index is not None and update_index == self._last_update_index
        duplicate_snapshot = bool(metric_snapshot) and metric_snapshot == self._last_metric_snapshot
        if current_values and not duplicate_update and not duplicate_snapshot:
            for key, value in current_values.items():
                self.metrics[key].append(value)
            self._last_update_index = update_index
            self._last_metric_snapshot = metric_snapshot
        
        should_print = self.verbose and self._print_every_steps and self.num_timesteps >= self._next_print
        if TRAIN_REDUCE_LOGGING and should_print:
            # Print only every 5x steps if reduced logging is enabled to save I/O overhead
            should_print = self.num_timesteps >= self._next_print * 5
            
        if should_print:
            self._next_print += self._print_every_steps
            summary = self.summary()
            print(
                f"[PPO] Step {self.num_timesteps:,} | "
                f"explained_variance={summary['explained_variance']:.3f} | "
                f"approx_kl={summary['approx_kl']:.3f} | "
                f"value_loss_mean(last10)={summary['value_loss_mean_last10']:.3f}"
            )
        return True

    def summary(self) -> dict[str, float | bool]:
        approx_kl = self.metrics["train/approx_kl"]
        explained_variance = self.metrics["train/explained_variance"]
        value_loss = self.metrics["train/value_loss"]
        last_kl = approx_kl[-1] if approx_kl else float("nan")
        last_ev = explained_variance[-1] if explained_variance else float("nan")
        recent_value_loss = value_loss[-10:] if value_loss else []
        value_loss_mean = float(np.mean(recent_value_loss)) if recent_value_loss else float("nan")
        value_loss_std = float(np.std(recent_value_loss)) if recent_value_loss else float("nan")
        diagnostic_sample_count = int(max(len(approx_kl), len(explained_variance), len(value_loss)))
        n_updates = int(self._last_update_index) if self._last_update_index is not None else None
        metrics_fresh = bool(diagnostic_sample_count > 0 and n_updates is not None)
        value_loss_stable = bool(
            bool(recent_value_loss) and np.isfinite(value_loss_mean) and np.isfinite(value_loss_std)
        )
        passes = bool(
            np.isfinite(last_kl)
            and np.isfinite(last_ev)
            and value_loss_stable
            and MIN_EXPLAINED_VARIANCE <= float(last_ev)
            and KL_MIN <= float(last_kl) <= KL_MAX
        )
        blockers: list[str] = []
        if not np.isfinite(last_ev) or float(last_ev) < MIN_EXPLAINED_VARIANCE:
            blockers.append(f"explained_variance below {MIN_EXPLAINED_VARIANCE:.2f}")
        if not np.isfinite(last_kl) or not (KL_MIN <= float(last_kl) <= KL_MAX):
            blockers.append(f"approx_kl outside [{KL_MIN:.2f}, {KL_MAX:.2f}]")
        if not value_loss_stable:
            blockers.append("value_loss is not stable")
        return {
            "approx_kl": float(last_kl),
            "explained_variance": float(last_ev),
            "value_loss_mean_last10": value_loss_mean,
            "value_loss_std_last10": value_loss_std,
            "value_loss_stable": value_loss_stable,
            "n_updates": n_updates,
            "diagnostic_sample_count": diagnostic_sample_count,
            "last_distinct_update_seen": n_updates,
            "metrics_fresh": metrics_fresh,
            "passes_thresholds": passes,
            "gate_passed": passes,
            "blockers": blockers,
        }


class TrainingHeartbeatCallback(BaseCallback):
    def __init__(
        self,
        *,
        out_path: str | Path,
        diagnostics_cb: TrainingDiagnosticsCallback,
        curriculum_cb: CurriculumCallback | None = None,
        run_id: str,
        symbol: str,
        checkpoints_root: str | Path,
        eval_cb: FullPathEvalCallback | None = None,
        fold_index: int | None = None,
        current_run_path: str | Path = CURRENT_TRAINING_RUN_PATH,
        every_steps: int = 5_000,
        total_timesteps: int | None = None,
        dataset_integrity_report_path: str | Path | None = None,
        dataset_integrity_verified: bool | None = None,
        baseline_report_path: str | Path | None = None,
        resume_model_path: str | Path | None = None,
        resume_vecnormalize_path: str | Path | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.out_path = Path(out_path)
        self.diagnostics_cb = diagnostics_cb
        self.curriculum_cb = curriculum_cb
        self.eval_cb = eval_cb
        self.run_id = str(run_id)
        self.symbol = str(symbol).upper()
        self.checkpoints_root = Path(checkpoints_root)
        self.fold_index = int(fold_index) if fold_index is not None else None
        self.current_run_path = Path(current_run_path)
        self.every_steps = max(int(every_steps), 0)
        self.total_timesteps = int(total_timesteps) if total_timesteps is not None else None
        self.dataset_integrity_report_path = (
            Path(dataset_integrity_report_path) if dataset_integrity_report_path is not None else None
        )
        self.dataset_integrity_verified = (
            bool(dataset_integrity_verified) if dataset_integrity_verified is not None else None
        )
        self.baseline_report_path = Path(baseline_report_path) if baseline_report_path is not None else None
        self.resume_model_path = Path(resume_model_path) if resume_model_path is not None else None
        self.resume_vecnormalize_path = (
            Path(resume_vecnormalize_path) if resume_vecnormalize_path is not None else None
        )
        self.fold_started_utc = datetime.now(timezone.utc).isoformat()
        self._fold_started_perf_counter = time.perf_counter()
        self._next_write = self.every_steps if self.every_steps else 0

    def _collect_training_diagnostics(self) -> dict[str, Any]:
        try:
            snapshots = self.training_env.env_method("get_training_diagnostics")
        except Exception:
            snapshots = []
        return aggregate_training_diagnostics(snapshots if isinstance(snapshots, list) else [])

    def _on_step(self) -> bool:
        if not self.every_steps:
            return True
        if self.num_timesteps < self._next_write:
            return True
        self._next_write += self.every_steps
        diagnostics_summary = self.diagnostics_cb.summary()
        now_utc = datetime.now(timezone.utc)
        elapsed_seconds = max((now_utc - datetime.fromisoformat(PROCESS_STARTED_UTC)).total_seconds(), 0.0)
        fold_elapsed_seconds = max(time.perf_counter() - self._fold_started_perf_counter, 0.0)
        steps_per_second = (
            float(self.num_timesteps) / float(fold_elapsed_seconds)
            if fold_elapsed_seconds > 0
            else None
        )
        progress_fraction = None
        estimated_remaining_seconds = None
        if self.total_timesteps:
            progress_fraction = min(float(self.num_timesteps) / float(self.total_timesteps), 1.0)
            if steps_per_second and np.isfinite(steps_per_second) and steps_per_second > 0:
                estimated_remaining_seconds = max(
                    float(self.total_timesteps - self.num_timesteps) / float(steps_per_second),
                    0.0,
                )
        latest_eval_metrics = copy.deepcopy(self.eval_cb.latest_metrics) if self.eval_cb and self.eval_cb.latest_metrics else None
        env_diagnostics = self._collect_training_diagnostics()
        payload = {
            "schema_version": HEARTBEAT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "symbol": self.symbol,
            "fold_index": self.fold_index,
            "checkpoints_root": str(self.checkpoints_root),
            "process_started_utc": PROCESS_STARTED_UTC,
            "fold_started_utc": self.fold_started_utc,
            "timestamp_utc": now_utc.isoformat(),
            "num_timesteps": int(self.num_timesteps),
            "total_timesteps": self.total_timesteps,
            "elapsed_seconds": elapsed_seconds,
            "fold_elapsed_seconds": fold_elapsed_seconds,
            "steps_per_second": steps_per_second,
            "progress_fraction": progress_fraction,
            "estimated_remaining_seconds": estimated_remaining_seconds,
            "n_updates": diagnostics_summary.get("n_updates"),
            "diagnostic_sample_count": diagnostics_summary.get("diagnostic_sample_count"),
            "ppo_diagnostics": diagnostics_summary,
            "latest_eval": latest_eval_metrics,
            "training_stage": TRAINING_STAGE,
            "curriculum_state": self.curriculum_cb.snapshot() if self.curriculum_cb is not None else None,
            "action_distribution": env_diagnostics["action_distribution"],
            "trade_diagnostics": env_diagnostics["trade_diagnostics"],
            "reward_components": env_diagnostics["reward_components"],
            "recovery_targets": STAGE_A_RECOVERY_TARGETS,
        }
        if self.dataset_integrity_report_path is not None:
            payload["dataset_integrity_report_path"] = str(self.dataset_integrity_report_path)
        if self.dataset_integrity_verified is not None:
            payload["dataset_integrity_verified"] = bool(self.dataset_integrity_verified)
        if self.baseline_report_path is not None:
            payload["baseline_report_path"] = str(self.baseline_report_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        if self.resume_model_path is not None:
            self.resume_model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.resume_model_path.with_suffix("")))
        if self.resume_vecnormalize_path is not None:
            vecnormalize = self.model.get_vec_normalize_env()
            if vecnormalize is not None:
                self.resume_vecnormalize_path.parent.mkdir(parents=True, exist_ok=True)
                vecnormalize.save(str(self.resume_vecnormalize_path))
        _write_current_training_run_context(
            run_id=self.run_id,
            symbol=self.symbol,
            checkpoints_root=self.checkpoints_root,
            state="training",
            fold_index=self.fold_index,
            heartbeat_path=self.out_path,
            out_path=self.current_run_path,
            num_timesteps=int(self.num_timesteps),
            total_timesteps=self.total_timesteps,
            progress_fraction=progress_fraction,
            estimated_remaining_seconds=estimated_remaining_seconds,
            dataset_integrity_report_path=self.dataset_integrity_report_path,
            dataset_integrity_verified=self.dataset_integrity_verified,
            baseline_report_path=self.baseline_report_path,
        )
        if self.verbose:
            print(f"[Heartbeat] Wrote -> {self.out_path}")
        return True


# ── Purged Walk-Forward Split ─────────────────────────────────────────────────

def purged_walk_forward_splits(
    df: pd.DataFrame,
    n_folds: int = N_FOLDS,
    test_frac: float = FOLD_TEST_FRAC,
    purge_gap: int = PURGE_GAP_BARS,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Anchored-expansion purged walk-forward folds.

    Each fold expands the training window forward in time.
    A purge_gap ensures indicator leakage cannot cross the boundary.

    Example (3 folds, test_frac=0.10, gap=200, N=2000):
      Total reserved for OOS = n_folds × test_frac × N = 600 bars
      Fold 0: train=[0:1200]  gap  val=[1400:1600]
      Fold 1: train=[0:1400]  gap  val=[1600:1800]
      Fold 2: train=[0:1600]  gap  val=[1800:2000]
    """
    n = len(df)
    minimums = training_data_minimums()
    min_val_size = int(minimums["min_val_bars"])
    val_size = max(min_val_size, int(n * test_frac))

    # In TRAIN_ENV_MODE=runtime, features are recomputed online via FeatureEngine, so a large
    # "purge gap" is usually unnecessary. Default to 0 unless explicitly overridden.
    effective_purge_gap_default = int(purge_gap)
    if TRAIN_ENV_MODE == "runtime":
        runtime_override = os.environ.get("TRAIN_PURGE_GAP_RUNTIME_BARS", "").strip()
        effective_purge_gap_default = int(runtime_override) if runtime_override else 0

    adaptive_gap = os.environ.get("TRAIN_ADAPTIVE_PURGE_GAP", "1") != "0"
    min_train_bars = int(max(minimums["min_train_bars"], int(WARMUP_BARS) + 20))
    min_gap_bars = int(os.environ.get("TRAIN_PURGE_GAP_MIN_BARS", "0"))

    # val windows anchored at the END, expanding backwards
    # val_end for last fold = n, for fold k = n - (n_folds - 1 - k) * val_size
    folds = []
    for k in range(n_folds):
        val_end   = n - (n_folds - 1 - k) * val_size
        val_start = val_end - val_size
        gap = int(effective_purge_gap_default)
        if adaptive_gap:
            # Shrink gaps for small datasets so folds remain usable.
            # We prioritize keeping at least `min_train_bars` training rows.
            max_gap_for_min_train = max(0, int(val_start) - int(min_train_bars))
            gap = min(gap, max_gap_for_min_train)
            if gap < min_gap_bars and val_start - min_gap_bars >= min_train_bars:
                gap = min_gap_bars
        train_end = val_start - gap

        rejection_reasons: list[str] = []
        if val_start >= val_end:
            rejection_reasons.append("invalid validation slice")
        if train_end < min_train_bars:
            rejection_reasons.append(f"train bars {train_end} < required {min_train_bars}")
        if (val_end - val_start) < min_val_size:
            rejection_reasons.append(f"validation bars {val_end - val_start} < required {min_val_size}")
        if rejection_reasons:
            print(f"  Fold {k}: skipped ({'; '.join(rejection_reasons)})")
            continue

        train_df = df.iloc[:train_end].copy()
        val_df   = df.iloc[val_start:val_end].copy()
        folds.append((train_df, val_df))
        print(f"  Fold {k}: train=[0:{train_end}]  gap={gap}bars  "
              f"val=[{val_start}:{val_end}]  ({len(val_df)} bars)")
    return folds


# ── Helpers ───────────────────────────────────────────────────────────────────

def linear_schedule(initial_value: float, min_value: float = 1e-6) -> Callable:
    def func(progress_remaining: float) -> float:
        return max(progress_remaining * initial_value, min_value)
    return func


def make_env(
    df,
    feature_cols,
    sl_opts,
    tp_opts,
    random_start=True,
    initial_slippage: float = 0.0,
    symbol: str = "EURUSD",
    scaler: StandardScaler | None = None,
    recovery_config: dict[str, Any] | None = None,
    bars: list[Any] | None = None,
):
    def _init():
        if TRAIN_ENV_MODE == "runtime":
            if scaler is None:
                raise RuntimeError("TRAIN_ENV_MODE=runtime requires a per-fold scaler.")
            from runtime_gym_env import RuntimeGymConfig, RuntimeGymEnv

            action_map = build_runtime_action_map(sl_opts, tp_opts)
            env = RuntimeGymEnv(
                symbol=symbol,
                bars_frame=df,
                bars=bars,
                scaler=scaler,
                action_map=action_map,
                config=RuntimeGymConfig(
                    commission_per_lot=float(TRAIN_COMMISSION_PER_LOT),
                    slippage_pips=float(initial_slippage),
                    partial_fill_ratio=float(TRAIN_PARTIAL_FILL_RATIO),
                    reward_scale=float(TRAIN_REWARD_SCALE),
                    drawdown_penalty=float(TRAIN_REWARD_DRAWDOWN_PENALTY),
                    transaction_penalty=float(TRAIN_REWARD_TRANSACTION_PENALTY),
                    reward_clip_low=float(TRAIN_REWARD_CLIP_LOW),
                    reward_clip_high=float(TRAIN_REWARD_CLIP_HIGH),
                    churn_min_hold_bars=int(TRAIN_CHURN_MIN_HOLD_BARS),
                    churn_action_cooldown=int(TRAIN_CHURN_ACTION_COOLDOWN),
                    churn_penalty_usd=float(TRAIN_CHURN_PENALTY_USD),
                    downside_risk_penalty=float(TRAIN_REWARD_DOWNSIDE_RISK_COEF),
                    turnover_penalty=float(TRAIN_REWARD_TURNOVER_COEF),
                    net_return_coef=float(TRAIN_REWARD_NET_RETURN_COEF),
                    random_start=bool(random_start),
                ),
                recovery_config=copy.deepcopy(recovery_config) if recovery_config is not None else None,
            )
        else:
            env = ForexTradingEnv(
                df=df,
                feature_columns=feature_cols,
                sl_options=sl_opts,
                tp_options=tp_opts,
                random_start=random_start,
                initial_equity=1_000.0,
                lot_size=0.01,
                max_slippage_pips=initial_slippage,  # starts at 0, annealed by curriculum
                use_trailing_stop=True,
                use_variable_spread=True,
                atr_scaled=True,
                vol_scaling=True,
                target_risk_pct=0.01,
                symbol=symbol,
            )
        env = ActionMasker(env, action_mask_fn)
        env = Monitor(env)
        return env
    return _init


def wrap_vecnormalize(vec_env, *, training: bool):
    return VecNormalize(
        vec_env,
        training=training,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=10.0,
    )


def sync_vecnormalize_stats(source: VecNormalize, target: VecNormalize) -> None:
    target.obs_rms = copy.deepcopy(source.obs_rms)
    target.ret_rms = copy.deepcopy(source.ret_rms)
    target.training = False
    target.norm_reward = False


def _curve_segment_metrics(
    equity_curve: list[float],
    timestamps: list[pd.Timestamp],
) -> dict[str, dict[str, float | int]]:
    labels = ("first", "middle", "last")
    segments: dict[str, dict[str, float | int]] = {}
    if not equity_curve:
        for label in labels:
            segments[label] = {
                "steps": 0,
                "final_equity": 0.0,
                "timed_sharpe": 0.0,
                "max_drawdown": 0.0,
            }
        return segments

    curve_len = len(equity_curve)
    time_len = len(timestamps)
    for label, indices in zip(labels, np.array_split(np.arange(curve_len), 3)):
        if len(indices) == 0:
            segments[label] = {
                "steps": 0,
                "final_equity": 0.0,
                "timed_sharpe": 0.0,
                "max_drawdown": 0.0,
            }
            continue
        start = int(indices[0])
        end = int(indices[-1]) + 1
        seg_curve = equity_curve[start:end]
        seg_timestamps = timestamps[start:min(end, time_len)] if time_len else []
        segments[label] = {
            "steps": int(len(seg_curve)),
            "final_equity": float(seg_curve[-1]),
            "timed_sharpe": compute_timed_sharpe(seg_curve, seg_timestamps),
            "max_drawdown": compute_max_drawdown(seg_curve),
        }
    return segments


def _extract_eval_trade_log(eval_env) -> list[dict[str, Any]]:
    """Extract closed trade log from eval env using a robust 3-strategy fallback chain.

    Strategy 1: env_method("get_trade_log") — works with RuntimeGymEnv via
        SubprocVecEnv/DummyVecEnv without requiring cross-process object serialization.
    Strategy 2: get_attr("_runtime") -> broker.trade_log — works when _runtime
        is directly accessible (DummyVecEnv with RuntimeGymEnv).
    Strategy 3: get_attr("trade_log") — ForexTradingEnv / legacy env fallback.
    """
    # Strategy 1: explicit env_method (most reliable via SubprocVecEnv)
    try:
        results = eval_env.env_method("get_trade_log")
        if results is not None and isinstance(results, list) and len(results) > 0:
            trace = results[0]
            if isinstance(trace, list):
                return [dict(item) for item in trace if isinstance(item, dict)]
    except Exception:
        pass

    # Strategy 2: get_attr("_runtime") -> broker.trade_log
    try:
        runtimes = eval_env.get_attr("_runtime")
        if runtimes:
            runtime = runtimes[0]
            broker = getattr(runtime, "broker", None)
            trade_log = getattr(broker, "trade_log", None)
            if isinstance(trade_log, list):
                return [dict(item) for item in trade_log if isinstance(item, dict)]
    except Exception:
        pass

    # Strategy 3: legacy ForexTradingEnv direct attribute
    try:
        trade_logs = eval_env.get_attr("trade_log")
        if trade_logs and isinstance(trade_logs[0], list):
            return [dict(item) for item in trade_logs[0] if isinstance(item, dict)]
    except Exception:
        pass

    log.warning(
        "evaluate_model: could not extract trade_log from eval env via any known strategy. "
        "trade_count and economic metrics will be zero. Check env type and SubprocVecEnv serialization."
    )
    return []


def _extract_eval_execution_log(eval_env) -> list[dict[str, Any]]:
    """Extract execution log from eval env using a robust 3-strategy fallback chain.

    Mirrors the strategy ordering of _extract_eval_trade_log.
    """
    # Strategy 1: explicit env_method (most reliable via SubprocVecEnv)
    try:
        results = eval_env.env_method("get_execution_log")
        if results is not None and isinstance(results, list) and len(results) > 0:
            trace = results[0]
            if isinstance(trace, list):
                return [dict(item) for item in trace if isinstance(item, dict)]
    except Exception:
        pass

    # Strategy 2: get_attr("_runtime") -> broker.execution_log
    try:
        runtimes = eval_env.get_attr("_runtime")
        if runtimes:
            runtime = runtimes[0]
            broker = getattr(runtime, "broker", None)
            execution_log = getattr(broker, "execution_log", None)
            if isinstance(execution_log, list):
                return [dict(item) for item in execution_log if isinstance(item, dict)]
    except Exception:
        pass

    # Strategy 3: legacy fallback
    try:
        exec_logs = eval_env.get_attr("execution_log")
        if exec_logs and isinstance(exec_logs[0], list):
            return [dict(item) for item in exec_logs[0] if isinstance(item, dict)]
    except Exception:
        pass

    return []


def evaluate_model(model, eval_env) -> tuple[list[float], dict[str, Any]]:
    obs = eval_env.reset()
    equity_curve: list[float] = []
    timestamps: list[pd.Timestamp] = []
    rewards: list[float] = []
    while True:
        masks  = eval_env.env_method("action_masks")
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        step_out   = eval_env.step(action)
        if len(step_out) == 4:
            obs, reward, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, reward, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])
        info = infos[0] if infos else {}
        reward_value = reward[0] if isinstance(reward, (list, tuple, np.ndarray)) else reward
        rewards.append(float(reward_value))
        if "equity" in info:
            equity_curve.append(float(info["equity"]))
        elif "total_equity_usd" in info:
            equity_curve.append(float(info["total_equity_usd"]))
        else:
            try:
                equity_curve.append(float(eval_env.get_attr("equity_usd")[0]))
            except Exception:
                equity_curve.append(1_000.0)
        timestamp_utc = info.get("timestamp_utc")
        if timestamp_utc:
            timestamps.append(pd.Timestamp(timestamp_utc))
        if done:
            break
    trade_log = _extract_eval_trade_log(eval_env)
    execution_log = _extract_eval_execution_log(eval_env)

    try:
        raw_env_diagnostics = eval_env.env_method("get_training_diagnostics") if hasattr(eval_env, "env_method") else []
    except Exception:
        raw_env_diagnostics = []

    env_diagnostics = aggregate_training_diagnostics(raw_env_diagnostics)

    # Use unified accounting from runtime_common
    accounting = build_evaluation_accounting(
        trade_log=trade_log,
        execution_diagnostics=env_diagnostics,
        execution_log_count=len(execution_log),
        initial_equity=1000.0,
    )
    val_status = validate_evaluation_accounting(accounting)

    metrics = {
        "final_equity": float(equity_curve[-1]) if equity_curve else 1000.0,
        "timed_sharpe": compute_timed_sharpe(equity_curve, timestamps),
        "max_drawdown": compute_max_drawdown(equity_curve),
        "steps": int(len(equity_curve)),
        "aux_total_shaped_reward": float(np.sum(rewards)) if rewards else 0.0,
        "aux_mean_step_shaped_reward": float(np.mean(rewards)) if rewards else 0.0,
        "segment_metrics": _curve_segment_metrics(equity_curve, timestamps),
        **accounting,
        "execution_diagnostics": env_diagnostics,
        "execution_event_count": int(len(execution_log)),
        "validation_status": val_status,
        "accounting_gap_detected": not val_status["passed"],
    }
    return equity_curve, metrics


def _split_holdout(df: pd.DataFrame, holdout_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    minimums = training_data_minimums()
    holdout_size = max(minimums["min_holdout_bars"], int(len(df) * holdout_frac))
    split_idx = len(df) - holdout_size
    if split_idx < minimums["min_train_bars"]:
        raise RuntimeError(
            f"Not enough rows ({len(df)}) to reserve a disjoint holdout of {holdout_size} bars "
            f"while keeping at least {minimums['min_train_bars']} training bars."
        )
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def _safe_pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _sign_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def _prepare_supervised_xy(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    horizon_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    if "Close" not in frame.columns:
        raise RuntimeError("Baseline gate requires a Close column.")
    missing = [col for col in feature_cols if col not in frame.columns]
    if missing:
        raise RuntimeError(f"Baseline gate missing feature columns: {missing}")

    dataset = frame.loc[:, [*feature_cols, "Close"]].copy()
    dataset["target"] = np.log(dataset["Close"].shift(-horizon_bars) / dataset["Close"])
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna(subset=[*feature_cols, "target"])
    if dataset.empty:
        return np.empty((0, len(feature_cols)), dtype=np.float64), np.empty(0, dtype=np.float64)
    x = dataset[feature_cols].to_numpy(dtype=np.float64, copy=True)
    y = dataset["target"].to_numpy(dtype=np.float64, copy=True)
    return x, y


def _baseline_prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | bool]:
    if y_true.size == 0 or y_pred.size == 0:
        return {
            "r2": float("nan"),
            "pearson_corr": float("nan"),
            "sign_accuracy": float("nan"),
            "pass_thresholds": False,
        }
    r2 = float(r2_score(y_true, y_pred))
    corr = _safe_pearson_corr(y_true, y_pred)
    sign_acc = _sign_accuracy(y_true, y_pred)
    passes = bool(
        np.isfinite(r2)
        and np.isfinite(corr)
        and np.isfinite(sign_acc)
        and r2 > BASELINE_R2_MIN
        and corr >= BASELINE_CORR_MIN
        and sign_acc >= BASELINE_SIGN_ACC_MIN
    )
    return {
        "r2": r2,
        "pearson_corr": corr,
        "sign_accuracy": sign_acc,
        "pass_thresholds": passes,
    }


def _fit_baseline_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, Any]:
    ridge = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    tree = HistGradientBoostingRegressor(
        max_depth=BASELINE_TREE_MAX_DEPTH,
        max_iter=BASELINE_TREE_MAX_ITER,
        random_state=SEED,
    )
    ridge.fit(x_train, y_train)
    tree.fit(x_train, y_train)
    return {
        "ridge": ridge,
        "hist_gradient_boosting": tree,
    }


def run_baseline_research_gate(
    *,
    symbol: str,
    trainable_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    feature_cols: list[str],
    out_path: str | Path,
    horizon_bars: int = BASELINE_TARGET_HORIZON_BARS,
) -> dict[str, Any]:
    return run_edge_baseline_research(
        symbol=symbol,
        trainable_frame=trainable_frame,
        holdout_frame=holdout_frame,
        folds=folds,
        feature_cols=feature_cols,
        out_path=out_path,
        horizon_bars=horizon_bars,
        commission_per_lot=float(TRAIN_COMMISSION_PER_LOT),
        slippage_pips=float(get_final_slippage_pips(TRAINING_RECOVERY_CONFIG)),
        min_edge_pips=float(BASELINE_MIN_EDGE_PIPS),
        probability_threshold=float(BASELINE_PROB_THRESHOLD),
        probability_margin=float(BASELINE_PROB_MARGIN),
        min_trade_count=int(MIN_EVAL_TRADE_COUNT),
    )


def _resolve_frame_position(frame: pd.DataFrame, start_index: Any) -> int:
    loc = frame.index.get_loc(start_index)
    if isinstance(loc, slice):
        return int(loc.start or 0)
    if isinstance(loc, (np.ndarray, list)):
        return int(loc[0]) if len(loc) else 0
    return int(loc)


def _prepend_runtime_warmup_context(full_frame: pd.DataFrame, segment_frame: pd.DataFrame) -> pd.DataFrame:
    if segment_frame.empty:
        raise RuntimeError("Evaluation segment is empty.")
    start_index = segment_frame.index[0]
    start_pos = _resolve_frame_position(full_frame, start_index)
    warmup_start = max(0, start_pos - int(WARMUP_BARS))
    warmup_frame = full_frame.iloc[warmup_start:start_pos].copy()
    return pd.concat([warmup_frame, segment_frame], axis=0)


def _fit_and_apply_fold_scaler(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    scaler = StandardScaler()
    scaler.fit(train_scaled.loc[:, FEATURE_COLS])
    train_scaled.loc[:, FEATURE_COLS] = scaler.transform(train_scaled.loc[:, FEATURE_COLS])
    val_scaled.loc[:, FEATURE_COLS] = scaler.transform(val_scaled.loc[:, FEATURE_COLS])
    return train_scaled, val_scaled, scaler


def _holdout_deployment_blockers(
    *,
    holdout_sharpe: float,
    holdout_max_drawdown: float,
    holdout_final_equity: float,
    holdout_profit_factor: float,
    holdout_expectancy: float,
    holdout_trade_count: int,
    initial_equity: float = 1_000.0,
) -> list[str]:
    blockers: list[str] = []
    if float(holdout_sharpe) < DEPLOY_TIMED_SHARPE_MIN:
        blockers.append(f"Holdout timed Sharpe {float(holdout_sharpe):.3f} < {DEPLOY_TIMED_SHARPE_MIN:.2f}")
    if float(holdout_max_drawdown) > DEPLOY_DD_MAX:
        blockers.append(f"Holdout max drawdown {float(holdout_max_drawdown):.1%} > {DEPLOY_DD_MAX:.0%}")
    if float(holdout_final_equity) < float(initial_equity):
        blockers.append(
            f"Holdout final equity {float(holdout_final_equity):.2f} < initial equity {float(initial_equity):.2f}"
        )
    if not np.isfinite(float(holdout_profit_factor)) or float(holdout_profit_factor) < DEPLOY_PROFIT_FACTOR_MIN:
        blockers.append(
            f"Holdout profit factor {float(holdout_profit_factor):.3f} < {DEPLOY_PROFIT_FACTOR_MIN:.2f}"
        )
    if float(holdout_expectancy) < DEPLOY_EXPECTANCY_MIN:
        blockers.append(f"Holdout expectancy {float(holdout_expectancy):.3f} < {DEPLOY_EXPECTANCY_MIN:.2f}")
    if int(holdout_trade_count) < MIN_EVAL_TRADE_COUNT:
        blockers.append(f"Holdout trades {int(holdout_trade_count)} < required {MIN_EVAL_TRADE_COUNT}")
    return blockers


def _deployment_candidate_rank(diagnostics: dict[str, Any]) -> tuple[float, float, float] | None:
    if not bool(diagnostics.get("deploy_ready", False)):
        return None
    return (
        float(diagnostics.get("holdout_sharpe", float("-inf"))),
        float(diagnostics.get("holdout_profit_factor", float("-inf"))),
        float(diagnostics.get("val_sharpe", float("-inf"))),
    )


def _archive_paths(paths: list[Path], *, archive_root: Path) -> list[str]:
    archived: list[str] = []
    seen: set[str] = set()
    archive_root.mkdir(parents=True, exist_ok=True)
    for raw_path in paths:
        path = Path(raw_path)
        normalized = str(path)
        if normalized in seen or not path.exists():
            continue
        seen.add(normalized)
        destination = archive_root / path.name
        suffix = 1
        while destination.exists():
            destination = archive_root / f"{path.stem}_{suffix}{path.suffix}"
            suffix += 1
        shutil.move(str(path), str(destination))
        archived.append(str(destination))
    return archived


def _clear_legacy_checkpoint_artifacts(*, run_id: str, checkpoints_root: Path = Path("checkpoints")) -> list[str]:
    legacy_paths = [
        *checkpoints_root.glob("fold_*"),
        checkpoints_root / "best_model.zip",
        checkpoints_root / "best_vecnormalize.pkl",
    ]
    archive_root = checkpoints_root / "archive" / run_id
    return _archive_paths(legacy_paths, archive_root=archive_root)


def _write_current_training_run_context(
    *,
    run_id: str,
    symbol: str,
    checkpoints_root: Path,
    state: str,
    fold_index: int | None = None,
    heartbeat_path: str | Path | None = None,
    out_path: str | Path = CURRENT_TRAINING_RUN_PATH,
    num_timesteps: int | None = None,
    total_timesteps: int | None = None,
    progress_fraction: float | None = None,
    estimated_remaining_seconds: float | None = None,
    dataset_integrity_report_path: str | Path | None = None,
    dataset_integrity_verified: bool | None = None,
    baseline_report_path: str | Path | None = None,
) -> Path:
    payload = {
        "schema_version": 1,
        "run_id": str(run_id),
        "symbol": str(symbol).upper(),
        "checkpoints_root": str(checkpoints_root),
        "process_started_utc": PROCESS_STARTED_UTC,
        "pid": int(os.getpid()),
        "state": str(state),
        "fold_index": int(fold_index) if fold_index is not None else None,
        "heartbeat_path": str(Path(heartbeat_path)) if heartbeat_path is not None else None,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if num_timesteps is not None:
        payload["num_timesteps"] = int(num_timesteps)
    if total_timesteps is not None:
        payload["total_timesteps"] = int(total_timesteps)
    if progress_fraction is not None:
        payload["progress_fraction"] = float(progress_fraction)
    if estimated_remaining_seconds is not None:
        payload["estimated_remaining_seconds"] = float(estimated_remaining_seconds)
    if dataset_integrity_report_path is not None:
        payload["dataset_integrity_report_path"] = str(Path(dataset_integrity_report_path))
    if dataset_integrity_verified is not None:
        payload["dataset_integrity_verified"] = bool(dataset_integrity_verified)
    if baseline_report_path is not None:
        payload["baseline_report_path"] = str(Path(baseline_report_path))
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return destination


def _resume_model_checkpoint_path(ckpt_dir: Path) -> Path:
    return ckpt_dir / RESUME_MODEL_NAME


def _resume_vecnormalize_checkpoint_path(ckpt_dir: Path) -> Path:
    return ckpt_dir / RESUME_VECNORMALIZE_NAME


def _candidate_scaler_artifact_path(ckpt_dir: Path, symbol: str) -> Path:
    return ckpt_dir / f"deployment_candidate_scaler_{symbol.upper()}.pkl"


def _load_training_resume_state(
    *,
    symbol: str,
    total_timesteps: int,
    current_run_path: Path = CURRENT_TRAINING_RUN_PATH,
) -> dict[str, Any] | None:
    if not current_run_path.exists():
        return None
    try:
        current_run = json.loads(current_run_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    current_symbol = str(current_run.get("symbol", "")).strip().upper()
    if current_symbol != symbol.strip().upper():
        return None

    state = str(current_run.get("state", "")).strip().lower()
    if state in {"completed", "failed_dataset_integrity", "failed_data_minimums", "failed_baseline_gate"}:
        return None

    checkpoints_root_raw = current_run.get("checkpoints_root")
    fold_index = current_run.get("fold_index")
    if not checkpoints_root_raw or fold_index is None:
        return None

    checkpoints_root = Path(str(checkpoints_root_raw))
    ckpt_dir = checkpoints_root / f"fold_{int(fold_index)}"
    resume_model_path = _resume_model_checkpoint_path(ckpt_dir)
    if not resume_model_path.exists():
        best_model_path = ckpt_dir / "best_model.zip"
        if best_model_path.exists():
            resume_model_path = best_model_path
        else:
            return None

    resume_vecnormalize_path = _resume_vecnormalize_checkpoint_path(ckpt_dir)
    if not resume_vecnormalize_path.exists():
        fallback_vecnormalize_path = ckpt_dir / BEST_VECNORMALIZE_NAME
        resume_vecnormalize_path = fallback_vecnormalize_path if fallback_vecnormalize_path.exists() else None

    run_id = str(current_run.get("run_id") or checkpoints_root.name.replace("run_", "", 1) or checkpoints_root.name)
    num_timesteps = int(current_run.get("num_timesteps", 0) or 0)

    return {
        "run_id": run_id,
        "symbol": current_symbol,
        "state": state,
        "checkpoints_root": checkpoints_root,
        "fold_index": int(fold_index),
        "model_path": resume_model_path,
        "vecnormalize_path": resume_vecnormalize_path,
        "num_timesteps": num_timesteps,
        "remaining_timesteps_hint": max(int(total_timesteps) - num_timesteps, 0),
    }


def _recover_completed_fold_state(
    *,
    run_id: str,
    checkpoints_root: Path,
    primary_symbol: str,
) -> tuple[
    float,
    dict[str, Any] | None,
    tuple[float, float, float] | None,
    dict[str, Any] | None,
    Path | None,
    Path | None,
    Path | None,
]:
    best_observed_sharpe = -np.inf
    best_observed_summary: dict[str, Any] | None = None
    candidate_rank: tuple[float, float, float] | None = None
    candidate_summary: dict[str, Any] | None = None
    candidate_model_source: Path | None = None
    candidate_vecnormalize_source: Path | None = None
    candidate_scaler_source: Path | None = None

    for metrics_path in sorted(checkpoints_root.glob("fold_*/training_diagnostics.json")):
        try:
            diagnostics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        ckpt_dir = metrics_path.parent
        sharpe = float(diagnostics.get("val_sharpe", -np.inf))
        if sharpe > best_observed_sharpe:
            best_observed_sharpe = sharpe
            best_observed_summary = _build_promoted_training_diagnostics(
                diagnostics,
                run_id=run_id,
                artifact_candidate_selected=False,
                artifact_candidate_reason="Recovered best-evaluated completed fold from resumed run state.",
            )

        current_candidate_rank = _deployment_candidate_rank(diagnostics)
        if current_candidate_rank is None:
            continue
        recovered_model_source = ckpt_dir / "deployment_candidate_model.zip"
        recovered_scaler_source = _candidate_scaler_artifact_path(ckpt_dir, primary_symbol)
        if not recovered_model_source.exists() or not recovered_scaler_source.exists():
            continue
        if candidate_rank is None or current_candidate_rank > candidate_rank:
            candidate_rank = current_candidate_rank
            candidate_summary = _build_promoted_training_diagnostics(
                diagnostics,
                run_id=run_id,
                artifact_candidate_selected=True,
                artifact_candidate_reason="Recovered deployment candidate from completed fold in resumed run.",
            )
            candidate_model_source = recovered_model_source
            recovered_vecnormalize_source = ckpt_dir / "deployment_candidate_vecnormalize.pkl"
            candidate_vecnormalize_source = (
                recovered_vecnormalize_source if recovered_vecnormalize_source.exists() else None
            )
            candidate_scaler_source = recovered_scaler_source

    return (
        best_observed_sharpe,
        best_observed_summary,
        candidate_rank,
        candidate_summary,
        candidate_model_source,
        candidate_vecnormalize_source,
        candidate_scaler_source,
    )


def _prime_eval_callback_from_history(eval_cb: FullPathEvalCallback) -> None:
    if eval_cb.history_path is None or not eval_cb.history_path.exists():
        return
    try:
        payload = json.loads(eval_cb.history_path.read_text(encoding="utf-8"))
    except Exception:
        return
    raw_history = list(payload.get("evaluations", []) or [])

    # Filter for integrity: skip legacy entries that fail unified accounting validation
    valid_history = []
    for entry in raw_history:
        try:
            # We treat entries without reconciliation info as legacy-invalid
            # or we can be slightly more permissive but here we favor truthfulness.
            validate_evaluation_payload(entry)
            valid_history.append(entry)
        except Exception as e:
            # Skip invalid entries silently but log once
            continue

    eval_cb.history = valid_history
    if valid_history:
        eval_cb.latest_metrics = valid_history[-1]

    best_metric = -float("inf")
    for entry in valid_history:
        raw_value = entry.get(eval_cb.metric_key, -float("inf"))
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric_value) and numeric_value > best_metric:
            best_metric = numeric_value
    if np.isfinite(best_metric):
        eval_cb.best_metric = float(best_metric)


def _clear_current_run_artifacts(
    *,
    primary_symbol: str,
    model_artifact_path: Path,
    vecnormalize_artifact_path: Path,
    run_id: str,
) -> list[str]:
    report_paths = deployment_paths(primary_symbol)
    paths_to_archive = [
        model_artifact_path,
        vecnormalize_artifact_path,
        Path(f"models/scaler_{primary_symbol}.pkl"),
        SCALER_PATH,
        Path(f"models/artifact_manifest_{primary_symbol}.json"),
        Path("models") / DEFAULT_MANIFEST_NAME,
        report_paths.diagnostics_path,
        report_paths.gate_path,
        report_paths.live_preflight_path,
        report_paths.ops_attestation_path,
    ]
    archive_root = Path("models") / "archive" / primary_symbol.lower() / run_id
    return _archive_paths(paths_to_archive, archive_root=archive_root)


def _build_promoted_training_diagnostics(
    diagnostics: dict[str, Any],
    *,
    run_id: str,
    artifact_candidate_selected: bool,
    artifact_candidate_reason: str,
) -> dict[str, Any]:
    payload = copy.deepcopy(diagnostics)
    payload["run_id"] = run_id
    payload["artifact_candidate_selected"] = bool(artifact_candidate_selected)
    payload["artifact_candidate_reason"] = str(artifact_candidate_reason)
    ticks_per_bar = int(
        payload.get(
            "bar_construction_ticks_per_bar",
            payload.get("ticks_per_bar", DEFAULT_BAR_CONSTRUCTION_TICKS_PER_BAR),
        )
    )
    payload["bar_construction_ticks_per_bar"] = ticks_per_bar
    payload["ticks_per_bar"] = ticks_per_bar
    return payload


def _publish_primary_candidate_artifacts(
    *,
    primary_symbol: str,
    model_artifact_path: Path,
    vecnormalize_artifact_path: Path,
    candidate_model_source: Path,
    candidate_vecnormalize_source: Path | None,
    candidate_scaler_source: Path,
    holdout_start_utc: str | None,
    dataset_path: str | Path,
    execution_cost_profile: dict[str, Any] | None = None,
    reward_profile: dict[str, Any] | None = None,
) -> None:
    if not candidate_scaler_source.exists():
        raise RuntimeError(f"Deployment candidate scaler is missing for {primary_symbol}: {candidate_scaler_source}")

    shutil.copyfile(candidate_model_source, model_artifact_path)
    if candidate_vecnormalize_source is not None and candidate_vecnormalize_source.exists():
        shutil.copyfile(candidate_vecnormalize_source, vecnormalize_artifact_path)
    elif vecnormalize_artifact_path.exists():
        vecnormalize_artifact_path.unlink()

    primary_scaler_path = Path(f"models/scaler_{primary_symbol}.pkl")
    shutil.copyfile(candidate_scaler_source, primary_scaler_path)
    shutil.copyfile(candidate_scaler_source, SCALER_PATH)

    action_map = build_runtime_action_map(list(ACTION_SL_MULTS), list(ACTION_TP_MULTS))
    observation_shape = [1, len(FEATURE_COLS) + STATE_FEATURE_COUNT]
    diagnostics_path = deployment_paths(primary_symbol).diagnostics_path
    manifest = create_manifest(
        strategy_symbol=primary_symbol,
        model_path=model_artifact_path,
        scaler_path=primary_scaler_path,
        vecnormalize_path=vecnormalize_artifact_path if vecnormalize_artifact_path.exists() else None,
        holdout_start_utc=holdout_start_utc,
        training_diagnostics_path=diagnostics_path,
        model_version=f"{model_artifact_path.stem}-v1",
        feature_columns=FEATURE_COLS,
        observation_shape=observation_shape,
        action_map=action_map,
        dataset_path=dataset_path,
        bar_construction_ticks_per_bar=BAR_CONSTRUCTION_TICKS_PER_BAR,
        execution_cost_profile=execution_cost_profile,
        reward_profile=reward_profile,
    )
    save_manifest(manifest, Path(f"models/artifact_manifest_{primary_symbol}.json"))
    save_manifest(manifest, Path("models") / DEFAULT_MANIFEST_NAME)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_runtime_dirs()
    log_config = configure_run_logging(
        "train_agent",
        symbol=TRAIN_SYMBOL or None,
        capture_print=True,
    )
    if TRAIN_ENV_MODE not in {"runtime", "legacy"}:
        raise RuntimeError(f"Unsupported TRAIN_ENV_MODE={TRAIN_ENV_MODE!r}. Expected 'runtime' or 'legacy'.")
    requested_envs = int(os.environ.get("TRAIN_NUM_ENVS", "0")) or None
    runtime_plan = configure_training_runtime(requested_envs)
    device = runtime_plan.device
    project_python = project_venv_python(Path(__file__).resolve().parent)
    configured_mp_executable = str(project_python) if project_python is not None else None
    if configured_mp_executable:
        mp.set_executable(configured_mp_executable)
    effective_envs = runtime_plan.env_workers
    force_dummy = os.environ.get("TRAIN_FORCE_DUMMY_VEC", "0") == "1"
    vec_env_type, vec_env_warnings = resolve_train_vec_env_type(
        requested_envs=requested_envs,
        effective_envs=effective_envs,
        force_dummy=force_dummy,
    )
    print(f"\nTraining on: {runtime_plan.accelerator_label}")
    print(
        f"CPU cores={runtime_plan.cpu_cores} | env_workers={runtime_plan.env_workers} | "
        f"torch_threads={runtime_plan.torch_threads} | interop_threads={runtime_plan.interop_threads}"
    )
    print(f"Runtime env mode: {TRAIN_ENV_MODE}")
    print(
        f"[Runtime] sys.executable={sys.executable} | sys.prefix={sys.prefix} | "
        f"multiprocessing_executable={configured_mp_executable} | "
        f"vec_env_type={vec_env_type} | env_workers={effective_envs}"
    )
    for warning in vec_env_warnings:
        print(f"[WARN] {warning}")
    
    use_amp = TRAIN_USE_AMP and str(device) == "cuda"
    amp_dtype = TRAIN_AMP_DTYPE
    msg_amp = " | AMP=on" if use_amp else " | AMP=off"
    if use_amp:
        msg_amp += f" ({amp_dtype})"
    
    print(
        f"[INFO] Supported training stack: MaskablePPO + RuntimeGymEnv + volume bars{msg_amp} | "
        f"bar_construction_ticks_per_bar={BAR_CONSTRUCTION_TICKS_PER_BAR}"
    )
    
    if not TRAIN_LEGACY_REQUIRE_BASELINE_GATE and not TRAIN_DEBUG_ALLOW_BASELINE_BYPASS:
        print(
            "[WARN] Ignoring deprecated TRAIN_REQUIRE_BASELINE_GATE=0. "
            "Use TRAIN_DEBUG_ALLOW_BASELINE_BYPASS=1 for an explicit debug-only bypass."
        )
    if TRAIN_DEBUG_ALLOW_BASELINE_BYPASS:
        print("[WARN] TRAIN_DEBUG_ALLOW_BASELINE_BYPASS=1 disables the baseline gate for debug only.")
    if TRAIN_STARTUP_SMOKE_ONLY:
        print("[StartupSmoke] Runtime setup complete; exiting before data load and training by request.")
        return
    # ── Load data ────────────────────────────────────────────────────────────
    # Try volume bars first (Phase 11), fall back to FOREX_MULTI_SET
    data_path = resolve_dataset_path()
    validate_dataset_bar_spec(
        dataset_path=data_path,
        expected_ticks_per_bar=BAR_CONSTRUCTION_TICKS_PER_BAR,
        metadata_required=True,
    )
    print(f"[INFO] Using dataset: {data_path}")
    if False:
        data_path = "data/FOREX_MULTI_SET.csv"
        print(f"[INFO] Volume bars not found — using {data_path}")
    else:
        pass

    df_raw = pd.read_csv(data_path, low_memory=False)
    df_raw = df_raw.dropna(subset=["Symbol"])
    df_raw = df_raw[df_raw["Symbol"].apply(lambda x: isinstance(x, str))]
    df_raw["Symbol"] = df_raw["Symbol"].astype(str).str.upper()

    if TRAIN_SYMBOL:
        df_raw = df_raw[df_raw["Symbol"] == TRAIN_SYMBOL].copy()
        if df_raw.empty:
            raise RuntimeError(f"No rows found for TRAIN_SYMBOL={TRAIN_SYMBOL} in dataset {data_path}.")
        print(f"[INFO] Training single-symbol model for {TRAIN_SYMBOL}")

    symbols = df_raw["Symbol"].unique().tolist()
    print(f"Symbols: {symbols}  |  Total rows: {len(df_raw):,}")
    if not symbols:
        raise RuntimeError("No symbols available after filtering dataset.")
    primary_symbol = TRAIN_SYMBOL or symbols[0]
    model_basename = f"model_{primary_symbol.lower()}_best"
    model_artifact_path = Path("models") / f"{model_basename}.zip"
    vecnormalize_artifact_path = Path("models") / f"{model_basename}_vecnormalize.pkl"
    resume_state = (
        _load_training_resume_state(
            symbol=primary_symbol,
            total_timesteps=TOTAL_TIMESTEPS,
            current_run_path=CURRENT_TRAINING_RUN_PATH,
        )
        if TRAIN_RESUME_LATEST
        else None
    )
    if resume_state is not None:
        run_id = str(resume_state["run_id"])
        run_checkpoints_root = Path(str(resume_state["checkpoints_root"]))
        print(
            f"[Resume] Found resumable run {run_id} for {primary_symbol}: "
            f"fold={resume_state['fold_index']} checkpoint={resume_state['model_path']}"
        )
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_checkpoints_root = Path("checkpoints") / f"run_{run_id}"
    run_checkpoints_root.mkdir(parents=True, exist_ok=True)
    set_log_context(symbol=primary_symbol, run_id=run_id)
    log.info(
        "Training run context established",
        extra={
            "event": "training_run_context",
            "checkpoint_root": run_checkpoints_root,
            "text_log_path": log_config.text_log_path,
            "jsonl_log_path": log_config.jsonl_log_path,
        },
    )
    print(f"[INFO] Text log -> {log_config.text_log_path}")
    print(f"[INFO] Event log -> {log_config.jsonl_log_path}")
    dataset_integrity_report_path = run_checkpoints_root / "dataset_integrity_report.json"
    point_in_time_verified = bool(TRAIN_POINT_IN_TIME_VERIFIED)
    _write_current_training_run_context(
        run_id=run_id,
        symbol=primary_symbol,
        checkpoints_root=run_checkpoints_root,
        state="validating_dataset_integrity",
        dataset_integrity_report_path=dataset_integrity_report_path,
    )
    try:
        dataset_integrity_report = validate_dataset_integrity(
            dataset_path=data_path,
            expected_ticks_per_bar=BAR_CONSTRUCTION_TICKS_PER_BAR,
            metadata_required=True,
            symbol=primary_symbol,
        )
    except Exception as exc:
        dataset_integrity_report = {
            "passed": False,
            "dataset_path": str(data_path),
            "expected_ticks_per_bar": int(BAR_CONSTRUCTION_TICKS_PER_BAR),
            "symbol": primary_symbol,
            "error": str(exc),
        }
        save_json_report(dataset_integrity_report, dataset_integrity_report_path)
        _write_current_training_run_context(
            run_id=run_id,
            symbol=primary_symbol,
            checkpoints_root=run_checkpoints_root,
            state="failed_dataset_integrity",
            dataset_integrity_report_path=dataset_integrity_report_path,
            dataset_integrity_verified=False,
        )
        raise
    save_json_report(dataset_integrity_report, dataset_integrity_report_path)
    dataset_integrity_verified = bool(TRAIN_DATASET_INTEGRITY_VERIFIED or dataset_integrity_report.get("passed", False))
    integrity_symbol_report = dataset_integrity_report["symbol_reports"].get(primary_symbol, {})
    print(
        "[INFO] Dataset integrity OK "
        f"for {primary_symbol}: bars={integrity_symbol_report.get('rows', 0):,} "
        f"raw_tick_rows={integrity_symbol_report.get('raw_tick_rows', 0):,} "
        f"max_gap_hours={integrity_symbol_report.get('max_gap_hours', 0.0):.2f}"
    )
    print(f"[INFO] Dataset integrity report -> {dataset_integrity_report_path}")
    legacy_checkpoint_archives = [] if resume_state is not None else _clear_legacy_checkpoint_artifacts(run_id=run_id)
    if legacy_checkpoint_archives:
        print(
            f"[INFO] Archived {len(legacy_checkpoint_archives)} legacy checkpoint artifacts "
            f"under checkpoints/archive/{run_id}"
        )
    _write_current_training_run_context(
        run_id=run_id,
        symbol=primary_symbol,
        checkpoints_root=run_checkpoints_root,
        state="starting",
        dataset_integrity_report_path=dataset_integrity_report_path,
        dataset_integrity_verified=dataset_integrity_verified,
    )

    # ── Config ───────────────────────────────────────────────────────────────
    SL_OPTS = list(ACTION_SL_MULTS)
    TP_OPTS = list(ACTION_TP_MULTS)
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # ── Per-symbol feature computation + purged walk-forward ─────────────────
    sym_folds: dict[str, list[tuple[pd.DataFrame, pd.DataFrame]]] = {}
    sym_computed_frames: dict[str, pd.DataFrame] = {}
    sym_trainable_frames: dict[str, pd.DataFrame] = {}
    sym_holdout_frames: dict[str, pd.DataFrame] = {}
    holdout_starts: dict[str, str] = {}
    holdout_sizes: dict[str, int] = {}

    for sym in symbols:
        sdf = df_raw[df_raw["Symbol"] == sym].copy()
        validate_symbol_bar_spec(
            sdf,
            expected_ticks_per_bar=BAR_CONSTRUCTION_TICKS_PER_BAR,
            symbol=sym,
        )
        computed = _compute_raw(sdf)
        computed.dropna(subset=[c for c in FEATURE_COLS if c in computed.columns], inplace=True)
        minimums = training_data_minimums()
        min_symbol_bars = minimums["min_train_bars"] + minimums["min_holdout_bars"]
        if len(computed) < min_symbol_bars:
            print(
                f"  Skipping {sym}: only {len(computed)} bars after warm-up; "
                f"need at least {min_symbol_bars} bars for train+holdout minimums"
            )
            continue
        trainable, holdout = _split_holdout(computed, HOLDOUT_FRAC)
        holdout_blockers = assess_training_data_sufficiency(
            train_bars=len(trainable),
            holdout_bars=len(holdout),
        )
        if holdout_blockers:
            print(f"  Skipping {sym}: {'; '.join(holdout_blockers)}")
            continue
        sym_computed_frames[sym] = computed
        sym_trainable_frames[sym] = trainable
        sym_holdout_frames[sym] = holdout
        holdout_starts[sym] = pd.Timestamp(holdout.index[0]).isoformat()
        holdout_sizes[sym] = int(len(holdout))

        # Purged walk-forward folds
        print(f"\n  {sym}: {len(computed)} bars — generating {N_FOLDS} purged folds")
        print(
            f"\n  {sym}: {len(trainable)} train/CV bars + {len(holdout)} holdout bars "
            f"â€” generating {N_FOLDS} purged folds"
        )
        folds = purged_walk_forward_splits(trainable, n_folds=N_FOLDS,
                                           test_frac=FOLD_TEST_FRAC,
                                           purge_gap=PURGE_GAP_BARS)
        if not folds:
            print(f"  Skipping {sym}: no folds satisfied minimum train/validation bar requirements.")
            continue
        sym_folds[sym] = folds

    if not sym_folds:
        minimums = training_data_minimums()
        _write_current_training_run_context(
            run_id=run_id,
            symbol=primary_symbol,
            checkpoints_root=run_checkpoints_root,
            state="failed_data_minimums",
            dataset_integrity_report_path=dataset_integrity_report_path,
            dataset_integrity_verified=dataset_integrity_verified,
        )
        raise RuntimeError(
            "No symbols had enough data. "
            f"Required minimums: train>={minimums['min_train_bars']}, "
            f"val>={minimums['min_val_bars']}, holdout>={minimums['min_holdout_bars']} bars."
        )
    feature_cols = FEATURE_COLS
    baseline_report_path = run_checkpoints_root / f"baseline_diagnostics_{primary_symbol}.json"
    baseline_report = run_baseline_research_gate(
        symbol=primary_symbol,
        trainable_frame=sym_trainable_frames[primary_symbol],
        holdout_frame=sym_holdout_frames[primary_symbol],
        folds=sym_folds[primary_symbol],
        feature_cols=feature_cols,
        out_path=baseline_report_path,
        horizon_bars=BASELINE_TARGET_HORIZON_BARS,
    )
    print(
        f"[BaselineGate] symbol={primary_symbol} gate_passed={baseline_report['gate_passed']} "
        f"passing_models={baseline_report.get('passing_models', [])}"
    )
    if not TRAIN_DEBUG_ALLOW_BASELINE_BYPASS and not bool(baseline_report["gate_passed"]):
        _write_current_training_run_context(
            run_id=run_id,
            symbol=primary_symbol,
            checkpoints_root=run_checkpoints_root,
            state="failed_baseline_gate",
            dataset_integrity_report_path=dataset_integrity_report_path,
            dataset_integrity_verified=dataset_integrity_verified,
            baseline_report_path=baseline_report_path,
        )
        raise RuntimeError("RL not justified: baseline gate failed.")

    execution_cost_profile = build_execution_cost_profile(
        slippage_pips=get_final_slippage_pips(TRAINING_RECOVERY_CONFIG)
    )
    reward_profile = build_reward_profile()

    archived_paths = _clear_current_run_artifacts(
        primary_symbol=primary_symbol,
        model_artifact_path=model_artifact_path,
        vecnormalize_artifact_path=vecnormalize_artifact_path,
        run_id=run_id,
    )
    if archived_paths:
        print(
            f"[INFO] Archived {len(archived_paths)} stale deploy artifacts for {primary_symbol} "
            f"under models/archive/{primary_symbol.lower()}/{run_id}"
        )

    # ── Training across folds ─────────────────────────────────────────────────
    resume_fold_index = int(resume_state["fold_index"]) if resume_state is not None else 0
    if resume_state is not None:
        (
            best_observed_sharpe,
            best_observed_summary,
            candidate_rank,
            candidate_summary,
            candidate_model_source,
            candidate_vecnormalize_source,
            candidate_scaler_source,
        ) = _recover_completed_fold_state(
            run_id=run_id,
            checkpoints_root=run_checkpoints_root,
            primary_symbol=primary_symbol,
        )
        recovered_folds = len(list(run_checkpoints_root.glob("fold_*/training_diagnostics.json")))
        if recovered_folds:
            print(f"[Resume] Recovered {recovered_folds} completed fold diagnostics from {run_checkpoints_root}.")
    else:
        best_observed_sharpe = -np.inf
        best_observed_summary = None
        candidate_rank = None
        candidate_summary = None
        candidate_model_source = None
        candidate_vecnormalize_source = None
        candidate_scaler_source = None

    for fold_idx in range(N_FOLDS):
        if resume_state is not None and fold_idx < resume_fold_index:
            print(f"[Resume] Skipping completed fold {fold_idx}.")
            continue
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1} / {N_FOLDS}")
        print(f"{'='*60}")

        # Gather train/val DataFrames across symbols for this fold
        fold_trains, fold_vals = [], []
        fold_scalers: dict[str, StandardScaler] = {}
        for sym, folds in sym_folds.items():
            if fold_idx < len(folds):
                fold_train, fold_val, fold_scaler = _fit_and_apply_fold_scaler(folds[fold_idx][0], folds[fold_idx][1])
                fold_trains.append(fold_train)
                fold_vals.append(fold_val)
                fold_scalers[sym] = fold_scaler

        if not fold_trains:
            continue

        train_df = pd.concat(fold_trains)
        val_df   = pd.concat(fold_vals)
        print(f"  Train: {len(train_df):,} bars  |  Val (purged): {len(val_df):,} bars")

        # Parallel training envs — starts with 0 slippage (Curriculum Phase 1)
        train_recovery_cfg = build_train_env_recovery_config(TRAINING_RECOVERY_CONFIG, env_workers=effective_envs)
        initial_curriculum_slippage = get_current_slippage_pips(0, TRAINING_RECOVERY_CONFIG)
        final_curriculum_slippage = get_final_slippage_pips(TRAINING_RECOVERY_CONFIG)
        sym_list = list(sym_folds.keys())
        # Pre-convert bars for all symbols in this fold to avoid O(N) conversion in every worker
        sym_bars_fold: dict[str, Any] = {}
        if TRAIN_ENV_MODE == "runtime":
            from runtime_gym_env import RuntimeGymEnv
            import tempfile
            for sym in sym_list:
                sdf = sym_folds[sym][fold_idx][0]
                bars_arr = RuntimeGymEnv._frame_to_bars(sdf)
                if TRAIN_SHARED_DATASET:
                    # Unique path to avoid Errno 22 / Access Denied on Windows re-runs
                    mmap_name = f"bars_{sym}_{fold_idx}_{os.getpid()}_{int(time.time())}.dat"
                    mmap_path = os.path.join(tempfile.gettempdir(), mmap_name)
                    mmap_arr = np.memmap(mmap_path, dtype=bars_arr.dtype, mode='w+', shape=bars_arr.shape)
                    mmap_arr[:] = bars_arr[:]
                    mmap_arr.flush()
                    # np.memmap pickles fine, the worker will reopen it
                    sym_bars_fold[sym] = mmap_arr
                else:
                    sym_bars_fold[sym] = bars_arr

        def make_parallel(rank: int):
            sym = sym_list[rank % len(sym_list)]
            if TRAIN_ENV_MODE == "runtime":
                sdf = sym_folds[sym][fold_idx][0]
                fold_scaler = fold_scalers.get(sym)
            else:
                sym_rows = train_df[train_df.get("Symbol", pd.Series(dtype=str)) == sym] \
                           if "Symbol" in train_df.columns else train_df
                sdf = sym_rows if len(sym_rows) > 300 else train_df
                fold_scaler = None
            return make_env(
                sdf,
                feature_cols,
                SL_OPTS,
                TP_OPTS,
                random_start=True,
                initial_slippage=initial_curriculum_slippage,
                symbol=sym,
                scaler=fold_scaler,
                recovery_config=train_recovery_cfg if TRAIN_ENV_MODE == "runtime" else None,
                bars=sym_bars_fold.get(sym),
            )

        val_symbol = primary_symbol
        if TRAIN_ENV_MODE == "runtime":
            trainable_frame = sym_trainable_frames.get(val_symbol)
            if trainable_frame is None:
                raise RuntimeError(f"Missing trainable frame for {val_symbol}.")
            val_segment = sym_folds[val_symbol][fold_idx][1]
            val_source = _prepend_runtime_warmup_context(trainable_frame, val_segment)
            val_scaler = fold_scalers.get(val_symbol)
            holdout_frame = sym_holdout_frames.get(val_symbol)
            full_frame = sym_computed_frames.get(val_symbol)
            if holdout_frame is None or full_frame is None:
                raise RuntimeError(f"Missing holdout/full computed frame for {val_symbol}.")
            holdout_source = _prepend_runtime_warmup_context(full_frame, holdout_frame)
            holdout_scaler = fold_scalers.get(val_symbol)
        else:
            val_rows = val_df[val_df.get("Symbol", pd.Series(dtype=str)) == val_symbol] \
                       if "Symbol" in val_df.columns else val_df
            val_source = val_rows if len(val_rows) > 300 else val_df
            val_scaler = None
            holdout_source = sym_holdout_frames.get(val_symbol)
            if holdout_source is None:
                raise RuntimeError(f"Missing holdout frame for {val_symbol}.")
            holdout_scaler = None

        train_fns = [make_parallel(i) for i in range(effective_envs)]
        if vec_env_type == "dummy":
            train_base_vec = DummyVecEnv(train_fns)
        else:
            train_base_vec = SubprocVecEnv(train_fns)

        ckpt_dir = run_checkpoints_root / f"fold_{fold_idx}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_vecnormalize_path = ckpt_dir / BEST_VECNORMALIZE_NAME
        resume_model_path = _resume_model_checkpoint_path(ckpt_dir)
        resume_vecnormalize_path = _resume_vecnormalize_checkpoint_path(ckpt_dir)
        resume_current_fold = bool(
            resume_state is not None
            and fold_idx == resume_fold_index
            and Path(str(resume_state["model_path"])).exists()
        )
        if resume_current_fold and resume_state.get("vecnormalize_path") is not None:
            train_vec = VecNormalize.load(str(resume_state["vecnormalize_path"]), train_base_vec)
            train_vec.training = True
            train_vec.norm_reward = False
        else:
            train_vec = wrap_vecnormalize(train_base_vec, training=True)

        print(
            f"[INFO] Supported training stack: MaskablePPO + RuntimeGymEnv + volume bars{msg_amp} | "
            f"bar_construction_ticks_per_bar={BAR_CONSTRUCTION_TICKS_PER_BAR}"
        )

        def build_single_eval_base_vec(
            source_frame: pd.DataFrame,
            scaler: StandardScaler | None,
            *,
            slippage_pips: float,
        ):
            return DummyVecEnv(
                [
                    make_env(
                        source_frame,
                        feature_cols,
                        SL_OPTS,
                        TP_OPTS,
                        random_start=False,
                        initial_slippage=slippage_pips,
                        symbol=val_symbol,
                        scaler=scaler,
                        recovery_config=None,
                    )
                ]
            )

        def build_single_eval_vec(
            source_frame: pd.DataFrame,
            scaler: StandardScaler | None,
            *,
            slippage_pips: float,
        ):
            return wrap_vecnormalize(
                build_single_eval_base_vec(source_frame, scaler, slippage_pips=slippage_pips),
                training=False,
            )

        val_vec = build_single_eval_vec(val_source, val_scaler, slippage_pips=final_curriculum_slippage)
        holdout_vec = build_single_eval_vec(
            holdout_source,
            holdout_scaler,
            slippage_pips=final_curriculum_slippage,
        )
        sync_vecnormalize_stats(train_vec, val_vec)
        sync_vecnormalize_stats(train_vec, holdout_vec)

        _write_current_training_run_context(
            run_id=run_id,
            symbol=primary_symbol,
            checkpoints_root=run_checkpoints_root,
            state="training",
            fold_index=fold_idx,
            heartbeat_path=ckpt_dir / "training_heartbeat.json",
            dataset_integrity_report_path=dataset_integrity_report_path,
            dataset_integrity_verified=dataset_integrity_verified,
            baseline_report_path=baseline_report_path,
        )

        tensorboard_log = "./tensorboard_log" if importlib.util.find_spec("tensorboard") is not None else None

        if resume_current_fold:
            ppo_class = MaskablePPO_AMP if use_amp else MaskablePPO
            model = ppo_class.load(str(resume_state["model_path"]), env=train_vec, device=device)
            
            if use_amp:
                # Re-inject AMP config if resuming
                setattr(model, "amp_dtype", (th.bfloat16 if amp_dtype == "bf16" else th.float16))
                setattr(model, "scaler", th.cuda.amp.GradScaler(enabled=(amp_dtype == "fp16")))

            remaining_timesteps = max(TOTAL_TIMESTEPS - int(getattr(model, "num_timesteps", 0) or 0), 0)
            print(
                f"[Resume] Loaded fold {fold_idx} checkpoint {resume_state['model_path']} | "
                f"num_timesteps={int(getattr(model, 'num_timesteps', 0) or 0):,} | "
                f"remaining={remaining_timesteps:,}"
            )
        else:
            ppo_class = MaskablePPO_AMP if use_amp else MaskablePPO
            ppo_kwargs = {}
            if use_amp:
                ppo_kwargs["amp_dtype"] = amp_dtype

            model = ppo_class(
                "MlpPolicy",
                train_vec,
                verbose=1,
                learning_rate=linear_schedule(PPO_LEARNING_RATE, PPO_MIN_LEARNING_RATE),
                n_steps=PPO_N_STEPS,
                batch_size=PPO_BATCH_SIZE,
                n_epochs=PPO_N_EPOCHS,
                ent_coef=get_current_ent_coef(0, TRAINING_RECOVERY_CONFIG),
                target_kl=PPO_TARGET_KL,
                policy_kwargs=dict(net_arch=dict(pi=[POLICY_WIDTH, POLICY_WIDTH], vf=[256, 256, 128])),
                tensorboard_log=tensorboard_log,
                device=device,
                seed=SEED + fold_idx,   # different seed per fold
                **ppo_kwargs
            )
            remaining_timesteps = TOTAL_TIMESTEPS

        # Callbacks: curriculum annealing + checkpoint + eval
        curriculum_cb = CurriculumCallback(
            TRAINING_RECOVERY_CONFIG,
            train_env=train_vec,
            verbose=1,
        )
        diagnostics_cb = TrainingDiagnosticsCallback(verbose=int(os.environ.get("TRAIN_PROGRESS_VERBOSE", "1")))
        heartbeat_default_steps = (
            max(int((TRAINING_RECOVERY_CONFIG.get("diagnostics", {}) or {}).get("heartbeat_every_n_updates", 10)), 1)
            * PPO_N_STEPS
            * max(effective_envs, 1)
        )
        heartbeat_every = int(os.environ.get("TRAIN_HEARTBEAT_EVERY_STEPS", str(heartbeat_default_steps)))
        if TRAIN_REDUCE_LOGGING:
            heartbeat_every = max(heartbeat_every, 20_000)
            
        eval_freq = max(TRAIN_EVAL_FREQ // max(effective_envs, 1), 1)
        if TRAIN_ASYNC_EVAL:
            # We delay saving vecnormalize and evaluating deeply by using a much larger frequency
            eval_freq = max(eval_freq, 50_000)

        eval_cb = FullPathEvalCallback(
            val_vec,
            train_vecnormalize=train_vec,
            eval_vecnormalize=val_vec,
            best_model_save_path=ckpt_dir,
            best_vecnormalize_path=best_vecnormalize_path,
            history_path=Path(ckpt_dir) / "full_path_evaluations.json",
            eval_freq=eval_freq,
            verbose=1,
        )
        if resume_current_fold:
            _prime_eval_callback_from_history(eval_cb)
        heartbeat_cb = TrainingHeartbeatCallback(
            out_path=Path(ckpt_dir) / "training_heartbeat.json",
            diagnostics_cb=diagnostics_cb,
            curriculum_cb=curriculum_cb,
            eval_cb=eval_cb,
            run_id=run_id,
            symbol=primary_symbol,
            checkpoints_root=run_checkpoints_root,
            fold_index=fold_idx,
            every_steps=heartbeat_every,
            total_timesteps=TOTAL_TIMESTEPS,
            dataset_integrity_report_path=dataset_integrity_report_path,
            dataset_integrity_verified=dataset_integrity_verified,
            baseline_report_path=baseline_report_path,
            resume_model_path=resume_model_path,
            resume_vecnormalize_path=resume_vecnormalize_path,
            verbose=0,
        )

        if remaining_timesteps > 0:
            if TRAIN_TORCH_COMPILE and hasattr(model, "policy"):
                # Windows/Triton availability check
                is_windows = sys.platform.startswith("win")
                try:
                    import triton
                    has_triton = True
                except ImportError:
                    has_triton = False

                if is_windows and not has_triton:
                    print("[Torch] Waning: Triton not found on Windows. Skipping torch.compile to avoid inducer crash (use a Triton-supported OS/backend for speedups).")
                else:
                    print(f"[Torch] Compiling policy network with mode={TRAIN_TORCH_COMPILE_MODE} ...")
                    compile_start = time.perf_counter()
                    try:
                        model.policy = torch.compile(model.policy, mode=TRAIN_TORCH_COMPILE_MODE)
                        print(f"[Torch] Compilation tracing initiated (elapsed: {time.perf_counter() - compile_start:.2f}s). Final code generation will happen on first forward pass.")
                    except Exception as e:
                        print(f"[Torch] Compilation failed: {e}. Falling back to standard policy.")

            model.learn(
                total_timesteps=remaining_timesteps,
                callback=[curriculum_cb, diagnostics_cb, eval_cb, heartbeat_cb],
                log_interval=max(TRAIN_LOG_INTERVAL, 1),
                tb_log_name=f"{primary_symbol.lower()}_fold_{fold_idx}",
                reset_num_timesteps=not resume_current_fold,
            )
        else:
            print(f"[Resume] Fold {fold_idx} already reached the requested total timesteps; skipping learn().")

        if not best_vecnormalize_path.exists():
            train_vec.save(str(best_vecnormalize_path))

        # Evaluate best checkpoint from this fold
        best_path = ckpt_dir / "best_model.zip"
        ppo_class = MaskablePPO_AMP if use_amp else MaskablePPO
        fold_model = ppo_class.load(str(best_path), device="cpu") if best_path.exists() else model
        if best_vecnormalize_path.exists():
            val_vec.close()
            holdout_vec.close()
            val_vec = VecNormalize.load(
                str(best_vecnormalize_path),
                build_single_eval_base_vec(val_source, val_scaler, slippage_pips=final_curriculum_slippage),
            )
            val_vec.training = False
            val_vec.norm_reward = False
            holdout_vec = VecNormalize.load(
                str(best_vecnormalize_path),
                build_single_eval_base_vec(
                    holdout_source,
                    holdout_scaler,
                    slippage_pips=final_curriculum_slippage,
                ),
            )
            holdout_vec.training = False
            holdout_vec.norm_reward = False

        _, evaluation_metrics = evaluate_model(fold_model, val_vec)
        _, holdout_metrics = evaluate_model(fold_model, holdout_vec)
        sharpe = float(evaluation_metrics["timed_sharpe"])
        max_dd = float(evaluation_metrics["max_drawdown"])
        fin_val = float(evaluation_metrics["final_equity"])
        val_profit_factor = float(evaluation_metrics.get("profit_factor", 0.0))
        val_expectancy = float(evaluation_metrics.get("expectancy", 0.0))
        val_trade_count = int(evaluation_metrics.get("trade_count", 0) or 0)
        holdout_sharpe = float(holdout_metrics["timed_sharpe"])
        holdout_max_dd = float(holdout_metrics["max_drawdown"])
        holdout_final_equity = float(holdout_metrics["final_equity"])
        holdout_profit_factor = float(holdout_metrics.get("profit_factor", 0.0))
        holdout_expectancy = float(holdout_metrics.get("expectancy", 0.0))
        holdout_trade_count = int(holdout_metrics.get("trade_count", 0) or 0)
        callback_summary = diagnostics_cb.summary()
        diagnostics = summarize_training_diagnostics(
            [
                {
                    "approx_kl": approx_kl,
                    "explained_variance": explained_variance,
                    "value_loss": value_loss,
                }
                for approx_kl, explained_variance, value_loss in zip(
                    diagnostics_cb.metrics["train/approx_kl"],
                    diagnostics_cb.metrics["train/explained_variance"],
                    diagnostics_cb.metrics["train/value_loss"],
                )
            ]
        )
        diagnostics["fold"] = fold_idx
        diagnostics["approx_kl"] = float(callback_summary["approx_kl"])
        diagnostics["explained_variance"] = float(callback_summary["explained_variance"])
        diagnostics["n_updates"] = callback_summary.get("n_updates")
        diagnostics["diagnostic_sample_count"] = callback_summary.get("diagnostic_sample_count")
        diagnostics["last_distinct_update_seen"] = callback_summary.get("last_distinct_update_seen")
        diagnostics["metrics_fresh"] = callback_summary.get("metrics_fresh")
        diagnostics["val_sharpe"] = float(sharpe)
        diagnostics["val_max_drawdown"] = float(max_dd)
        diagnostics["val_final_equity"] = float(fin_val)
        diagnostics["val_profit_factor"] = float(val_profit_factor)
        diagnostics["val_expectancy"] = float(val_expectancy)
        diagnostics["val_trade_count"] = int(val_trade_count)
        diagnostics["holdout_sharpe"] = float(holdout_sharpe)
        diagnostics["holdout_max_drawdown"] = float(holdout_max_dd)
        diagnostics["holdout_final_equity"] = float(holdout_final_equity)
        diagnostics["holdout_profit_factor"] = float(holdout_profit_factor)
        diagnostics["holdout_expectancy"] = float(holdout_expectancy)
        diagnostics["holdout_trade_count"] = int(holdout_trade_count)
        diagnostics["baseline_gate_passed"] = bool(baseline_report["gate_passed"])
        diagnostics["baseline_gate_bypassed_debug"] = bool(TRAIN_DEBUG_ALLOW_BASELINE_BYPASS)
        diagnostics["baseline_target_horizon_bars"] = int(BASELINE_TARGET_HORIZON_BARS)
        diagnostics["baseline_report_path"] = str(baseline_report_path)
        diagnostics["eval_protocol_valid"] = True
        diagnostics["full_path_eval_used"] = True
        diagnostics["full_path_validation_metrics"] = evaluation_metrics
        diagnostics["holdout_metrics"] = holdout_metrics
        diagnostics["segment_metrics"] = {
            "validation": evaluation_metrics.get("segment_metrics", {}),
            "holdout": holdout_metrics.get("segment_metrics", {}),
        }
        diagnostics["train_bars"] = int(len(train_df))
        diagnostics["val_bars"] = int(len(val_df))
        diagnostics["holdout_bars"] = int(holdout_sizes.get(primary_symbol, 0))
        diagnostics["point_in_time_verified"] = bool(point_in_time_verified)
        diagnostics["dataset_integrity_verified"] = bool(dataset_integrity_verified)
        diagnostics["dataset_integrity_report_path"] = str(dataset_integrity_report_path)
        diagnostics["execution_cost_profile"] = dict(execution_cost_profile)
        diagnostics["reward_profile"] = dict(reward_profile)
        diagnostics["env_workers"] = int(effective_envs)
        diagnostics["vec_env_type"] = vec_env_type
        diagnostics["bar_construction_ticks_per_bar"] = int(BAR_CONSTRUCTION_TICKS_PER_BAR)
        diagnostics["ticks_per_bar"] = int(BAR_CONSTRUCTION_TICKS_PER_BAR)
        data_sufficiency_blockers = assess_training_data_sufficiency(
            train_bars=int(len(train_df)),
            val_bars=int(len(val_df)),
            holdout_bars=int(holdout_sizes.get(primary_symbol, 0)),
        )
        diagnostics.setdefault("blockers", [])
        diagnostics["data_sufficiency_passed"] = not data_sufficiency_blockers
        if data_sufficiency_blockers:
            diagnostics["blockers"] = list(dict.fromkeys([*diagnostics["blockers"], *data_sufficiency_blockers]))
            diagnostics["gate_passed"] = False
        holdout_gate_blockers = _holdout_deployment_blockers(
            holdout_sharpe=holdout_sharpe,
            holdout_max_drawdown=holdout_max_dd,
            holdout_final_equity=holdout_final_equity,
            holdout_profit_factor=holdout_profit_factor,
            holdout_expectancy=holdout_expectancy,
            holdout_trade_count=holdout_trade_count,
        )
        diagnostics["holdout_gate_passed"] = not holdout_gate_blockers
        if holdout_gate_blockers:
            diagnostics["blockers"] = list(dict.fromkeys([*diagnostics["blockers"], *holdout_gate_blockers]))
        diagnostics["deploy_ready"] = bool(
            diagnostics["gate_passed"]
            and max_dd <= DEPLOY_DD_MAX
            and not holdout_gate_blockers
        )
        metrics_path = ckpt_dir / "training_diagnostics.json"
        save_json_report(diagnostics, metrics_path)

        if sharpe > best_observed_sharpe:
            best_observed_sharpe = sharpe
            best_observed_summary = _build_promoted_training_diagnostics(
                diagnostics,
                run_id=run_id,
                artifact_candidate_selected=False,
                artifact_candidate_reason=(
                    "Best-evaluated fold from this run did not meet deployment artifact criteria."
                    if not diagnostics["deploy_ready"]
                    else "Best-evaluated fold from this run is deployment eligible but was not selected as the canonical candidate."
                ),
            )

        print(
            f"\n  Fold {fold_idx} VAL: equity=${fin_val:,.2f}  "
            f"Sharpe={sharpe:.3f}  MaxDD={max_dd:.1%}  "
            f"PF={val_profit_factor:.2f}  Trades={val_trade_count}"
        )
        print(
            "  PPO diagnostics: "
            f"explained_variance={diagnostics.get('explained_variance', 0.0):.3f} "
            f"approx_kl={diagnostics.get('approx_kl', 0.0):.3f} "
            f"value_loss_mean={diagnostics.get('value_loss_mean', 0.0):.3f}"
        )

        current_candidate_rank = _deployment_candidate_rank(diagnostics)
        if current_candidate_rank is not None and (candidate_rank is None or current_candidate_rank > candidate_rank):
            candidate_rank = current_candidate_rank
            candidate_summary = _build_promoted_training_diagnostics(
                diagnostics,
                run_id=run_id,
                artifact_candidate_selected=True,
                artifact_candidate_reason="Selected deployment artifact candidate from the current run.",
            )
            candidate_model_source = ckpt_dir / "deployment_candidate_model.zip"
            candidate_scaler_source = _candidate_scaler_artifact_path(ckpt_dir, primary_symbol)
            fold_model.save(str(candidate_model_source.with_suffix("")))
            primary_scaler = fold_scalers.get(primary_symbol)
            if primary_scaler is None:
                raise RuntimeError(f"Deployment candidate is missing a scaler for {primary_symbol}.")
            joblib.dump(primary_scaler, candidate_scaler_source)
            if best_vecnormalize_path.exists():
                candidate_vecnormalize_source = ckpt_dir / "deployment_candidate_vecnormalize.pkl"
                shutil.copyfile(best_vecnormalize_path, candidate_vecnormalize_source)
            else:
                candidate_vecnormalize_source = None
            print(
                "  Selected deployment candidate "
                f"(holdout_sharpe={holdout_sharpe:.3f}, val_sharpe={sharpe:.3f}) -> {candidate_model_source}"
            )
        elif diagnostics["deploy_ready"]:
            print("  Fold eligible but not promoted: an earlier fold has a stronger holdout ranking.")
        else:
            print("  Fold rejected for deployment: diagnostics, validation drawdown, or holdout gate failed.")

        train_vec.close()
        val_vec.close()
        holdout_vec.close()

    # ── Final cross-fold summary ──────────────────────────────────────────────
    if best_observed_summary is None:
        raise RuntimeError("Training completed without producing fold diagnostics.")

    print(f"\n{'='*60}")
    print(f"Training complete. Best observed Sharpe across folds: {best_observed_sharpe:.3f}")
    canonical_summary = candidate_summary or best_observed_summary
    _write_current_training_run_context(
        run_id=run_id,
        symbol=primary_symbol,
        checkpoints_root=run_checkpoints_root,
        state="completed",
        dataset_integrity_report_path=dataset_integrity_report_path,
        dataset_integrity_verified=dataset_integrity_verified,
        baseline_report_path=baseline_report_path,
    )
    report_paths = deployment_paths(primary_symbol)
    save_json_report(canonical_summary, report_paths.diagnostics_path)
    print(f"Canonical diagnostics -> {report_paths.diagnostics_path}")
    if candidate_summary is not None and candidate_model_source is not None and candidate_scaler_source is not None:
        _publish_primary_candidate_artifacts(
            primary_symbol=primary_symbol,
            model_artifact_path=model_artifact_path,
            vecnormalize_artifact_path=vecnormalize_artifact_path,
            candidate_model_source=candidate_model_source,
            candidate_vecnormalize_source=candidate_vecnormalize_source,
            candidate_scaler_source=candidate_scaler_source,
            holdout_start_utc=holdout_starts.get(primary_symbol),
            dataset_path=data_path,
            execution_cost_profile=execution_cost_profile,
            reward_profile=reward_profile,
        )
        print(f"Deployment candidate saved -> {model_artifact_path}")
    else:
        print("No deployment artifact candidate was selected from this run. Canonical model/manifests remain cleared.")
    print(f"Run evaluate_oos.py for final out-of-sample validation.")


if __name__ == "__main__":
    main()
