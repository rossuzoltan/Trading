from __future__ import annotations
import os
import copy
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
from trading_config import (
    DEFAULT_MIN_LEARNING_RATE,
    DEFAULT_TARGET_KL,
    DEFAULT_SLIPPAGE_START_PIPS,
    DEFAULT_SLIPPAGE_END_PIPS,
    DEFAULT_CHURN_MIN_HOLD_BARS,
    DEFAULT_CHURN_ACTION_COOLDOWN,
    DEFAULT_CHURN_PENALTY_USD,
    DEFAULT_ENTRY_SPREAD_Z_LIMIT,
    DEFAULT_REWARD_DOWNSIDE_RISK_COEF,
    DEFAULT_REWARD_TURNOVER_COEF,
    DEFAULT_REWARD_DRAWDOWN_COEF,
    DEFAULT_REWARD_NET_RETURN_COEF,
    resolve_bar_construction_ticks_per_bar,
)

def _resolve_training_experiment_profile(profile_name: str) -> dict[str, Any]:
    normalized = str(profile_name or "").strip().lower()
    if normalized in {"", "default", "none"}:
        return {}
    if normalized in {"reward_strip", "finalboss_reward_strip"}:
        return {
            "reward_downside_risk_coef": 0.0,
            "reward_turnover_coef": 0.0,
            "churn_penalty_usd": 0.0,
            "reward_net_return_coef": 1.0,
            "reward_scale": 1000.0,
            "reward_clip_low": -10.0,
            "reward_clip_high": 10.0,
            "participation_bonus_enabled": False,
        }
    if normalized in {"reward_strip_hard_churn", "finalboss_reward_strip_hard_churn"}:
        profile = _resolve_training_experiment_profile("reward_strip")
        profile.update(
            {
                "churn_min_hold_bars": 5,
                "churn_action_cooldown": 3,
            }
        )
        return profile
    if normalized in {"reward_strip_hard_churn_alpha_gate", "finalboss_reward_strip_hard_churn_alpha_gate"}:
        profile = _resolve_training_experiment_profile("reward_strip_hard_churn")
        profile.update(
            {
                "alpha_gate_enabled": True,
                "alpha_gate_model": "auto",
            }
        )
        return profile
    if normalized in {"reward_strip_rehab_safer_alpha_gate", "finalboss_reward_strip_rehab_safer_alpha_gate"}:
        profile = _resolve_training_experiment_profile("reward_strip")
        profile.update(
            {
                "churn_min_hold_bars": 8,
                "churn_action_cooldown": 5,
                "entry_spread_z_limit": 0.75,
                "alpha_gate_enabled": True,
                "alpha_gate_model": "auto",
                "alpha_gate_warmup_steps": 100_000,
                "alpha_gate_warmup_threshold_delta": 0.10,
                "alpha_gate_warmup_margin_scale": 0.0,
                "ppo_learning_rate": 8e-4,
                "ppo_target_kl": 0.02,
                "adaptive_kl_lr": True,
                "adaptive_kl_max_lr": 0.003,
                "adaptive_kl_low": 0.002,
                "adaptive_kl_up_mult": 2.5,
                "recovery_entropy_mid": 0.01,
                "recovery_entropy_final": 0.003,
                "fail_fast_enabled": True,
                "fail_fast_warmup_steps": 60_000,
                "fail_fast_consecutive": 3,
                "fail_fast_sparse_alpha_gate_block_rate": 0.94,
                "fail_fast_approx_kl_max": 0.001,
                "fail_fast_explained_variance_max": 0.20,
                "fail_fast_max_trade_count": 12,
            }
        )
        return profile
    if normalized in {"reward_strip_profit_rehab_v1", "finalboss_reward_strip_profit_rehab_v1"}:
        profile = _resolve_training_experiment_profile("reward_strip")
        profile.update(
            {
                "alpha_gate_enabled": False,
                "churn_min_hold_bars": 12,
                "churn_action_cooldown": 8,
                "entry_spread_z_limit": 0.50,
                "churn_penalty_usd": 1.0,
                "reward_downside_risk_coef": 0.25,
                "reward_turnover_coef": 0.20,
                "reward_net_return_coef": 1.0,
                "reward_scale": 1000.0,
                "reward_clip_low": -10.0,
                "reward_clip_high": 10.0,
                "ppo_learning_rate": 4e-4,
                "ppo_target_kl": 0.015,
                "fail_fast_enabled": True,
                "fail_fast_warmup_steps": 40_000,
                "fail_fast_consecutive": 2,
                "fail_fast_overtrade_trade_count": 180,
                "fail_fast_cost_share_min": 1.0,
            }
        )
        return profile
    raise ValueError(
        "Unsupported TRAIN_EXPERIMENT_PROFILE="
        f"{profile_name!r}. Expected one of: reward_strip, reward_strip_hard_churn, "
        "reward_strip_hard_churn_alpha_gate, reward_strip_rehab_safer_alpha_gate, "
        "reward_strip_profit_rehab_v1."
    )


def _apply_profile_override(current_value: Any, env_var_name: str, override_value: Any) -> Any:
    if env_var_name in os.environ:
        return current_value
    return override_value


# ── Curriculum config ─────────────────────────────────────────────────────────
_BONUS_ANNEAL_START_STEP = 1_000_000
_BONUS_ANNEAL_END_STEP   = 2_500_000
_PARTICIPATION_BONUS_UNTIL_DEFAULT = 2_500_000
_ENTROPY_PHASE_1_UNTIL_DEFAULT = 1_000_000
_ENTROPY_PHASE_2_UNTIL_DEFAULT = 2_500_000
_ENTROPY_MIN_FLOOR = 0.002  # Raised from 0.001 to prevent total policy freeze

TRAIN_EXPERIMENT_PROFILE = os.environ.get("TRAIN_EXPERIMENT_PROFILE", "default").strip().lower()
TRAIN_EXPERIMENT_PROFILE_SETTINGS = _resolve_training_experiment_profile(TRAIN_EXPERIMENT_PROFILE)

SLIPPAGE_START   = DEFAULT_SLIPPAGE_START_PIPS
SLIPPAGE_END     = DEFAULT_SLIPPAGE_END_PIPS
TOTAL_TIMESTEPS  = int(os.environ.get("TRAIN_TOTAL_TIMESTEPS", "3000000"))
TRAIN_SYMBOL     = os.environ.get("TRAIN_SYMBOL", os.environ.get("TRADING_SYMBOL", "EURUSD")).strip().upper()
TRAIN_ENV_MODE = os.environ.get("TRAIN_ENV_MODE", "runtime").strip().lower()
TRAIN_ACTION_SPACE_MODE = os.environ.get("TRAIN_ACTION_SPACE_MODE", "simple").strip().lower() or "simple"
TRAIN_SIMPLE_ACTION_SL_MULT = float(os.environ.get("TRAIN_SIMPLE_ACTION_SL_MULT", "1.5"))
TRAIN_SIMPLE_ACTION_TP_MULT = float(os.environ.get("TRAIN_SIMPLE_ACTION_TP_MULT", "3.0"))
TRAIN_MODEL_DIR = Path(os.environ.get("TRAIN_MODEL_DIR", "models"))
PPO_LEARNING_RATE = float(os.environ.get("TRAIN_PPO_LEARNING_RATE", "4e-4"))
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
_TRAIN_EVAL_FREQ_RAW = os.environ.get("TRAIN_EVAL_FREQ")
TRAIN_EVAL_FREQ = int(_TRAIN_EVAL_FREQ_RAW or "20000")
TRAIN_LOG_INTERVAL = int(os.environ.get("TRAIN_LOG_INTERVAL", "5"))

TRAIN_TORCH_COMPILE = os.environ.get("TRAIN_TORCH_COMPILE", "0") == "1"
TRAIN_TORCH_COMPILE_MODE = os.environ.get("TRAIN_TORCH_COMPILE_MODE", "default").strip()
TRAIN_REDUCE_LOGGING = os.environ.get("TRAIN_REDUCE_LOGGING", "1") == "1"
TRAIN_ASYNC_EVAL = os.environ.get("TRAIN_ASYNC_EVAL", "0") == "1"
TRAIN_SHARED_DATASET = os.environ.get("TRAIN_SHARED_DATASET", "1") == "1"
TRAIN_USE_AMP = os.environ.get("TRAIN_USE_AMP", "0") == "1"
TRAIN_AMP_DTYPE = os.environ.get("TRAIN_AMP_DTYPE", "bf16").strip().lower()
TRAIN_EXPORT_BEST_FOLD = os.environ.get("TRAIN_EXPORT_BEST_FOLD", "0") == "1"
TRAIN_BENCH_SPS = os.environ.get("TRAIN_BENCH_SPS", "0") == "1"
TRAIN_EVAL_STOCHASTIC_RUNS = max(int(os.environ.get("TRAIN_EVAL_STOCHASTIC_RUNS", "0")), 0)
TRAIN_MINIMAL_POST_COST_REWARD = os.environ.get("TRAIN_MINIMAL_POST_COST_REWARD", "1") == "1"
TRAIN_FORCE_FAST_WINDOW_BENCHMARK = os.environ.get("TRAIN_FORCE_FAST_WINDOW_BENCHMARK", "0") == "1"
TRAIN_REQUIRE_RL_BEAT_BASELINE = os.environ.get("TRAIN_REQUIRE_RL_BEAT_BASELINE", "1") == "1"
TRAIN_ENABLE_ENHANCED_LOGGING = os.environ.get(
    "TRAIN_ENABLE_ENHANCED_LOGGING",
    "0" if TRAIN_REDUCE_LOGGING else "1",
) == "1"
TRAIN_ADAPTIVE_KL_LR = os.environ.get("TRAIN_ADAPTIVE_KL_LR", "0") == "1"
TRAIN_ADAPTIVE_KL_MAX_LR = float(os.environ.get("TRAIN_ADAPTIVE_KL_MAX_LR", "0.002"))
TRAIN_ADAPTIVE_KL_LOW = float(os.environ.get("TRAIN_ADAPTIVE_KL_LOW", "0.005"))
TRAIN_ADAPTIVE_KL_HIGH = float(os.environ.get("TRAIN_ADAPTIVE_KL_HIGH", "0.05"))
TRAIN_ADAPTIVE_KL_UP_MULT = float(os.environ.get("TRAIN_ADAPTIVE_KL_UP_MULT", "1.5"))
TRAIN_ADAPTIVE_KL_DOWN_MULT = float(os.environ.get("TRAIN_ADAPTIVE_KL_DOWN_MULT", "0.7"))
TRAIN_SPARSE_ALPHA_GATE_WARN_RATE = float(os.environ.get("TRAIN_SPARSE_ALPHA_GATE_WARN_RATE", "0.90"))
TRAIN_COLLAPSE_WARMUP_STEPS = max(int(os.environ.get("TRAIN_COLLAPSE_WARMUP_STEPS", "500000")), 0)
TRAIN_COLLAPSE_CONSECUTIVE = max(int(os.environ.get("TRAIN_COLLAPSE_CONSECUTIVE", "2")), 1)

if TRAIN_REDUCE_LOGGING:
    TRAIN_LOG_INTERVAL = max(TRAIN_LOG_INTERVAL, 20)
    if _TRAIN_EVAL_FREQ_RAW is None:
        # Default low-noise runs should not burn large chunks of wall time on dense
        # full-path evals unless the caller explicitly asked for them.
        TRAIN_EVAL_FREQ = max(TRAIN_EVAL_FREQ, 50000)
if TRAIN_FORCE_FAST_WINDOW_BENCHMARK and not TRAIN_BENCH_SPS:
    raise RuntimeError(
        "TRAIN_FORCE_FAST_WINDOW_BENCHMARK=1 is benchmark-only. "
        "Set TRAIN_BENCH_SPS=1 when using this falsifier."
    )
if TRAIN_ASYNC_EVAL:
    TRAIN_EVAL_FREQ = max(TRAIN_EVAL_FREQ, 100000)


TRAIN_CHURN_MIN_HOLD_BARS = int(os.environ.get("TRAIN_CHURN_MIN_HOLD_BARS", str(DEFAULT_CHURN_MIN_HOLD_BARS)))
TRAIN_CHURN_ACTION_COOLDOWN = int(os.environ.get("TRAIN_CHURN_ACTION_COOLDOWN", str(DEFAULT_CHURN_ACTION_COOLDOWN)))
TRAIN_CHURN_PENALTY_USD = float(os.environ.get("TRAIN_CHURN_PENALTY_USD", str(DEFAULT_CHURN_PENALTY_USD)))
TRAIN_ENTRY_SPREAD_Z_LIMIT = float(os.environ.get("TRAIN_ENTRY_SPREAD_Z_LIMIT", str(DEFAULT_ENTRY_SPREAD_Z_LIMIT)))

TRAIN_RISK_MAX_DRAWDOWN_FRACTION = float(os.environ.get("TRAIN_RISK_MAX_DRAWDOWN_FRACTION", "0.50"))
TRAIN_RISK_DAILY_LOSS_FRACTION = float(os.environ.get("TRAIN_RISK_DAILY_LOSS_FRACTION", "0.25"))
EVAL_RISK_MAX_DRAWDOWN_FRACTION = float(os.environ.get("EVAL_RISK_MAX_DRAWDOWN_FRACTION", "1.00"))
EVAL_RISK_DAILY_LOSS_FRACTION = float(os.environ.get("EVAL_RISK_DAILY_LOSS_FRACTION", "1.00"))

TRAIN_REWARD_DOWNSIDE_RISK_COEF = float(os.environ.get("TRAIN_REWARD_DOWNSIDE_RISK_COEF", str(DEFAULT_REWARD_DOWNSIDE_RISK_COEF)))
TRAIN_REWARD_TURNOVER_COEF = float(os.environ.get("TRAIN_REWARD_TURNOVER_COEF", str(DEFAULT_REWARD_TURNOVER_COEF)))
TRAIN_REWARD_NET_RETURN_COEF = float(os.environ.get("TRAIN_REWARD_NET_RETURN_COEF", str(DEFAULT_REWARD_NET_RETURN_COEF)))
TRAIN_REWARD_DRAWDOWN_PENALTY = float(os.environ.get("TRAIN_REWARD_DRAWDOWN_PENALTY", str(DEFAULT_REWARD_DRAWDOWN_COEF)))
PPO_ENT_COEF = float(os.environ.get("TRAIN_PPO_ENT_COEF", "0.05"))
TRAIN_DEBUG_ALLOW_BASELINE_BYPASS = os.environ.get("TRAIN_DEBUG_ALLOW_BASELINE_BYPASS", "0") == "1"
TRAIN_DEPRECATED_REQUIRE_BASELINE_GATE_FLAG = os.environ.get("TRAIN_REQUIRE_BASELINE_GATE", "1") != "0"
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
TRAIN_REWARD_SCALE = float(os.environ.get("TRAIN_REWARD_SCALE", "2500.0"))
TRAIN_REWARD_DRAWDOWN_PENALTY = TRAIN_REWARD_DRAWDOWN_PENALTY  # Consistently using value above
TRAIN_REWARD_TRANSACTION_PENALTY = float(os.environ.get("TRAIN_REWARD_TRANSACTION_PENALTY", "1.0"))
TRAIN_REWARD_CLIP_LOW = float(os.environ.get("TRAIN_REWARD_CLIP_LOW", "-10.0"))
TRAIN_REWARD_CLIP_HIGH = float(os.environ.get("TRAIN_REWARD_CLIP_HIGH", "10.0"))
TRAIN_TX_PENALTY_START = float(os.environ.get("TRAIN_TX_PENALTY_START", str(TRAIN_REWARD_TRANSACTION_PENALTY)))
TRAIN_TX_PENALTY_END = float(os.environ.get("TRAIN_TX_PENALTY_END", str(TRAIN_REWARD_TRANSACTION_PENALTY)))
TRAIN_TX_PENALTY_RAMP_STEPS = max(int(os.environ.get("TRAIN_TX_PENALTY_RAMP_STEPS", "0")), 0)
TRAIN_DRAWDOWN_PENALTY_START = float(os.environ.get("TRAIN_DRAWDOWN_PENALTY_START", str(TRAIN_REWARD_DRAWDOWN_PENALTY)))
TRAIN_DRAWDOWN_PENALTY_END = float(os.environ.get("TRAIN_DRAWDOWN_PENALTY_END", str(TRAIN_REWARD_DRAWDOWN_PENALTY)))
TRAIN_DRAWDOWN_PENALTY_RAMP_STEPS = max(int(os.environ.get("TRAIN_DRAWDOWN_PENALTY_RAMP_STEPS", "0")), 0)
TRAIN_WINDOW_SIZE = int(os.environ.get("TRAIN_WINDOW_SIZE", "1"))
TRAIN_ALPHA_GATE_ENABLED = os.environ.get("TRAIN_ALPHA_GATE_ENABLED", "0") == "1"
TRAIN_ALPHA_GATE_MODEL = os.environ.get("TRAIN_ALPHA_GATE_MODEL", "auto").strip().lower() or "auto"
TRAIN_ALPHA_GATE_WARMUP_STEPS = max(int(os.environ.get("TRAIN_ALPHA_GATE_WARMUP_STEPS", "0")), 0)
TRAIN_ALPHA_GATE_WARMUP_THRESHOLD_DELTA = max(float(os.environ.get("TRAIN_ALPHA_GATE_WARMUP_THRESHOLD_DELTA", "0.0")), 0.0)
TRAIN_ALPHA_GATE_WARMUP_MARGIN_SCALE = min(max(float(os.environ.get("TRAIN_ALPHA_GATE_WARMUP_MARGIN_SCALE", "1.0")), 0.0), 1.0)
TRAIN_FAIL_FAST_ENABLED = os.environ.get("TRAIN_FAIL_FAST_ENABLED", "0") == "1"
TRAIN_FAIL_FAST_WARMUP_STEPS = max(int(os.environ.get("TRAIN_FAIL_FAST_WARMUP_STEPS", "0")), 0)
TRAIN_FAIL_FAST_CONSECUTIVE = max(int(os.environ.get("TRAIN_FAIL_FAST_CONSECUTIVE", "2")), 1)
TRAIN_FAIL_FAST_SPARSE_ALPHA_GATE_BLOCK_RATE = min(
    max(float(os.environ.get("TRAIN_FAIL_FAST_SPARSE_ALPHA_GATE_BLOCK_RATE", "0.94")), 0.0),
    1.0,
)
TRAIN_FAIL_FAST_APPROX_KL_MAX = max(float(os.environ.get("TRAIN_FAIL_FAST_APPROX_KL_MAX", "0.001")), 0.0)
TRAIN_FAIL_FAST_EXPLAINED_VARIANCE_MAX = float(os.environ.get("TRAIN_FAIL_FAST_EXPLAINED_VARIANCE_MAX", "0.20"))
TRAIN_FAIL_FAST_MAX_TRADE_COUNT = max(int(os.environ.get("TRAIN_FAIL_FAST_MAX_TRADE_COUNT", "12")), 0)
TRAIN_FAIL_FAST_OVERTRADE_TRADE_COUNT = max(int(os.environ.get("TRAIN_FAIL_FAST_OVERTRADE_TRADE_COUNT", "0")), 0)
TRAIN_FAIL_FAST_COST_SHARE_MIN = max(float(os.environ.get("TRAIN_FAIL_FAST_COST_SHARE_MIN", "0.0")), 0.0)
TRAIN_DIRECTION_CONCENTRATION_WARN_SHARE = min(
    max(float(os.environ.get("TRAIN_DIRECTION_CONCENTRATION_WARN_SHARE", "0.90")), 0.0),
    1.0,
)
TRAIN_THIN_EDGE_COST_SHARE_WARN = max(float(os.environ.get("TRAIN_THIN_EDGE_COST_SHARE_WARN", "0.75")), 0.0)
TRAIN_THIN_EDGE_EXPECTANCY_WARN_USD = float(os.environ.get("TRAIN_THIN_EDGE_EXPECTANCY_WARN_USD", "0.25"))
TRAIN_TRUTH_RUN_FAIL_FAST_ENABLED = os.environ.get("TRAIN_TRUTH_RUN_FAIL_FAST_ENABLED", "1") == "1"
TRAIN_TRUTH_RUN_FAIL_FAST_WARMUP_STEPS = max(int(os.environ.get("TRAIN_TRUTH_RUN_FAIL_FAST_WARMUP_STEPS", "500000")), 0)
TRAIN_TRUTH_RUN_FAIL_FAST_CONSECUTIVE = max(int(os.environ.get("TRAIN_TRUTH_RUN_FAIL_FAST_CONSECUTIVE", "4")), 1)
TRAIN_TRUTH_RUN_FAIL_FAST_REJECT_STREAK = max(
    int(os.environ.get("TRAIN_TRUTH_RUN_FAIL_FAST_REJECT_STREAK", "6")),
    1,
)
HEARTBEAT_SCHEMA_VERSION = 2
PROCESS_STARTED_UTC = datetime.now(timezone.utc).isoformat()
TRAINING_STAGE = os.environ.get("TRAINING_STAGE", "stage_a_unlock").strip().lower() or "stage_a_unlock"
TRAIN_EXPERIMENT_PROFILE = os.environ.get("TRAIN_EXPERIMENT_PROFILE", "").strip().lower()
TRAIN_EXPERIMENT_PROFILE_SETTINGS = _resolve_training_experiment_profile(TRAIN_EXPERIMENT_PROFILE)




def _timed_recovery_step(*, fraction: float, minimum: int, maximum: int | None = None) -> int:
    candidate = max(int(TOTAL_TIMESTEPS * float(fraction)), int(minimum))
    if maximum is not None:
        candidate = min(candidate, int(maximum))
    return min(candidate, max(int(TOTAL_TIMESTEPS), 1))


def linear_ramp_value(*, step: int, start: float, end: float, ramp_steps: int) -> float:
    if ramp_steps <= 0:
        return float(end)
    progress = min(max(float(step) / float(ramp_steps), 0.0), 1.0)
    return float(start + ((end - start) * progress))


_PHASE_1_UNTIL_DEFAULT = _timed_recovery_step(fraction=1.0 / 3.0, minimum=50_000)
_PHASE_2_UNTIL_DEFAULT = max(
    _timed_recovery_step(fraction=2.0 / 3.0, minimum=max(_PHASE_1_UNTIL_DEFAULT + 1, 100_000)),
    _PHASE_1_UNTIL_DEFAULT + 1,
)
def get_entropy_coef(current_step: int) -> float:
    if current_step < _ENTROPY_PHASE_1_UNTIL_DEFAULT:
        return 0.02
    if current_step < _ENTROPY_PHASE_2_UNTIL_DEFAULT:
        return 0.005
    return _ENTROPY_MIN_FLOOR
_ENTROPY_PHASE_1_UNTIL_DEFAULT = _timed_recovery_step(fraction=0.15, minimum=25_000)
_ENTROPY_PHASE_2_UNTIL_DEFAULT = max(
    _timed_recovery_step(
        fraction=0.40,
        maximum=1_200_000,
        minimum=max(_ENTROPY_PHASE_1_UNTIL_DEFAULT + 1, 75_000),
    ),
    _ENTROPY_PHASE_1_UNTIL_DEFAULT + 1,
)
_PARTICIPATION_BONUS_UNTIL_DEFAULT = _timed_recovery_step(fraction=0.33, minimum=20_000, maximum=1_000_000)
_BONUS_ANNEAL_START_STEP = 1_000_000
_BONUS_ANNEAL_END_STEP = 2_500_000

TRAINING_RECOVERY_CONFIG: dict[str, Any] = {
    "training_stage": TRAINING_STAGE,
    "slippage_curriculum": {
        "enabled": os.environ.get("TRAIN_RECOVERY_SLIPPAGE_ENABLED", "1") != "0",
        "mode": os.environ.get("TRAIN_RECOVERY_SLIPPAGE_MODE", "staircase").strip().lower() or "staircase",
        "phases": [
            {
                "until_step": int(os.environ.get("TRAIN_RECOVERY_PHASE_1_UNTIL", str(_PHASE_1_UNTIL_DEFAULT))),
                "slippage_pips": float(os.environ.get("TRAIN_RECOVERY_PHASE_1_SLIPPAGE_PIPS", "0.05")),
            },
            {
                "until_step": int(os.environ.get("TRAIN_RECOVERY_PHASE_2_UNTIL", str(_PHASE_2_UNTIL_DEFAULT))),
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
        "mode": os.environ.get("TRAIN_RECOVERY_PARTICIPATION_BONUS_MODE", "per_bar"),
        "bonus_value": float(os.environ.get("TRAIN_RECOVERY_PARTICIPATION_BONUS_VALUE", "0.0025")),
        "active_until_step": int(
            os.environ.get("TRAIN_RECOVERY_PARTICIPATION_BONUS_UNTIL", str(_PARTICIPATION_BONUS_UNTIL_DEFAULT))
        ),
        "cooldown_steps": int(os.environ.get("TRAIN_RECOVERY_PARTICIPATION_BONUS_COOLDOWN", "32")),
        "only_from_flat": os.environ.get("TRAIN_RECOVERY_PARTICIPATION_ONLY_FROM_FLAT", "1") != "0",
        "max_bonus_per_episode": int(os.environ.get("TRAIN_RECOVERY_PARTICIPATION_MAX_PER_EPISODE", "5000")),
    },
    "entropy_schedule": {
        "enabled": os.environ.get("TRAIN_RECOVERY_ENTROPY_SCHEDULE_ENABLED", "1") != "0",
        "initial_ent_coef": float(os.environ.get("TRAIN_RECOVERY_ENTROPY_INITIAL", "0.02")),
        "mid_ent_coef": float(os.environ.get("TRAIN_RECOVERY_ENTROPY_MID", "0.005")),
        "final_ent_coef": float(os.environ.get("TRAIN_RECOVERY_ENTROPY_FINAL", str(_ENTROPY_MIN_FLOOR))),
        "phase_1_until": int(
            os.environ.get("TRAIN_RECOVERY_ENTROPY_PHASE_1_UNTIL", str(_ENTROPY_PHASE_1_UNTIL_DEFAULT))
        ),
        "phase_2_until": int(
            os.environ.get("TRAIN_RECOVERY_ENTROPY_PHASE_2_UNTIL", str(_ENTROPY_PHASE_2_UNTIL_DEFAULT))
        ),
    },
    "diagnostics": {
        "log_action_distribution": os.environ.get("TRAIN_RECOVERY_LOG_ACTION_DISTRIBUTION", "1") != "0",
        "log_reward_components": os.environ.get("TRAIN_RECOVERY_LOG_REWARD_COMPONENTS", "1") != "0",
        "heartbeat_every_n_updates": int(os.environ.get("TRAIN_RECOVERY_HEARTBEAT_EVERY_N_UPDATES", "10")),
    },
}

if TRAIN_EXPERIMENT_PROFILE_SETTINGS:
    TRAIN_CHURN_MIN_HOLD_BARS = int(
        _apply_profile_override(
            TRAIN_CHURN_MIN_HOLD_BARS,
            "TRAIN_CHURN_MIN_HOLD_BARS",
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get("churn_min_hold_bars", TRAIN_CHURN_MIN_HOLD_BARS),
        )
    )
    TRAIN_CHURN_ACTION_COOLDOWN = int(
        _apply_profile_override(
            TRAIN_CHURN_ACTION_COOLDOWN,
            "TRAIN_CHURN_ACTION_COOLDOWN",
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get("churn_action_cooldown", TRAIN_CHURN_ACTION_COOLDOWN),
        )
    )
    TRAIN_CHURN_PENALTY_USD = float(
        _apply_profile_override(
            TRAIN_CHURN_PENALTY_USD,
            "TRAIN_CHURN_PENALTY_USD",
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get("churn_penalty_usd", TRAIN_CHURN_PENALTY_USD),
        )
    )
    TRAIN_ENTRY_SPREAD_Z_LIMIT = float(
        _apply_profile_override(
            TRAIN_ENTRY_SPREAD_Z_LIMIT,
            "TRAIN_ENTRY_SPREAD_Z_LIMIT",
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get("entry_spread_z_limit", TRAIN_ENTRY_SPREAD_Z_LIMIT),
        )
    )
    TRAIN_REWARD_DOWNSIDE_RISK_COEF = float(
        _apply_profile_override(
            TRAIN_REWARD_DOWNSIDE_RISK_COEF,
            "TRAIN_REWARD_DOWNSIDE_RISK_COEF",
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get("reward_downside_risk_coef", TRAIN_REWARD_DOWNSIDE_RISK_COEF),
        )
    )
    TRAIN_REWARD_TURNOVER_COEF = float(
        _apply_profile_override(
            TRAIN_REWARD_TURNOVER_COEF,
            "TRAIN_REWARD_TURNOVER_COEF",
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get("reward_turnover_coef", TRAIN_REWARD_TURNOVER_COEF),
        )
    )
    TRAIN_REWARD_NET_RETURN_COEF = float(
        _apply_profile_override(
            TRAIN_REWARD_NET_RETURN_COEF,
            "TRAIN_REWARD_NET_RETURN_COEF",
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get("reward_net_return_coef", TRAIN_REWARD_NET_RETURN_COEF),
        )
    )
    TRAIN_REWARD_SCALE = float(
        _apply_profile_override(
            TRAIN_REWARD_SCALE,
            "TRAIN_REWARD_SCALE",
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get("reward_scale", TRAIN_REWARD_SCALE),
        )
    )
    TRAIN_REWARD_CLIP_LOW = float(
        _apply_profile_override(
            TRAIN_REWARD_CLIP_LOW,
            "TRAIN_REWARD_CLIP_LOW",
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get("reward_clip_low", TRAIN_REWARD_CLIP_LOW),
        )
    )
    TRAIN_REWARD_CLIP_HIGH = float(
        _apply_profile_override(
            TRAIN_REWARD_CLIP_HIGH,
            "TRAIN_REWARD_CLIP_HIGH",
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get("reward_clip_high", TRAIN_REWARD_CLIP_HIGH),
        )
    )
    if "TRAIN_ALPHA_GATE_ENABLED" not in os.environ:
        TRAIN_ALPHA_GATE_ENABLED = bool(
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get("alpha_gate_enabled", TRAIN_ALPHA_GATE_ENABLED)
        )
    if "TRAIN_ALPHA_GATE_MODEL" not in os.environ:
        TRAIN_ALPHA_GATE_MODEL = str(
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get("alpha_gate_model", TRAIN_ALPHA_GATE_MODEL)
        ).strip().lower() or TRAIN_ALPHA_GATE_MODEL
    TRAIN_ALPHA_GATE_WARMUP_STEPS = int(
        _apply_profile_override(
            TRAIN_ALPHA_GATE_WARMUP_STEPS,
            "TRAIN_ALPHA_GATE_WARMUP_STEPS",
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get("alpha_gate_warmup_steps", TRAIN_ALPHA_GATE_WARMUP_STEPS),
        )
    )
    TRAIN_ALPHA_GATE_WARMUP_THRESHOLD_DELTA = float(
        _apply_profile_override(
            TRAIN_ALPHA_GATE_WARMUP_THRESHOLD_DELTA,
            "TRAIN_ALPHA_GATE_WARMUP_THRESHOLD_DELTA",
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get(
                "alpha_gate_warmup_threshold_delta",
                TRAIN_ALPHA_GATE_WARMUP_THRESHOLD_DELTA,
            ),
        )
    )
    TRAIN_ALPHA_GATE_WARMUP_MARGIN_SCALE = min(
        max(
            float(
                _apply_profile_override(
                    TRAIN_ALPHA_GATE_WARMUP_MARGIN_SCALE,
                    "TRAIN_ALPHA_GATE_WARMUP_MARGIN_SCALE",
                    TRAIN_EXPERIMENT_PROFILE_SETTINGS.get(
                        "alpha_gate_warmup_margin_scale",
                        TRAIN_ALPHA_GATE_WARMUP_MARGIN_SCALE,
                    ),
                )
            ),
            0.0,
        ),
        1.0,
    )
    PPO_LEARNING_RATE = float(
        _apply_profile_override(
            PPO_LEARNING_RATE,
            "TRAIN_PPO_LEARNING_RATE",
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get("ppo_learning_rate", PPO_LEARNING_RATE),
        )
    )
    PPO_TARGET_KL = float(
        _apply_profile_override(
            PPO_TARGET_KL,
            "TRAIN_PPO_TARGET_KL",
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get("ppo_target_kl", PPO_TARGET_KL),
        )
    )
    if "TRAIN_ADAPTIVE_KL_LR" not in os.environ:
        TRAIN_ADAPTIVE_KL_LR = bool(
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get("adaptive_kl_lr", TRAIN_ADAPTIVE_KL_LR)
        )
    TRAIN_ADAPTIVE_KL_MAX_LR = float(
        _apply_profile_override(
            TRAIN_ADAPTIVE_KL_MAX_LR,
            "TRAIN_ADAPTIVE_KL_MAX_LR",
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get("adaptive_kl_max_lr", TRAIN_ADAPTIVE_KL_MAX_LR),
        )
    )
    TRAIN_ADAPTIVE_KL_LOW = float(
        _apply_profile_override(
            TRAIN_ADAPTIVE_KL_LOW,
            "TRAIN_ADAPTIVE_KL_LOW",
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get("adaptive_kl_low", TRAIN_ADAPTIVE_KL_LOW),
        )
    )
    TRAIN_ADAPTIVE_KL_UP_MULT = float(
        _apply_profile_override(
            TRAIN_ADAPTIVE_KL_UP_MULT,
            "TRAIN_ADAPTIVE_KL_UP_MULT",
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get("adaptive_kl_up_mult", TRAIN_ADAPTIVE_KL_UP_MULT),
        )
    )
    if "TRAIN_FAIL_FAST_ENABLED" not in os.environ:
        TRAIN_FAIL_FAST_ENABLED = bool(
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get("fail_fast_enabled", TRAIN_FAIL_FAST_ENABLED)
        )
    TRAIN_FAIL_FAST_WARMUP_STEPS = int(
        _apply_profile_override(
            TRAIN_FAIL_FAST_WARMUP_STEPS,
            "TRAIN_FAIL_FAST_WARMUP_STEPS",
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get("fail_fast_warmup_steps", TRAIN_FAIL_FAST_WARMUP_STEPS),
        )
    )
    TRAIN_FAIL_FAST_CONSECUTIVE = int(
        _apply_profile_override(
            TRAIN_FAIL_FAST_CONSECUTIVE,
            "TRAIN_FAIL_FAST_CONSECUTIVE",
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get("fail_fast_consecutive", TRAIN_FAIL_FAST_CONSECUTIVE),
        )
    )
    TRAIN_FAIL_FAST_SPARSE_ALPHA_GATE_BLOCK_RATE = min(
        max(
            float(
                _apply_profile_override(
                    TRAIN_FAIL_FAST_SPARSE_ALPHA_GATE_BLOCK_RATE,
                    "TRAIN_FAIL_FAST_SPARSE_ALPHA_GATE_BLOCK_RATE",
                    TRAIN_EXPERIMENT_PROFILE_SETTINGS.get(
                        "fail_fast_sparse_alpha_gate_block_rate",
                        TRAIN_FAIL_FAST_SPARSE_ALPHA_GATE_BLOCK_RATE,
                    ),
                )
            ),
            0.0,
        ),
        1.0,
    )
    TRAIN_FAIL_FAST_APPROX_KL_MAX = float(
        _apply_profile_override(
            TRAIN_FAIL_FAST_APPROX_KL_MAX,
            "TRAIN_FAIL_FAST_APPROX_KL_MAX",
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get("fail_fast_approx_kl_max", TRAIN_FAIL_FAST_APPROX_KL_MAX),
        )
    )
    TRAIN_FAIL_FAST_EXPLAINED_VARIANCE_MAX = float(
        _apply_profile_override(
            TRAIN_FAIL_FAST_EXPLAINED_VARIANCE_MAX,
            "TRAIN_FAIL_FAST_EXPLAINED_VARIANCE_MAX",
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get(
                "fail_fast_explained_variance_max",
                TRAIN_FAIL_FAST_EXPLAINED_VARIANCE_MAX,
            ),
        )
    )
    TRAIN_FAIL_FAST_MAX_TRADE_COUNT = int(
        _apply_profile_override(
            TRAIN_FAIL_FAST_MAX_TRADE_COUNT,
            "TRAIN_FAIL_FAST_MAX_TRADE_COUNT",
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get("fail_fast_max_trade_count", TRAIN_FAIL_FAST_MAX_TRADE_COUNT),
        )
    )
    TRAIN_FAIL_FAST_OVERTRADE_TRADE_COUNT = int(
        _apply_profile_override(
            TRAIN_FAIL_FAST_OVERTRADE_TRADE_COUNT,
            "TRAIN_FAIL_FAST_OVERTRADE_TRADE_COUNT",
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get(
                "fail_fast_overtrade_trade_count",
                TRAIN_FAIL_FAST_OVERTRADE_TRADE_COUNT,
            ),
        )
    )
    TRAIN_FAIL_FAST_COST_SHARE_MIN = float(
        _apply_profile_override(
            TRAIN_FAIL_FAST_COST_SHARE_MIN,
            "TRAIN_FAIL_FAST_COST_SHARE_MIN",
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get(
                "fail_fast_cost_share_min",
                TRAIN_FAIL_FAST_COST_SHARE_MIN,
            ),
        )
    )
    TRAINING_RECOVERY_CONFIG["entropy_schedule"]["mid_ent_coef"] = float(
        _apply_profile_override(
            TRAINING_RECOVERY_CONFIG["entropy_schedule"]["mid_ent_coef"],
            "TRAIN_RECOVERY_ENTROPY_MID",
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get(
                "recovery_entropy_mid",
                TRAINING_RECOVERY_CONFIG["entropy_schedule"]["mid_ent_coef"],
            ),
        )
    )
    TRAINING_RECOVERY_CONFIG["entropy_schedule"]["final_ent_coef"] = max(
        float(
            _apply_profile_override(
                TRAINING_RECOVERY_CONFIG["entropy_schedule"]["final_ent_coef"],
                "TRAIN_RECOVERY_ENTROPY_FINAL",
                TRAIN_EXPERIMENT_PROFILE_SETTINGS.get(
                    "recovery_entropy_final",
                    TRAINING_RECOVERY_CONFIG["entropy_schedule"]["final_ent_coef"],
                ),
            )
        ),
        float(_ENTROPY_MIN_FLOOR),
    )
    if "TRAIN_RECOVERY_PARTICIPATION_BONUS_ENABLED" not in os.environ:
        TRAINING_RECOVERY_CONFIG["participation_bonus"]["enabled"] = bool(
            TRAIN_EXPERIMENT_PROFILE_SETTINGS.get(
                "participation_bonus_enabled",
                TRAINING_RECOVERY_CONFIG["participation_bonus"]["enabled"],
            )
        )

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
