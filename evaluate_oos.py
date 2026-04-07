from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from interpreter_guard import ensure_project_venv

ensure_project_venv(project_root=Path(__file__).resolve().parent, script_path=__file__)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

from artifact_manifest import (
    dataset_id_for_path,
    load_manifest,
    load_validated_model,
    load_validated_scaler,
    load_validated_vecnormalize,
)
from dataset_validation import validate_symbol_bar_spec
from execution.replay_broker import ReplayBroker
from risk.risk_engine import RiskEngine, RiskLimits
from runtime.runtime_engine import ModelPolicy, RuntimeEngine, RuntimeSnapshot

from edge_research import fit_baseline_alpha_gate
from feature_engine import FEATURE_COLS, FeatureEngine, WARMUP_BARS, _compute_raw
from project_paths import ensure_runtime_dirs, resolve_dataset_path, resolve_manifest_path, validate_dataset_bar_spec
from run_logging import configure_run_logging, set_log_context
from runtime_common import (
    STATE_FEATURE_COUNT,
    build_evaluation_accounting,
    compute_max_drawdown,
    compute_timed_sharpe,
    compute_trade_metrics,
    deserialize_action_map,
    runtime_options_from_training_payload,
    validate_evaluation_accounting,
    validate_evaluation_payload,
)
from runtime_gym_env import BarView, RuntimeGymEnv, TrainingDiagnostics
from trading_config import deployment_paths
from train_agent import (
    FOLD_TEST_FRAC,
    HOLDOUT_FRAC,
    N_FOLDS,
    PURGE_GAP_BARS,
    TRAINING_RECOVERY_CONFIG,
    _fit_and_apply_fold_scaler,
    _prepend_runtime_warmup_context,
    _split_holdout,
    aggregate_training_diagnostics,
    build_execution_cost_profile,
    build_reward_profile,
    build_runtime_action_map,
    get_final_slippage_pips,
    purged_walk_forward_splits,
)
from validation_metrics import build_deployment_gate, load_json_report, save_json_report


TARGET_SYM = os.environ.get("EVAL_SYMBOL", "EURUSD").strip().upper() or "EURUSD"
EVAL_MANIFEST_PATH = os.environ.get("EVAL_MANIFEST_PATH")
_eval_output_dir_env = os.environ.get("EVAL_OUTPUT_DIR")
EVAL_OUTPUT_DIR = (
    Path(_eval_output_dir_env)
    if _eval_output_dir_env
    else (Path(EVAL_MANIFEST_PATH).resolve().parent if EVAL_MANIFEST_PATH else Path("models"))
)
EVAL_MAX_BARS = int(os.environ.get("EVAL_MAX_BARS", "0") or "0")
EVAL_SKIP_PLOT = os.environ.get("EVAL_SKIP_PLOT", "0") == "1"
CURRENT_RUN_CONTEXT_PATH = Path("checkpoints") / "current_training_run.json"
log = logging.getLogger("evaluate_oos")


@dataclass
class ReplayContext:
    symbol: str
    source: str
    dataset_path: Path
    action_map: tuple[Any, ...]
    model: Any
    obs_normalizer: Any | None
    scaler: Any
    execution_cost_profile: dict[str, float]
    reward_profile: dict[str, Any]
    warmup_frame: pd.DataFrame
    replay_frame: pd.DataFrame
    replay_feature_frame: pd.DataFrame
    full_feature_frame: pd.DataFrame
    trainable_feature_frame: pd.DataFrame
    holdout_feature_frame: pd.DataFrame
    holdout_start_utc: str | None
    diagnostics_path: Path | None
    manifest_path: Path | None
    artifact_metadata: dict[str, Any]
    runtime_options: dict[str, Any]


def _clone_context_with_segment(context: ReplayContext, segment: pd.DataFrame) -> ReplayContext:
    if segment.empty:
        raise RuntimeError("Segment must not be empty.")
    warmup_source = _prepend_runtime_warmup_context(context.full_feature_frame, segment)
    warmup_count = max(len(warmup_source) - len(segment), 0)
    return replace(
        context,
        warmup_frame=warmup_source.iloc[:warmup_count].copy(),
        replay_frame=segment.copy(),
        replay_feature_frame=segment.copy(),
        holdout_feature_frame=segment.copy(),
        holdout_start_utc=pd.Timestamp(segment.index[0]).isoformat(),
    )


def _with_cost_stress(context: ReplayContext, *, slippage_multiplier: float) -> ReplayContext:
    stressed = dict(context.execution_cost_profile)
    stressed["slippage_pips"] = float(stressed.get("slippage_pips", 0.25)) * float(slippage_multiplier)
    return replace(context, execution_cost_profile=stressed)


def _best_model_name(models: dict[str, Any]) -> str | None:
    if not models:
        return None
    return max(
        models.items(),
        key=lambda item: (
            float(((item[1] or {}).get("metrics", {}) or {}).get("expectancy_usd", 0.0)),
            float(((item[1] or {}).get("metrics", {}) or {}).get("profit_factor", 0.0)),
            float(((item[1] or {}).get("metrics", {}) or {}).get("trade_count", 0.0)),
        ),
    )[0]


def _flat_provider(**_: object) -> int:
    return 0


def _always_long_provider(*, position_direction: int, action_map, **_: object) -> int:
    return _target_direction_to_action_index(
        action_map=action_map,
        position_direction=int(position_direction or 0),
        target_direction=1,
    )


def _always_short_provider(*, position_direction: int, action_map, **_: object) -> int:
    return _target_direction_to_action_index(
        action_map=action_map,
        position_direction=int(position_direction or 0),
        target_direction=-1,
    )


def _resolve_action_indexes(action_map) -> dict[str, int]:
    indexes = {"hold": 0, "close": 0, "long": 0, "short": 0}
    for idx, action in enumerate(action_map):
        if action.action_type.value == "HOLD":
            indexes["hold"] = int(idx)
        elif action.action_type.value == "CLOSE":
            indexes["close"] = int(idx)
        elif action.action_type.value == "OPEN" and int(action.direction or 0) > 0:
            indexes["long"] = int(idx)
        elif action.action_type.value == "OPEN" and int(action.direction or 0) < 0:
            indexes["short"] = int(idx)
    return indexes


def _target_direction_to_action_index(*, action_map, position_direction: int, target_direction: int) -> int:
    indexes = _resolve_action_indexes(action_map)
    current_direction = int(position_direction or 0)
    desired_direction = int(target_direction or 0)
    if current_direction == desired_direction:
        return indexes["hold"]
    if desired_direction == 0:
        return indexes["close"] if current_direction != 0 else indexes["hold"]
    if current_direction == 0:
        return indexes["long"] if desired_direction > 0 else indexes["short"]
    return indexes["close"]


def _trend_provider(*, feature_row, position_direction: int, action_map, **_: object) -> int:
    ma20_slope = float(feature_row.get("ma20_slope", 0.0) or 0.0)
    ma50_slope = float(feature_row.get("ma50_slope", 0.0) or 0.0)
    target_direction = 0
    if ma20_slope > 0.0 and ma50_slope > 0.0:
        target_direction = 1
    elif ma20_slope < 0.0 and ma50_slope < 0.0:
        target_direction = -1
    return _target_direction_to_action_index(
        action_map=action_map,
        position_direction=int(position_direction or 0),
        target_direction=target_direction,
    )


def _mean_reversion_provider(*, feature_row, position_direction: int, action_map, **_: object) -> int:
    spread_z = float(feature_row.get("spread_z", 0.0) or 0.0)
    target_direction = 0
    if spread_z <= -1.0:
        target_direction = 1
    elif spread_z >= 1.0:
        target_direction = -1
    return _target_direction_to_action_index(
        action_map=action_map,
        position_direction=int(position_direction or 0),
        target_direction=target_direction,
    )


RUNTIME_BASELINE_PROVIDERS: dict[str, Callable[..., int]] = {
    "flat": _flat_provider,
    "always_long": _always_long_provider,
    "always_short": _always_short_provider,
    "mean_reversion": _mean_reversion_provider,
    "trend_rule": _trend_provider,
    "runtime_flat": _flat_provider,
    "runtime_always_long": _always_long_provider,
    "runtime_always_short": _always_short_provider,
    "runtime_mean_reversion": _mean_reversion_provider,
    "runtime_trend": _trend_provider,
}


def _metrics_profitable(metrics: dict[str, Any] | None) -> bool:
    if not metrics:
        return False
    validation_status = dict(metrics.get("validation_status", {}) or {})
    if validation_status and not bool(validation_status.get("passed", False)):
        return False
    return bool(int(metrics.get("trade_count", 0) or 0) > 0 and float(metrics.get("net_pnl_usd", 0.0)) > 0.0)


def _compute_expectancy_slice(trades: list[dict[str, Any]]) -> dict[str, float]:
    if not trades:
        return {"trade_count": 0.0, "net_pnl_usd": 0.0, "expectancy_usd": 0.0, "win_rate": 0.0}
    wins = sum(1 for trade in trades if float(trade.get("net_pnl_usd", 0.0)) > 0.0)
    total = sum(float(trade.get("net_pnl_usd", 0.0)) for trade in trades)
    return {
        "trade_count": float(len(trades)),
        "net_pnl_usd": float(total),
        "expectancy_usd": float(total / max(len(trades), 1)),
        "win_rate": float(wins / max(len(trades), 1)),
    }


def _direction_expectancy(trade_log: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    longs = [dict(trade) for trade in trade_log if int(trade.get("direction", 0) or 0) > 0]
    shorts = [dict(trade) for trade in trade_log if int(trade.get("direction", 0) or 0) < 0]
    return {
        "long": _compute_expectancy_slice(longs),
        "short": _compute_expectancy_slice(shorts),
    }


def _hour_bucket_expectancy(
    trade_log: list[dict[str, Any]],
    execution_log: list[dict[str, Any]],
) -> dict[str, Any] | None:
    close_events = [
        dict(event)
        for event in execution_log
        if isinstance(event, dict) and str(event.get("side", "")).lower() == "close" and event.get("time_msc") is not None
    ]
    matched = min(len(close_events), len(trade_log))
    if matched <= 0:
        return None
    buckets: dict[int, list[dict[str, Any]]] = {}
    for idx in range(matched):
        hour = int(pd.Timestamp(int(close_events[idx]["time_msc"]), unit="ms", tz="UTC").hour)
        buckets.setdefault(hour, []).append(dict(trade_log[idx]))
    return {
        "matched_trade_count": int(matched),
        "close_hour_utc": {
            str(hour): _compute_expectancy_slice(trades)
            for hour, trades in sorted(buckets.items(), key=lambda item: item[0])
        },
    }


def _pnl_concentration(trade_log: list[dict[str, Any]]) -> dict[str, float]:
    pnl_values = sorted((float(trade.get("net_pnl_usd", 0.0)) for trade in trade_log), reverse=True)
    if not pnl_values:
        return {
            "top_1_share_of_abs_net_pnl": 0.0,
            "top_3_share_of_abs_net_pnl": 0.0,
            "top_5_share_of_abs_net_pnl": 0.0,
        }
    denominator = max(sum(abs(value) for value in pnl_values), 1e-6)
    return {
        "top_1_share_of_abs_net_pnl": float(sum(abs(value) for value in pnl_values[:1]) / denominator),
        "top_3_share_of_abs_net_pnl": float(sum(abs(value) for value in pnl_values[:3]) / denominator),
        "top_5_share_of_abs_net_pnl": float(sum(abs(value) for value in pnl_values[:5]) / denominator),
    }


def _build_reject_fast_diagnostics(
    *,
    replay_metrics: dict[str, Any],
    trade_log: list[dict[str, Any]],
    execution_log: list[dict[str, Any]],
) -> dict[str, Any]:
    trade_count = int(replay_metrics.get("trade_count", 0) or 0)
    gross_pnl_usd = float(replay_metrics.get("gross_pnl_usd", 0.0))
    total_cost_usd = float(replay_metrics.get("total_transaction_cost_usd", 0.0))
    diagnostics = {
        "trade_count_vs_net_pnl": {
            "trade_count": trade_count,
            "net_pnl_usd": float(replay_metrics.get("net_pnl_usd", 0.0)),
            "net_pnl_per_trade_usd": float(replay_metrics.get("expectancy_usd", 0.0)),
            "trades_per_100_bars": float(100.0 * trade_count / max(int(replay_metrics.get("steps", 0) or 0), 1)),
        },
        "cost_share_of_gross_pnl": {
            "gross_pnl_usd": gross_pnl_usd,
            "total_transaction_cost_usd": total_cost_usd,
            "cost_share_of_abs_gross_pnl": float(total_cost_usd / max(abs(gross_pnl_usd), 1e-6)),
        },
        "expectancy_by_direction": _direction_expectancy(trade_log),
        "pnl_concentration": _pnl_concentration(trade_log),
    }
    hour_bucket = _hour_bucket_expectancy(trade_log, execution_log)
    if hour_bucket is not None:
        diagnostics["time_bucket_expectancy"] = hour_bucket
    return diagnostics


def _build_decision_summary(
    *,
    reject_fast_diagnostics: dict[str, Any],
    runtime_parity_verdict: dict[str, Any],
) -> dict[str, Any]:
    flags: list[str] = []
    cost_share = float(
        ((reject_fast_diagnostics.get("cost_share_of_gross_pnl") or {}).get("cost_share_of_abs_gross_pnl", 0.0))
    )
    if cost_share >= 1.0:
        flags.append("overtrading_or_weak_entry_quality")
    expectancy_by_direction = dict(reject_fast_diagnostics.get("expectancy_by_direction", {}) or {})
    long_expectancy = float(((expectancy_by_direction.get("long") or {}).get("expectancy_usd", 0.0)))
    short_expectancy = float(((expectancy_by_direction.get("short") or {}).get("expectancy_usd", 0.0)))
    long_trades = int(((expectancy_by_direction.get("long") or {}).get("trade_count", 0.0)) or 0)
    short_trades = int(((expectancy_by_direction.get("short") or {}).get("trade_count", 0.0)) or 0)
    if (long_trades > 0 and short_trades > 0 and ((long_expectancy > 0.0) != (short_expectancy > 0.0))) or (
        long_trades == 0 < short_trades or short_trades == 0 < long_trades
    ):
        flags.append("direction_concentration")
    top_3_share = float(((reject_fast_diagnostics.get("pnl_concentration") or {}).get("top_3_share_of_abs_net_pnl", 0.0)))
    if top_3_share >= 0.8:
        flags.append("pnl_concentrated_in_few_trades")
    if bool(runtime_parity_verdict.get("fragile_under_cost_stress", False)):
        flags.append("fragile_under_slippage_stress")
    if str(runtime_parity_verdict.get("best_runtime_baseline") or "") == "runtime_flat":
        flags.append("no_trade_baseline_preferred")
    if "no_trade_baseline_preferred" in flags:
        verdict = "reject_trade_deployment"
    elif flags:
        verdict = "needs_targeted_ablation"
    else:
        verdict = "no_immediate_reject_flag"
    return {
        "verdict": verdict,
        "flags": flags,
        "metrics_snapshot": {
            "cost_share_of_abs_gross_pnl": cost_share,
            "long_expectancy_usd": long_expectancy,
            "short_expectancy_usd": short_expectancy,
            "top_3_share_of_abs_net_pnl": top_3_share,
            "best_runtime_baseline": runtime_parity_verdict.get("best_runtime_baseline"),
            "fragile_under_cost_stress": bool(runtime_parity_verdict.get("fragile_under_cost_stress", False)),
        },
    }


def _load_research_baseline_summary(training_diagnostics: dict[str, Any] | None) -> dict[str, Any] | None:
    if not training_diagnostics:
        return None
    baseline_report_path = training_diagnostics.get("baseline_report_path")
    if not baseline_report_path:
        return None
    report_path = Path(str(baseline_report_path))
    if not report_path.exists():
        return None
    baseline_report = load_json_report(report_path)
    holdout_models = dict((baseline_report.get("holdout_metrics", {}) or {}).get("models", {}) or {})
    best_baseline = _best_model_name(holdout_models)
    best_metrics = dict((holdout_models.get(best_baseline, {}) or {}).get("metrics", {}) or {}) if best_baseline else {}
    viable = bool(int(best_metrics.get("trade_count", 0) or 0) > 0 and float(best_metrics.get("net_pnl_usd", 0.0)) > 0.0)
    return {
        "report_path": str(report_path),
        "target_definition": dict(baseline_report.get("target_definition", {}) or {}),
        "best_baseline": best_baseline,
        "best_baseline_metrics": best_metrics or None,
        "research_baseline_viable": viable,
    }


def _resolve_execution_cost_profile(manifest) -> dict[str, float]:
    profile = dict(getattr(manifest, "execution_cost_profile", None) or {})
    return {
        "commission_per_lot": float(profile.get("commission_per_lot", 7.0)),
        "slippage_pips": float(profile.get("slippage_pips", 0.25)),
        "partial_fill_ratio": float(profile.get("partial_fill_ratio", 1.0)),
    }


def _resolve_reward_profile(manifest) -> dict[str, Any]:
    profile = dict(getattr(manifest, "reward_profile", None) or {})
    return {
        "reward_scale": float(profile.get("reward_scale", 10_000.0)),
        "drawdown_penalty": float(profile.get("drawdown_penalty", 2.0)),
        "transaction_penalty": float(profile.get("transaction_penalty", 1.0)),
        "reward_clip_low": float(profile.get("reward_clip_low", -5.0)),
        "reward_clip_high": float(profile.get("reward_clip_high", 5.0)),
        "minimal_post_cost_reward": bool(profile.get("minimal_post_cost_reward", True)),
    }


def _load_training_runtime_options(diagnostics_path: Path | None) -> dict[str, Any]:
    payload = load_json_report(diagnostics_path) if diagnostics_path is not None and diagnostics_path.exists() else {}
    return runtime_options_from_training_payload(payload, default_window_size=1)


def _resolve_checkpoint_diagnostics_path(fold_dir: Path) -> Path | None:
    for candidate in (
        fold_dir / "training_diagnostics.json",
        fold_dir / "training_diagnostics_EURUSD.json",
        fold_dir / "best_training_diagnostics.json",
        fold_dir / "training_heartbeat.json",
    ):
        if candidate.exists():
            return candidate
    return None


def _frame_to_bars(frame: pd.DataFrame) -> list[BarView]:
    raw_bars = RuntimeGymEnv._frame_to_bars(frame)
    return [BarView(row) for row in raw_bars]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_symbol_raw_frame(
    *,
    symbol: str,
    dataset_path: Path,
    expected_ticks_per_bar: int | None = None,
) -> pd.DataFrame:
    if expected_ticks_per_bar is not None:
        validate_dataset_bar_spec(
            dataset_path=dataset_path,
            expected_ticks_per_bar=int(expected_ticks_per_bar),
            metadata_required=True,
        )
    raw = pd.read_csv(dataset_path, low_memory=False, parse_dates=["Gmt time"])
    raw = raw.loc[raw["Symbol"].astype(str).str.upper() == symbol.upper()].copy()
    raw["Gmt time"] = pd.to_datetime(raw["Gmt time"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["Gmt time"]).set_index("Gmt time").sort_index()
    if expected_ticks_per_bar is not None:
        validate_symbol_bar_spec(raw.reset_index(), expected_ticks_per_bar=int(expected_ticks_per_bar), symbol=symbol.upper())
    if len(raw) <= WARMUP_BARS + 10:
        raise RuntimeError(f"Not enough raw bars for {symbol.upper()}: {len(raw)}")
    return raw


def _load_promoted_manifest_context(symbol: str) -> ReplayContext | None:
    try:
        manifest_path = resolve_manifest_path(symbol=symbol, preferred=EVAL_MANIFEST_PATH)
    except FileNotFoundError:
        return None
    manifest = load_manifest(manifest_path)
    dataset_path = resolve_dataset_path()
    manifest_ticks = manifest.bar_construction_ticks_per_bar or manifest.ticks_per_bar
    raw = _load_symbol_raw_frame(symbol=symbol, dataset_path=dataset_path, expected_ticks_per_bar=manifest_ticks)
    featured = _compute_raw(raw).dropna(subset=list(FEATURE_COLS))
    if manifest.holdout_start_utc:
        holdout_start = pd.Timestamp(manifest.holdout_start_utc)
        replay_frame = raw.loc[holdout_start:].copy()
        train_frame = raw.loc[raw.index < holdout_start].copy()
        warmup_frame = train_frame.iloc[-max(WARMUP_BARS * 3, 300):].copy()
        holdout_feature_frame = featured.loc[featured.index >= holdout_start].copy()
        trainable_feature_frame = featured.loc[featured.index < holdout_start].copy()
    else:
        raw_split_idx = int(len(raw) * float(HOLDOUT_FRAC))
        split_pos = max(len(raw) - raw_split_idx, 1)
        split_ts = raw.index[split_pos]
        warmup_start = max(0, split_pos - max(WARMUP_BARS * 3, 300))
        warmup_frame = raw.iloc[warmup_start:split_pos].copy()
        replay_frame = raw.iloc[split_pos:].copy()
        holdout_feature_frame = featured.loc[featured.index >= split_ts].copy()
        trainable_feature_frame = featured.loc[featured.index < split_ts].copy()
    if replay_frame.empty or warmup_frame.empty or holdout_feature_frame.empty:
        raise RuntimeError(f"Holdout split produced empty replay, warmup, or feature frames for {symbol}.")
    if EVAL_MAX_BARS and int(EVAL_MAX_BARS) > 0:
        max_bars = int(EVAL_MAX_BARS)
        if len(replay_frame) > max_bars:
            replay_frame = replay_frame.iloc[-max_bars:].copy()
        if len(holdout_feature_frame) > max_bars:
            holdout_feature_frame = holdout_feature_frame.iloc[-max_bars:].copy()
        if len(trainable_feature_frame) > max_bars:
            trainable_feature_frame = trainable_feature_frame.iloc[-max_bars:].copy()

    dataset_id = dataset_id_for_path(dataset_path)
    action_map = deserialize_action_map(manifest.action_map)
    diagnostics_path = Path(manifest.training_diagnostics_path) if manifest.training_diagnostics_path else None
    runtime_options = _load_training_runtime_options(diagnostics_path)
    expected_observation_shape = list(getattr(manifest, "observation_shape", None) or [])
    if not expected_observation_shape:
        expected_observation_shape = [int(runtime_options["window_size"]), len(FEATURE_COLS) + STATE_FEATURE_COUNT]
    model = load_validated_model(
        manifest,
        expected_symbol=symbol,
        expected_action_map=action_map,
        expected_observation_shape=expected_observation_shape,
        expected_dataset_id=dataset_id,
    )
    obs_normalizer = load_validated_vecnormalize(
        manifest,
        expected_symbol=symbol,
        expected_action_map=action_map,
        expected_observation_shape=expected_observation_shape,
        expected_dataset_id=dataset_id,
    )
    scaler = load_validated_scaler(
        manifest,
        expected_symbol=symbol,
        expected_action_map=action_map,
        expected_observation_shape=expected_observation_shape,
        expected_dataset_id=dataset_id,
    )
    return ReplayContext(
        symbol=symbol.upper(),
        source="promoted_manifest",
        dataset_path=dataset_path,
        action_map=tuple(action_map),
        model=model,
        obs_normalizer=obs_normalizer,
        scaler=scaler,
        execution_cost_profile=_resolve_execution_cost_profile(manifest),
        reward_profile=_resolve_reward_profile(manifest),
        warmup_frame=warmup_frame,
        replay_frame=replay_frame,
        replay_feature_frame=holdout_feature_frame,
        full_feature_frame=featured,
        trainable_feature_frame=trainable_feature_frame,
        holdout_feature_frame=holdout_feature_frame,
        holdout_start_utc=pd.Timestamp(holdout_feature_frame.index[0]).isoformat() if not holdout_feature_frame.empty else None,
        diagnostics_path=diagnostics_path,
        manifest_path=manifest_path,
        artifact_metadata={
            "manifest_path": str(manifest_path),
            "model_path": str(manifest.model_path),
            "vecnormalize_path": str(manifest.vecnormalize_path) if manifest.vecnormalize_path else None,
            "training_diagnostics_path": str(diagnostics_path) if diagnostics_path is not None else None,
        },
        runtime_options=runtime_options,
    )


def _artifact_paths_from_fold_dir(fold_dir: Path, *, prefer_resume: bool) -> tuple[Path, Path] | None:
    candidate_pairs = [
        ("resume_model.zip", "resume_vecnormalize.pkl"),
        ("best_model.zip", "best_vecnormalize.pkl"),
    ]
    if not prefer_resume:
        candidate_pairs.reverse()
    for model_name, vec_name in candidate_pairs:
        model_path = fold_dir / model_name
        vec_path = fold_dir / vec_name
        if model_path.exists() and vec_path.exists():
            return model_path, vec_path
    return None


def _select_checkpoint_artifacts(symbol: str) -> dict[str, Any]:
    current_run = _load_json(CURRENT_RUN_CONTEXT_PATH) or {}
    candidates: list[dict[str, Any]] = []
    if str(current_run.get("symbol", "")).strip().upper() == symbol.upper():
        checkpoints_root = Path(str(current_run.get("checkpoints_root", "")))
        fold_index = int(current_run.get("fold_index", 0) or 0)
        current_fold_dir = checkpoints_root / f"fold_{fold_index}"
        current_paths = _artifact_paths_from_fold_dir(current_fold_dir, prefer_resume=True)
        if current_paths is not None:
            model_path, vecnormalize_path = current_paths
            candidates.append(
                {
                    "score": 100,
                    "checkpoints_root": checkpoints_root,
                    "fold_index": fold_index,
                    "fold_dir": current_fold_dir,
                    "model_path": model_path,
                    "vecnormalize_path": vecnormalize_path,
                    "diagnostics_path": _resolve_checkpoint_diagnostics_path(current_fold_dir),
                    "current_run": current_run,
                }
            )
        for fold_dir in sorted(checkpoints_root.glob("fold_*")):
            if fold_dir == current_fold_dir:
                continue
            best_paths = _artifact_paths_from_fold_dir(fold_dir, prefer_resume=False)
            if best_paths is None:
                continue
            model_path, vecnormalize_path = best_paths
            candidates.append(
                {
                    "score": 80 if _resolve_checkpoint_diagnostics_path(fold_dir) is not None else 60,
                    "checkpoints_root": checkpoints_root,
                    "fold_index": int(fold_dir.name.replace("fold_", "", 1)),
                    "fold_dir": fold_dir,
                    "model_path": model_path,
                    "vecnormalize_path": vecnormalize_path,
                    "diagnostics_path": _resolve_checkpoint_diagnostics_path(fold_dir),
                    "current_run": current_run,
                }
            )
    if not candidates:
        for run_dir in sorted(Path("checkpoints").glob("run_*")):
            for fold_dir in sorted(run_dir.glob("fold_*")):
                best_paths = _artifact_paths_from_fold_dir(fold_dir, prefer_resume=False)
                if best_paths is None:
                    continue
                model_path, vecnormalize_path = best_paths
                diagnostics_path = _resolve_checkpoint_diagnostics_path(fold_dir)
                candidates.append(
                    {
                        "score": 40 if diagnostics_path is not None else 20,
                        "checkpoints_root": run_dir,
                        "fold_index": int(fold_dir.name.replace("fold_", "", 1)),
                        "fold_dir": fold_dir,
                        "model_path": model_path,
                        "vecnormalize_path": vecnormalize_path,
                        "diagnostics_path": diagnostics_path,
                        "current_run": None,
                    }
                )
    if not candidates:
        raise FileNotFoundError(
            f"No promoted manifest or checkpoint artifacts found for {symbol.upper()}. "
            "Run train_agent.py first."
        )
    candidates.sort(
        key=lambda item: (
            int(item["score"]),
            float(item["model_path"].stat().st_mtime),
        ),
        reverse=True,
    )
    return candidates[0]


def _load_checkpoint_replay_context(symbol: str) -> ReplayContext:
    artifact = _select_checkpoint_artifacts(symbol)
    dataset_path = resolve_dataset_path()
    expected_ticks_per_bar = int(os.environ.get("TRAIN_BAR_TICKS", os.environ.get("BAR_TICKS_PER_BAR", "2000")))
    raw = _load_symbol_raw_frame(
        symbol=symbol,
        dataset_path=dataset_path,
        expected_ticks_per_bar=expected_ticks_per_bar,
    )
    featured = _compute_raw(raw).dropna(subset=list(FEATURE_COLS))
    trainable_feature_frame, holdout_feature_frame = _split_holdout(featured, HOLDOUT_FRAC)
    folds = purged_walk_forward_splits(
        trainable_feature_frame,
        n_folds=N_FOLDS,
        test_frac=FOLD_TEST_FRAC,
        purge_gap=PURGE_GAP_BARS,
    )
    fold_index = int(artifact["fold_index"])
    if not folds:
        raise RuntimeError("No runtime-compatible purged walk-forward folds were available for checkpoint replay.")
    if fold_index >= len(folds):
        fold_index = len(folds) - 1
    fold_train, fold_val = folds[fold_index]
    _, _, scaler = _fit_and_apply_fold_scaler(fold_train, fold_val)
    holdout_source = _prepend_runtime_warmup_context(featured, holdout_feature_frame)
    warmup_count = max(len(holdout_source) - len(holdout_feature_frame), 0)
    warmup_frame = holdout_source.iloc[:warmup_count].copy()
    replay_frame = holdout_source.iloc[warmup_count:].copy()
    if replay_frame.empty or warmup_frame.empty:
        raise RuntimeError("Checkpoint replay context produced an empty warmup or replay frame.")

    model = MaskablePPO.load(str(artifact["model_path"]), device="cpu")
    diagnostics_raw = artifact.get("diagnostics_path")
    diagnostics_path = Path(diagnostics_raw) if diagnostics_raw and Path(diagnostics_raw).exists() else None
    runtime_options = _load_training_runtime_options(diagnostics_path)
    expected_obs_shape = [int(runtime_options["window_size"]), len(FEATURE_COLS) + STATE_FEATURE_COUNT]
    model_obs_shape = [int(value) for value in model.observation_space.shape]
    if model_obs_shape != expected_obs_shape:
        if len(model_obs_shape) == 2 and int(model_obs_shape[1]) == int(expected_obs_shape[1]):
            runtime_options["window_size"] = int(model_obs_shape[0])
        else:
            raise RuntimeError(
                f"Checkpoint model observation shape mismatch: expected {expected_obs_shape}, got {model_obs_shape}."
            )
    with Path(artifact["vecnormalize_path"]).open("rb") as handle:
        obs_normalizer = pickle.load(handle)
    if hasattr(obs_normalizer, "training"):
        obs_normalizer.training = False
    if hasattr(obs_normalizer, "norm_reward"):
        obs_normalizer.norm_reward = False

    action_map = tuple(build_runtime_action_map())
    if int(model.action_space.n) != int(len(action_map)):
        raise RuntimeError(
            f"Checkpoint model action count mismatch: model={int(model.action_space.n)}, runtime={len(action_map)}."
        )

    execution_cost_profile = {
        key: float(value)
        for key, value in build_execution_cost_profile(
            slippage_pips=get_final_slippage_pips(TRAINING_RECOVERY_CONFIG)
        ).items()
        if key in {"commission_per_lot", "slippage_pips", "partial_fill_ratio"}
    }
    reward_profile = {
        key: float(value)
        for key, value in build_reward_profile().items()
    }
    return ReplayContext(
        symbol=symbol.upper(),
        source="checkpoint_fallback",
        dataset_path=dataset_path,
        action_map=action_map,
        model=model,
        obs_normalizer=obs_normalizer,
        scaler=scaler,
        execution_cost_profile=execution_cost_profile,
        reward_profile=reward_profile,
        warmup_frame=warmup_frame,
        replay_frame=replay_frame,
        replay_feature_frame=holdout_feature_frame,
        full_feature_frame=featured,
        trainable_feature_frame=trainable_feature_frame,
        holdout_feature_frame=holdout_feature_frame,
        holdout_start_utc=pd.Timestamp(holdout_feature_frame.index[0]).isoformat() if not holdout_feature_frame.empty else None,
        diagnostics_path=diagnostics_path,
        manifest_path=None,
        artifact_metadata={
            "run_id": str((artifact.get("current_run") or {}).get("run_id") or artifact["checkpoints_root"].name.replace("run_", "", 1)),
            "checkpoints_root": str(artifact["checkpoints_root"]),
            "fold_index": int(artifact["fold_index"]),
            "fold_dir": str(artifact["fold_dir"]),
            "model_path": str(artifact["model_path"]),
            "vecnormalize_path": str(artifact["vecnormalize_path"]),
            "diagnostics_path": str(diagnostics_path) if diagnostics_path is not None else None,
        },
        runtime_options=runtime_options,
    )


def load_replay_context(symbol: str | None = None) -> ReplayContext:
    resolved_symbol = str(symbol or TARGET_SYM).strip().upper() or TARGET_SYM
    promoted = _load_promoted_manifest_context(resolved_symbol)
    if promoted is not None:
        return promoted
    return _load_checkpoint_replay_context(resolved_symbol)


def _build_runtime(
    *,
    context: ReplayContext,
    use_policy: bool,
    disable_alpha_gate: bool = False,
) -> tuple[RuntimeEngine, ReplayBroker]:
    feature_engine = FeatureEngine.from_scaler(context.scaler)
    feature_engine.warm_up(context.warmup_frame)
    broker = ReplayBroker(symbol=context.symbol, **context.execution_cost_profile)
    snapshot = RuntimeSnapshot(last_equity=1_000.0, high_water_mark=1_000.0, day_start_equity=1_000.0)
    risk_engine = RiskEngine(RiskLimits(), snapshot=snapshot, initial_equity=1_000.0)
    runtime_options = dict(context.runtime_options or {})
    reward_profile = dict(context.reward_profile or {})
    reward_profile.pop("minimal_post_cost_reward", None)
    alpha_gate = None
    if bool(runtime_options.get("alpha_gate_enabled", False)) and not disable_alpha_gate:
        alpha_gate = fit_baseline_alpha_gate(
            symbol=context.symbol,
            train_frame=context.trainable_feature_frame,
            feature_cols=FEATURE_COLS,
            horizon_bars=int(runtime_options.get("baseline_target_horizon_bars", 10)),
            commission_per_lot=float(context.execution_cost_profile.get("commission_per_lot", 7.0)),
            slippage_pips=float(context.execution_cost_profile.get("slippage_pips", 0.25)),
            min_edge_pips=float(runtime_options.get("alpha_gate_min_edge_pips", 0.0)),
            probability_threshold=float(runtime_options.get("alpha_gate_probability_threshold", 0.55)),
            probability_margin=float(runtime_options.get("alpha_gate_probability_margin", 0.05)),
            model_preference=str(runtime_options.get("alpha_gate_model", "auto")),
        )
    runtime = RuntimeEngine(
        symbol=context.symbol,
        feature_engine=feature_engine,
        policy=ModelPolicy(context.model, context.action_map, obs_normalizer=context.obs_normalizer)
        if use_policy
        else SimpleNamespace(),
        broker=broker,
        action_map=context.action_map,
        risk_engine=risk_engine,
        snapshot=snapshot,
        state_store=None,
        window_size=int(runtime_options.get("window_size", 1)),
        minimal_post_cost_reward=bool(runtime_options.get("minimal_post_cost_reward", True)),
        force_fast_window_benchmark=bool(runtime_options.get("force_fast_window_benchmark", False)),
        alpha_gate=alpha_gate,
        churn_min_hold_bars=int(runtime_options.get("churn_min_hold_bars", 0)),
        churn_action_cooldown=int(runtime_options.get("churn_action_cooldown", 0)),
        entry_spread_z_limit=float(runtime_options.get("entry_spread_z_limit", 1.5)),
        **reward_profile,
    )
    runtime.startup_reconcile()
    return runtime, broker


def _align_reward_components_for_mode(
    reward_components: dict[str, Any],
    *,
    reward_value: float,
    minimal_post_cost_reward: bool,
) -> dict[str, Any]:
    aligned = dict(reward_components)
    if not minimal_post_cost_reward:
        return aligned
    aligned["pnl_reward"] = float(reward_value)
    aligned["holding_penalty_applied"] = 0.0
    aligned["participation_bonus_applied"] = 0.0
    aligned["slippage_penalty_applied"] = 0.0
    aligned["drawdown_penalty_applied"] = 0.0
    aligned["transaction_penalty_applied"] = 0.0
    aligned["turnover_penalty_applied"] = 0.0
    aligned["downside_risk_penalty_applied"] = 0.0
    aligned["rapid_reversal_penalty_applied"] = 0.0
    aligned["net_return_adjustment_applied"] = 0.0
    aligned["pre_bonus_reward"] = float(reward_value)
    aligned["final_reward_unclipped"] = float(reward_value)
    aligned["reward_raw_unclipped"] = float(reward_value)
    aligned["reward_unclipped"] = float(reward_value)
    aligned["reward_clipped"] = float(reward_value)
    aligned["net_reward_with_bonus"] = float(reward_value)
    aligned["reward_unclipped_net"] = float(reward_value)
    aligned["final_reward_clipped_low"] = 0.0
    aligned["final_reward_clipped_high"] = 0.0
    aligned["clip_hit_high"] = 0.0
    aligned["clip_hit_low"] = 0.0
    return aligned


def run_replay(
    *,
    symbol: str | None = None,
    replay_context: ReplayContext | None = None,
    action_index_provider: Callable[..., int] | None = None,
    disable_alpha_gate: bool = False,
) -> tuple[list[float], list[pd.Timestamp], list[dict[str, Any]], list[dict[str, Any]], dict[str, object]]:
    context = replay_context or load_replay_context(symbol)
    runtime, broker = _build_runtime(
        context=context,
        use_policy=action_index_provider is None,
        disable_alpha_gate=disable_alpha_gate,
    )
    minimal_post_cost_reward = bool(getattr(runtime, "minimal_post_cost_reward", False))

    equity_curve: list[float] = []
    timestamps: list[pd.Timestamp] = []
    diagnostics = TrainingDiagnostics()
    replay_bars = _frame_to_bars(context.replay_frame)
    replay_features = context.replay_feature_frame
    for bar_index, bar in enumerate(replay_bars):
        prev_position = int(runtime.confirmed_position.direction)
        prev_position_duration = int(runtime.confirmed_position.time_in_trade_bars)
        trade_log_before = len(getattr(broker, "trade_log", []))
        execution_log_before = len(getattr(broker, "execution_log", []))
        action_index_override = None
        if action_index_provider is not None:
            feature_row = replay_features.iloc[bar_index] if bar_index < len(replay_features) else pd.Series(dtype=float)
            action_index_override = int(
                action_index_provider(
                    bar_index=bar_index,
                    bar=bar,
                    feature_row=feature_row,
                    position_direction=prev_position,
                    position_duration=prev_position_duration,
                    action_map=context.action_map,
                )
            )
        result = runtime.process_bar(bar, action_index_override=action_index_override)
        reward_components = _align_reward_components_for_mode(
            dict(result.reward_components),
            reward_value=float(result.reward),
            minimal_post_cost_reward=minimal_post_cost_reward,
        )
        reward_components.setdefault("holding_penalty_applied", 0.0)
        reward_components.setdefault("participation_bonus_applied", 0.0)
        reward_components["forced_close_applied"] = 0.0
        turnover_lots = float(reward_components.get("turnover_lots", 0.0))
        entry_signal_direction = 0
        if (
            result.action.action_type.value == "OPEN"
            and prev_position == 0
            and result.action.direction is not None
            and result.submit_result is not None
            and bool(getattr(result.submit_result, "accepted", False))
        ):
            entry_signal_direction = int(result.action.direction)

        if bar_index == len(replay_bars) - 1 and int(result.position_direction) != 0:
            flatten_result = runtime.force_flatten(bar, reason="FORCED_EVAL_CLOSE")
            if bool(flatten_result.get("forced_close", False)):
                turnover_lots += float(flatten_result.get("turnover_lots", 0.0))
                reward_components = dict(
                    runtime._build_reward_components(
                        float(flatten_result.get("equity", result.equity)),
                        current_price=float(bar.close),
                        turnover_lots=turnover_lots,
                        avg_spread=float(bar.avg_spread),
                    )
                )
                forced_reward_value = (
                    float(flatten_result.get("equity", result.equity)) - float(runtime.last_equity)
                    if minimal_post_cost_reward
                    else float(reward_components.get("reward_clipped", result.reward))
                )
                reward_components = _align_reward_components_for_mode(
                    reward_components,
                    reward_value=float(forced_reward_value),
                    minimal_post_cost_reward=minimal_post_cost_reward,
                )
                reward_components["forced_close_applied"] = 1.0
                result.equity = float(flatten_result.get("equity", result.equity))
                result.reward = float(forced_reward_value)
                result.position_direction = 0

        trade_log_slice = [dict(item) for item in broker.trade_log[trade_log_before:] if isinstance(item, dict)]
        execution_log_slice = [
            dict(item) for item in getattr(broker, "execution_log", [])[execution_log_before:] if isinstance(item, dict)
        ]
        entry_filled_direction = 0
        for event in execution_log_slice:
            if str(event.get("side", "")).lower() == "open":
                entry_filled_direction = int(event.get("direction", 0) or 0)
                break
        diagnostics.record_step(
            action=result.action,
            submit_result=result.submit_result,
            prev_position=prev_position,
            new_position=int(result.position_direction),
            prev_position_duration=prev_position_duration,
            entry_signal_direction=entry_signal_direction,
            entry_filled_direction=entry_filled_direction,
            executed_events=execution_log_slice,
            closed_trades=trade_log_slice,
            reward_components=reward_components,
            reward=float(reward_components.get("reward_clipped", result.reward)),
        )
        equity_curve.append(float(result.equity))
        ts = pd.Timestamp(bar.timestamp)
        timestamps.append(ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC"))
    return equity_curve, timestamps, broker.trade_log, getattr(broker, "execution_log", []), diagnostics.snapshot()


def _evaluate_policy(
    *,
    replay_context: ReplayContext,
    action_index_provider: Callable[..., int] | None,
    disable_alpha_gate: bool = False,
) -> dict[str, Any]:
    equity_curve, timestamps, trade_log, execution_log, diagnostics = run_replay(
        replay_context=replay_context,
        action_index_provider=action_index_provider,
        disable_alpha_gate=disable_alpha_gate,
    )
    execution_diagnostics = aggregate_training_diagnostics([diagnostics])
    accounting = build_evaluation_accounting(
        trade_log=trade_log,
        execution_diagnostics=execution_diagnostics,
        execution_log_count=len(execution_log),
        initial_equity=1000.0,
    )
    validation_status = validate_evaluation_accounting(accounting)
    metrics = {
        "final_equity": float(equity_curve[-1]) if equity_curve else 1000.0,
        "total_return": float(((equity_curve[-1] if equity_curve else 1000.0) - 1000.0) / 1000.0),
        "timed_sharpe": float(compute_timed_sharpe(equity_curve, timestamps)),
        "max_drawdown": float(compute_max_drawdown(equity_curve)),
        "steps": int(len(equity_curve)),
        **accounting,
        "validation_status": validation_status,
        "accounting_gap_detected": not bool(validation_status.get("passed", False)),
    }
    return {
        "metrics": metrics,
        "trade_log": trade_log,
        "execution_log": execution_log,
        "timestamps": timestamps,
        "diagnostics": diagnostics,
        "execution_diagnostics": execution_diagnostics,
    }


def _evaluate_runtime_baselines(*, replay_context: ReplayContext) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for name in (
        "runtime_flat",
        "runtime_always_long",
        "runtime_always_short",
        "runtime_mean_reversion",
        "runtime_trend",
    ):
        provider = RUNTIME_BASELINE_PROVIDERS[name]
        payload = _evaluate_policy(
            replay_context=replay_context,
            action_index_provider=provider,
            disable_alpha_gate=True,
        )
        results[name] = {"metrics": payload["metrics"]}
    return results


def _evaluate_research_best_runtime_baseline(
    *,
    replay_context: ReplayContext,
    research_summary: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not research_summary:
        return None
    best_baseline = str(research_summary.get("best_baseline") or "").strip()
    if not best_baseline:
        return None
    provider = RUNTIME_BASELINE_PROVIDERS.get(best_baseline)
    if provider is None:
        return {
            "best_baseline": best_baseline,
            "supported": False,
            "metrics": None,
        }
    payload = _evaluate_policy(
        replay_context=replay_context,
        action_index_provider=provider,
        disable_alpha_gate=True,
    )
    return {
        "best_baseline": best_baseline,
        "supported": True,
        "metrics": payload["metrics"],
    }


def _build_runtime_parity_verdict(
    *,
    context: ReplayContext,
    replay_metrics: dict[str, Any],
    training_diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    runtime_baselines = _evaluate_runtime_baselines(replay_context=context)
    best_runtime_baseline = _best_model_name(runtime_baselines)
    best_runtime_metrics = (
        dict((runtime_baselines.get(best_runtime_baseline, {}) or {}).get("metrics", {}) or {})
        if best_runtime_baseline
        else {}
    )
    research_summary = _load_research_baseline_summary(training_diagnostics)
    research_best_runtime_replay = _evaluate_research_best_runtime_baseline(
        replay_context=context,
        research_summary=research_summary,
    )
    gate_off_payload = _evaluate_policy(
        replay_context=context,
        action_index_provider=None,
        disable_alpha_gate=True,
    )
    stress_results: dict[str, Any] = {}
    for multiplier in (1.5, 2.0):
        stressed_context = _with_cost_stress(context, slippage_multiplier=multiplier)
        stressed_payload = _evaluate_policy(
            replay_context=stressed_context,
            action_index_provider=None,
        )
        stress_results[f"{multiplier:.1f}x"] = {
            "slippage_multiplier": float(multiplier),
            "metrics": stressed_payload["metrics"],
            "profitable": _metrics_profitable(stressed_payload["metrics"]),
        }
    base_profitable = _metrics_profitable(replay_metrics)
    fragile_under_cost_stress = bool(
        base_profitable and any(not bool(result.get("profitable", False)) for result in stress_results.values())
    )
    runtime_baseline_viable = _metrics_profitable(best_runtime_metrics)
    research_baseline_viable = bool((research_summary or {}).get("research_baseline_viable", False))
    return {
        "base_replay_profitable": base_profitable,
        "rl_beats_best_runtime_baseline": bool(
            float(replay_metrics.get("net_pnl_usd", 0.0)) > float(best_runtime_metrics.get("net_pnl_usd", 0.0))
        ),
        "best_runtime_baseline": best_runtime_baseline,
        "best_runtime_baseline_metrics": best_runtime_metrics or None,
        "runtime_baseline_viable": runtime_baseline_viable,
        "research_baseline_summary": research_summary,
        "research_best_exact_runtime_replay": research_best_runtime_replay,
        "gate_off_replay_metrics": gate_off_payload["metrics"],
        "research_vs_runtime_parity_aligned": bool(not research_baseline_viable or runtime_baseline_viable),
        "runtime_holdout_models": runtime_baselines,
        "slippage_stress": stress_results,
        "fragile_under_cost_stress": fragile_under_cost_stress,
    }


def main() -> None:
    ensure_runtime_dirs()
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    context = load_replay_context(TARGET_SYM)
    log_config = configure_run_logging(
        "evaluate_oos",
        symbol=TARGET_SYM,
        capture_print=True,
    )
    set_log_context(symbol=TARGET_SYM)
    log.info(
        "Replay evaluation starting",
        extra={
            "event": "evaluation_start",
            "text_log_path": log_config.text_log_path,
            "jsonl_log_path": log_config.jsonl_log_path,
            "artifact_source": context.source,
        },
    )
    equity_curve, timestamps, trade_log, execution_log, diagnostics = run_replay(replay_context=context)
    final_equity = equity_curve[-1] if equity_curve else 1_000.0
    timed_sharpe = compute_timed_sharpe(equity_curve, timestamps)
    max_dd = compute_max_drawdown(equity_curve)
    execution_diagnostics = aggregate_training_diagnostics([diagnostics])
    # Build core accounting summary using unified helpers
    accounting = build_evaluation_accounting(
        trade_log=trade_log,
        execution_diagnostics=execution_diagnostics,
        execution_log_count=len(execution_log),
        initial_equity=1000.0,
    )
    val_status = validate_evaluation_accounting(accounting)
    trade_metrics = accounting  # Alias for schema compatibility and internal references

    replay_metrics = {
        "final_equity": float(final_equity),
        "total_return": float((final_equity - 1000.0) / 1000.0),
        "timed_sharpe": float(timed_sharpe),
        "max_drawdown": float(max_dd),
        "steps": int(len(equity_curve)),
        **accounting,
        "validation_status": val_status,
        "accounting_gap_detected": not val_status["passed"],
    }

    win_rate = float(replay_metrics["win_rate"])
    profit_factor = float(replay_metrics["profit_factor"])
    expectancy = float(replay_metrics["expectancy_usd"])
    expectancy_pips = float(replay_metrics["expectancy_pips"])
    n_trades = int(replay_metrics["trade_count"])
    avg_holding_bars = float(replay_metrics["avg_holding_bars"])
    total_return = float(replay_metrics["total_return"])
    report_paths = deployment_paths(TARGET_SYM, model_dir=EVAL_OUTPUT_DIR)
    training_diagnostics = (
        load_json_report(context.diagnostics_path)
        if context.diagnostics_path is not None and context.diagnostics_path.exists()
        else None
    )
    reject_fast_diagnostics = _build_reject_fast_diagnostics(
        replay_metrics=replay_metrics,
        trade_log=trade_log,
        execution_log=execution_log,
    )
    runtime_parity_verdict = _build_runtime_parity_verdict(
        context=context,
        replay_metrics=replay_metrics,
        training_diagnostics=training_diagnostics,
    )
    decision_summary = _build_decision_summary(
        reject_fast_diagnostics=reject_fast_diagnostics,
        runtime_parity_verdict=runtime_parity_verdict,
    )
    replay_metrics["runtime_parity_verdict"] = runtime_parity_verdict

    print("=" * 60)
    print("Replay OOS Evaluation")
    print("=" * 60)
    print(f"Artifact source: {context.source}")
    print(f"Initial equity : $1,000.00")
    print(f"Final equity   : ${final_equity:,.2f}")
    print(f"Total return   : {total_return:.1%}")
    print(f"Timed Sharpe   : {timed_sharpe:.3f}")
    print(f"Max drawdown   : {max_dd:.1%}")
    print(f"Win rate       : {win_rate:.1%}")
    print(f"Profit factor  : {profit_factor:.3f}")
    print(f"Expectancy     : ${expectancy:,.2f}/trade ({expectancy_pips:.3f} pips/trade)")
    print(f"Total trades   : {n_trades}")
    print(f"Gross PnL      : ${float(replay_metrics['gross_pnl_usd']):,.2f}")
    print(f"Net PnL        : ${float(replay_metrics['net_pnl_usd']):,.2f}")
    print(f"Txn costs paid : ${float(replay_metrics['total_transaction_cost_usd']):,.2f}")
    print(f"Avg hold bars  : {avg_holding_bars:.2f}")
    print(f"Orders exec'd  : {int(replay_metrics['executed_order_count'])}")
    print(f"Forced closes  : {int(replay_metrics['forced_close_count'])}")
    print(f"Best runtime baseline : {runtime_parity_verdict.get('best_runtime_baseline') or 'n/a'}")
    print(f"Base replay profitable: {bool(runtime_parity_verdict.get('base_replay_profitable', False))}")
    print(f"Fragile @ stress      : {bool(runtime_parity_verdict.get('fragile_under_cost_stress', False))}")
    print(f"Decision summary      : {decision_summary.get('verdict')}")
    print("=" * 60)

    replay_report = {
        "symbol": TARGET_SYM,
        "artifact_source": context.source,
        "artifact_metadata": context.artifact_metadata,
        "replay_metrics": replay_metrics,
        "trade_metrics": trade_metrics,
        "diagnostics": diagnostics,
        "execution_diagnostics": execution_diagnostics,
        "execution_log_count": int(len(execution_log)),
        "trade_log_count": int(len(trade_log)),
        "reject_fast_diagnostics": reject_fast_diagnostics,
        "runtime_parity_verdict": runtime_parity_verdict,
        "decision_summary": decision_summary,
        "reward_shaping_in_eval": False,
    }
    validate_evaluation_payload(replay_report)
    replay_report_path = EVAL_OUTPUT_DIR / f"replay_report_{TARGET_SYM.lower()}.json"
    save_json_report(replay_report, replay_report_path)
    gate = build_deployment_gate(
        symbol=TARGET_SYM,
        replay_metrics=replay_metrics,
        training_diagnostics=training_diagnostics,
    )
    save_json_report(gate, report_paths.gate_path)

    if gate["approved_for_live"]:
        print("Deployment candidate: replay and training metrics pass thresholds")
    else:
        print("Not deployment ready: deployment gate failed")
        for blocker in gate["blockers"]:
            print(f"BLOCKER: {blocker}")

    out_path = EVAL_OUTPUT_DIR / f"equity_curve_oos_{TARGET_SYM.lower()}.png"
    if not EVAL_SKIP_PLOT:
        plt.figure(figsize=(12, 5))
        plt.plot(equity_curve, color="#2ecc71", linewidth=1.5)
        plt.axhline(1_000, color="grey", linestyle="--", alpha=0.5, label="Start")
        plt.title(f"OOS Replay Equity - {TARGET_SYM}  TimedSharpe={timed_sharpe:.2f}  MaxDD={max_dd:.1%}")
        plt.xlabel("Replay bar")
        plt.ylabel("Equity ($)")
        plt.tight_layout()
        plt.savefig(out_path)
        print(f"Saved -> {out_path}")
    print(f"Saved -> {replay_report_path}")
    print(f"Saved -> {report_paths.gate_path}")
    log.info(
        "Replay evaluation complete",
        extra={
            "event": "evaluation_complete",
            "artifact_source": context.source,
            "equity_curve_path": out_path,
            "replay_report_path": replay_report_path,
            "deployment_gate_path": report_paths.gate_path,
            "trade_count": n_trades,
            "executed_order_count": int(len(execution_log)),
            "forced_close_count": int((diagnostics.get("trade_stats", {}) or {}).get("forced_close_count", 0)),
            "total_transaction_cost_usd": float(trade_metrics["total_transaction_cost_usd"]),
            "timed_sharpe": float(timed_sharpe),
            "max_drawdown": float(max_dd),
            "final_equity": float(final_equity),
        },
    )


if __name__ == "__main__":
    main()
