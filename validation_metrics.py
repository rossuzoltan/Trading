from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_config import (
    APPROX_KL_MAX,
    APPROX_KL_MIN,
    DEPLOY_DD_MAX,
    DEPLOY_EXPECTANCY_MIN,
    DEPLOY_PROFIT_FACTOR_MIN,
    DEPLOY_TIMED_SHARPE_MIN,
    EXPLAINED_VARIANCE_MIN,
    VALUE_LOSS_STABILITY_MAX,
)
from trading_config import deployment_paths


def _load_ops_attestation(symbol: str) -> dict[str, Any] | None:
    paths = deployment_paths(symbol)
    if not paths.ops_attestation_path.exists():
        return None
    return load_json_report(paths.ops_attestation_path)


def _training_bool(payload: dict[str, Any] | None, key: str, default: bool = False) -> bool:
    if not payload:
        return default
    return bool(payload.get(key, default))


def compute_timed_sharpe(equity_curve: list[float], timestamps: list[pd.Timestamp]) -> float:
    if len(equity_curve) < 2 or len(timestamps) < 2:
        return 0.0
    curve = np.asarray(equity_curve, dtype=np.float64)
    times = pd.to_datetime(timestamps, utc=True)
    log_returns = np.diff(np.log(np.maximum(curve, 1e-6)))
    delta_years = np.diff(times.view("int64")) / 1e9 / (365.25 * 24 * 3600.0)
    valid = delta_years > 0
    if not np.any(valid):
        return 0.0
    normalized = log_returns[valid] / np.sqrt(delta_years[valid])
    if len(normalized) == 0 or np.std(normalized) == 0:
        return 0.0
    return float(np.mean(normalized) / np.std(normalized))


def compute_max_drawdown(equity_curve: list[float]) -> float:
    curve = np.asarray(equity_curve, dtype=np.float64)
    if len(curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(curve)
    return float(np.max((peak - curve) / np.maximum(peak, 1e-6)))


def compute_win_rate(trade_log: list[dict[str, Any]]) -> float:
    if not trade_log:
        return 0.0
    wins = sum(1 for trade in trade_log if float(trade.get("net_pips", 0.0)) > 0)
    return float(wins / len(trade_log))


def compute_profit_factor(trade_log: list[dict[str, Any]]) -> float:
    profits = [float(trade.get("net_pips", 0.0)) for trade in trade_log if float(trade.get("net_pips", 0.0)) > 0]
    losses = [-float(trade.get("net_pips", 0.0)) for trade in trade_log if float(trade.get("net_pips", 0.0)) < 0]
    gross_profit = sum(profits)
    gross_loss = sum(losses)
    if gross_loss <= 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def compute_expectancy(trade_log: list[dict[str, Any]]) -> float:
    if not trade_log:
        return 0.0
    net_pips = np.asarray([float(trade.get("net_pips", 0.0)) for trade in trade_log], dtype=np.float64)
    return float(np.mean(net_pips))


def summarize_training_diagnostics(samples: list[dict[str, float]]) -> dict[str, Any]:
    if not samples:
        return {
            "sample_count": 0,
            "explained_variance": None,
            "approx_kl": None,
            "value_loss": None,
            "value_loss_stability": None,
            "gate_passed": False,
            "blockers": ["No training diagnostics were recorded."],
        }

    metrics: dict[str, list[float]] = {}
    for sample in samples:
        for key, value in sample.items():
            metrics.setdefault(key, []).append(float(value))

    summary: dict[str, Any] = {"sample_count": len(samples)}
    for key, values in metrics.items():
        arr = np.asarray(values, dtype=np.float64)
        summary[key] = float(arr[-1])
        summary[f"{key}_mean"] = float(np.mean(arr))
        summary[f"{key}_min"] = float(np.min(arr))
        summary[f"{key}_max"] = float(np.max(arr))

    value_losses = np.asarray(metrics.get("value_loss", []), dtype=np.float64)
    if len(value_losses) == 0:
        value_loss_stability = float("inf")
    else:
        median = float(np.median(np.abs(value_losses)))
        scale = max(median, 1e-6)
        value_loss_stability = float(np.percentile(np.abs(value_losses), 95) / scale)
    summary["value_loss_stability"] = value_loss_stability

    blockers: list[str] = []
    explained_variance = summary.get("explained_variance")
    approx_kl = summary.get("approx_kl")
    if explained_variance is None or float(explained_variance) < EXPLAINED_VARIANCE_MIN:
        blockers.append(f"explained_variance below {EXPLAINED_VARIANCE_MIN:.2f}")
    if approx_kl is None or not (APPROX_KL_MIN <= float(approx_kl) <= APPROX_KL_MAX):
        blockers.append(f"approx_kl outside [{APPROX_KL_MIN:.2f}, {APPROX_KL_MAX:.2f}]")
    if not np.isfinite(value_loss_stability) or value_loss_stability > VALUE_LOSS_STABILITY_MAX:
        blockers.append(f"value_loss stability above {VALUE_LOSS_STABILITY_MAX:.1f}")

    summary["gate_passed"] = not blockers
    summary["blockers"] = blockers
    return summary


def training_data_minimums() -> dict[str, int]:
    return {
        "min_train_bars": int(os.environ.get("TRAIN_MIN_TRAIN_BARS", os.environ.get("DEPLOY_MIN_TRAIN_BARS", "5000"))),
        "min_val_bars": int(os.environ.get("TRAIN_MIN_VAL_BARS", "200")),
        "min_holdout_bars": int(os.environ.get("TRAIN_MIN_HOLDOUT_BARS", os.environ.get("DEPLOY_MIN_HOLDOUT_BARS", "500"))),
    }


def assess_training_data_sufficiency(
    *,
    train_bars: int,
    holdout_bars: int,
    val_bars: int | None = None,
) -> list[str]:
    minimums = training_data_minimums()
    blockers: list[str] = []
    if int(train_bars) < minimums["min_train_bars"]:
        blockers.append(f"Train bars {int(train_bars)} < required {minimums['min_train_bars']}")
    if val_bars is not None and int(val_bars) < minimums["min_val_bars"]:
        blockers.append(f"Validation bars {int(val_bars)} < required {minimums['min_val_bars']}")
    if int(holdout_bars) < minimums["min_holdout_bars"]:
        blockers.append(f"Holdout bars {int(holdout_bars)} < required {minimums['min_holdout_bars']}")
    return blockers


def build_deployment_gate(
    *,
    symbol: str,
    replay_metrics: dict[str, Any],
    training_diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    timed_sharpe = float(replay_metrics.get("timed_sharpe", 0.0))
    max_drawdown = float(replay_metrics.get("max_drawdown", 1.0))
    profit_factor = float(replay_metrics.get("profit_factor", 0.0))
    expectancy = float(replay_metrics.get("expectancy", 0.0))

    if timed_sharpe < DEPLOY_TIMED_SHARPE_MIN:
        blockers.append(f"Timed Sharpe {timed_sharpe:.3f} < {DEPLOY_TIMED_SHARPE_MIN:.2f}")
    if max_drawdown > DEPLOY_DD_MAX:
        blockers.append(f"Max drawdown {max_drawdown:.1%} > {DEPLOY_DD_MAX:.0%}")
    if profit_factor < DEPLOY_PROFIT_FACTOR_MIN:
        blockers.append(f"Profit factor {profit_factor:.3f} < {DEPLOY_PROFIT_FACTOR_MIN:.2f}")
    if expectancy < DEPLOY_EXPECTANCY_MIN:
        blockers.append(f"Expectancy {expectancy:.3f} < {DEPLOY_EXPECTANCY_MIN:.2f}")

    if training_diagnostics is None:
        blockers.append("Training diagnostics missing.")
    else:
        blockers.extend(training_diagnostics.get("blockers", []))
        if not training_diagnostics.get("blockers") and not bool(
            training_diagnostics.get("gate_passed", training_diagnostics.get("passes_thresholds", False))
        ):
            blockers.append("Training diagnostics thresholds failed.")

        for key in ("baseline_gate_passed", "eval_protocol_valid", "full_path_eval_used"):
            if key not in training_diagnostics:
                blockers.append(f"Training diagnostics missing {key}.")
            elif training_diagnostics.get(key) is not True:
                blockers.append(f"Training diagnostics {key} must be true.")

        train_bars = int(training_diagnostics.get("train_bars", 0) or 0)
        val_bars_raw = training_diagnostics.get("val_bars")
        val_bars = int(val_bars_raw) if val_bars_raw is not None else None
        holdout_bars = int(training_diagnostics.get("holdout_bars", 0) or 0)
        blockers.extend(
            assess_training_data_sufficiency(
                train_bars=train_bars,
                val_bars=val_bars,
                holdout_bars=holdout_bars,
            )
        )

        if not _training_bool(training_diagnostics, "point_in_time_verified", False):
            blockers.append("Point-in-time integrity is not verified.")
        if not _training_bool(training_diagnostics, "dataset_integrity_verified", False):
            blockers.append("Dataset integrity is not verified.")

    require_attestation = os.environ.get("LIVE_REQUIRE_OPS_ATTESTATION", "1") != "0"
    min_shadow_days = int(os.environ.get("DEPLOY_MIN_SHADOW_DAYS", "14"))
    ops_attestation = _load_ops_attestation(symbol)
    if require_attestation:
        if not ops_attestation:
            blockers.append("Ops attestation missing (shadow/paper validation not evidenced).")
        else:
            shadow_days = int(ops_attestation.get("shadow_days_completed", 0) or 0)
            if shadow_days < min_shadow_days:
                blockers.append(f"Shadow days {shadow_days} < required {min_shadow_days}")
            if not bool(ops_attestation.get("execution_drift_ok", False)):
                blockers.append("Execution drift not attested OK.")
            if not bool(ops_attestation.get("position_reconciliation_ok", False)):
                blockers.append("Position reconciliation not attested OK.")

    return {
        "symbol": symbol.upper(),
        "approved_for_live": not blockers,
        "replay_metrics": replay_metrics,
        "training_diagnostics": training_diagnostics,
        "ops_attestation": ops_attestation,
        "blockers": blockers,
    }


def save_json_report(payload: dict[str, Any], path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def load_json_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
