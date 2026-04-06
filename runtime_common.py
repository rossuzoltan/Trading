from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from domain.enums import ActionType
from domain.models import ActionSpec, ConfirmedPosition
from symbol_utils import price_to_pips

STATE_FEATURE_COUNT = 4
TRAINING_RUNTIME_OPTION_KEYS = (
    "training_window_size",
    "training_churn_min_hold_bars",
    "training_churn_action_cooldown",
    "training_entry_spread_z_limit",
    "training_alpha_gate_enabled",
    "training_alpha_gate_model",
    "training_alpha_gate_probability_threshold",
    "training_alpha_gate_probability_margin",
    "training_alpha_gate_min_edge_pips",
    "baseline_target_horizon_bars",
)


def build_action_map(sl_options: Sequence[float], tp_options: Sequence[float]) -> tuple[ActionSpec, ...]:
    if not sl_options:
        raise ValueError("sl_options must contain at least one value.")
    if not tp_options:
        raise ValueError("tp_options must contain at least one value.")
    return build_simple_action_map(sl_value=float(sl_options[0]), tp_value=float(tp_options[0]))


def build_simple_action_map(
    *,
    sl_value: float = 1.0,
    tp_value: float = 1.0,
) -> tuple[ActionSpec, ...]:
    return (
        ActionSpec(ActionType.HOLD),
        ActionSpec(ActionType.CLOSE),
        ActionSpec(ActionType.OPEN, direction=1, sl_value=float(sl_value), tp_value=float(tp_value)),
        ActionSpec(ActionType.OPEN, direction=-1, sl_value=float(sl_value), tp_value=float(tp_value)),
    )


def serialize_action_map(action_map: Sequence[ActionSpec]) -> list[dict[str, float | int | str | None]]:
    return [
        {
            "action_type": action.action_type.value,
            "direction": action.direction,
            "sl_value": action.sl_value,
            "tp_value": action.tp_value,
        }
        for action in action_map
    ]


def deserialize_action_map(
    raw_action_map: Sequence[dict[str, float | int | str | None]],
) -> tuple[ActionSpec, ...]:
    actions: list[ActionSpec] = []
    for raw in raw_action_map:
        action_type = ActionType(str(raw.get("action_type", ActionType.HOLD.value)))
        direction_raw = raw.get("direction")
        sl_raw = raw.get("sl_value")
        tp_raw = raw.get("tp_value")
        actions.append(
            ActionSpec(
                action_type=action_type,
                direction=int(direction_raw) if direction_raw is not None else None,
                sl_value=float(sl_raw) if sl_raw is not None else None,
                tp_value=float(tp_raw) if tp_raw is not None else None,
            )
        )
    return tuple(actions)


def unrealised_pips(position: ConfirmedPosition, current_price: float, symbol: str) -> float:
    if position.is_flat:
        return 0.0
    assert position.entry_price is not None
    price_delta = float(current_price) - float(position.entry_price)
    if position.direction < 0:
        price_delta = -price_delta
    return price_to_pips(symbol, price_delta)


def build_state_vector(
    position: ConfirmedPosition,
    *,
    current_price: float,
    symbol: str,
) -> np.ndarray:
    pnl_sign = 0.0
    if not position.is_flat:
        pnl_sign = float(np.sign(unrealised_pips(position, current_price, symbol)))
    state = np.array(
        [
            float(position.direction),
            min(float(position.time_in_trade_bars) / 1000.0, 1.0),
            pnl_sign,
            np.clip(float(position.last_reward), -1.0, 1.0),
        ],
        dtype=np.float32,
    )
    return state


def build_observation(
    feature_rows: np.ndarray,
    *,
    position: ConfirmedPosition,
    current_price: float,
    symbol: str,
    window_size: int = 1,
) -> np.ndarray:
    rows = np.asarray(feature_rows, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.shape[0] > window_size:
        rows = rows[-window_size:]
    if rows.shape[0] < window_size:
        pad_row = rows[0] if rows.size else np.zeros(0, dtype=np.float32)
        pad = np.tile(pad_row, (window_size - rows.shape[0], 1))
        rows = np.vstack([pad, rows]) if rows.size else pad
    state = build_state_vector(position, current_price=current_price, symbol=symbol)
    state_block = np.tile(state, (window_size, 1))
    obs = np.hstack([rows, state_block]).astype(np.float32)
    return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


def build_action_mask(
    action_map: Sequence[ActionSpec],
    *,
    position: ConfirmedPosition,
    spread_z: float = 0.0,
) -> np.ndarray:
    mask = np.zeros(len(action_map), dtype=bool)
    if not action_map:
        return mask
    mask[0] = True
    if position.is_flat:
        if float(spread_z) < 1.5:
            mask[2:] = True
    else:
        mask[1] = True
    return mask


def apply_execution_action_guards(
    mask: Sequence[bool],
    *,
    position: ConfirmedPosition,
    spread_z: float,
    entry_spread_z_limit: float,
    churn_min_hold_bars: int,
    current_bar_index: int,
    last_close_bar_index: int | None,
    churn_action_cooldown: int,
) -> np.ndarray:
    guarded = np.asarray(mask, dtype=bool).copy()
    if guarded.size == 0:
        return guarded
    if position.is_flat and float(spread_z) >= float(entry_spread_z_limit):
        guarded[2:] = False
    if not position.is_flat and int(churn_min_hold_bars) > 0:
        if int(position.time_in_trade_bars) < int(churn_min_hold_bars):
            forced_hold = np.zeros_like(guarded)
            forced_hold[0] = True
            return forced_hold
    if position.is_flat and int(churn_action_cooldown) > 0 and last_close_bar_index is not None:
        bars_since_close = int(current_bar_index) - int(last_close_bar_index)
        if bars_since_close < int(churn_action_cooldown):
            forced_hold = np.zeros_like(guarded)
            forced_hold[0] = True
            return forced_hold
    return guarded


def runtime_options_from_training_payload(
    payload: Mapping[str, Any] | None,
    *,
    default_window_size: int = 1,
) -> dict[str, Any]:
    source = dict(payload or {})
    return {
        "window_size": int(source.get("training_window_size", default_window_size) or default_window_size),
        "churn_min_hold_bars": int(source.get("training_churn_min_hold_bars", 0) or 0),
        "churn_action_cooldown": int(source.get("training_churn_action_cooldown", 0) or 0),
        "entry_spread_z_limit": float(source.get("training_entry_spread_z_limit", 1.5)),
        "minimal_post_cost_reward": bool(source.get("training_minimal_post_cost_reward", False)),
        "force_fast_window_benchmark": bool(source.get("training_force_fast_window_benchmark", False)),
        "alpha_gate_enabled": bool(source.get("training_alpha_gate_enabled", False)),
        "alpha_gate_model": str(source.get("training_alpha_gate_model", "auto") or "auto"),
        "alpha_gate_probability_threshold": float(source.get("training_alpha_gate_probability_threshold", 0.55)),
        "alpha_gate_probability_margin": float(source.get("training_alpha_gate_probability_margin", 0.05)),
        "alpha_gate_min_edge_pips": float(source.get("training_alpha_gate_min_edge_pips", 0.0)),
        "baseline_target_horizon_bars": int(source.get("baseline_target_horizon_bars", 10) or 10),
    }


def action_label(action: ActionSpec) -> str:
    if action.action_type == ActionType.HOLD:
        return "FLAT"
    if action.action_type == ActionType.CLOSE:
        return "CLOSE"
    if int(action.direction or 0) > 0:
        return "LONG"
    if int(action.direction or 0) < 0:
        return "SHORT"
    return action.action_type.value


def flatten_feature_window(rows: Iterable[Sequence[float]]) -> np.ndarray:
    return np.asarray(list(rows), dtype=np.float32)


def compute_timed_sharpe(equity_curve: Sequence[float], timestamps: Sequence[pd.Timestamp]) -> float:
    if len(equity_curve) < 2 or len(timestamps) < 2:
        return 0.0
    curve = np.asarray(equity_curve, dtype=np.float64)
    times = pd.to_datetime(list(timestamps), utc=True)
    log_returns = np.diff(np.log(np.maximum(curve, 1e-6)))
    delta_years = np.diff(times.view("int64")) / 1e9 / (365.25 * 24 * 3600.0)
    valid = delta_years > 0
    if not np.any(valid):
        return 0.0
    normalized = log_returns[valid] / np.sqrt(delta_years[valid])
    if len(normalized) == 0 or np.std(normalized) == 0:
        return 0.0
    return float(np.mean(normalized) / np.std(normalized))


def compute_max_drawdown(equity_curve: Sequence[float]) -> float:
    curve = np.asarray(equity_curve, dtype=np.float64)
    if len(curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(curve)
    return float(np.max((peak - curve) / np.maximum(peak, 1e-6)))


def compute_trade_metrics(
    trade_log: Sequence[dict],
    *,
    initial_equity: float = 1_000.0,
) -> dict[str, float]:
    if not trade_log:
        return {
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "expectancy_usd": 0.0,
            "expectancy_pips": 0.0,
            "win_rate": 0.0,
            "trade_count": 0.0,
            "gross_pnl_usd": 0.0,
            "net_pnl_usd": 0.0,
            "gross_profit_usd": 0.0,
            "gross_loss_usd": 0.0,
            "total_transaction_cost_usd": 0.0,
            "total_commission_usd": 0.0,
            "total_spread_slippage_cost_usd": 0.0,
            "total_spread_cost_usd": 0.0,
            "total_slippage_cost_usd": 0.0,
            "forced_close_count": 0.0,
            "avg_holding_bars": 0.0,
            "avg_win_usd": 0.0,
            "avg_loss_usd": 0.0,
            "win_loss_asymmetry": 0.0,
        }

    pnl_series: list[float] = []
    net_pips_series: list[float] = []
    gross_pnl_usd = 0.0
    total_transaction_cost_usd = 0.0
    total_commission_usd = 0.0
    total_spread_slippage_cost_usd = 0.0
    total_spread_cost_usd = 0.0
    total_slippage_cost_usd = 0.0
    forced_close_count = 0
    holding_bars: list[float] = []
    prev_equity = float(initial_equity)
    for trade in trade_log:
        gross_pnl_usd += float(trade.get("gross_pnl_usd", 0.0))
        total_transaction_cost_usd += float(trade.get("transaction_cost_usd", 0.0))
        total_commission_usd += float(trade.get("commission_usd", 0.0))
        total_spread_slippage_cost_usd += float(trade.get("spread_slippage_cost_usd", 0.0))
        total_spread_cost_usd += float(trade.get("spread_cost_usd", 0.0))
        total_slippage_cost_usd += float(trade.get("slippage_cost_usd", 0.0))
        forced_close_count += int(bool(trade.get("forced_close", False)))
        if "holding_bars" in trade:
            holding_bars.append(float(trade.get("holding_bars", 0.0)))
        net_pips_series.append(float(trade.get("net_pips", 0.0)))
        if "equity" in trade:
            current_equity = float(trade["equity"])
            pnl_series.append(current_equity - prev_equity)
            prev_equity = current_equity
        elif "net_pnl_usd" in trade:
            pnl_series.append(float(trade["net_pnl_usd"]))
        else:
            pnl_series.append(float(trade.get("net_pips", 0.0)))

    wins = [pnl for pnl in pnl_series if pnl > 0]
    losses = [pnl for pnl in pnl_series if pnl < 0]
    gross_profit = float(sum(wins))
    gross_loss = float(abs(sum(losses)))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    expectancy_usd = float(np.mean(pnl_series)) if pnl_series else 0.0
    expectancy_pips = float(np.mean(net_pips_series)) if net_pips_series else 0.0
    win_rate = len(wins) / len(pnl_series) if pnl_series else 0.0
    avg_win_usd = float(np.mean(wins)) if wins else 0.0
    avg_loss_usd = float(np.mean(losses)) if losses else 0.0
    win_loss_asymmetry = avg_win_usd / abs(avg_loss_usd) if avg_loss_usd < 0 else (float("inf") if avg_win_usd > 0 else 0.0)
    net_pnl_usd = float(sum(pnl_series))
    avg_holding_bars = float(np.mean(holding_bars)) if holding_bars else 0.0
    return {
        "profit_factor": float(profit_factor),
        "expectancy": expectancy_usd,
        "expectancy_usd": expectancy_usd,
        "expectancy_pips": expectancy_pips,
        "win_rate": float(win_rate),
        "trade_count": float(len(pnl_series)),
        "gross_pnl_usd": float(gross_pnl_usd),
        "net_pnl_usd": net_pnl_usd,
        "gross_profit_usd": gross_profit,
        "gross_loss_usd": gross_loss,
        "total_transaction_cost_usd": float(total_transaction_cost_usd),
        "total_commission_usd": float(total_commission_usd),
        "total_spread_slippage_cost_usd": float(total_spread_slippage_cost_usd),
        "total_spread_cost_usd": float(total_spread_cost_usd),
        "total_slippage_cost_usd": float(total_slippage_cost_usd),
        "forced_close_count": float(forced_close_count),
        "avg_holding_bars": avg_holding_bars,
        "avg_win_usd": avg_win_usd,
        "avg_loss_usd": avg_loss_usd,
        "win_loss_asymmetry": float(win_loss_asymmetry),
    }


def build_trade_metric_reconciliation(
    *,
    trade_metrics: dict[str, float],
    trade_diagnostics: dict | None = None,
    economics: dict | None = None,
    trade_log_count: int | None = None,
    execution_log_count: int | None = None,
    amount_tolerance: float = 1e-6,
) -> dict[str, object]:
    trade_diagnostics = dict(trade_diagnostics or {})
    economics = dict(economics or {})
    checks: dict[str, dict[str, object]] = {}
    mismatches: list[str] = []

    def _numeric_check(name: str, left: float, right: float, *, tolerance: float = amount_tolerance) -> None:
        diff = float(abs(float(left) - float(right)))
        passed = bool(diff <= float(tolerance))
        checks[name] = {
            "passed": passed,
            "left": float(left),
            "right": float(right),
            "abs_diff": diff,
            "tolerance": float(tolerance),
        }
        if not passed:
            mismatches.append(name)

    def _count_check(name: str, left: int, right: int) -> None:
        passed = int(left) == int(right)
        checks[name] = {
            "passed": passed,
            "left": int(left),
            "right": int(right),
        }
        if not passed:
            mismatches.append(name)

    trade_count = int(trade_metrics.get("trade_count", 0.0) or 0)
    forced_close_count = int(trade_metrics.get("forced_close_count", 0.0) or 0)
    avg_holding_bars = float(trade_metrics.get("avg_holding_bars", 0.0) or 0.0)

    if trade_log_count is not None:
        _count_check("trade_count_vs_trade_log", trade_count, int(trade_log_count))
    if "closed_trade_count" in trade_diagnostics:
        _count_check("trade_count_vs_diagnostics", trade_count, int(trade_diagnostics.get("closed_trade_count", 0)))
    if "forced_close_count" in trade_diagnostics:
        _count_check(
            "forced_close_count_vs_diagnostics",
            forced_close_count,
            int(trade_diagnostics.get("forced_close_count", 0)),
        )
    if execution_log_count is not None and "order_executed_count" in trade_diagnostics:
        _count_check(
            "executed_order_count_vs_execution_log",
            int(trade_diagnostics.get("order_executed_count", 0)),
            int(execution_log_count),
        )
    if "position_duration_sum" in trade_diagnostics and "position_duration_count" in trade_diagnostics:
        duration_count = int(trade_diagnostics.get("position_duration_count", 0) or 0)
        diagnostics_avg_holding = (
            float(trade_diagnostics.get("position_duration_sum", 0.0) or 0.0) / float(duration_count)
            if duration_count > 0
            else 0.0
        )
        _numeric_check("avg_holding_bars_vs_diagnostics", avg_holding_bars, diagnostics_avg_holding)

    economics_key_map = {
        "gross_pnl_usd": "gross_pnl_usd",
        "net_pnl_usd": "net_pnl_usd",
        "total_transaction_cost_usd": "transaction_cost_usd",
        "total_commission_usd": "commission_usd",
        "total_spread_slippage_cost_usd": "spread_slippage_cost_usd",
        "total_spread_cost_usd": "spread_cost_usd",
        "total_slippage_cost_usd": "slippage_cost_usd",
    }
    for metric_key, economics_key in economics_key_map.items():
        if economics_key not in economics:
            continue
        _numeric_check(
            f"{metric_key}_vs_diagnostics",
            float(trade_metrics.get(metric_key, 0.0) or 0.0),
            float(economics.get(economics_key, 0.0) or 0.0),
        )

    return {
        "passed": not mismatches,
        "mismatch_fields": mismatches,
        "checks": checks,
    }


def build_evaluation_accounting(
    *,
    trade_log: Sequence[dict],
    execution_diagnostics: dict[str, Any],
    execution_log_count: int,
    initial_equity: float = 1000.0,
) -> dict[str, Any]:
    """
    Transform trade log + execution diagnostics into a single normalized accounting summary.
    """
    trade_diagnostics = dict(
        execution_diagnostics.get("trade_diagnostics", execution_diagnostics.get("trade_stats", {})) or {}
    )
    economics = dict(execution_diagnostics.get("economics", {}))
    trade_metrics = compute_trade_metrics(trade_log, initial_equity=initial_equity)
    reconciliation = build_trade_metric_reconciliation(
        trade_metrics=trade_metrics,
        trade_diagnostics=trade_diagnostics,
        economics=economics,
        trade_log_count=len(trade_log),
        execution_log_count=execution_log_count,
    )

    # Flatten into a single summary
    summary = {
        **trade_metrics,
        "metrics_reconciliation": reconciliation,
        "metric_reconciliation": reconciliation,
    }
    # Ensure top-level fields match downstream expectations if they differ
    summary["executed_order_count"] = int(trade_diagnostics.get("order_executed_count", 0))
    summary["forced_close_count"] = int(trade_diagnostics.get("forced_close_count", 0))

    return summary


def validate_evaluation_accounting(accounting: dict[str, Any]) -> dict[str, Any]:
    """
    Assert that trade counts derived from the trade log match the closed trade counts from diagnostics.
    """
    reconcile = accounting.get("metrics_reconciliation") or accounting.get("metric_reconciliation", {})
    passed = bool(reconcile.get("passed", False))
    mismatches = reconcile.get("mismatch_fields", [])

    if not passed:
        reason = f"Accounting reconciliation failed for fields: {', '.join(mismatches)}"
    else:
        reason = "Accounting reconciliation passed."

    return {
        "passed": passed,
        "reason": reason,
        "mismatches": mismatches,
    }


def validate_evaluation_payload(payload: dict[str, Any]) -> None:
    """
    Ensure required summary fields exist and are consistent before serialization.
    Fail closed if reconciliation failed.
    """
    # 1. Check top-level metrics for reconciliation success
    metrics = payload.get("replay_metrics") or payload
    if not isinstance(metrics, dict):
        raise ValueError("Evaluation payload missing valid metrics dictionary.")

    reconcile = metrics.get("metrics_reconciliation") or metrics.get("metric_reconciliation", {})
    if not bool(reconcile.get("passed", False)):
        mismatches = reconcile.get("mismatch_fields", [])
        raise RuntimeError(f"Cannot serialize evaluation: reconciliation failed for {mismatches}")

    # 2. Ensure critical fields exist
    required_fields = ["trade_count", "final_equity", "net_pnl_usd", "profit_factor"]
    missing = [f for f in required_fields if f not in metrics]
    if missing:
        raise ValueError(f"Evaluation metrics missing required fields: {missing}")
