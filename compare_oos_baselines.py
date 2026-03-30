from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from evaluate_oos import run_replay
from runtime_common import (
    build_evaluation_accounting,
    compute_max_drawdown,
    compute_timed_sharpe,
    validate_evaluation_accounting,
)
from train_agent import _prepend_runtime_warmup_context, aggregate_training_diagnostics


_TARGET = Path(__file__).resolve().parent / "tools" / "compare_oos_baselines.py"
__file__ = str(_TARGET)
exec(compile(_TARGET.read_text(encoding="utf-8"), str(_TARGET), "exec"), globals())


def _flat_provider(**_: object) -> int:
    return 0


def _clone_context_with_segment(context, segment: pd.DataFrame):
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


def _with_cost_stress(context, *, slippage_multiplier: float):
    stressed = dict(context.execution_cost_profile)
    stressed["slippage_pips"] = float(stressed.get("slippage_pips", 0.25)) * float(slippage_multiplier)
    return replace(context, execution_cost_profile=stressed)


def _evaluate_policy(*, replay_context, action_index_provider):
    equity_curve, timestamps, trade_log, execution_log, diagnostics = run_replay(
        replay_context=replay_context,
        action_index_provider=action_index_provider,
    )
    accounting = build_evaluation_accounting(
        trade_log=trade_log,
        execution_diagnostics=aggregate_training_diagnostics([diagnostics]),
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
    }
