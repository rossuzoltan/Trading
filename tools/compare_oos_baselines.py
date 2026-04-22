from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge_research import run_edge_baseline_research
from evaluate_oos import (
    _best_model_name as _shared_best_model_name,
    _build_action_index_provider as _shared_build_action_index_provider,
    _build_runtime_parity_verdict as _shared_build_runtime_parity_verdict,
    _evaluate_policy as _shared_evaluate_policy,
    _evaluate_runtime_baselines as _shared_evaluate_runtime_baselines,
    _flat_provider as _shared_flat_provider,
    _mean_reversion_provider as _shared_mean_reversion_provider,
    _trend_provider as _shared_trend_provider,
    _with_cost_stress as _shared_with_cost_stress,
    load_replay_context,
)
from feature_engine import FEATURE_COLS
from runtime_common import ActionType
from validation_metrics import load_json_report, save_json_report


def _flat_provider(**_: object) -> int:
    return _shared_flat_provider(**_)


def _resolve_action_indexes(action_map) -> dict[str, int]:
    indexes = {"hold": 0, "close": 0, "long": 0, "short": 0}
    for idx, action in enumerate(action_map):
        if action.action_type == ActionType.HOLD:
            indexes["hold"] = int(idx)
        elif action.action_type == ActionType.CLOSE:
            indexes["close"] = int(idx)
        elif action.action_type == ActionType.OPEN and int(action.direction or 0) > 0:
            indexes["long"] = int(idx)
        elif action.action_type == ActionType.OPEN and int(action.direction or 0) < 0:
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


def _trend_provider(
    *,
    feature_row,
    position_direction: int,
    action_map,
    **_: object,
) -> int:
    return _shared_trend_provider(
        feature_row=feature_row,
        position_direction=position_direction,
        action_map=action_map,
        **_,
    )


def _mean_reversion_provider(
    *,
    feature_row,
    position_direction: int,
    action_map,
    **_: object,
) -> int:
    return _shared_mean_reversion_provider(
        feature_row=feature_row,
        position_direction=position_direction,
        action_map=action_map,
        **_,
    )


def _evaluate_policy(*, replay_context, action_index_provider, disable_alpha_gate: bool = False):
    return _shared_evaluate_policy(
        replay_context=replay_context,
        action_index_provider=action_index_provider,
        disable_alpha_gate=disable_alpha_gate,
    )


def _evaluate_runtime_baselines(*, replay_context) -> dict[str, dict[str, Any]]:
    return _shared_evaluate_runtime_baselines(replay_context=replay_context)


def _build_folds(trainable_frame: pd.DataFrame, *, validation_frac: float) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    validation_size = max(int(len(trainable_frame) * float(validation_frac)), 1)
    validation_start = max(len(trainable_frame) - validation_size, 1)
    train_fold = trainable_frame.iloc[:validation_start].copy()
    val_fold = trainable_frame.iloc[validation_start:].copy()
    if train_fold.empty or val_fold.empty:
        raise RuntimeError("Validation split produced an empty training or validation fold.")
    return [(train_fold, val_fold)]


def _best_model_name(models: dict[str, Any]) -> str | None:
    return _shared_best_model_name(models)


def build_baseline_comparison(
    *,
    symbol: str,
    report_path: Path | None = None,
    horizon_bars: int = 10,
    validation_frac: float = 0.15,
    min_edge_pips: float = 0.0,
) -> dict[str, Any]:
    replay_context = load_replay_context(symbol)
    cost_profile = dict(replay_context.execution_cost_profile)
    trainable_frame = replay_context.trainable_feature_frame.copy()
    holdout_frame = replay_context.holdout_feature_frame.copy()
    if trainable_frame.empty or holdout_frame.empty:
        raise RuntimeError("Replay context did not provide a usable trainable/holdout split.")
    folds = _build_folds(trainable_frame, validation_frac=validation_frac)
    out_path = report_path or Path("models") / f"baseline_comparison_{symbol.lower()}.json"
    baseline_out_path = out_path.with_name(f"baseline_holdout_{symbol.lower()}.json")
    baseline_report = run_edge_baseline_research(
        symbol=symbol,
        trainable_frame=trainable_frame,
        holdout_frame=holdout_frame,
        folds=folds,
        feature_cols=list(FEATURE_COLS),
        out_path=baseline_out_path,
        horizon_bars=int(horizon_bars),
        commission_per_lot=float(cost_profile["commission_per_lot"]),
        slippage_pips=float(cost_profile["slippage_pips"]),
        min_edge_pips=float(min_edge_pips),
        probability_threshold=0.55,
        probability_margin=0.05,
        min_trade_count=20,
    )
    replay_report_path = Path("models") / f"replay_report_{symbol.lower()}.json"
    replay_report = load_json_report(replay_report_path) if replay_report_path.exists() else None
    if replay_report is None:
        action_index_provider = _shared_build_action_index_provider(replay_context)
        replay_report = {
            "replay_metrics": _evaluate_policy(
                replay_context=replay_context,
                action_index_provider=action_index_provider,
                disable_alpha_gate=bool(action_index_provider is not None),
            )["metrics"]
        }
    replay_metrics = dict((replay_report or {}).get("replay_metrics", {}) or {})
    training_diagnostics = (
        load_json_report(replay_context.diagnostics_path)
        if replay_context.diagnostics_path is not None and replay_context.diagnostics_path.exists()
        else None
    )
    runtime_parity_verdict = dict(replay_metrics.get("runtime_parity_verdict", {}) or {})
    if not runtime_parity_verdict:
        runtime_parity_verdict = _shared_build_runtime_parity_verdict(
            context=replay_context,
            replay_metrics=replay_metrics,
            training_diagnostics=training_diagnostics,
        )
        replay_metrics["runtime_parity_verdict"] = runtime_parity_verdict
    holdout_models = dict((baseline_report.get("holdout_metrics", {}) or {}).get("models", {}) or {})
    best_baseline = _best_model_name(holdout_models)
    best_baseline_metrics = dict((holdout_models.get(best_baseline, {}) or {}).get("metrics", {}) or {}) if best_baseline else {}
    best_runtime_baseline = runtime_parity_verdict.get("best_runtime_baseline")
    best_runtime_baseline_metrics = dict(runtime_parity_verdict.get("best_runtime_baseline_metrics", {}) or {})
    runtime_holdout_models = dict(runtime_parity_verdict.get("runtime_holdout_models", {}) or {})

    comparison = {
        "symbol": symbol.upper(),
        "cost_profile": cost_profile,
        "target_definition": dict(baseline_report.get("target_definition", {}) or {}),
        "replay_report_path": str(replay_report_path) if replay_report_path.exists() else None,
        "baseline_report_path": str(baseline_out_path),
        "rl_replay_metrics": replay_metrics or None,
        "runtime_holdout_models": runtime_holdout_models,
        "best_runtime_baseline": best_runtime_baseline,
        "best_runtime_baseline_metrics": best_runtime_baseline_metrics or None,
        "runtime_parity_verdict": runtime_parity_verdict,
        "baseline_holdout_models": holdout_models,
        "best_baseline": best_baseline,
        "best_baseline_metrics": best_baseline_metrics or None,
        "comparison": {
            "rl_trade_count": replay_metrics.get("trade_count"),
            "rl_net_pnl_usd": replay_metrics.get("net_pnl_usd"),
            "rl_profit_factor": replay_metrics.get("profit_factor"),
            "runtime_baseline_trade_count": best_runtime_baseline_metrics.get("trade_count") if best_runtime_baseline_metrics else None,
            "runtime_baseline_net_pnl_usd": best_runtime_baseline_metrics.get("net_pnl_usd") if best_runtime_baseline_metrics else None,
            "runtime_baseline_profit_factor": best_runtime_baseline_metrics.get("profit_factor") if best_runtime_baseline_metrics else None,
            "baseline_trade_count": best_baseline_metrics.get("trade_count") if best_baseline_metrics else None,
            "baseline_net_pnl_usd": best_baseline_metrics.get("net_pnl_usd") if best_baseline_metrics else None,
            "baseline_profit_factor": best_baseline_metrics.get("profit_factor") if best_baseline_metrics else None,
        },
    }
    save_json_report(comparison, out_path)
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare OOS RL replay against simple baselines under manifest costs.")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--horizon-bars", type=int, default=10)
    parser.add_argument("--validation-frac", type=float, default=0.15)
    parser.add_argument("--min-edge-pips", type=float, default=0.0)
    args = parser.parse_args()
    report = build_baseline_comparison(
        symbol=str(args.symbol).upper(),
        report_path=Path(args.report_path) if args.report_path else None,
        horizon_bars=int(args.horizon_bars),
        validation_frac=float(args.validation_frac),
        min_edge_pips=float(args.min_edge_pips),
    )
    out = [
        f"# Baseline Comparison: {args.symbol}",
        "",
        "| Model | Trades | Net PnL (USD) | Profit Factor | Expectancy (USD) | Win Rate | Trades / Bar |",
        "| --- | --- | --- | --- | --- | --- | --- |"
    ]

    def _row(name: str, metrics: dict[str, Any] | None) -> str:
        if not metrics:
            return f"| **{name}** | N/A | N/A | N/A | N/A | N/A | N/A |"
        t = int(metrics.get("trade_count", 0))
        pnl = float(metrics.get("net_pnl_usd", 0.0))
        pf = float(metrics.get("profit_factor", 0.0))
        ex = float(metrics.get("expectancy_usd", 0.0))
        wr = float(metrics.get("win_rate", 0.0))
        tpb = float(metrics.get("trades_per_bar", 0.0))
        return f"| **{name}** | {t} | ${pnl:.2f} | {pf:.2f} | ${ex:.2f} | {wr:.1%} | {tpb:.4f} |"

    rl_metrics = report.get("rl_replay_metrics")
    out.append(_row("RL Agent (OOS Replay)", rl_metrics))

    runtime_models = report.get("runtime_holdout_models", {})
    for name, data in runtime_models.items():
        out.append(_row(f"Runtime baseline: {name}", data.get("metrics")))

    models = report.get("baseline_holdout_models", {})
    for name, data in models.items():
        out.append(_row(f"Baseline: {name}", data.get("metrics")))
    
    out.append("")
    out.append("## Verdict")
    out.append("")
    
    rl_pnl = float(rl_metrics.get("net_pnl_usd", 0.0)) if rl_metrics else 0.0
    best_runtime_baseline_pnl = float((report.get("best_runtime_baseline_metrics") or {}).get("net_pnl_usd", 0.0))
    best_baseline_pnl = float((report.get("best_baseline_metrics") or {}).get("net_pnl_usd", 0.0))
    best_runtime_name = report.get("best_runtime_baseline", "None")
    best_name = report.get("best_baseline", "None")
    
    if not rl_metrics:
        out.append("RL Replay metrics missing. Verdict: CANNOT COMPARE.")
    else:
        if report.get("best_runtime_baseline_metrics") is not None:
            if rl_pnl > best_runtime_baseline_pnl:
                out.append(f"**RL Agent BEATS the best runtime baseline ({best_runtime_name})** on net_pnl_usd (${rl_pnl:.2f} vs ${best_runtime_baseline_pnl:.2f}).")
            else:
                out.append(f"**RL Agent DOES NOT BEAT the best runtime baseline ({best_runtime_name})** on net_pnl_usd (${rl_pnl:.2f} vs ${best_runtime_baseline_pnl:.2f}).")
        if rl_pnl > best_baseline_pnl:
            out.append(f"**RL Agent BEATS the best research baseline ({best_name})** on net_pnl_usd (${rl_pnl:.2f} vs ${best_baseline_pnl:.2f}).")
        else:
            out.append(f"**RL Agent DOES NOT BEAT the best research baseline ({best_name})** on net_pnl_usd (${rl_pnl:.2f} vs ${best_baseline_pnl:.2f}).")

    print("\n".join(out))
    
    md_out_path = Path("models") / f"baseline_comparison_{str(args.symbol).lower()}.md"
    md_out_path.write_text("\n".join(out), encoding="utf-8")
    print(f"\nReport written to {md_out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
