from __future__ import annotations

import logging
import os
import math
from pathlib import Path

from interpreter_guard import ensure_project_venv

ensure_project_venv(project_root=Path(__file__).resolve().parent, script_path=__file__)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from artifact_manifest import (
    dataset_id_for_path,
    load_manifest,
    load_validated_model,
    load_validated_scaler,
    load_validated_vecnormalize,
)
from dataset_validation import validate_symbol_bar_spec
from event_pipeline import (
    JsonStateStore,
    ModelPolicy,
    ReplayBroker,
    RiskEngine,
    RiskLimits,
    RuntimeEngine,
    RuntimeSnapshot,
    VolumeBar,
)
from feature_engine import FEATURE_COLS, FeatureEngine, WARMUP_BARS
from project_paths import ensure_runtime_dirs, resolve_dataset_path, resolve_manifest_path, validate_dataset_bar_spec
from run_logging import configure_run_logging, set_log_context
from runtime_common import (
    STATE_FEATURE_COUNT,
    compute_max_drawdown,
    compute_timed_sharpe,
    compute_trade_metrics,
    deserialize_action_map,
)
from runtime_gym_env import TrainingDiagnostics
from trading_config import deployment_paths
from validation_metrics import build_deployment_gate, load_json_report, save_json_report


TARGET_SYM = os.environ.get("EVAL_SYMBOL", "EURUSD").strip().upper() or "EURUSD"
log = logging.getLogger("evaluate_oos")


def _resolve_execution_cost_profile(manifest) -> dict[str, float]:
    profile = dict(getattr(manifest, "execution_cost_profile", None) or {})
    return {
        "commission_per_lot": float(profile.get("commission_per_lot", 7.0)),
        "slippage_pips": float(profile.get("slippage_pips", 0.25)),
        "partial_fill_ratio": float(profile.get("partial_fill_ratio", 1.0)),
    }


def _resolve_reward_profile(manifest) -> dict[str, float]:
    profile = dict(getattr(manifest, "reward_profile", None) or {})
    return {
        "reward_scale": float(profile.get("reward_scale", 10_000.0)),
        "drawdown_penalty": float(profile.get("drawdown_penalty", 2.0)),
        "transaction_penalty": float(profile.get("transaction_penalty", 1.0)),
        "reward_clip_low": float(profile.get("reward_clip_low", -5.0)),
        "reward_clip_high": float(profile.get("reward_clip_high", 5.0)),
    }


def _frame_to_bars(frame: pd.DataFrame) -> list[VolumeBar]:
    bars: list[VolumeBar] = []
    for timestamp, row in frame.iterrows():
        time_delta_s = float(row.get("time_delta_s", 0.0))
        end_time_msc = int(pd.Timestamp(timestamp).timestamp() * 1000)
        bars.append(
            VolumeBar(
                timestamp=pd.Timestamp(timestamp).to_pydatetime(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
                avg_spread=float(row.get("avg_spread", 0.0)),
                time_delta_s=time_delta_s,
                start_time_msc=end_time_msc,
                end_time_msc=end_time_msc + int(max(time_delta_s, 0.0) * 1000),
            )
        )
    return bars


def run_replay() -> tuple[list[float], list[pd.Timestamp], list[dict], list[dict], dict[str, object]]:
    dataset_path = resolve_dataset_path()
    manifest_path = resolve_manifest_path(symbol=TARGET_SYM)
    manifest = load_manifest(manifest_path)
    manifest_ticks = manifest.bar_construction_ticks_per_bar or manifest.ticks_per_bar
    execution_cost_profile = _resolve_execution_cost_profile(manifest)
    reward_profile = _resolve_reward_profile(manifest)
    if manifest_ticks is not None:
        validate_dataset_bar_spec(
            dataset_path=dataset_path,
            expected_ticks_per_bar=int(manifest_ticks),
            metadata_required=True,
        )
    dataset_id = dataset_id_for_path(dataset_path)

    action_map = deserialize_action_map(manifest.action_map)
    observation_shape = [1, len(FEATURE_COLS) + STATE_FEATURE_COUNT]
    model = load_validated_model(
        manifest,
        expected_symbol=TARGET_SYM,
        expected_action_map=action_map,
        expected_observation_shape=observation_shape,
        expected_dataset_id=dataset_id,
    )
    obs_normalizer = load_validated_vecnormalize(
        manifest,
        expected_symbol=TARGET_SYM,
        expected_action_map=action_map,
        expected_observation_shape=observation_shape,
        expected_dataset_id=dataset_id,
    )
    scaler = load_validated_scaler(
        manifest,
        expected_symbol=TARGET_SYM,
        expected_action_map=action_map,
        expected_observation_shape=observation_shape,
        expected_dataset_id=dataset_id,
    )

    raw = pd.read_csv(dataset_path, low_memory=False, parse_dates=["Gmt time"])
    raw = raw[raw["Symbol"].astype(str).str.upper() == TARGET_SYM].copy()
    raw["Gmt time"] = pd.to_datetime(raw["Gmt time"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["Gmt time"]).set_index("Gmt time").sort_index()
    if manifest_ticks is not None:
        validate_symbol_bar_spec(
            raw.reset_index(),
            expected_ticks_per_bar=int(manifest_ticks),
            symbol=TARGET_SYM,
        )
    if len(raw) <= WARMUP_BARS + 10:
        raise RuntimeError(f"Not enough raw bars for {TARGET_SYM}: {len(raw)}")

    if manifest.holdout_start_utc:
        holdout_start = pd.Timestamp(manifest.holdout_start_utc)
        replay_frame = raw.loc[holdout_start:].copy()
        train_frame = raw.loc[raw.index < holdout_start].copy()
        warmup_frame = train_frame.iloc[-max(WARMUP_BARS * 3, 300) :].copy()
    else:
        split_idx = int(len(raw) * 0.85)
        warmup_start = max(0, split_idx - max(WARMUP_BARS * 3, 300))
        warmup_frame = raw.iloc[warmup_start:split_idx].copy()
        replay_frame = raw.iloc[split_idx:].copy()
    if replay_frame.empty or warmup_frame.empty:
        raise RuntimeError(f"Holdout split produced empty replay or warmup frame for {TARGET_SYM}.")

    feature_engine = FeatureEngine.from_scaler(scaler)
    feature_engine.warm_up(warmup_frame)
    broker = ReplayBroker(symbol=TARGET_SYM, **execution_cost_profile)
    snapshot = RuntimeSnapshot(last_equity=1_000.0, high_water_mark=1_000.0, day_start_equity=1_000.0)
    risk_engine = RiskEngine(RiskLimits(), snapshot=snapshot, initial_equity=1_000.0)
    runtime = RuntimeEngine(
        symbol=TARGET_SYM,
        feature_engine=feature_engine,
        policy=ModelPolicy(model, action_map, obs_normalizer=obs_normalizer),
        broker=broker,
        action_map=action_map,
        risk_engine=risk_engine,
        snapshot=snapshot,
        state_store=None,
        **reward_profile,
    )
    runtime.startup_reconcile()

    equity_curve: list[float] = []
    timestamps: list[pd.Timestamp] = []
    diagnostics = TrainingDiagnostics()
    replay_bars = _frame_to_bars(replay_frame)
    for bar_index, bar in enumerate(replay_bars):
        prev_position = int(runtime.confirmed_position.direction)
        prev_position_duration = int(runtime.confirmed_position.time_in_trade_bars)
        trade_log_before = len(getattr(broker, "trade_log", []))
        execution_log_before = len(getattr(broker, "execution_log", []))
        prev_equity = float(runtime.last_equity)
        result = runtime.process_bar(bar)
        reward_components = dict(result.reward_components)
        reward_components.setdefault("holding_penalty_applied", 0.0)
        reward_components.setdefault("participation_bonus_applied", 0.0)
        reward_components["pnl_reward"] = float(
            reward_components.get("reward_raw_unclipped", reward_components.get("pnl_reward", 0.0))
        )
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
                reward_components["pnl_reward"] = float(
                    reward_components.get("reward_raw_unclipped", reward_components.get("pnl_reward", 0.0))
                )
                reward_components["holding_penalty_applied"] = 0.0
                reward_components["participation_bonus_applied"] = 0.0
                reward_components["forced_close_applied"] = 1.0
                result.equity = float(flatten_result.get("equity", result.equity))
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


def main() -> None:
    ensure_runtime_dirs()
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
        },
    )
    equity_curve, timestamps, trade_log, execution_log, diagnostics = run_replay()
    final_equity = equity_curve[-1] if equity_curve else 1_000.0
    timed_sharpe = compute_timed_sharpe(equity_curve, timestamps)
    max_dd = compute_max_drawdown(equity_curve)
    trade_metrics = compute_trade_metrics(trade_log, initial_equity=1_000.0)
    win_rate = float(trade_metrics["win_rate"])
    profit_factor = float(trade_metrics["profit_factor"])
    expectancy = float(trade_metrics["expectancy_usd"])
    expectancy_pips = float(trade_metrics["expectancy_pips"])
    n_trades = int(trade_metrics["trade_count"])
    avg_holding_bars = float((diagnostics.get("trade_stats", {}) or {}).get("position_duration_sum", 0.0)) / float(
        max(int((diagnostics.get("trade_stats", {}) or {}).get("position_duration_count", 0)), 1)
    )
    total_return = (final_equity - 1_000.0) / 1_000.0
    report_paths = deployment_paths(TARGET_SYM)
    training_diagnostics = None
    manifest_path = resolve_manifest_path(symbol=TARGET_SYM)
    manifest = load_manifest(manifest_path)
    diagnostics_path = Path(manifest.training_diagnostics_path) if manifest.training_diagnostics_path else report_paths.diagnostics_path
    if diagnostics_path.exists():
        training_diagnostics = load_json_report(diagnostics_path)

    print("=" * 60)
    print("Replay OOS Evaluation")
    print("=" * 60)
    print(f"Initial equity : $1,000.00")
    print(f"Final equity   : ${final_equity:,.2f}")
    print(f"Total return   : {total_return:.1%}")
    print(f"Timed Sharpe   : {timed_sharpe:.3f}")
    print(f"Max drawdown   : {max_dd:.1%}")
    print(f"Win rate       : {win_rate:.1%}")
    print(f"Profit factor  : {profit_factor:.3f}")
    print(f"Expectancy     : ${expectancy:,.2f}/trade ({expectancy_pips:.3f} pips/trade)")
    print(f"Total trades   : {n_trades}")
    print(f"Gross PnL      : ${float(trade_metrics['gross_pnl_usd']):,.2f}")
    print(f"Net PnL        : ${float(trade_metrics['net_pnl_usd']):,.2f}")
    print(f"Txn costs paid : ${float(trade_metrics['total_transaction_cost_usd']):,.2f}")
    print(f"Avg hold bars  : {avg_holding_bars:.2f}")
    print(f"Orders exec'd  : {int((diagnostics.get('trade_stats', {}) or {}).get('order_executed_count', 0))}")
    print(f"Forced closes  : {int((diagnostics.get('trade_stats', {}) or {}).get('forced_close_count', 0))}")
    print("=" * 60)

    replay_metrics = {
        "final_equity": float(final_equity),
        "total_return": float(total_return),
        "timed_sharpe": float(timed_sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "expectancy": float(expectancy),
        "expectancy_usd": float(expectancy),
        "expectancy_pips": float(expectancy_pips),
        "trade_count": int(n_trades),
        "gross_pnl_usd": float(trade_metrics["gross_pnl_usd"]),
        "net_pnl_usd": float(trade_metrics["net_pnl_usd"]),
        "total_transaction_cost_usd": float(trade_metrics["total_transaction_cost_usd"]),
        "total_commission_usd": float(trade_metrics["total_commission_usd"]),
        "total_spread_slippage_cost_usd": float(trade_metrics["total_spread_slippage_cost_usd"]),
        "total_spread_cost_usd": float(trade_metrics["total_spread_cost_usd"]),
        "total_slippage_cost_usd": float(trade_metrics["total_slippage_cost_usd"]),
        "avg_holding_bars": float(avg_holding_bars),
        "avg_win_usd": float(trade_metrics["avg_win_usd"]),
        "avg_loss_usd": float(trade_metrics["avg_loss_usd"]),
        "win_loss_asymmetry": float(trade_metrics["win_loss_asymmetry"]),
        "forced_close_count": int((diagnostics.get("trade_stats", {}) or {}).get("forced_close_count", 0)),
        "executed_order_count": int((diagnostics.get("trade_stats", {}) or {}).get("order_executed_count", 0)),
        "action_distribution": dict(diagnostics.get("action_counts", {}) or {}),
        "trade_diagnostics": dict(diagnostics.get("trade_stats", {}) or {}),
        "economics": dict(diagnostics.get("economics", {}) or {}),
        "economic_metrics_primary": True,
        "reward_shaping_excluded_from_primary_metrics": True,
    }
    replay_report = {
        "symbol": TARGET_SYM,
        "replay_metrics": replay_metrics,
        "trade_metrics": trade_metrics,
        "diagnostics": diagnostics,
        "execution_log_count": int(len(execution_log)),
        "trade_log_count": int(len(trade_log)),
        "reward_shaping_in_eval": False,
    }
    replay_report_path = Path(f"models/replay_report_{TARGET_SYM.lower()}.json")
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

    out_path = Path(f"models/equity_curve_oos_{TARGET_SYM.lower()}.png")
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
