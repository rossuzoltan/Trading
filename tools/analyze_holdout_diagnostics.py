from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifact_manifest import load_manifest
from dataset_validation import validate_symbol_bar_spec
from feature_engine import FEATURE_COLS, WARMUP_BARS, _compute_raw
from project_paths import resolve_dataset_path, resolve_manifest_path, validate_dataset_bar_spec
from train_agent import (
    ACTION_SL_MULTS,
    ACTION_TP_MULTS,
    FOLD_TEST_FRAC,
    N_FOLDS,
    evaluate_model,
    get_final_slippage_pips,
    make_env,
    purged_walk_forward_splits,
    TRAINING_RECOVERY_CONFIG,
    _extract_eval_completed_episode_audit,
)
from training_status import build_status_summary
from validation_metrics import save_json_report


def _maybe_load_manifest(symbol: str) -> Any | None:
    manifest_path = resolve_manifest_path(symbol=symbol)
    if not manifest_path.exists():
        return None
    return load_manifest(manifest_path)


def _load_symbol_frame(symbol: str, manifest: Any | None) -> pd.DataFrame:
    dataset_path = resolve_dataset_path()
    manifest_ticks = None
    if manifest is not None:
        manifest_ticks = getattr(manifest, "bar_construction_ticks_per_bar", None) or getattr(manifest, "ticks_per_bar", None)
    if manifest_ticks is not None:
        validate_dataset_bar_spec(
            dataset_path=dataset_path,
            expected_ticks_per_bar=int(manifest_ticks),
            metadata_required=True,
        )
    raw = pd.read_csv(dataset_path, low_memory=False, parse_dates=["Gmt time"])
    raw = raw.loc[raw["Symbol"].astype(str).str.upper() == symbol.upper()].copy()
    raw["Gmt time"] = pd.to_datetime(raw["Gmt time"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["Gmt time"]).set_index("Gmt time").sort_index()
    if manifest_ticks is not None:
        validate_symbol_bar_spec(raw.reset_index(), expected_ticks_per_bar=int(manifest_ticks), symbol=symbol.upper())
    return _compute_raw(raw).dropna(subset=list(FEATURE_COLS)).copy()


def _split_trainable_holdout(frame: pd.DataFrame, manifest: Any | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout_start_utc = getattr(manifest, "holdout_start_utc", None) if manifest is not None else None
    if holdout_start_utc:
        holdout_start = pd.Timestamp(holdout_start_utc)
        trainable = frame.loc[frame.index < holdout_start].copy()
        holdout = frame.loc[holdout_start:].copy()
    else:
        holdout_size = max(500, int(len(frame) * 0.15))
        split_idx = len(frame) - holdout_size
        trainable = frame.iloc[:split_idx].copy()
        holdout = frame.iloc[split_idx:].copy()
    if trainable.empty or holdout.empty:
        raise RuntimeError("Holdout split produced an empty trainable or holdout frame.")
    return trainable, holdout


def _resolve_frame_position(frame: pd.DataFrame, start_index: pd.Timestamp) -> int:
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


def _resolve_checkpoint_paths(symbol: str, fold_index: int | None) -> tuple[Path, Path | None, int]:
    summary = build_status_summary(symbol, Path("checkpoints"))
    current_run = summary.get("current_run") or {}
    checkpoints_root = Path(str(current_run.get("checkpoints_root"))) if current_run.get("checkpoints_root") else Path("checkpoints")
    if fold_index is None and current_run.get("fold_index") is not None:
        fold_index = int(current_run["fold_index"])
    if fold_index is not None:
        fold_dir = checkpoints_root / f"fold_{int(fold_index)}"
        best_model_path = fold_dir / "best_model.zip"
        if not best_model_path.exists():
            raise RuntimeError(f"Missing best model checkpoint: {best_model_path}")
        vecnormalize_path = fold_dir / "best_vecnormalize.pkl"
        return best_model_path, vecnormalize_path if vecnormalize_path.exists() else None, int(fold_index)

    candidates = list(checkpoints_root.glob("fold_*/best_model.zip"))
    candidates.extend(Path("checkpoints").glob("run_*/fold_*/best_model.zip"))
    if not candidates:
        raise RuntimeError("No best_model.zip checkpoint found.")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    best_model_path = candidates[0]
    vecnormalize_path = best_model_path.parent / "best_vecnormalize.pkl"
    return best_model_path, vecnormalize_path if vecnormalize_path.exists() else None, int(best_model_path.parent.name.split("_")[-1])


def _evaluate_holdout_checkpoint(symbol: str, fold_index: int | None) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _maybe_load_manifest(symbol)
    full_frame = _load_symbol_frame(symbol, manifest)
    trainable_frame, holdout_frame = _split_trainable_holdout(full_frame, manifest)
    folds = purged_walk_forward_splits(trainable_frame, n_folds=N_FOLDS, test_frac=FOLD_TEST_FRAC)
    best_model_path, best_vecnormalize_path, resolved_fold_index = _resolve_checkpoint_paths(symbol, fold_index)
    if resolved_fold_index >= len(folds):
        raise RuntimeError(f"Fold index {resolved_fold_index} out of range for {len(folds)} available folds.")
    train_fold, _val_fold = folds[resolved_fold_index]
    scaler = StandardScaler()
    scaler.fit(train_fold.loc[:, FEATURE_COLS])
    holdout_source = _prepend_runtime_warmup_context(full_frame, holdout_frame)
    slippage_pips = float(get_final_slippage_pips(TRAINING_RECOVERY_CONFIG))

    base_vec = DummyVecEnv(
        [
            make_env(
                holdout_source,
                list(FEATURE_COLS),
                list(ACTION_SL_MULTS),
                list(ACTION_TP_MULTS),
                random_start=False,
                initial_slippage=slippage_pips,
                symbol=symbol,
                scaler=scaler,
                recovery_config=None,
            )
        ]
    )
    if best_vecnormalize_path is not None:
        holdout_vec = VecNormalize.load(str(best_vecnormalize_path), base_vec)
        holdout_vec.training = False
        holdout_vec.norm_reward = False
    else:
        holdout_vec = VecNormalize(base_vec, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0, clip_reward=10.0)
    model = MaskablePPO.load(str(best_model_path), device="cpu")
    try:
        _, metrics = evaluate_model(model, holdout_vec)
        audit = _extract_eval_completed_episode_audit(holdout_vec)
    finally:
        holdout_vec.close()
    if not audit:
        raise RuntimeError("Completed episode audit missing after holdout evaluation.")
    return metrics, audit


def _bucket_holding_bars(value: int) -> str:
    bars = int(value)
    if bars <= 2:
        return "01-02"
    if bars <= 5:
        return "03-05"
    if bars <= 10:
        return "06-10"
    if bars <= 20:
        return "11-20"
    return "21+"


def _sum_trade_field(trades: list[dict[str, Any]], field: str) -> float:
    return float(sum(float(trade.get(field, 0.0) or 0.0) for trade in trades))


def _group_trade_pnl(trades: list[dict[str, Any]], key_fn) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        key = str(key_fn(trade))
        buckets.setdefault(key, []).append(trade)
    summary: dict[str, dict[str, float]] = {}
    for key, bucket_trades in buckets.items():
        summary[key] = {
            "trade_count": float(len(bucket_trades)),
            "gross_pnl_usd": _sum_trade_field(bucket_trades, "gross_pnl_usd"),
            "net_pnl_usd": _sum_trade_field(bucket_trades, "net_pnl_usd"),
            "transaction_cost_usd": _sum_trade_field(bucket_trades, "transaction_cost_usd"),
        }
    return summary


def _top_loss_drivers(metrics: dict[str, Any], trade_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    long_trades = [trade for trade in trade_log if int(trade.get("direction", 0) or 0) > 0]
    short_trades = [trade for trade in trade_log if int(trade.get("direction", 0) or 0) < 0]
    forced_trades = [trade for trade in trade_log if bool(trade.get("forced_close", False))]
    candidates = [
        {
            "driver": "transaction_costs",
            "loss_usd": max(float(metrics.get("total_transaction_cost_usd", 0.0) or 0.0), 0.0),
        },
        {
            "driver": "negative_gross_signal",
            "loss_usd": max(-float(metrics.get("gross_pnl_usd", 0.0) or 0.0), 0.0),
        },
        {
            "driver": "forced_closes",
            "loss_usd": max(-_sum_trade_field(forced_trades, "net_pnl_usd"), 0.0),
        },
        {
            "driver": "long_book",
            "loss_usd": max(-_sum_trade_field(long_trades, "net_pnl_usd"), 0.0),
        },
        {
            "driver": "short_book",
            "loss_usd": max(-_sum_trade_field(short_trades, "net_pnl_usd"), 0.0),
        },
    ]
    ranked = [item for item in candidates if float(item["loss_usd"]) > 0.0]
    ranked.sort(key=lambda item: float(item["loss_usd"]), reverse=True)
    return ranked[:3]


def _build_markdown_report(symbol: str, payload: dict[str, Any]) -> str:
    metrics = dict(payload.get("metrics", {}) or {})
    top_drivers = list(payload.get("top_loss_drivers", []) or [])
    lines = [
        f"# Holdout Diagnostics - {symbol.upper()}",
        "",
        "## Core Metrics",
        "",
        f"- Final equity: {float(metrics.get('final_equity', 0.0)):.2f}",
        f"- Timed Sharpe: {float(metrics.get('timed_sharpe', 0.0)):.3f}",
        f"- Max drawdown: {float(metrics.get('max_drawdown', 0.0)):.3%}",
        f"- Trade count: {int(metrics.get('trade_count', 0) or 0)}",
        f"- Hold fraction: {float(metrics.get('hold_fraction', 0.0)):.3f}",
        f"- Trades per 1000 steps: {float(metrics.get('trades_per_1000_steps', 0.0)):.2f}",
        f"- Churn ratio: {float(metrics.get('churn_ratio', 0.0)):.3f}",
        "",
        "## Signal vs Costs",
        "",
        f"- Gross PnL before costs: {float(metrics.get('gross_pnl_usd', 0.0)):.2f}",
        f"- Total transaction cost: {float(metrics.get('total_transaction_cost_usd', 0.0)):.2f}",
        f"- Net PnL after costs: {float(metrics.get('net_pnl_usd', 0.0)):.2f}",
        f"- Cost per trade: {float(payload.get('cost_per_trade_usd', 0.0)):.2f}",
        f"- Cost per 1000 steps: {float(payload.get('cost_per_1000_steps_usd', 0.0)):.2f}",
        "",
        "## Top Loss Drivers",
        "",
    ]
    if not top_drivers:
        lines.append("- No positive loss drivers identified from the current trade log.")
    else:
        for driver in top_drivers:
            lines.append(f"- {driver['driver']}: {float(driver['loss_usd']):.2f} USD")
    lines.extend(
        [
            "",
            "## Holding Duration Buckets",
            "",
            "| bucket | trades | gross_pnl_usd | net_pnl_usd | transaction_cost_usd |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for bucket, bucket_metrics in dict(payload.get("pnl_by_holding_duration_bucket", {}) or {}).items():
        lines.append(
            f"| {bucket} | {int(bucket_metrics['trade_count'])} | {float(bucket_metrics['gross_pnl_usd']):.2f} | "
            f"{float(bucket_metrics['net_pnl_usd']):.2f} | {float(bucket_metrics['transaction_cost_usd']):.2f} |"
        )
    return "\n".join(lines) + "\n"


def build_holdout_diagnostics(symbol: str, fold_index: int | None = None, out_path: Path | None = None) -> dict[str, Any]:
    metrics, audit = _evaluate_holdout_checkpoint(symbol, fold_index)
    trade_log = [dict(item) for item in list(audit.get("trade_log", []) or []) if isinstance(item, dict)]
    long_trades = [trade for trade in trade_log if int(trade.get("direction", 0) or 0) > 0]
    short_trades = [trade for trade in trade_log if int(trade.get("direction", 0) or 0) < 0]
    forced_trades = [trade for trade in trade_log if bool(trade.get("forced_close", False))]
    cost_per_trade = (
        float(metrics.get("total_transaction_cost_usd", 0.0) or 0.0) / float(max(int(metrics.get("trade_count", 0) or 0), 1))
        if int(metrics.get("trade_count", 0) or 0) > 0
        else 0.0
    )
    cost_per_1000_steps = (
        1000.0 * float(metrics.get("total_transaction_cost_usd", 0.0) or 0.0) / float(max(int(metrics.get("steps", 0) or 0), 1))
    )
    payload = {
        "symbol": symbol.upper(),
        "metrics": metrics,
        "trade_log_count": int(len(trade_log)),
        "gross_alpha_before_costs_usd": float(metrics.get("gross_pnl_usd", 0.0) or 0.0),
        "cost_per_trade_usd": float(cost_per_trade),
        "cost_per_1000_steps_usd": float(cost_per_1000_steps),
        "pnl_before_after_costs": {
            "gross_pnl_usd": float(metrics.get("gross_pnl_usd", 0.0) or 0.0),
            "net_pnl_usd": float(metrics.get("net_pnl_usd", 0.0) or 0.0),
            "transaction_cost_usd": float(metrics.get("total_transaction_cost_usd", 0.0) or 0.0),
            "commission_usd": float(metrics.get("commission_usd", metrics.get("total_commission_usd", 0.0)) or 0.0),
            "spread_slippage_cost_usd": float(
                metrics.get("spread_slippage_cost_usd", metrics.get("total_spread_slippage_cost_usd", 0.0)) or 0.0
            ),
        },
        "pnl_by_holding_duration_bucket": _group_trade_pnl(
            trade_log,
            key_fn=lambda trade: _bucket_holding_bars(int(trade.get("holding_bars", 0) or 0)),
        ),
        "pnl_by_action_transition_bucket": _group_trade_pnl(
            trade_log,
            key_fn=lambda trade: f"{'long' if int(trade.get('direction', 0) or 0) > 0 else 'short'}:{str(trade.get('reason', 'unknown')).lower()}",
        ),
        "forced_close_contribution": {
            "trade_count": int(len(forced_trades)),
            "net_pnl_usd": _sum_trade_field(forced_trades, "net_pnl_usd"),
        },
        "long_vs_short_contribution": {
            "long_net_pnl_usd": _sum_trade_field(long_trades, "net_pnl_usd"),
            "short_net_pnl_usd": _sum_trade_field(short_trades, "net_pnl_usd"),
            "long_trade_count": int(len(long_trades)),
            "short_trade_count": int(len(short_trades)),
        },
        "top_loss_drivers": _top_loss_drivers(metrics, trade_log),
    }
    output_path = out_path or Path("models") / f"holdout_diagnostics_{symbol.lower()}.json"
    markdown_path = output_path.with_suffix(".md")
    save_json_report(payload, output_path)
    markdown_path.write_text(_build_markdown_report(symbol, payload), encoding="utf-8")
    payload["markdown_path"] = str(markdown_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-evaluate a holdout checkpoint and write detailed loss diagnostics.")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--fold-index", type=int, default=None)
    parser.add_argument("--out-path", default=None)
    args = parser.parse_args()

    report = build_holdout_diagnostics(
        symbol=str(args.symbol).upper(),
        fold_index=int(args.fold_index) if args.fold_index is not None else None,
        out_path=Path(args.out_path) if args.out_path else None,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
