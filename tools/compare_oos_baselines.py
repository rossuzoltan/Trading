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

from artifact_manifest import load_manifest
from dataset_validation import validate_symbol_bar_spec
from edge_research import run_edge_baseline_research
from evaluate_oos import _resolve_execution_cost_profile
from feature_engine import FEATURE_COLS, WARMUP_BARS, _compute_raw
from project_paths import resolve_dataset_path, resolve_manifest_path, validate_dataset_bar_spec
from validation_metrics import load_json_report, save_json_report


def _load_symbol_frame(*, symbol: str, manifest: Any) -> pd.DataFrame:
    dataset_path = resolve_dataset_path()
    manifest_ticks = manifest.bar_construction_ticks_per_bar or manifest.ticks_per_bar
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
    if len(raw) <= WARMUP_BARS + 10:
        raise RuntimeError(f"Not enough bars for {symbol.upper()}: {len(raw)}")
    return raw


def _split_trainable_holdout(frame: pd.DataFrame, manifest: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    if manifest.holdout_start_utc:
        holdout_start = pd.Timestamp(manifest.holdout_start_utc)
        trainable = frame.loc[frame.index < holdout_start].copy()
        holdout = frame.loc[holdout_start:].copy()
    else:
        split_idx = int(len(frame) * 0.85)
        trainable = frame.iloc[:split_idx].copy()
        holdout = frame.iloc[split_idx:].copy()
    if trainable.empty or holdout.empty:
        raise RuntimeError("Holdout split produced an empty trainable or holdout frame.")
    return trainable, holdout


def _build_folds(trainable_frame: pd.DataFrame, *, validation_frac: float) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    validation_size = max(int(len(trainable_frame) * float(validation_frac)), 1)
    validation_start = max(len(trainable_frame) - validation_size, 1)
    train_fold = trainable_frame.iloc[:validation_start].copy()
    val_fold = trainable_frame.iloc[validation_start:].copy()
    if train_fold.empty or val_fold.empty:
        raise RuntimeError("Validation split produced an empty training or validation fold.")
    return [(train_fold, val_fold)]


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


def build_baseline_comparison(
    *,
    symbol: str,
    report_path: Path | None = None,
    horizon_bars: int = 10,
    validation_frac: float = 0.15,
    min_edge_pips: float = 0.0,
) -> dict[str, Any]:
    manifest = load_manifest(resolve_manifest_path(symbol=symbol))
    cost_profile = _resolve_execution_cost_profile(manifest)
    raw_frame = _load_symbol_frame(symbol=symbol, manifest=manifest)
    featured = _compute_raw(raw_frame)
    featured = featured.dropna(subset=list(FEATURE_COLS))
    trainable_frame, holdout_frame = _split_trainable_holdout(featured, manifest)
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
    replay_metrics = dict((replay_report or {}).get("replay_metrics", {}) or {})
    holdout_models = dict((baseline_report.get("holdout_metrics", {}) or {}).get("models", {}) or {})
    best_baseline = _best_model_name(holdout_models)
    best_baseline_metrics = dict((holdout_models.get(best_baseline, {}) or {}).get("metrics", {}) or {}) if best_baseline else {}

    comparison = {
        "symbol": symbol.upper(),
        "cost_profile": cost_profile,
        "target_definition": dict(baseline_report.get("target_definition", {}) or {}),
        "replay_report_path": str(replay_report_path) if replay_report_path.exists() else None,
        "baseline_report_path": str(baseline_out_path),
        "rl_replay_metrics": replay_metrics or None,
        "baseline_holdout_models": holdout_models,
        "best_baseline": best_baseline,
        "best_baseline_metrics": best_baseline_metrics or None,
        "comparison": {
            "rl_trade_count": replay_metrics.get("trade_count"),
            "rl_net_pnl_usd": replay_metrics.get("net_pnl_usd"),
            "rl_profit_factor": replay_metrics.get("profit_factor"),
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

    models = report.get("baseline_holdout_models", {})
    for name, data in models.items():
        out.append(_row(f"Baseline: {name}", data.get("metrics")))
    
    out.append("")
    out.append("## Verdict")
    out.append("")
    
    rl_pnl = float(rl_metrics.get("net_pnl_usd", 0.0)) if rl_metrics else 0.0
    best_baseline_pnl = float((report.get("best_baseline_metrics") or {}).get("net_pnl_usd", 0.0))
    best_name = report.get("best_baseline", "None")
    
    if not rl_metrics:
        out.append("RL Replay metrics missing. Verdict: CANNOT COMPARE.")
    elif rl_pnl > best_baseline_pnl:
        out.append(f"**RL Agent BEATS the best simple baseline ({best_name})** on net_pnl_usd (${rl_pnl:.2f} vs ${best_baseline_pnl:.2f}).")
    else:
        out.append(f"**RL Agent DOES NOT BEAT the best simple baseline ({best_name})** on net_pnl_usd (${rl_pnl:.2f} vs ${best_baseline_pnl:.2f}).")

    print("\n".join(out))
    
    md_out_path = Path("models") / f"baseline_comparison_{str(args.symbol).lower()}.md"
    md_out_path.write_text("\n".join(out), encoding="utf-8")
    print(f"\nReport written to {md_out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
