from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from edge_research import run_edge_baseline_research as _run_edge_baseline_research
from feature_engine import FEATURE_COLS, _compute_raw


@dataclass(frozen=True)
class EdgeResearchConfig:
    symbol: str = "EURUSD"
    dataset_path: str | Path | None = None
    report_path: str | Path | None = None
    horizon_bars: int = 10
    validation_frac: float = 0.15
    holdout_frac: float = 0.15
    min_edge_pips: float = 0.0
    commission_per_lot: float = 7.0
    slippage_pips: float = 0.25
    ticks_per_bar: int | None = None


def _load_symbol_frame(config: EdgeResearchConfig) -> pd.DataFrame:
    if config.dataset_path is None:
        raise RuntimeError("EdgeResearchConfig.dataset_path is required.")
    frame = pd.read_csv(config.dataset_path, low_memory=False, parse_dates=["Gmt time"])
    frame = frame.loc[frame["Symbol"].astype(str).str.upper() == config.symbol.upper()].copy()
    frame["Gmt time"] = pd.to_datetime(frame["Gmt time"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["Gmt time"]).set_index("Gmt time").sort_index()
    return _compute_raw(frame)


def run_edge_research(config: EdgeResearchConfig) -> dict[str, Any]:
    frame = _load_symbol_frame(config)
    holdout_size = max(int(len(frame) * float(config.holdout_frac)), 1)
    validation_size = max(int(len(frame) * float(config.validation_frac)), 1)
    holdout_start = max(len(frame) - holdout_size, 1)
    trainable = frame.iloc[:holdout_start].copy()
    holdout = frame.iloc[holdout_start:].copy()
    validation_start = max(len(trainable) - validation_size, 1)
    folds = [
        (trainable.iloc[:validation_start].copy(), trainable.iloc[validation_start:].copy()),
    ]
    out_path = Path(config.report_path) if config.report_path is not None else Path("models") / f"edge_research_{config.symbol.lower()}.json"
    baseline_report = _run_edge_baseline_research(
        symbol=config.symbol,
        trainable_frame=trainable,
        holdout_frame=holdout,
        folds=folds,
        feature_cols=list(FEATURE_COLS),
        out_path=out_path,
        horizon_bars=int(config.horizon_bars),
        commission_per_lot=float(config.commission_per_lot),
        slippage_pips=float(config.slippage_pips),
        min_edge_pips=float(config.min_edge_pips),
        probability_threshold=0.55,
        probability_margin=0.05,
        min_trade_count=20,
    )
    holdout_models = dict((baseline_report.get("holdout_metrics", {}) or {}).get("models", {}))
    best_holdout_baseline = None
    if holdout_models:
        best_holdout_baseline = max(
            holdout_models.items(),
            key=lambda item: (
                float(((item[1] or {}).get("metrics", {}) or {}).get("expectancy_usd", 0.0)),
                float(((item[1] or {}).get("metrics", {}) or {}).get("profit_factor", 0.0)),
                float(((item[1] or {}).get("metrics", {}) or {}).get("trade_count", 0.0)),
            ),
        )[0]
    return {
        "symbol": str(config.symbol).upper(),
        "config": asdict(config),
        "target_definition": dict(baseline_report.get("target_definition", {}) or {}),
        "baselines": holdout_models,
        "fold_metrics": list(baseline_report.get("fold_metrics", []) or []),
        "passing_models": list(baseline_report.get("passing_models", []) or []),
        "best_holdout_baseline": best_holdout_baseline,
        "edge_found": bool(baseline_report.get("gate_passed", False)),
        "gate_passed": bool(baseline_report.get("gate_passed", False)),
        "report_path": str(out_path),
    }


def run_edge_baseline_research(
    *,
    symbol: str,
    trainable_frame,
    holdout_frame,
    folds,
    feature_cols,
    out_path,
    horizon_bars: int,
    commission_per_lot: float,
    slippage_pips: float,
    min_edge_pips: float,
    probability_threshold: float = 0.55,
    probability_margin: float = 0.05,
    min_trade_count: int = 20,
) -> dict[str, Any]:
    return _run_edge_baseline_research(
        symbol=symbol,
        trainable_frame=trainable_frame,
        holdout_frame=holdout_frame,
        folds=folds,
        feature_cols=list(feature_cols),
        out_path=out_path,
        horizon_bars=int(horizon_bars),
        commission_per_lot=float(commission_per_lot),
        slippage_pips=float(slippage_pips),
        min_edge_pips=float(min_edge_pips),
        probability_threshold=float(probability_threshold),
        probability_margin=float(probability_margin),
        min_trade_count=int(min_trade_count),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run cost-adjusted non-RL edge research baselines.")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--horizon-bars", type=int, default=10)
    parser.add_argument("--validation-frac", type=float, default=0.15)
    parser.add_argument("--holdout-frac", type=float, default=0.15)
    parser.add_argument("--min-edge-pips", type=float, default=0.0)
    parser.add_argument("--commission-per-lot", type=float, default=7.0)
    parser.add_argument("--slippage-pips", type=float, default=0.25)
    parser.add_argument("--ticks-per-bar", type=int, default=None)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    report = run_edge_research(
        EdgeResearchConfig(
            symbol=str(args.symbol).upper(),
            dataset_path=args.dataset_path,
            report_path=args.report_path,
            horizon_bars=int(args.horizon_bars),
            validation_frac=float(args.validation_frac),
            holdout_frac=float(args.holdout_frac),
            min_edge_pips=float(args.min_edge_pips),
            commission_per_lot=float(args.commission_per_lot),
            slippage_pips=float(args.slippage_pips),
            ticks_per_bar=args.ticks_per_bar,
        )
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
