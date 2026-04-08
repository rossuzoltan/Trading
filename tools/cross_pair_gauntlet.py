"""
Experiment G: Cross-Pair Raw Rule Gauntlet
-------------------------------------------
Evaluates the raw `runtime_mean_reversion` rule across EURUSD, GBPUSD, and USDJPY.
Horizon: 10,000 ticks/bar.
Scoreboard: flat, always_long, always_short, trend, mean_reversion.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_oos import (
    _evaluate_policy,
    _evaluate_runtime_baselines,
    _load_symbol_raw_frame,
    ReplayContext,
    WARMUP_BARS,
    HOLDOUT_FRAC,
    EVAL_MAX_BARS,
)
from feature_engine import FEATURE_COLS, _compute_raw
from project_paths import resolve_dataset_path
from run_logging import configure_run_logging
from runtime_common import ActionSpec, ActionType
from validation_metrics import save_json_report

log = logging.getLogger("cross_pair_gauntlet")

# Standard sparse action map for the gauntlet
GAUNTLET_ACTION_MAP = tuple([
    ActionSpec(ActionType.HOLD),
    ActionSpec(ActionType.OPEN, direction=1, sl_value=1.5, tp_value=3.0),
    ActionSpec(ActionType.OPEN, direction=-1, sl_value=1.5, tp_value=3.0),
    ActionSpec(ActionType.CLOSE),
])

def create_raw_replay_context(symbol: str, ticks_per_bar: int) -> ReplayContext:
    dataset_path = resolve_dataset_path()
    raw = _load_symbol_raw_frame(
        symbol=symbol,
        dataset_path=dataset_path,
        expected_ticks_per_bar=ticks_per_bar
    )
    featured = _compute_raw(raw).dropna(subset=list(FEATURE_COLS))

    # Standard 15% holdout split if no manifest provided
    raw_split_idx = int(len(raw) * float(HOLDOUT_FRAC))
    split_pos = max(len(raw) - raw_split_idx, 1)
    split_ts = raw.index[split_pos]
    warmup_start = max(0, split_pos - max(WARMUP_BARS * 3, 300))
    warmup_frame = raw.iloc[warmup_start:split_pos].copy()
    replay_frame = raw.iloc[split_pos:].copy()
    holdout_feature_frame = featured.loc[featured.index >= split_ts].copy()
    trainable_feature_frame = featured.loc[featured.index < split_ts].copy()

    if EVAL_MAX_BARS and int(EVAL_MAX_BARS) > 0:
        max_bars = int(EVAL_MAX_BARS)
        if len(replay_frame) > max_bars:
            replay_frame = replay_frame.iloc[-max_bars:].copy()
        if len(holdout_feature_frame) > max_bars:
            holdout_feature_frame = holdout_feature_frame.iloc[-max_bars:].copy()

    return ReplayContext(
        symbol=symbol.upper(),
        source="gauntlet_script",
        dataset_path=dataset_path,
        action_map=GAUNTLET_ACTION_MAP,
        model=None,
        obs_normalizer=None,
        scaler=None,
        execution_cost_profile={
            "commission_per_lot": 7.0,
            "slippage_pips": 0.25
        },
        reward_profile={},
        warmup_frame=warmup_frame,
        replay_frame=replay_frame,
        replay_feature_frame=holdout_feature_frame,
        full_feature_frame=featured,
        trainable_feature_frame=trainable_feature_frame,
        holdout_feature_frame=holdout_feature_frame,
        holdout_start_utc=pd.Timestamp(holdout_feature_frame.index[0]).isoformat(),
        diagnostics_path=None,
        manifest_path=None,
        artifact_metadata={},
        runtime_options={"window_size": 1},
    )

def _sanitize_payload(obj):
    if isinstance(obj, dict):
        return {str(k): _sanitize_payload(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_payload(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks-per-bar", type=int, default=10000)
    args = parser.parse_args()

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    configure_run_logging("cross_pair_gauntlet", capture_print=False)

    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    ticks_per_bar = args.ticks_per_bar
    
    gauntlet_results = {}

    for symbol in symbols:
        log.info(f"--- Evaluating {symbol} ({ticks_per_bar} ticks/bar) ---")
        try:
            context = create_raw_replay_context(symbol, ticks_per_bar)
            log.info(f"  Holdout Bars: {len(context.replay_frame)}")
            
            baselines = _evaluate_runtime_baselines(replay_context=context)
            gauntlet_results[symbol] = baselines
            
            mr_net = baselines["runtime_mean_reversion"]["metrics"]["net_pnl_usd"]
            log.info(f"  {symbol} Mean Reversion Net: ${mr_net:.2f}")
            
        except Exception as e:
            log.exception(f"Failed to evaluate {symbol}: {e}")

    # Generate Scoreboard Markdown
    lines = [f"# Experiment G: Cross-Pair Raw Rule Gauntlet ({ticks_per_bar} ticks/bar)"]
    lines.append("\nScoreboard side-by-side comparison for Mean Reversion vs Anchors.\n")

    for symbol in symbols:
        if symbol not in gauntlet_results:
            continue
            
        res = gauntlet_results[symbol]
        lines.append(f"## {symbol} Performance")
        lines.append("| Baseline | Trades | Net USD | Gross Profit | Gross Loss | Costs | PF | Expct | WinRate |")
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        
        for name in ["runtime_flat", "runtime_always_long", "runtime_always_short", "runtime_trend", "runtime_mean_reversion"]:
            m = res[name]["metrics"]
            trades = int(m.get("trade_count", 0))
            net = float(m.get("net_pnl_usd", 0.0))
            gross_p = float(m.get("gross_profit_usd", 0.0))
            gross_l = float(m.get("gross_loss_usd", 0.0))
            costs = float(m.get("total_cost_usd", 0.0))
            pf = float(m.get("profit_factor", 0.0))
            expct = float(m.get("expectancy_usd", 0.0))
            wr = float(m.get("win_rate", 0.0))
            
            net_str = f"**${net:.2f}**" if net > 0 else f"${net:.2f}"
            pf_str = f"**{pf:.2f}**" if pf > 1.0 else f"{pf:.2f}"
            
            lines.append(f"| {name} | {trades} | {net_str} | ${gross_p:.2f} | ${gross_l:.2f} | ${costs:.2f} | {pf_str} | ${expct:.2f} | {wr:.1%} |")
        lines.append("")

    summary_path = Path("artifacts") / f"cross_pair_gauntlet_results_{ticks_per_bar}.md"
    summary_path.parent.mkdir(exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    
    # Save full JSON report
    report_path = Path("models") / f"gauntlet_report_{ticks_per_bar}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_json_report(_sanitize_payload(gauntlet_results), report_path)
    
    log.info(f"Gauntlet complete. Results saved to {summary_path}")

if __name__ == "__main__":
    main()
