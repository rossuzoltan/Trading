"""
optimize_rules.py — Automated Candidate Generator for Rule-First Strategies

1. Sweeps rule parameter grids across multiple rule families.
2. Runs exact-runtime replay evaluation on the TRAIN holdout set (or training set).
3. Applies strict stability constraints (PF, Expectancy, Win Rate, Direction cap).
4. Outputs a Markdown report ranking the best candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import multiprocessing
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge_research import fit_baseline_alpha_gate
from feature_engine import FEATURE_COLS
from evaluate_oos import (
    load_replay_context,
    run_replay,
    _rule_action_provider,
    _target_direction_to_action_index,
)

def _run_worker(base_ctx, config, alpha_gate):
    """Worker function for multiprocessing.

    Uses an explicit deep-isolated copy of the mutable DataFrame fields to
    prevent cross-worker state contamination in a multiprocessing Pool.
    dataclasses.replace() is a *shallow* copy — two workers sharing the same
    DataFrame object can race on index/column mutations inside _compute_raw.
    """
    try:
        from dataclasses import replace as dc_replace
        # Deep-isolate the DataFrame fields that are mutated during replay
        ctx = dc_replace(
            base_ctx,
            replay_frame=base_ctx.replay_frame.copy() if base_ctx.replay_frame is not None and not base_ctx.replay_frame.empty else base_ctx.replay_frame,
            trainable_feature_frame=base_ctx.trainable_feature_frame.copy() if base_ctx.trainable_feature_frame is not None and not base_ctx.trainable_feature_frame.empty else base_ctx.trainable_feature_frame,
            holdout_feature_frame=base_ctx.holdout_feature_frame.copy() if base_ctx.holdout_feature_frame is not None and not base_ctx.holdout_feature_frame.empty else base_ctx.holdout_feature_frame,
        )
        from tools.optimize_rules import _run_single_variant
        return _run_single_variant(ctx, config, alpha_gate=alpha_gate)
    except Exception as e:
        print(f"Worker Error evaluating {config}: {e}")
        return None

log = logging.getLogger("optimize_rules")

def _hash_file(filepath: Path) -> str:
    if not filepath.exists():
        return "Not_Found"
    return hashlib.sha256(filepath.read_bytes()).hexdigest()[:8]


# ── Configuration Grid ───────────────────────────────────────────────────────

def build_parameter_grid(*, include_regime_guard_variants: bool = False) -> list[dict[str, Any]]:
    candidates = []
    regime_profiles = [dict()]
    if include_regime_guard_variants:
        regime_profiles.append(
            {
                "min_vol_norm_atr": 0.00005,
                "max_abs_log_return": 0.0035,
                "max_abs_body_size": 3.0,
            }
        )

    # 1. mean_reversion (Standard & Aggressive)
    # Grid includes max_spread_z=0.5 which matches the current RC1 manifest value.
    # NOTE: 0.5 was previously missing from the search space — this is now corrected.
    for long_threshold in [-0.5, -0.75, -1.0, -1.25, -1.5, -1.75, -2.0]:
        for short_threshold in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
            for max_spread_z in [0.5, 0.75, 1.0, 1.25]:  # 0.5 added — was missing; matches RC1 manifest
                for max_abs_ma20_slope, max_abs_ma50_slope in [(0.15, 0.08), (0.20, 0.10), (0.25, 0.15), (0.5, 0.3)]:
                    for regime_profile in regime_profiles:
                        params = {
                            "long_threshold": long_threshold,
                            "short_threshold": short_threshold,
                            "max_spread_z": max_spread_z,
                            "max_time_delta_z": 2.5,
                            "max_abs_ma20_slope": max_abs_ma20_slope,
                            "max_abs_ma50_slope": max_abs_ma50_slope,
                        }
                        params.update(regime_profile)
                        candidates.append({
                            "rule_family": "mean_reversion",
                            "params": params,
                        })

    # 2. pro_mean_reversion (RSI/ADX based)
    for adx_threshold in [20.0, 30.0, 45.0]:
        for rsi_oversold in [25.0, 35.0, 45.0]:
            for long_pz in [-0.75, -1.25, -1.75]:
                for short_pz in [0.75, 1.25, 1.75]:
                    for hurst in [False, True]:
                        candidates.append({
                            "rule_family": "pro_mean_reversion",
                            "params": {
                                "adx_threshold": adx_threshold,
                                "rsi_oversold": rsi_oversold,
                                "rsi_overbought": 100.0 - rsi_oversold,
                                "long_pz": long_pz,
                                "short_pz": short_pz,
                                "hurst_filter": hurst
                            }
                        })

    # 3. macd_trend (Momentum)
    for macdh_threshold in [0.0, 0.00005, 0.0002]:
        for require_ma_alignment in [True, False]:
            for adx_trend in [0.0, 20.0, 30.0]:
                for hurst in [False, True]:
                    candidates.append({
                        "rule_family": "macd_trend",
                        "params": {
                            "macdh_threshold": macdh_threshold,
                            "require_ma_alignment": require_ma_alignment,
                            "adx_trend_threshold": adx_trend,
                            "hurst_filter": hurst
                        }
                    })

    # 4. volatility_breakout (Bollinger)
    for mean_revert in [True, False]:
        for threshold_up in [0.7, 0.85, 1.0, 1.15, 1.3]:
            for threshold_down in [0.3, 0.15, 0.0, -0.15, -0.3]:
                candidates.append({
                    "rule_family": "volatility_breakout",
                    "params": {
                        "mean_revert": mean_revert,
                        "threshold_up": threshold_up,
                        "threshold_down": threshold_down,
                    }
                })

    # 5. microstructure_bounce (High Frequency)
    for td_threshold in [-1.5, -2.5, -3.5]:
        for long_pz in [-0.5, -1.0, -1.5]:
            for short_pz in [0.5, 1.0, 1.5]:
                candidates.append({
                    "rule_family": "microstructure_bounce",
                    "params": {
                        "td_threshold": td_threshold,
                        "long_pz": long_pz,
                        "short_pz": short_pz,
                        "spread_max_z": 1.5
                    }
                })

    return candidates


# ── Metric extraction and Constraints ────────────────────────────────────────

def _rollover_share(trade_log: list[dict], rollover_hours: list[int] = [21, 22, 23, 0]) -> float:
    if not trade_log:
        return 0.0
    count = 0
    import pandas as pd
    for t in trade_log:
        ts = t.get("entry_time_msc") or t.get("open_time_msc")  # Try to grab timestamp from either key
        if not ts:
            continue
        hour = pd.Timestamp(ts, unit="ms", tz="UTC").hour
        if hour in rollover_hours:
            count += 1
    return count / len(trade_log)


def _direction_metrics(trade_log: list[dict]) -> tuple[float, float, int, int]:
    if not trade_log:
        return 0.0, 0.0, 0, 0
    longs = sum(1 for t in trade_log if int(t.get("direction", 0) or 0) > 0)
    shorts = sum(1 for t in trade_log if int(t.get("direction", 0) or 0) < 0)
    total = len(trade_log)
    return longs / total, shorts / total, longs, shorts


def extract_constrained_metrics(
    equity_curve: list[float], 
    timestamps: list, 
    trade_log: list[dict], 
    execution_log: list[dict], 
    diagnostics: dict,
) -> dict[str, Any]:
    from evaluate_oos import aggregate_training_diagnostics
    from runtime_common import build_evaluation_accounting

    execution_diagnostics = aggregate_training_diagnostics([diagnostics])
    accounting = build_evaluation_accounting(
        trade_log=trade_log,
        execution_diagnostics=execution_diagnostics,
        execution_log_count=len(execution_log),
        initial_equity=1_000.0,
    )
    
    net_pnl = float(accounting.get("net_pnl_usd", 0.0))
    pf = float(accounting.get("profit_factor", 0.0))
    expectancy = float(accounting.get("expectancy_usd", 0.0))
    trades = int(float(accounting.get("trade_count", 0.0)))
    win_rate = float(accounting.get("win_rate_pct", accounting.get("win_rate", 0.0)))
    is_valid = bool(accounting.get("is_valid", True))
    
    roll_hours = [21, 22, 23, 0]
    roll_share = _rollover_share(trade_log, rollover_hours=roll_hours)
    long_share, short_share, long_trades, short_trades = _direction_metrics(trade_log)
    max_direction_share = max(long_share, short_share)
    
    # Calculate Max Drawdown safely
    max_dd = 0.0
    if equity_curve:
        import numpy as np
        curve = np.array(equity_curve)
        peak = np.maximum.accumulate(curve)
        max_dd = float(np.max((peak - curve) / np.maximum(peak, 1e-6)))

    # Apply Strict Stability Constraints
    status = "REJECTED"
    reason = []
    
    confidence_band = "stable"
    if trades < 5:
        reason.append(f"Too few trades ({trades} < 5)")
        confidence_band = "rejected"
    elif trades < 10:
        confidence_band = "exploratory"
    if pf < 1.25:
        reason.append(f"Low PF ({pf:.2f} < 1.25)")
    if expectancy <= 0:
        reason.append(f"Negative Expectancy (${expectancy:.2f})")
    if max_dd > 0.15:
        reason.append(f"Max DD too high ({max_dd:.1%} > 15%)")
    if max_direction_share > 0.85 and trades >= 10:
        reason.append(f"Direction skewed (Max {max_direction_share:.1%} > 85%)")
    if roll_share > 0.20:
        reason.append(f"High Rollover Exposure ({roll_share:.1%} > 20%)")
    if not is_valid:
        reason.append("Accounting Validation Failed")

    # --- Cost-margin guard (Finding #2: cost model eats 70% of gross) ---
    # Reject if net PnL per trade is less than 2x the estimated round-trip
    # slippage cost — this ensures a live cost buffer exists.
    # At 0.25 pip slippage + commission ~$0.84/trade, require expectancy > $1.68.
    # This is a forward-looking safety check, not a backfit.
    MIN_LIVE_EDGE_MULTIPLE = 2.0   # net must be ≥ 2× estimated cost buffer
    ESTIMATED_RT_SLIP_USD = 0.84   # $0.84 estimated round-trip cost at 0.1 lot
    min_required_expectancy = MIN_LIVE_EDGE_MULTIPLE * ESTIMATED_RT_SLIP_USD
    if trades >= 10 and expectancy > 0 and expectancy < min_required_expectancy:
        reason.append(
            f"Thin live edge: expectancy ${expectancy:.2f} < {MIN_LIVE_EDGE_MULTIPLE:.0f}× cost buffer ${min_required_expectancy:.2f}"
        )

    if not reason:
        status = "PASSED"
        
    return {
        "status": status,
        "reject_reason": " | ".join(reason) if reason else "",
        "net_pnl": net_pnl,
        "pf": pf,
        "expectancy": expectancy,
        "trades": trades,
        "win_rate": win_rate,
        "max_dd": max_dd,
        "long_share": long_share,
        "short_share": short_share,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "roll_share": roll_share,
        "rollover_hours": roll_hours,
        "accounting_valid": is_valid,
        "confidence_band": confidence_band,
    }


# ── Execution Task ───────────────────────────────────────────────────────────

def _run_single_variant(replay_context: Any, config: dict[str, Any], alpha_gate: Any | None = None) -> dict[str, Any]:
    """Runs replay on the TRAIN portion (which is trainable_feature_frame).

    IMPORTANT: Uses dataclasses.replace() for all context mutations to avoid
    mutating the shared base object — even inside a worker, the object may be
    reused across multiple configs if the pool is reused.
    """
    import evaluate_oos
    from dataclasses import replace as dc_replace

    rule_family = config["rule_family"]
    params = config["params"]

    # Swap to TRAIN frame and inject rule config — use replace(), never direct mutation
    train_frame = replay_context.trainable_feature_frame
    if train_frame is not None and not train_frame.empty:
        replay_context = dc_replace(
            replay_context,
            replay_frame=train_frame,
            rule_family=rule_family,
            rule_params=params,
        )
    else:
        replay_context = dc_replace(
            replay_context,
            rule_family=rule_family,
            rule_params=params,
        )

    # Track how many times the rule logic itself fired
    signal_counts = {"long": 0, "short": 0}
    
    def action_index_provider_with_telemetry(*args, **kwargs):
        from strategies.rule_logic import compute_rule_direction
        f_dict = kwargs["feature_row"].to_dict() if hasattr(kwargs["feature_row"], "to_dict") else dict(kwargs["feature_row"])
        direction = compute_rule_direction(rule_family, f_dict, params)
        if direction == 1: signal_counts["long"] += 1
        elif direction == -1: signal_counts["short"] += 1
        return evaluate_oos._rule_action_provider(*args, **kwargs, rule_family=rule_family, rule_params=params)

    try:
        equity_curve, timestamps, trade_log, execution_log, diagnostics = evaluate_oos.run_replay(
            replay_context=replay_context,
            action_index_provider=action_index_provider_with_telemetry,
            disable_alpha_gate=False if alpha_gate else True,
            alpha_gate=alpha_gate,
        )
        
        metrics = extract_constrained_metrics(equity_curve, timestamps, trade_log, execution_log, diagnostics)
        metrics["signal_longs"] = signal_counts["long"]
        metrics["signal_shorts"] = signal_counts["short"]
        return {"config": config, **metrics}
    except Exception as e:
        return {"config": config, "status": "ERROR", "error": str(e)}



# ── Report Generation ────────────────────────────────────────────────────────

def _write_report(results: list[dict], args: argparse.Namespace, sys_hashes: dict):
    output_path = Path(f"artifacts/optimization_report_{args.symbol}_{args.stage}.md")
    passed = [r for r in results if r["status"] == "PASSED"]
    # Rank by Net PnL / Expectancy
    passed.sort(key=lambda x: (x["net_pnl"], x["expectancy"]), reverse=True)
    
    rejected = [r for r in results if r["status"] == "REJECTED"]
    
    lines = [
        f"# Automated Rule Candidate Generation — {args.symbol}",
        f"**Stage:** `{args.stage}`",
        f"**Manifest Path:** `{args.manifest_path}`",
        f"**Evaluator Hash (`evaluate_oos.py`):** `{sys_hashes['evaluator']}`",
        f"**Rule Logic Hash (`strategies/rule_logic.py`):** `{sys_hashes['rule_logic']}`",
        f"**Manifest Hash:** `{sys_hashes['manifest']}`",
        "",
        "**Objective:** Maximize Net PnL & Expectancy subject to strict stability constraints.",
        "**Method:** Exact-runtime evaluation over parameter grid.",
        "",
        "## Passed Candidates (Ranked)",
        "| Rank | Rule Family | Params | Net PnL | PF | Expectancy | Trades (L/S) | Signal (L/S) | Win% | MaxDD | Acc.Valid | L/S Mix | Confidence |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|"
    ]
    
    for i, r in enumerate(passed):
        param_str = ", ".join(f"{k}={v}" for k,v in r["config"]["params"].items())
        ls_mix = f"{r['long_share']:.0%}/{r['short_share']:.0%}"
        valid_mark = "✔️" if r.get("accounting_valid", False) else "❌"
        lines.append(
            f"| {i+1} | {r['config']['rule_family']} | `{param_str}` | "
            f"${r['net_pnl']:.2f} | {r['pf']:.2f} | ${r['expectancy']:.2f} | "
            f"{r['trades']} ({r.get('long_trades',0)}/{r.get('short_trades',0)}) | "
            f"{r.get('signal_longs',0)}/{r.get('signal_shorts',0)} | {r['win_rate']:.1%} | {r['max_dd']:.1%} | {valid_mark} | {ls_mix} | {r.get('confidence_band','stable')} |"
        )

        
    lines.append("")
    lines.append("## Top 5 Rejected Constraints Example")
    lines.append("| Rule Family | Params | Net PnL | PF | Trades | Rejection Reason |")
    lines.append("|---|---|---:|---:|---:|---|")
    
    for r in rejected[:5]:
        param_str = ", ".join(f"{k}={v}" for k,v in r["config"]["params"].items())
        pnl = r.get("net_pnl", 0.0)
        pf = r.get("pf", 0.0)
        trades = r.get("trades", 0)
        reason = r.get("reject_reason", "Unknown")
        lines.append(f"| {r['config']['rule_family']} | `{param_str}` | ${pnl:.2f} | {pf:.2f} | {trades} | {reason} |")
        
    output_path.parent.mkdir(exist_ok=True, parents=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    
    json_path = Path(f"artifacts/optimization_report_{args.symbol}_{args.stage}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"hashes": sys_hashes, "results": results}, f, indent=2)
        
    print(f"\nMarkdown Report written to: {output_path}")
    print(f"JSON Output written to: {json_path}")
    
    print("\n--- RECOMMENDATION ---")
    print("The Candidate Shortlist must now be validated with the Walk-Forward logic or Holdout set.")
    print("Do not manually update strategies/rule_logic.py without this final confirm stage.")


def main():
    logging.basicConfig(level=logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--manifest-path", help="Path to manifest.json")
    parser.add_argument("--dataset-path", help="Override dataset logic")
    parser.add_argument("--ticks-per-bar", type=int, help="Override ticks per bar")
    parser.add_argument("--stage", choices=["train", "validation", "holdout"], default="train", help="Which subset to optimize on")
    parser.add_argument("--use-alpha-gate", action="store_true", help="Enable self-learning filter")
    parser.add_argument(
        "--alpha-gate-model",
        choices=["auto", "logistic_pair", "xgboost_pair", "lightgbm_pair", "ridge_signed_target", "ridge"],
        default="logistic_pair",
        help="AlphaGate model backend preference.",
    )
    parser.add_argument(
        "--enable-regime-guard-sweep",
        action="store_true",
        help="Include mean-reversion variants with optional regime guard thresholds.",
    )
    parser.add_argument("--limit", type=int, help="Limit number of variants to evaluate")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    candidates = build_parameter_grid(include_regime_guard_variants=bool(args.enable_regime_guard_sweep))
    
    if args.manifest_path:
        manifest_path = Path(args.manifest_path)
    else:
        candidates_files = list(Path(ROOT / "models" / "rc1").glob(f"*{symbol.lower()}*/manifest.json"))
        if not candidates_files:
            raise SystemExit(f"No RC1 manifest found for {symbol}.")
        manifest_path = candidates_files[0]
        args.manifest_path = str(manifest_path)
    
    # Very important: Sets EVAL_MANIFEST_PATH *before* evaluate_oos executes its loading
    os.environ["EVAL_MANIFEST_PATH"] = str(manifest_path)
    if args.dataset_path:
        os.environ["DATASET_PATH"] = str(args.dataset_path)
    if args.ticks_per_bar:
        os.environ["BAR_TICKS_PER_BAR"] = str(args.ticks_per_bar)
    print(f"Using manifest context: {manifest_path}")
    
    sys_hashes = {
        "manifest": _hash_file(manifest_path),
        "evaluator": _hash_file(ROOT / "evaluate_oos.py"),
        "rule_logic": _hash_file(ROOT / "strategies/rule_logic.py"),
    }
    
    print(f"Loading base replay context for {symbol}...")
    import evaluate_oos
    base_context = evaluate_oos.load_replay_context(symbol)

    alpha_gate = None
    if args.use_alpha_gate:
        requested_alpha_model = str(args.alpha_gate_model)
        if requested_alpha_model == "xgboost_pair" and importlib.util.find_spec("xgboost") is None:
            print("Requested alpha_gate_model=xgboost_pair, but xgboost is not installed. Proceeding without AlphaGate.")
        elif requested_alpha_model == "lightgbm_pair" and importlib.util.find_spec("lightgbm") is None:
            print("Requested alpha_gate_model=lightgbm_pair, but lightgbm is not installed. Proceeding without AlphaGate.")
        print(f"Fitting AlphaGate (Self-Learning Filter) on {symbol} training data...")
        # Relaxed gate parameters for rule-filtering (Pivot to Profit)
        alpha_gate = fit_baseline_alpha_gate(
            symbol=symbol,
            train_frame=base_context.trainable_feature_frame,
            feature_cols=FEATURE_COLS,
            horizon_bars=25, # Match ATR resolution better
            commission_per_lot=7.0,
            slippage_pips=0.25,
            min_edge_pips=0.0,
            probability_threshold=0.51, # Aggressive profit mode
            probability_margin=0.01, # Tightened margin for higher density
            model_preference=requested_alpha_model,
        )
        if alpha_gate:
            print(
                f"AlphaGate fitted: model={alpha_gate.model_kind}, "
                f"PF={alpha_gate.fit_profit_factor:.2f}, Trades={alpha_gate.fit_trade_count}"
            )
        else:
            print("AlphaGate fitting failed (insufficient data). Proceeding without gate.")
    
    # Handle Stage subsetting via proper copy
    from dataclasses import replace
    if args.stage == "train":
        if base_context.trainable_feature_frame is None or base_context.trainable_feature_frame.empty:
            raise SystemExit("No trainable frame available for train stage")
        base_context = replace(base_context,
            replay_feature_frame=base_context.trainable_feature_frame,
            replay_frame=base_context.trainable_feature_frame
        )
    elif args.stage == "holdout":
        if base_context.holdout_feature_frame is None or base_context.holdout_feature_frame.empty:
            SystemExit("No holdout frame available")
        base_context = replace(base_context,
            replay_feature_frame=base_context.holdout_feature_frame,
            replay_frame=base_context.holdout_feature_frame
        )
    
    print(f"Starting parallel generation over {len(candidates)} candidate configurations [{args.stage.upper()} STAGE]...")
    
    # Pack tasks for workers
    tasks = []
    for i, config in enumerate(candidates):
        if args.limit and i >= args.limit:
            break
        tasks.append((base_context, config, alpha_gate))

    # Run in parallel
    cpu_count = max(1, multiprocessing.cpu_count() - 1)
    with multiprocessing.Pool(processes=cpu_count) as pool:
        raw_results = pool.starmap(_run_worker, tasks)
        results = [r for r in raw_results if r is not None]

    _write_report(results, args, sys_hashes)

if __name__ == "__main__":
    main()
