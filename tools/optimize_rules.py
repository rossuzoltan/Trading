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
import json
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("optimize_rules")

def _hash_file(filepath: Path) -> str:
    if not filepath.exists():
        return "Not_Found"
    return hashlib.sha256(filepath.read_bytes()).hexdigest()[:8]


# ── Configuration Grid ───────────────────────────────────────────────────────

def build_parameter_grid() -> list[dict[str, Any]]:
    candidates = []

    # 1. mean_reversion (Control / Spread-based)
    for threshold in [0.8, 1.0, 1.2, 1.5, 2.0]:
        candidates.append({
            "rule_family": "mean_reversion",
            "params": {"threshold": threshold}
        })

    # 2. price_mean_reversion
    for threshold in [1.0, 1.5, 2.0, 2.5, 3.0]:
        candidates.append({
            "rule_family": "price_mean_reversion",
            "params": {"threshold": threshold}
        })

    # 3. price_mr_spread_filter
    for threshold in [1.0, 1.5, 2.0, 2.5, 3.0]:
        candidates.append({
            "rule_family": "price_mr_spread_filter",
            "params": {"threshold": threshold}
        })

    # 4. combined_mr
    for threshold in [1.0, 1.2, 1.5, 2.0]:
        candidates.append({
            "rule_family": "combined_mr",
            "params": {"threshold": threshold}
        })

    # 5. trend
    for threshold in [0.0, 0.5, 1.0]:
        candidates.append({
            "rule_family": "trend",
            "params": {"threshold": threshold}
        })

    # 6. volatility_breakout
    for mean_revert in [True, False]:
        for threshold_up in [0.8, 0.9, 1.0, 1.1]:
            for threshold_down in [0.2, 0.1, 0.0, -0.1]:
                candidates.append({
                    "rule_family": "volatility_breakout",
                    "params": {
                        "mean_revert": mean_revert,
                        "threshold_up": threshold_up,
                        "threshold_down": threshold_down
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


def _direction_share(trade_log: list[dict]) -> tuple[float, float]:
    if not trade_log:
        return 0.0, 0.0
    longs = sum(1 for t in trade_log if int(t.get("direction", 0) or 0) > 0)
    shorts = sum(1 for t in trade_log if int(t.get("direction", 0) or 0) < 0)
    return longs / len(trade_log), shorts / len(trade_log)


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
    long_share, short_share = _direction_share(trade_log)
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
    
    if trades < 10:
        reason.append(f"Too few trades ({trades} < 10)")
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
        "roll_share": roll_share,
        "rollover_hours": roll_hours,
        "accounting_valid": is_valid,
    }


# ── Execution Task ───────────────────────────────────────────────────────────

def _run_single_variant(replay_context: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Runs replay on the TRAIN portion (which is trainable_feature_frame)"""
    import evaluate_oos
    from functools import partial
    

    # We want to optimize on the TRAIN set, NOT the HOLDOUT.
    # evaluate_oos uses replay_context.replay_frame, which is normally the holdout.
    # We must explicitly swap it to use the trainable_feature_frame for the search phase,
    # just like the model training does.
    if replay_context.trainable_feature_frame is not None and not replay_context.trainable_feature_frame.empty:
        # Instead of directly mutating, we should ideally use replace on context in a higher scope,
        # but since we already got a copy from main loop, we safely assign it here as the object is ours.
        replay_context.replay_frame = replay_context.trainable_feature_frame
    
    rule_family = config["rule_family"]
    params = config["params"]
    
    replay_context.rule_family = rule_family
    replay_context.rule_params = params

    action_index_provider = partial(
        evaluate_oos._rule_action_provider,
        rule_family=rule_family,
        rule_params=params,
    )

    try:
        equity_curve, timestamps, trade_log, execution_log, diagnostics = evaluate_oos.run_replay(
            replay_context=replay_context,
            action_index_provider=action_index_provider,
            disable_alpha_gate=True,
        )
        
        metrics = extract_constrained_metrics(equity_curve, timestamps, trade_log, execution_log, diagnostics)
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
        "| Rank | Rule Family | Params | Net PnL | PF | Expectancy | Trades | Win% | MaxDD | Acc.Valid | Rollover | L/S Mix |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|"
    ]
    
    for i, r in enumerate(passed):
        param_str = ", ".join(f"{k}={v}" for k,v in r["config"]["params"].items())
        ls_mix = f"{r['long_share']:.0%}/{r['short_share']:.0%}"
        valid_mark = "✔️" if r.get("accounting_valid", False) else "❌"
        lines.append(
            f"| {i+1} | {r['config']['rule_family']} | `{param_str}` | "
            f"${r['net_pnl']:.2f} | {r['pf']:.2f} | ${r['expectancy']:.2f} | "
            f"{r['trades']} | {r['win_rate']:.1%} | {r['max_dd']:.1%} | {valid_mark} | {r.get('roll_share', 0):.1%} | {ls_mix} |"
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
    args = parser.parse_args()

    symbol = args.symbol.upper()
    candidates = build_parameter_grid()
    
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
    
    print(f"Starting generation over {len(candidates)} candidate configurations [{args.stage.upper()} STAGE]...")
    results = []
    
    # Run sequentially for debugging, or multiprocessing if robust
    for i, config in enumerate(candidates):
        import traceback
        try:
            # We copy via replace since it's a dataclass, preventing state leakage.
            ctx = replace(base_context)
            res = _run_single_variant(ctx, config)
            if res.get("status") == "PASSED":
                print(f"[{i+1}/{len(candidates)}] \033[92mPASSED\033[0m: {config['rule_family']} -> PnL: ${res.get('net_pnl', 0):.2f}")
            elif res.get("status") == "REJECTED":
                print(f"[{i+1}/{len(candidates)}] \033[93mREJECTED\033[0m: {config['rule_family']} ({res.get('reject_reason')})")
            else:
                print(f"[{i+1}/{len(candidates)}] \033[91mERROR\033[0m: {config['rule_family']} ({res.get('error')})")
            results.append(res)
        except Exception:
            traceback.print_exc()

    _write_report(results, args, sys_hashes)

if __name__ == "__main__":
    main()
