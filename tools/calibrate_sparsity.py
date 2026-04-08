"""
Experiment 1a+ / Tree Challenger: EURUSD 10k Sparsity Calibration

Runs the same sweep on both the HGB and Tree models under identical conditions:
  - same dataset, features, cost model, replay logic, threshold semantics
  - only the model object differs
Produces a unified scoreboard for comparison.
"""
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_oos import (
    _evaluate_policy,
    _evaluate_runtime_baselines,
    _target_direction_to_action_index,
)
from feature_engine import FEATURE_COLS
from run_logging import configure_run_logging
from replay_selector import load_selector_replay_context
from selector_manifest import load_selector_manifest, load_validated_selector_model

log = logging.getLogger("calibrate_sparsity")


def _create_provider(name, model, threshold_pips=0.0, top_k_thresh=None, hybrid_rule=None):
    audit_stats = Counter()
    gate = top_k_thresh if top_k_thresh is not None else threshold_pips

    def provider(*, feature_row, position_direction, action_map, **_):
        vec = np.array([feature_row[c] for c in FEATURE_COLS]).reshape(1, -1)
        pred = model.predict(vec)[0]
        target_direction = 0

        pre_long  = pred >  gate
        pre_short = pred < -gate

        if hybrid_rule == "trend":
            ma20 = float(feature_row.get("ma20_slope", 0.0) or 0.0)
            ma50 = float(feature_row.get("ma50_slope", 0.0) or 0.0)
            if not (ma20 > 0 and ma50 > 0): pre_long  = False
            if not (ma20 < 0 and ma50 < 0): pre_short = False

        if hybrid_rule == "mean_rev":
            sz = float(feature_row.get("spread_z", 0.0) or 0.0)
            if not (sz <= -1.0): pre_long  = False
            if not (sz >=  1.0): pre_short = False

        if pre_long:
            target_direction = 1;  audit_stats["long"] += 1
        elif pre_short:
            target_direction = -1; audit_stats["short"] += 1
        else:
            audit_stats["flat"] += 1

        return _target_direction_to_action_index(
            action_map=action_map,
            position_direction=int(position_direction or 0),
            target_direction=target_direction,
        )

    provider.audit_stats = audit_stats
    provider.name = name
    return provider


def _run_sweep(context, model, model_tag, preds, total_bars):
    abs_preds = np.abs(preds)
    p90 = np.percentile(abs_preds, 90)
    p95 = np.percentile(abs_preds, 95)
    p98 = np.percentile(abs_preds, 98)
    p99 = np.percentile(abs_preds, 99)

    log.info(f"[{model_tag}] Top-K thresholds | 10%={p90:.3f}  5%={p95:.3f}  2%={p98:.3f}  1%={p99:.3f}")

    variants = [
        # absolute thresholds
        (f"{model_tag}_raw",              0.0,  None, None),
        (f"{model_tag}_abs0.5",           0.5,  None, None),
        (f"{model_tag}_abs1.0",           1.0,  None, None),
        (f"{model_tag}_abs1.5",           1.5,  None, None),
        (f"{model_tag}_abs2.0",           2.0,  None, None),
        (f"{model_tag}_abs3.0",           3.0,  None, None),
        # top-K
        (f"{model_tag}_top10pct",         0.0,  p90, None),
        (f"{model_tag}_top5pct",          0.0,  p95, None),
        (f"{model_tag}_top2pct",          0.0,  p98, None),
        (f"{model_tag}_top1pct",          0.0,  p99, None),
        # hybrid rule-filtered
        (f"{model_tag}_abs1.0+trend",     1.0,  None, "trend"),
        (f"{model_tag}_abs1.0+mean_rev",  1.0,  None, "mean_rev"),
        (f"{model_tag}_top10+trend",      0.0,  p90, "trend"),
        (f"{model_tag}_top10+mean_rev",   0.0,  p90, "mean_rev"),
    ]

    results = {}
    for name, thresh, topk, hybrid in variants:
        p = _create_provider(name, model, thresh, topk, hybrid)
        res = _evaluate_policy(replay_context=context, action_index_provider=p)
        res["audit"] = dict(p.audit_stats)
        results[name] = res
        m = res["metrics"]
        trades = int(m.get("trade_count", 0))
        net    = float(m.get("net_pnl_usd", 0.0))
        pf     = float(m.get("profit_factor", 0.0))
        log.info(f"  {name:<35} | trades={trades:>4} | net=${net:>8.2f} | PF={pf:.2f}")

    return results


def _fmt_row(name, data, audit=None):
    m = data.get("metrics", {})
    trades = int(m.get("trade_count", 0))
    net    = float(m.get("net_pnl_usd", 0.0))
    pf     = float(m.get("profit_factor", 0.0))
    expct  = float(m.get("expectancy_usd", 0.0))
    wr     = float(m.get("win_rate", 0.0))

    in_band = 30 <= trades <= 150
    trades_s = f"**{trades}**" if in_band else str(trades)
    net_s    = f"**${net:.2f}**" if net > 0 else f"${net:.2f}"

    ls = flat = "-"
    if audit:
        ls   = f"{audit.get('long',0)}/{audit.get('short',0)}"
        flat = str(audit.get("flat", 0))

    return f"| {name} | {trades_s} | {net_s} | {pf:.2f} | ${expct:.2f} | {wr:.1%} | {ls} | {flat} |"


def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    configure_run_logging("calibrate_sparsity", capture_print=False)

    symbol = "EURUSD"
    log.info("Loading replay context from HGB model (used as replay engine base)...")
    context = load_selector_replay_context(symbol)   # loads eurusd_10k (HGB)

    # --- Load HGB model ---
    hgb_dir  = Path("models/selector/eurusd_10k")
    hgb_manifest = load_selector_manifest(hgb_dir / "selector_manifest.json")
    hgb_model    = load_validated_selector_model(hgb_manifest, expected_symbol=symbol)

    # --- Load Tree model ---
    tree_dir  = Path("models/selector/eurusd_10000_tree")
    tree_manifest = load_selector_manifest(tree_dir / "selector_manifest.json")
    tree_model    = load_validated_selector_model(tree_manifest, expected_symbol=symbol)

    holdout_X = context.replay_feature_frame.loc[:, FEATURE_COLS].to_numpy(dtype=np.float64)

    log.info("Computing HGB predictions on holdout...")
    hgb_preds  = hgb_model.predict(holdout_X)
    log.info("Computing Tree predictions on holdout...")
    tree_preds = tree_model.predict(holdout_X)

    log.info("=== HGB Sweep ===")
    hgb_results  = _run_sweep(context,  hgb_model, "hgb",  hgb_preds,  len(holdout_X))

    log.info("=== Tree Sweep ===")
    tree_results = _run_sweep(context, tree_model, "tree", tree_preds, len(holdout_X))

    log.info("=== Anchor Baselines ===")
    baselines = _evaluate_runtime_baselines(replay_context=context)

    # ---- Build unified markdown scoreboard ----
    header = ("| Policy | Trades | Net USD | PF | Expct | WinRate | L/S | Flat |")
    sep    = ("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    lines = ["# Experiment Tree Challenger vs HGB — EURUSD 10k\n",
             "**Target density band: 30–150 trades (bolded)**\n",
             "## HGB Variants\n", header, sep]
    for name, data in hgb_results.items():
        lines.append(_fmt_row(name, data, data.get("audit")))

    lines += ["\n## Tree Variants\n", header, sep]
    for name, data in tree_results.items():
        lines.append(_fmt_row(name, data, data.get("audit")))

    anchor_order = ["runtime_flat","runtime_always_long","runtime_always_short",
                    "runtime_mean_reversion","runtime_trend"]
    lines += ["\n## Anchor Baselines\n", header, sep]
    for name in anchor_order:
        if name in baselines:
            lines.append(_fmt_row(name, baselines[name]))

    Path("artifacts").mkdir(exist_ok=True)
    out = Path("artifacts/tree_challenger_results.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Scoreboard saved to {out}")


if __name__ == "__main__":
    main()
