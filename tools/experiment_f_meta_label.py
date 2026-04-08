"""
Experiment F: Meta-label on runtime_mean_reversion
---------------------------------------------------
The mean_reversion rule is the candidate generator.
A supervised classifier decides: allow this entry or stay flat.

Protocol:
- Candidate set: bars where spread_z <= -1.0 (long) OR spread_z >= 1.0 (short)
- Label: binary (profitable) and continuous (net_pips) from exact N-bar outcome
- Train: LogisticRegression + GradientBoosting classifier on candidates only
- Replay: exact-runtime engine with raw rule vs rule+filter
- Compare: must beat raw runtime_mean_reversion to be worth anything
"""
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge_research import _prepare_targets
from evaluate_oos import _evaluate_policy, _evaluate_runtime_baselines, _target_direction_to_action_index
from feature_engine import FEATURE_COLS, _compute_raw
from project_paths import resolve_dataset_path
from run_logging import configure_run_logging
from train_agent import _split_holdout
from replay_selector import load_selector_replay_context

log = logging.getLogger("experiment_f")

# Mean-reversion rule conditions (mirrors runtime engine)
SPREAD_Z_COL = "spread_z"
MR_LONG_THRESH  = -1.0  # spread_z <= this → long candidate
MR_SHORT_THRESH =  1.0  # spread_z >= this → short candidate

HORIZON_BARS     = 10
COMMISSION       = 7.0
SLIPPAGE_PIPS    = 0.25
MIN_EDGE_PIPS    = 0.0


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------
def _extract_candidates(featured: pd.DataFrame) -> pd.DataFrame:
    """Return rows where the mean_reversion rule would fire, with direction."""
    sz = featured[SPREAD_Z_COL].fillna(0.0)
    longs  = featured[sz <= MR_LONG_THRESH].copy()
    longs["rule_direction"] = 1
    shorts = featured[sz >= MR_SHORT_THRESH].copy()
    shorts["rule_direction"] = -1
    candidates = pd.concat([longs, shorts]).sort_index()
    return candidates


# ---------------------------------------------------------------------------
# Meta-provider factories
# ---------------------------------------------------------------------------
def _make_raw_rule_provider():
    """Exact replica of runtime_mean_reversion."""
    audit = Counter()
    def provider(*, feature_row, position_direction, action_map, **_):
        sz = float(feature_row.get(SPREAD_Z_COL, 0.0) or 0.0)
        target = 0
        if sz <= MR_LONG_THRESH:
            target = 1;  audit["long"] += 1
        elif sz >= MR_SHORT_THRESH:
            target = -1; audit["short"] += 1
        else:
            audit["flat"] += 1
        return _target_direction_to_action_index(
            action_map=action_map,
            position_direction=int(position_direction or 0),
            target_direction=target,
        )
    provider.audit = audit
    provider.name = "raw_mean_rev"
    return provider


def _make_filtered_provider(name, clf, threshold, top_k_thresh=None):
    """Rule fires first; classifier gates the entry."""
    audit = Counter()

    def provider(*, feature_row, position_direction, action_map, **_):
        sz = float(feature_row.get(SPREAD_Z_COL, 0.0) or 0.0)
        rule_long  = sz <= MR_LONG_THRESH
        rule_short = sz >= MR_SHORT_THRESH

        if not rule_long and not rule_short:
            audit["flat_no_rule"] += 1
            return _target_direction_to_action_index(
                action_map=action_map,
                position_direction=int(position_direction or 0),
                target_direction=0,
            )

        # Rule fires → ask the classifier
        vec = np.array([feature_row[c] for c in FEATURE_COLS]).reshape(1, -1)
        try:
            prob_allow = float(clf.predict_proba(vec)[0, 1])
        except Exception:
            prob_allow = 0.0

        gate = top_k_thresh if top_k_thresh is not None else threshold
        allowed = prob_allow >= gate

        if not allowed:
            audit["flat_rejected"] += 1
            return _target_direction_to_action_index(
                action_map=action_map,
                position_direction=int(position_direction or 0),
                target_direction=0,
            )

        target = 1 if rule_long else -1
        audit["allowed_long" if target > 0 else "allowed_short"] += 1
        return _target_direction_to_action_index(
            action_map=action_map,
            position_direction=int(position_direction or 0),
            target_direction=target,
        )

    provider.audit = audit
    provider.name = name
    return provider


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    configure_run_logging("experiment_f", capture_print=False)

    symbol = "EURUSD"
    log.info("Loading replay context...")
    context = load_selector_replay_context(symbol)

    # Full featured frame (train + holdout)
    full_featured = context.full_feature_frame
    trainable_featured, holdout_featured = _split_holdout(full_featured, 0.15)

    log.info(f"Trainable bars: {len(trainable_featured)} | Holdout bars: {len(holdout_featured)}")

    # ---- Build targets on TRAINING candidates only ----
    log.info("Synthesising targets on trainable frame...")
    prepared = _prepare_targets(
        trainable_featured,
        symbol=symbol,
        feature_cols=list(FEATURE_COLS),
        horizon_bars=HORIZON_BARS,
        commission_per_lot=COMMISSION,
        slippage_pips=SLIPPAGE_PIPS,
        min_edge_pips=MIN_EDGE_PIPS,
    )

    # Extract candidates where rule fires on training set
    train_candidates = _extract_candidates(prepared)
    log.info(f"Training candidates (rule fires): {len(train_candidates)} "
             f"(long={int((train_candidates['rule_direction']==1).sum())}, "
             f"short={int((train_candidates['rule_direction']==-1).sum())})")

    if len(train_candidates) < 20:
        log.error("Too few training candidates. Aborting.")
        return

    # Binary label: was the trade profitable post-cost?
    # signed_target > 0 means we expected profit in the rule direction
    train_candidates = train_candidates.copy()
    # For a long candidate: profit if signed_target > 0 (long expectancy positive)
    # For a short candidate: profit if signed_target < 0 (short expectancy positive)
    train_candidates["is_profitable"] = (
        ((train_candidates["rule_direction"] == 1) & (train_candidates["signed_target"] > 0)) |
        ((train_candidates["rule_direction"] == -1) & (train_candidates["signed_target"] < 0))
    ).astype(int)

    positive_rate = train_candidates["is_profitable"].mean()
    log.info(f"Positive label rate: {positive_rate:.1%}")

    X_train = train_candidates.loc[:, FEATURE_COLS].to_numpy(dtype=np.float64)
    y_train = train_candidates["is_profitable"].to_numpy()

    # ---- Train classifiers ----
    clfs = {}

    log.info("Training LogisticRegression...")
    lr = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.1, max_iter=500))])
    lr.fit(X_train, y_train)
    clfs["logistic"] = lr

    log.info("Training HistGradientBoostingClassifier...")
    hgb_clf = HistGradientBoostingClassifier(max_iter=100, max_depth=4, learning_rate=0.05, random_state=42)
    hgb_clf.fit(X_train, y_train)
    clfs["hgb_clf"] = hgb_clf

    # ---- Compute holdout probabilities for top-K thresholds ----
    holdout_candidates = _extract_candidates(holdout_featured)
    log.info(f"Holdout candidates: {len(holdout_candidates)}")

    def _holdout_probs(clf):
        if len(holdout_candidates) == 0:
            return np.array([])
        X_hc = holdout_candidates.loc[:, FEATURE_COLS].to_numpy(dtype=np.float64)
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(X_hc)[:, 1]
        return clf.predict(X_hc)

    # Variants to sweep
    variants = []
    for clf_name, clf in clfs.items():
        hprobs = _holdout_probs(clf)
        p50 = float(np.percentile(hprobs, 50)) if len(hprobs) else 0.5
        p70 = float(np.percentile(hprobs, 70)) if len(hprobs) else 0.7
        p80 = float(np.percentile(hprobs, 80)) if len(hprobs) else 0.8
        p90 = float(np.percentile(hprobs, 90)) if len(hprobs) else 0.9
        log.info(f"[{clf_name}] prob percentiles: p50={p50:.3f} p70={p70:.3f} p80={p80:.3f} p90={p90:.3f}")
        for tname, gate in [("p50", p50), ("p70", p70), ("p80", p80), ("p90", p90)]:
            variants.append((f"{clf_name}_{tname}", clf, gate, None))

    # ---- Replay ----
    results = {}

    log.info("Evaluating raw mean_reversion rule (baseline)...")
    raw_provider = _make_raw_rule_provider()
    raw_res = _evaluate_policy(replay_context=context, action_index_provider=raw_provider)
    raw_res["audit"] = dict(raw_provider.audit)
    results["raw_mean_rev_rule"] = raw_res
    m = raw_res["metrics"]
    log.info(f"  raw_mean_rev_rule | trades={int(m.get('trade_count',0))} | net=${float(m.get('net_pnl_usd',0)):.2f} | PF={float(m.get('profit_factor',0)):.2f}")

    for name, clf, gate, topk in variants:
        log.info(f"Evaluating filtered provider: {name} (gate={gate:.3f})...")
        p = _make_filtered_provider(name, clf, threshold=gate, top_k_thresh=topk)
        res = _evaluate_policy(replay_context=context, action_index_provider=p)
        res["audit"] = dict(p.audit)
        results[name] = res
        m = res["metrics"]
        log.info(f"  {name:<35} | trades={int(m.get('trade_count',0)):>4} | net=${float(m.get('net_pnl_usd',0)):>8.2f} | PF={float(m.get('profit_factor',0)):.2f}")

    log.info("Evaluating anchor baselines...")
    baselines = _evaluate_runtime_baselines(replay_context=context)

    # ---- Build scoreboard ----
    raw_net = float(results["raw_mean_rev_rule"]["metrics"].get("net_pnl_usd", 0.0))
    raw_trades = int(results["raw_mean_rev_rule"]["metrics"].get("trade_count", 0))

    header = "| Policy | Trades | Net USD | PF | Expct | WinRate | Allowed | Rejected | Flat (no rule) |"
    sep    = "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"

    def _row(name, data):
        m = data.get("metrics", {})
        trades = int(m.get("trade_count", 0))
        net    = float(m.get("net_pnl_usd", 0.0))
        pf     = float(m.get("profit_factor", 0.0))
        expct  = float(m.get("expectancy_usd", 0.0))
        wr     = float(m.get("win_rate", 0.0))
        audit  = data.get("audit", {})
        allowed   = audit.get("allowed_long", 0) + audit.get("allowed_short", 0)
        rejected  = audit.get("flat_rejected", 0)
        flat_norule = audit.get("flat_no_rule", 0)
        beat  = " ✅" if (net > raw_net and trades >= 5) else ""
        net_s = f"**${net:.2f}{beat}**" if net > 0 else f"${net:.2f}"
        return f"| {name} | {trades} | {net_s} | {pf:.2f} | ${expct:.2f} | {wr:.1%} | {allowed} | {rejected} | {flat_norule} |"

    def _row_simple(name, data):
        m = data.get("metrics", {})
        net = float(m.get("net_pnl_usd", 0.0))
        net_s = f"**${net:.2f}**" if net > 0 else f"${net:.2f}"
        return (f"| {name} | {int(m.get('trade_count',0))} | {net_s} | "
                f"{float(m.get('profit_factor',0)):.2f} | ${float(m.get('expectancy_usd',0)):.2f} | "
                f"{float(m.get('win_rate',0)):.1%} | - | - | - |")

    lines = [
        "# Experiment F: Meta-label on runtime_mean_reversion — EURUSD 10k\n",
        f"> **Raw rule baseline:** {raw_trades} trades | ${raw_net:.2f} net | "
        f"PF {float(results['raw_mean_rev_rule']['metrics'].get('profit_factor',0)):.2f}\n",
        "> ✅ = beats raw rule (net > raw and trades >= 5)\n",
        "## Filtered Selectors\n", header, sep,
    ]
    lines.append(_row("raw_mean_rev_rule", results["raw_mean_rev_rule"]))
    for name in variants:
        lines.append(_row(name[0], results[name[0]]))

    lines += ["\n## Anchor Baselines\n", header, sep]
    for bname in ["runtime_flat", "runtime_always_long", "runtime_always_short",
                  "runtime_mean_reversion", "runtime_trend"]:
        if bname in baselines:
            lines.append(_row_simple(bname, baselines[bname]))

    Path("artifacts").mkdir(exist_ok=True)
    out = Path("artifacts/experiment_f_results.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Scoreboard saved to {out}")


if __name__ == "__main__":
    main()
