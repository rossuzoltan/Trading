"""
ab_trigger_comparison.py  —  Exact-runtime A/B trigger semantics experiment

Runs 4 challenger rule variants through the same evaluate_oos replay engine,
same cost model, same pair, same holdout data.  Reports all required metrics.

Variants:
  A. mean_reversion         — current RC1 rule (spread_z trigger)
  B. price_mean_reversion   — pure price_z trigger
  C. price_mr_spread_filter — price_z trigger + spread stability gate
  D. combined_mr            — both spread_z AND price_z must agree

Usage:
    python tools/ab_trigger_comparison.py --symbol EURUSD
    python tools/ab_trigger_comparison.py --symbol EURUSD --output ab_report.md
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("ab_trigger_comparison")


# ── Metric extraction ────────────────────────────────────────────────────────


def _rollover_trades(trade_log: list[dict]) -> int:
    """Count trades opened during UTC rollover hours (22-23)."""
    count = 0
    for t in trade_log:
        ts = t.get("entry_time_msc") or t.get("open_time_msc")
        if ts is None:
            continue
        import pandas as pd
        hour = pd.Timestamp(ts, unit="ms", tz="UTC").hour
        if hour in (22, 23):
            count += 1
    return count


def _direction_mix(trade_log: list[dict]) -> tuple[int, int]:
    longs = sum(1 for t in trade_log if int(t.get("direction", 0) or 0) > 0)
    shorts = sum(1 for t in trade_log if int(t.get("direction", 0) or 0) < 0)
    return longs, shorts


def _extract_metrics(
    equity_curve, timestamps, trade_log, execution_log, diagnostics, *, variant_name: str
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
    longs, shorts = _direction_mix(trade_log)
    rollover = _rollover_trades(trade_log)
    return {
        "variant": variant_name,
        "net_pnl": float(accounting.get("net_pnl_usd", 0.0)),
        "gross_pnl": float(accounting.get("gross_pnl_usd", 0.0)),
        "total_cost": float(accounting.get("total_transaction_cost_usd", 0.0)),
        "profit_factor": float(accounting.get("profit_factor", 0.0)),
        "expectancy": float(accounting.get("expectancy_usd", 0.0)),
        "trade_count": int(float(accounting.get("trade_count", 0.0))),
        "win_rate": float(accounting.get("win_rate", 0.0)) * 100.0,
        "longs": longs,
        "shorts": shorts,
        "rollover_trades": rollover,
    }


# ── Run a single variant through evaluate_oos ────────────────────────────────

def _run_variant(
    symbol: str,
    manifest_path: Path,
    rule_family_override: str,
) -> dict[str, Any]:
    import evaluate_oos
    from functools import partial

    replay_context = evaluate_oos.load_replay_context(symbol)
    # Override rule_family / rule_params on the context
    replay_context.rule_family = rule_family_override
    # rule_params stay the same as the RC1 manifest (threshold, sl, tp)

    # Build the RULE action provider — exactly as evaluate_oos.main() does for RULE manifests
    action_index_provider = partial(
        evaluate_oos._rule_action_provider,
        rule_family=rule_family_override,
        rule_params=replay_context.rule_params,
    )

    equity_curve, timestamps, trade_log, execution_log, diagnostics = evaluate_oos.run_replay(
        replay_context=replay_context,
        action_index_provider=action_index_provider,
        disable_alpha_gate=True,
    )
    return _extract_metrics(
        equity_curve, timestamps, trade_log, execution_log, diagnostics,
        variant_name=rule_family_override,
    )


# ── Markdown table rendering ─────────────────────────────────────────────────

def _render_report(results: list[dict], symbol: str, baseline_net: float) -> str:
    header = "| Variant | Net PnL | Gross PnL | Cost | PF | Expectancy | Trades | Win% | Longs | Shorts | Rollover |"
    sep    = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"

    rows = []
    for r in results:
        tag = " ← **RC1**" if r["variant"] == "mean_reversion" else ""
        beat = " 🟢" if r["net_pnl"] > baseline_net and r["variant"] != "mean_reversion" else ""
        rows.append(
            f"| {r['variant']}{tag}{beat} "
            f"| ${r['net_pnl']:.2f} "
            f"| ${r['gross_pnl']:.2f} "
            f"| ${r['total_cost']:.2f} "
            f"| {r['profit_factor']:.2f} "
            f"| ${r['expectancy']:.2f} "
            f"| {r['trade_count']} "
            f"| {r['win_rate']:.1f}% "
            f"| {r['longs']} "
            f"| {r['shorts']} "
            f"| {r['rollover_trades']} |"
        )

    verdict_lines = ["\n## Verdict"]
    best = max(results, key=lambda x: x["net_pnl"])
    current = next(r for r in results if r["variant"] == "mean_reversion")

    if best["variant"] == "mean_reversion":
        verdict_lines.append("**A (spread_z / current RC1) wins.** Do not replace the trigger.")
    elif best["net_pnl"] > current["net_pnl"] * 1.05:
        verdict_lines.append(
            f"**{best['variant'].upper()} beats current RC1** by "
            f"${best['net_pnl'] - current['net_pnl']:.2f} net PnL. "
            f"Consider promoting to challenger RC2."
        )
    else:
        verdict_lines.append(
            f"**{best['variant']}** is marginally better but not decisively. "
            "Hold current RC1 rule; gather more shadow evidence first."
        )

    return "\n".join([
        f"# A/B Trigger Semantics Comparison — {symbol}",
        "",
        "Exact-runtime replay, same holdout data, same cost model.",
        "Four rule variants tested against identical baselines.",
        "",
        "## Results",
        header, sep,
        *rows,
        *verdict_lines,
        "",
        "---",
        "*Generated by tools/ab_trigger_comparison.py*",
    ])


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Exact-runtime A/B trigger semantics experiment")
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--manifest", default="")
    p.add_argument("--output", default="")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    symbol = args.symbol.upper()

    manifest_path = Path(args.manifest) if args.manifest else None
    if manifest_path is None:
        candidates = sorted(Path(ROOT / "models" / "rc1").glob(f"*{symbol.lower()}*/manifest.json"))
        if not candidates:
            raise SystemExit(f"No RC1 manifest for {symbol}. Pass --manifest.")
        manifest_path = candidates[0]

    out_path = Path(args.output) if args.output else manifest_path.parent / "ab_trigger_comparison.md"

    variants = [
        ("A", "mean_reversion"),
        ("B", "price_mean_reversion"),
        ("C", "price_mr_spread_filter"),
        ("D", "combined_mr"),
        ("E", "pro_mean_reversion"),
    ]

    results = []
    for label, rule_family in variants:
        log.info("Running variant %s (%s)…", label, rule_family)
        try:
            result = _run_variant(symbol, manifest_path, rule_family)
            result["label"] = label
            results.append(result)
            log.info(
                "  %s: net=$%.2f  trades=%d  PF=%.2f  rollover=%d",
                rule_family, result["net_pnl"], result["trade_count"],
                result["profit_factor"], result["rollover_trades"],
            )
        except Exception as exc:
            log.error("Variant %s FAILED: %s", label, exc)

    if not results:
        raise SystemExit("All variants failed.")

    baseline_net = next(
        (r["net_pnl"] for r in results if r["variant"] == "mean_reversion"), 0.0
    )
    report = _render_report(results, symbol, baseline_net)
    out_path.write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
