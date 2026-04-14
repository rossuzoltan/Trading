from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_oos import (
    RUNTIME_BASELINE_PROVIDERS,
    _evaluate_policy,
    _load_promoted_manifest_context,
    _selector_action_provider,
    build_evaluation_accounting,
)
from selector_manifest import _file_sha256, load_selector_manifest
from rule_selector import RuleSelector

log = logging.getLogger("verify_v1_rc")
DEFAULT_BASELINES = (
    "runtime_flat",
    "runtime_always_long",
    "runtime_always_short",
    "runtime_trend",
    "runtime_mean_reversion",
)


def get_detailed_metrics(trade_log: list[dict[str, Any]], execution_log: list[dict[str, Any]], *, initial_equity: float = 1000.0) -> dict[str, Any]:
    accounting = build_evaluation_accounting(
        trade_log=trade_log,
        execution_diagnostics={},
        execution_log_count=len(execution_log),
        initial_equity=initial_equity,
    )
    long_count = sum(
        1 for trade in trade_log if float(trade.get("direction", trade.get("long_short", 0)) or 0) > 0
    )
    short_count = sum(
        1 for trade in trade_log if float(trade.get("direction", trade.get("long_short", 0)) or 0) < 0
    )
    return {
        "net_pnl_usd": float(accounting.get("net_pnl_usd", 0.0)),
        "gross_pnl_usd": float(accounting.get("gross_pnl_usd", 0.0)),
        "total_cost_usd": float(accounting.get("total_transaction_cost_usd", 0.0)),
        "profit_factor": float(accounting.get("profit_factor", 0.0)),
        "expectancy_usd": float(accounting.get("expectancy_usd", 0.0)),
        "trade_count": int(accounting.get("trade_count", 0)),
        "win_rate": float(accounting.get("win_rate", 0.0)),
        "long_count": int(long_count),
        "short_count": int(short_count),
    }


def verify_component_hashes(payload: dict[str, Any]) -> dict[str, str]:
    expected_evaluator_hash = _file_sha256(ROOT / "evaluate_oos.py")
    expected_logic_hash = _file_sha256(ROOT / "strategies" / "rule_logic.py")
    if str(payload.get("evaluator_hash") or "") != expected_evaluator_hash:
        raise RuntimeError("Truth-engine drift detected: evaluator_hash does not match evaluate_oos.py.")
    if str(payload.get("logic_hash") or "") != expected_logic_hash:
        raise RuntimeError("Rule-logic drift detected: logic_hash does not match strategies/rule_logic.py.")
    return {
        "evaluator_hash": expected_evaluator_hash,
        "logic_hash": expected_logic_hash,
    }


def validate_manifest_truth_requirements(manifest_path: Path) -> dict[str, str]:
    manifest = load_selector_manifest(
        manifest_path,
        verify_manifest_hash=True,
        strict_manifest_hash=True,
        require_component_hashes=True,
    )
    if manifest.release_stage != "paper_live_candidate":
        raise RuntimeError(
            f"RC1 certification requires release_stage='paper_live_candidate', got {manifest.release_stage!r}."
        )
    if manifest.live_trading_approved:
        raise RuntimeError("RC1 certification refused because live_trading_approved=true.")
    verified = verify_component_hashes(
        {
            "evaluator_hash": manifest.evaluator_hash,
            "logic_hash": manifest.logic_hash,
        }
    )
    return {
        "manifest_hash": manifest.manifest_hash,
        "evaluator_hash": verified["evaluator_hash"],
        "logic_hash": verified["logic_hash"],
    }


def _certification_failures(
    *,
    manifest: Any,
    component_hashes: dict[str, Any],
    rc_metrics: dict[str, Any],
    baselines: dict[str, dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    if getattr(manifest, "release_stage", None) != "paper_live_candidate":
        failures.append("release_stage is not paper_live_candidate")
    if bool(getattr(manifest, "live_trading_approved", False)):
        failures.append("live_trading_approved must remain false")
    if getattr(manifest, "evaluator_hash", "") != component_hashes.get("evaluator_hash_current"):
        failures.append("evaluator_hash drift detected")
    if getattr(manifest, "logic_hash", "") != component_hashes.get("logic_hash_current"):
        failures.append("logic_hash drift detected")
    if not bool(rc_metrics.get("validation_passed", True)):
        failures.append("validation parity failed")
    if int(rc_metrics.get("trade_count", 0)) <= 0:
        failures.append("candidate produced no trades")
    rc_net = float(rc_metrics.get("net_pnl_usd", 0.0))
    for baseline_name, metrics in baselines.items():
        if rc_net < float(metrics.get("net_pnl_usd", 0.0)):
            failures.append(f"candidate underperformed {baseline_name}")
    return failures


def _main_replay_metrics(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], Any]:
    manifest = load_selector_manifest(
        manifest_path,
        verify_manifest_hash=True,
        strict_manifest_hash=True,
        require_component_hashes=True,
    )
    os.environ["EVAL_MANIFEST_PATH"] = str(manifest_path)
    context = _load_promoted_manifest_context(manifest.strategy_symbol)
    if context is None:
        raise RuntimeError(f"Failed to load replay context from {manifest_path}")
    action_index_provider = None
    if context.engine_type == "RULE" and context.rule_family:
        selector = RuleSelector(manifest_path)
        action_index_provider = partial(_selector_action_provider, selector=selector)
    payload = _evaluate_policy(
        replay_context=context,
        action_index_provider=action_index_provider,
        disable_alpha_gate=False,
    )
    metrics = get_detailed_metrics(payload["trade_log"], payload["execution_log"])
    replay_bars = int(len(context.replay_frame))
    metrics["replay_bars"] = replay_bars
    metrics["trades_per_bar"] = float(metrics["trade_count"] / replay_bars) if replay_bars > 0 else 0.0
    metrics["validation_passed"] = bool(((payload.get("metrics", {}) or {}).get("validation_status", {}) or {}).get("passed", False))
    return metrics, payload["trade_log"], payload["execution_log"], context


def certify_manifest(manifest_path: Path) -> dict[str, Any]:
    hash_evidence = validate_manifest_truth_requirements(manifest_path)
    manifest = load_selector_manifest(
        manifest_path,
        verify_manifest_hash=True,
        strict_manifest_hash=True,
        require_component_hashes=True,
    )
    rc_metrics, _, _, context = _main_replay_metrics(manifest_path)

    baselines: dict[str, Any] = {}
    for baseline_name in DEFAULT_BASELINES:
        payload = _evaluate_policy(
            replay_context=context,
            action_index_provider=RUNTIME_BASELINE_PROVIDERS[baseline_name],
            disable_alpha_gate=True,
        )
        baselines[baseline_name] = get_detailed_metrics(payload["trade_log"], payload["execution_log"])

    failures = _certification_failures(
        manifest=manifest,
        component_hashes={
            "evaluator_hash_current": hash_evidence["evaluator_hash"],
            "logic_hash_current": hash_evidence["logic_hash"],
        },
        rc_metrics=rc_metrics,
        baselines=baselines,
    )
    if failures:
        raise RuntimeError("RC1 certification failed: " + "; ".join(failures))

    return {
        "name": manifest_path.parent.name,
        "symbol": manifest.strategy_symbol,
        "ticks_per_bar": int(manifest.ticks_per_bar or manifest.bar_construction_ticks_per_bar or 0),
        "release_stage": manifest.release_stage,
        "live_trading_approved": bool(manifest.live_trading_approved),
        "manifest_hash": hash_evidence["manifest_hash"],
        "evaluator_hash": hash_evidence["evaluator_hash"],
        "logic_hash": hash_evidence["logic_hash"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rc_candidate": rc_metrics,
        "baselines": baselines,
        "certification_failures": failures,
    }


def generate_reports(result: dict[str, Any], output_dir: Path) -> None:
    json_path = output_dir / "baseline_scoreboard_rc1.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    rc_metrics = result["rc_candidate"]
    lines = [
        f"# Baseline Scoreboard - {result['name']}",
        "",
        f"* **Symbol**: {result['symbol']}",
        f"* **Horizon**: {result['ticks_per_bar']} ticks",
        f"* **Stage**: `{result['release_stage']}`",
        f"* **Manifest Hash**: `{result['manifest_hash']}`",
        f"* **Evaluator Hash**: `{result['evaluator_hash']}`",
        f"* **Logic Hash**: `{result['logic_hash']}`",
        "",
        "## Comparison Table",
        "| Policy | Net PnL | Trades | Win Rate | PF | Cost | Long | Short |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| **RC1 candidate** | **${rc_metrics['net_pnl_usd']:.2f}** | **{rc_metrics['trade_count']}** | "
            f"**{rc_metrics['win_rate']:.1%}** | **{rc_metrics['profit_factor']:.2f}** | "
            f"**${rc_metrics['total_cost_usd']:.2f}** | **{rc_metrics['long_count']}** | **{rc_metrics['short_count']}** |"
        ),
    ]
    for baseline_name, baseline_metrics in result["baselines"].items():
        lines.append(
            f"| {baseline_name} | ${baseline_metrics['net_pnl_usd']:.2f} | {baseline_metrics['trade_count']} | "
            f"{baseline_metrics['win_rate']:.1%} | {baseline_metrics['profit_factor']:.2f} | "
            f"${baseline_metrics['total_cost_usd']:.2f} | {baseline_metrics['long_count']} | {baseline_metrics['short_count']} |"
        )
    lines.extend(
        [
            "",
            "## Safety Notes",
            "* Certified only against the current evaluator and rule-logic hashes.",
            "* `live_trading_approved` remains `false`; this scoreboard is not a live-trading approval.",
            "",
            "*Generated by tools/verify_v1_rc.py*",
        ]
    )
    (output_dir / "baseline_scoreboard_rc1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Global Parity Report - Bot v1 RC1",
        "",
        "| Pack | Symbol | Horizon | Net PnL | Trades | Stage |",
        "| :--- | :--- | ---: | ---: | ---: | :--- |",
    ]
    for result in results:
        candidate = result["rc_candidate"]
        lines.append(
            f"| {result['name']} | {result['symbol']} | {result['ticks_per_bar']} | "
            f"${candidate['net_pnl_usd']:.2f} | {candidate['trade_count']} | {result['release_stage']} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Bot v1 RC1 packs and write per-pack baseline scoreboards.")
    parser.add_argument("manifest_paths", nargs="*", help="Optional explicit manifest paths. Defaults to models/rc1/*/manifest.json.")
    return parser.parse_args()


def _discover_manifest_paths(args: argparse.Namespace) -> list[Path]:
    if args.manifest_paths:
        return [Path(path).resolve() for path in args.manifest_paths]
    return sorted((ROOT / "models" / "rc1").glob("*/manifest.json"))


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest_paths = _discover_manifest_paths(args)
    if not manifest_paths:
        log.error("No RC1 manifests found to certify.")
        return 1

    results: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        log.info("Certifying %s", manifest_path)
        result = certify_manifest(manifest_path)
        generate_reports(result, manifest_path.parent)
        results.append(result)
        log.info(
            "  -> %s net=$%.2f trades=%d",
            result["name"],
            result["rc_candidate"]["net_pnl_usd"],
            result["rc_candidate"]["trade_count"],
        )

    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "parity_report_rc1.md").write_text(_summary_markdown(results), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
