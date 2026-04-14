from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from selector_manifest import load_selector_manifest, validate_paper_live_candidate_manifest
from tools.paper_live_gate import _historical_replay_status, _load_json, _scoreboard_path
from trading_config import deployment_paths
from run_logging import configure_run_logging, set_log_context


log = logging.getLogger("pre_test_gate")


def build_pre_test_gate(*, manifest_path: str | Path) -> dict[str, Any]:
    resolved_manifest_path = Path(manifest_path)
    manifest = load_selector_manifest(
        resolved_manifest_path,
        verify_manifest_hash=True,
        strict_manifest_hash=True,
        require_component_hashes=True,
    )
    validate_paper_live_candidate_manifest(manifest)

    blockers: list[str] = []
    warnings: list[str] = []

    set_log_context(symbol=manifest.strategy_symbol, event="pre_test_gate")
    log.info(
        "Starting pre-test gate.",
        extra={
            "manifest_path": str(resolved_manifest_path),
            "manifest_hash": manifest.manifest_hash,
            "ticks_per_bar": int(manifest.ticks_per_bar or manifest.bar_construction_ticks_per_bar or 0),
        },
    )

    scoreboard = _load_json(_scoreboard_path(resolved_manifest_path))
    if not scoreboard:
        blockers.append(f"missing_scoreboard:{_scoreboard_path(resolved_manifest_path)}")
        rc_candidate: dict[str, Any] = {}
        baselines: dict[str, Any] = {}
    else:
        rc_candidate = dict(scoreboard.get("rc_candidate", {}) or {})
        baselines = dict(scoreboard.get("baselines", {}) or {})

    trade_count = int(rc_candidate.get("trade_count", 0) or 0)
    net_pnl = float(rc_candidate.get("net_pnl_usd", 0.0) or 0.0)
    profit_factor = float(rc_candidate.get("profit_factor", 0.0) or 0.0)
    expectancy = float(rc_candidate.get("expectancy_usd", 0.0) or 0.0)
    long_count = int(rc_candidate.get("long_count", 0) or 0)
    short_count = int(rc_candidate.get("short_count", 0) or 0)

    if scoreboard:
        if trade_count < 20:
            blockers.append(f"trade_count_below_floor:{trade_count}")
        if net_pnl <= 0.0:
            blockers.append(f"non_positive_net_pnl:{net_pnl:.2f}")
        if profit_factor < 1.10:
            blockers.append(f"profit_factor_below_floor:{profit_factor:.3f}")
        if expectancy <= 0.0:
            blockers.append(f"non_positive_expectancy:{expectancy:.3f}")
        if trade_count >= 10 and (long_count == 0 or short_count == 0):
            blockers.append(f"one_sided_replay:longs={long_count},shorts={short_count}")

        for baseline_name in ("runtime_flat", "runtime_always_short", "runtime_trend"):
            baseline = dict(baselines.get(baseline_name, {}) or {})
            if not baseline:
                blockers.append(f"missing_baseline:{baseline_name}")
                continue
            baseline_net = float(baseline.get("net_pnl_usd", 0.0) or 0.0)
            if net_pnl < baseline_net:
                blockers.append(f"underperformed_{baseline_name}:{net_pnl:.2f}<{baseline_net:.2f}")
        log.info(
            "Loaded RC scoreboard metrics.",
            extra={
                "trade_count": trade_count,
                "net_pnl_usd": net_pnl,
                "profit_factor": profit_factor,
                "expectancy_usd": expectancy,
                "long_count": long_count,
                "short_count": short_count,
            },
        )
    else:
        log.warning("RC scoreboard is missing.", extra={"scoreboard_path": str(_scoreboard_path(resolved_manifest_path))})

    historical_status = _historical_replay_status(
        resolved_manifest_path.parent / "mt5_historical_replay_report.json",
        expected_manifest_hash=manifest.manifest_hash,
        expected_logic_hash=manifest.logic_hash,
        expected_evaluator_hash=manifest.evaluator_hash,
    )
    if not historical_status.get("present", False):
        blockers.append("missing_historical_replay_report")
        historical_payload: dict[str, Any] = {}
    else:
        historical_payload = dict(historical_status.get("payload", {}) or {})
        if not historical_status.get("ok", False):
            blockers.extend([f"historical_replay:{item}" for item in historical_status.get("blockers", [])])
        live_trades_per_bar = historical_payload.get("live_trades_per_bar")
        replay_trades_per_bar = historical_payload.get("replay_trades_per_bar")
        if live_trades_per_bar is not None and replay_trades_per_bar is not None and float(replay_trades_per_bar) > 0.0:
            ratio = float(live_trades_per_bar) / float(replay_trades_per_bar)
            if ratio > 1.25:
                warnings.append(f"elevated_live_signal_density:{ratio:.3f}")
        log.info(
            "Historical replay status loaded.",
            extra={
                "historical_ok": bool(historical_status.get("ok", False)),
                "historical_verdict": historical_status.get("overall_verdict"),
                "historical_blockers": list(historical_status.get("blockers", []) or []),
            },
        )

    alpha_gate_cfg = dict(manifest.alpha_gate or {})
    if bool(alpha_gate_cfg.get("enabled", False)):
        alpha_gate_path = Path(str(alpha_gate_cfg.get("model_path") or "").strip())
        if not alpha_gate_path.exists():
            blockers.append(f"missing_alpha_gate_artifact:{alpha_gate_path}")

    deployment = deployment_paths(manifest.strategy_symbol)
    preflight_present = bool(deployment.live_preflight_path.exists())
    restart_present = bool((Path("models") / f"restart_drill_{manifest.strategy_symbol.lower()}.json").exists())
    ops_present = bool(deployment.ops_attestation_path.exists())
    if not preflight_present:
        warnings.append("missing_preflight_evidence")
    if not restart_present:
        warnings.append("missing_restart_drill_evidence")
    if not ops_present:
        warnings.append("missing_ops_attestation_evidence")

    payload = {
        "symbol": manifest.strategy_symbol,
        "manifest_path": str(resolved_manifest_path),
        "manifest_hash": manifest.manifest_hash,
        "ticks_per_bar": int(manifest.ticks_per_bar or manifest.bar_construction_ticks_per_bar or 0),
        "rc_metrics": rc_candidate,
        "historical_replay_status": historical_status,
        "alpha_gate_enabled": bool(alpha_gate_cfg.get("enabled", False)),
        "blockers": blockers,
        "warnings": warnings,
        "ready_for_test": not blockers,
    }
    out_path = resolved_manifest_path.parent / "pre_test_gate.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown_path = resolved_manifest_path.parent / "pre_test_gate.md"
    markdown_path.write_text(render_pre_test_gate_markdown(payload), encoding="utf-8")
    log.info(
        "Finished pre-test gate.",
        extra={
            "ready_for_test": bool(payload["ready_for_test"]),
            "blocker_count": len(blockers),
            "warning_count": len(warnings),
            "output_json": str(out_path),
            "output_markdown": str(markdown_path),
        },
    )
    return payload


def render_pre_test_gate_markdown(payload: dict[str, Any]) -> str:
    rc_metrics = dict(payload.get("rc_metrics", {}) or {})
    historical = dict(payload.get("historical_replay_status", {}) or {})
    blockers = list(payload.get("blockers", []) or [])
    warnings = list(payload.get("warnings", []) or [])
    lines = [
        f"# Pre-Test Gate - {payload.get('symbol', 'UNKNOWN')}",
        "",
        f"* Manifest: `{payload.get('manifest_path', '')}`",
        f"* Manifest hash: `{payload.get('manifest_hash', '')}`",
        f"* Ready for test: `{payload.get('ready_for_test', False)}`",
        "",
        "## RC Metrics",
        f"* Net PnL USD: `{rc_metrics.get('net_pnl_usd')}`",
        f"* Profit factor: `{rc_metrics.get('profit_factor')}`",
        f"* Expectancy USD: `{rc_metrics.get('expectancy_usd')}`",
        f"* Trade count: `{rc_metrics.get('trade_count')}`",
        f"* Long / short: `{rc_metrics.get('long_count')} / {rc_metrics.get('short_count')}`",
        "",
        "## Historical Replay",
        f"* Present: `{historical.get('present')}`",
        f"* OK: `{historical.get('ok')}`",
        f"* Verdict: `{historical.get('overall_verdict')}`",
    ]
    if blockers:
        lines.extend(["", "## Blockers", *[f"* `{item}`" for item in blockers]])
    else:
        lines.extend(["", "## Blockers", "* None"])
    if warnings:
        lines.extend(["", "## Warnings", *[f"* `{item}`" for item in warnings]])
    else:
        lines.extend(["", "## Warnings", "* None"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-fast pre-test gate for RC candidates.")
    parser.add_argument("--manifest-path", "--manifest", dest="manifest_path", required=True)
    args = parser.parse_args()
    run_id = Path(args.manifest_path).resolve().parent.name
    configure_run_logging(component="pre_test_gate", symbol=None, run_id=run_id, logger_name="pre_test_gate")
    payload = build_pre_test_gate(manifest_path=args.manifest_path)
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("ready_for_test", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
