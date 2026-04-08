from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_live_metrics import (
    MIN_PROMOTION_ACTIONABLE_EVENTS,
    MIN_PROMOTION_TRADING_DAYS,
    compute_drift_metrics,
    has_two_consecutive_critical_reviews,
    load_shadow_events,
    render_shadow_summary_markdown,
    resolve_paper_live_gate_paths,
    resolve_shadow_evidence_paths,
    summarize_shadow_events,
    weekly_shadow_reviews,
)
from selector_manifest import load_selector_manifest, validate_paper_live_candidate_manifest
from trading_config import deployment_paths
from validation_metrics import load_json_report


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _scoreboard_path(manifest_path: Path) -> Path:
    return manifest_path.parent / "baseline_scoreboard_rc1.json"


def _baseline_holdout_path(manifest_path: Path, symbol: str) -> Path:
    return manifest_path.parent / f"baseline_holdout_{symbol.lower()}.json"


def _historical_replay_path(manifest_path: Path) -> Path:
    return manifest_path.parent / "mt5_historical_replay_report.json"


def _extract_replay_reference(
    *,
    manifest_path: Path,
    symbol: str,
    rule_family: str,
    scoreboard: dict[str, Any],
    replay_reference_path: Path | None,
) -> dict[str, Any]:
    if replay_reference_path is not None and replay_reference_path.exists():
        payload = _load_json(replay_reference_path) or {}
        if "metrics" in payload:
            metrics = dict(payload.get("metrics", {}) or {})
            trade_count = float(metrics.get("trade_count", 0.0) or 0.0)
            steps = float(metrics.get("steps", 0.0) or 0.0)
            return {
                "reference_source": str(replay_reference_path),
                "signal_density": (trade_count / steps) if steps > 0 else None,
                "would_open_density": (trade_count / steps) if steps > 0 else None,
                "long_share": _long_share_from_metrics(metrics),
            }

    holdout_payload = _load_json(_baseline_holdout_path(manifest_path, symbol))
    if holdout_payload:
        model_payload = (
            holdout_payload.get("holdout_metrics", {})
            .get("models", {})
            .get(rule_family, {})
            .get("metrics", {})
        )
        if model_payload:
            return {
                "reference_source": str(_baseline_holdout_path(manifest_path, symbol)),
                "signal_density": model_payload.get("trades_per_bar"),
                "would_open_density": model_payload.get("trades_per_bar"),
                "long_share": _long_share_from_metrics(model_payload),
            }

    raw_anchor = dict(scoreboard.get("baselines", {}).get("runtime_mean_reversion", {}) or {})
    return {
        "reference_source": "baseline_scoreboard_rc1",
        "signal_density": None,
        "would_open_density": None,
        "long_share": _long_share_from_metrics(raw_anchor),
    }


def _long_share_from_metrics(metrics: dict[str, Any]) -> float | None:
    long_count = metrics.get("long_count")
    short_count = metrics.get("short_count")
    if long_count is None or short_count is None:
        long_count = metrics.get("long_trade_count")
        short_count = metrics.get("short_trade_count")
    if long_count is None or short_count is None:
        return None
    total = float(long_count) + float(short_count)
    if total <= 0:
        return 0.0
    return float(long_count) / total


def _build_baseline_comparison(scoreboard: dict[str, Any]) -> dict[str, Any]:
    rc_candidate = dict(scoreboard.get("rc_candidate", {}) or {})
    baselines = dict(scoreboard.get("baselines", {}) or {})
    mandatory_names = ("runtime_flat", "runtime_always_short", "runtime_trend")
    mandatory = {name: dict(baselines.get(name, {}) or {}) for name in mandatory_names}
    raw_anchor = dict(baselines.get("runtime_mean_reversion", {}) or {})

    mandatory_pass = all(
        float(rc_candidate.get("net_pnl_usd", 0.0) or 0.0) >= float(payload.get("net_pnl_usd", 0.0) or 0.0)
        for payload in mandatory.values()
    )
    raw_anchor_pass = True
    if raw_anchor:
        raw_anchor_pass = (
            float(rc_candidate.get("profit_factor", 0.0) or 0.0) >= float(raw_anchor.get("profit_factor", 0.0) or 0.0)
            and float(rc_candidate.get("expectancy_usd", 0.0) or 0.0) >= float(raw_anchor.get("expectancy_usd", 0.0) or 0.0)
        )
    same_as_raw_anchor = bool(raw_anchor) and all(
        float(rc_candidate.get(key, 0.0) or 0.0) == float(raw_anchor.get(key, 0.0) or 0.0)
        for key in ("net_pnl_usd", "profit_factor", "expectancy_usd", "trade_count")
    )

    return {
        "mandatory_baselines": mandatory,
        "raw_anchor_baseline": raw_anchor,
        "deployed_anchor_rc": rc_candidate,
        "mandatory_baseline_pass": mandatory_pass,
        "raw_anchor_baseline_pass": raw_anchor_pass,
        "same_logic_as_raw_anchor": same_as_raw_anchor,
    }


def _status_payload(path: Path | None, *, ok_field: str | None = None) -> dict[str, Any]:
    payload = _load_json(path)
    if payload is None:
        return {"present": False, "ok": False, "path": str(path) if path is not None else None}
    ok = bool(payload.get(ok_field, False)) if ok_field else True
    return {"present": True, "ok": ok, "path": str(path), "payload": payload}


def _historical_replay_status(path: Path | None) -> dict[str, Any]:
    payload = _load_json(path)
    if payload is None:
        return {"present": False, "ok": False, "path": str(path) if path is not None else None}
    verdict = str(payload.get("overall_verdict", "") or "").upper()
    ok = verdict not in {"DRIFT_CRITICAL", "NO_DATA", ""}
    return {
        "present": True,
        "ok": ok,
        "path": str(path),
        "overall_verdict": verdict,
        "payload": payload,
    }


def build_paper_live_gate(
    *,
    manifest_path: str | Path,
    shadow_dir: str | Path | None = None,
    replay_reference_path: str | Path | None = None,
    restart_drill_path: str | Path | None = None,
    preflight_path: str | Path | None = None,
    ops_attestation_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    resolved_manifest_path = Path(manifest_path)
    manifest = load_selector_manifest(resolved_manifest_path, verify_manifest_hash=True)
    validate_paper_live_candidate_manifest(manifest)

    scoreboard = _load_json(_scoreboard_path(resolved_manifest_path))
    if not scoreboard:
        raise FileNotFoundError(f"Missing RC1 scoreboard: {_scoreboard_path(resolved_manifest_path)}")

    baseline_comparison = _build_baseline_comparison(scoreboard)
    shadow_paths = resolve_shadow_evidence_paths(
        symbol=manifest.strategy_symbol,
        manifest_hash=manifest.manifest_hash,
        base_dir=shadow_dir,
    )
    shadow_events = load_shadow_events(shadow_paths.events_path)
    shadow_summary = summarize_shadow_events(shadow_events)
    replay_reference = _extract_replay_reference(
        manifest_path=resolved_manifest_path,
        symbol=manifest.strategy_symbol,
        rule_family=str(manifest.rule_family or "mean_reversion"),
        scoreboard=scoreboard,
        replay_reference_path=Path(replay_reference_path) if replay_reference_path is not None else None,
    )
    drift_metrics = compute_drift_metrics(shadow_summary, replay_reference=replay_reference)
    weekly_reviews = weekly_shadow_reviews(shadow_events, replay_reference=replay_reference)
    sufficient_shadow_window = bool(shadow_summary.get("evidence_sufficient", False))

    deployment = deployment_paths(manifest.strategy_symbol)
    restart_status = _status_payload(
        Path(restart_drill_path) if restart_drill_path is not None else Path("models") / f"restart_drill_{manifest.strategy_symbol.lower()}.json",
        ok_field="state_restored_ok",
    )
    if restart_status.get("present") and restart_status.get("ok"):
        payload = dict(restart_status.get("payload", {}) or {})
        restart_status["ok"] = bool(
            payload.get("state_restored_ok", False)
            and payload.get("confirmed_position_restored_ok", False)
            and payload.get("startup_reconcile_ok", False)
        )

    preflight_status = _status_payload(
        Path(preflight_path) if preflight_path is not None else deployment.live_preflight_path,
        ok_field="approved_for_live_runtime",
    )
    ops_status = _status_payload(
        Path(ops_attestation_path) if ops_attestation_path is not None else deployment.ops_attestation_path,
        ok_field="approved",
    )
    historical_replay_status = _historical_replay_status(_historical_replay_path(resolved_manifest_path))

    reasons: list[str] = []
    rc_certification_pass = bool(scoreboard.get("rc_candidate"))
    if not baseline_comparison["mandatory_baseline_pass"]:
        reasons.append("deployed anchor underperformed a mandatory baseline")
    if not baseline_comparison["raw_anchor_baseline_pass"]:
        reasons.append("deployed anchor underperformed the raw anchor baseline on PF or expectancy")
    if not shadow_summary.get("evidence_sufficient", False):
        reasons.append(
            f"shadow evidence below threshold: need {MIN_PROMOTION_TRADING_DAYS} trading days and {MIN_PROMOTION_ACTIONABLE_EVENTS} actionable events"
        )
    if sufficient_shadow_window and drift_metrics.get("critical", False):
        reasons.append("shadow window shows critical replay-vs-shadow drift")
    if has_two_consecutive_critical_reviews(weekly_reviews):
        reasons.append("two consecutive eligible weekly reviews show critical drift")
    if not restart_status.get("ok", False):
        reasons.append("restart drill failed or missing")
    if not preflight_status.get("ok", False):
        reasons.append("preflight failed or missing")
    if not ops_status.get("ok", False):
        reasons.append("ops attestation failed or missing")
    if historical_replay_status.get("present") and not historical_replay_status.get("ok", False):
        reasons.append("historical MT5 replay shows critical drift")

    promotion_pass = bool(
        rc_certification_pass
        and baseline_comparison["mandatory_baseline_pass"]
        and baseline_comparison["raw_anchor_baseline_pass"]
        and sufficient_shadow_window
        and not drift_metrics.get("critical", False)
        and not has_two_consecutive_critical_reviews(weekly_reviews)
        and restart_status.get("ok", False)
        and preflight_status.get("ok", False)
        and ops_status.get("ok", False)
    )
    demotion_triggered = bool(
        not baseline_comparison["mandatory_baseline_pass"]
        or not baseline_comparison["raw_anchor_baseline_pass"]
        or (sufficient_shadow_window and drift_metrics.get("critical", False))
        or has_two_consecutive_critical_reviews(weekly_reviews)
        or not restart_status.get("ok", False)
        or not preflight_status.get("ok", False)
        or not ops_status.get("ok", False)
    )

    if promotion_pass:
        anchor_status = "paper_live_profitable"
        final_verdict = "paper_live_profitable"
        verdict_reason = "promotion gate passed"
    elif demotion_triggered:
        anchor_status = "demoted"
        final_verdict = "demoted"
        verdict_reason = "; ".join(reasons) if reasons else "demotion gate triggered"
    else:
        anchor_status = "candidate"
        final_verdict = "candidate"
        verdict_reason = "; ".join(reasons) if reasons else "promotion evidence still incomplete"

    payload = {
        "symbol": manifest.strategy_symbol,
        "ticks_per_bar": int(manifest.ticks_per_bar or manifest.bar_construction_ticks_per_bar or 0),
        "manifest_hash": manifest.manifest_hash,
        "logic_hash": manifest.logic_hash,
        "evaluator_hash": manifest.evaluator_hash,
        "release_stage": manifest.release_stage,
        "anchor_status": anchor_status,
        "replay_metrics": dict(scoreboard.get("rc_candidate", {}) or {}),
        "baseline_comparison": baseline_comparison,
        "shadow_window_start": shadow_summary.get("shadow_window_start"),
        "shadow_window_end": shadow_summary.get("shadow_window_end"),
        "shadow_summary_stats": shadow_summary,
        "drift_metrics": drift_metrics,
        "restart_status": restart_status,
        "ops_attestation_status": ops_status,
        "preflight_status": preflight_status,
        "historical_replay_status": historical_replay_status,
        "final_verdict": final_verdict,
        "verdict_reason": verdict_reason,
        "weekly_reviews": weekly_reviews,
        "replay_reference": replay_reference,
    }

    gate_paths = resolve_paper_live_gate_paths(
        symbol=manifest.strategy_symbol,
        manifest_hash=manifest.manifest_hash,
        base_dir=output_dir,
    )
    gate_paths.root_dir.mkdir(parents=True, exist_ok=True)
    gate_paths.json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    gate_paths.markdown_path.write_text(render_paper_live_gate_markdown(payload), encoding="utf-8")
    if shadow_events and not shadow_paths.summary_json_path.exists():
        shadow_paths.summary_json_path.write_text(json.dumps(shadow_summary, indent=2) + "\n", encoding="utf-8")
    if shadow_events and not shadow_paths.summary_markdown_path.exists():
        shadow_paths.summary_markdown_path.write_text(render_shadow_summary_markdown(shadow_summary), encoding="utf-8")
    return payload


def render_paper_live_gate_markdown(payload: dict[str, Any]) -> str:
    replay_metrics = dict(payload.get("replay_metrics", {}) or {})
    baseline = dict(payload.get("baseline_comparison", {}) or {})
    drift = dict(payload.get("drift_metrics", {}) or {})
    lines = [
        f"# Paper-Live Gate - {payload.get('symbol', 'UNKNOWN')}",
        "",
        f"* Manifest hash: `{payload.get('manifest_hash', '')}`",
        f"* Verdict: `{payload.get('final_verdict', 'unknown')}`",
        f"* Anchor status: `{payload.get('anchor_status', 'unknown')}`",
        f"* Reason: {payload.get('verdict_reason', '')}",
        "",
        "## Replay",
        f"* Net PnL USD: `{replay_metrics.get('net_pnl_usd')}`",
        f"* Profit factor: `{replay_metrics.get('profit_factor')}`",
        f"* Expectancy USD: `{replay_metrics.get('expectancy_usd')}`",
        f"* Trade count: `{replay_metrics.get('trade_count')}`",
        "",
        "## Baseline Comparison",
        f"* Mandatory baseline pass: `{baseline.get('mandatory_baseline_pass')}`",
        f"* Raw anchor baseline pass: `{baseline.get('raw_anchor_baseline_pass')}`",
        f"* Same logic as raw anchor: `{baseline.get('same_logic_as_raw_anchor')}`",
        "",
        "## Shadow Window",
        f"* Start: `{payload.get('shadow_window_start')}`",
        f"* End: `{payload.get('shadow_window_end')}`",
        f"* Evidence sufficient: `{payload.get('shadow_summary_stats', {}).get('evidence_sufficient')}`",
        f"* Trading days: `{payload.get('shadow_summary_stats', {}).get('trading_days')}`",
        f"* Actionable events: `{payload.get('shadow_summary_stats', {}).get('actionable_event_count')}`",
        "",
        "## Drift",
        f"* Verdict: `{drift.get('verdict')}`",
        f"* Critical: `{drift.get('critical')}`",
        f"* Signal density ratio: `{drift.get('signal_density_ratio')}`",
        f"* Would-open density ratio: `{drift.get('would_open_density_ratio')}`",
        f"* Spread rejection delta pp: `{drift.get('spread_rejection_delta_pp')}`",
        f"* Session rejection delta pp: `{drift.get('session_rejection_delta_pp')}`",
        f"* Directional occupancy delta pp: `{drift.get('directional_occupancy_delta_pp')}`",
        "",
        "## Historical Replay",
        f"* Present: `{payload.get('historical_replay_status', {}).get('present')}`",
        f"* Verdict: `{payload.get('historical_replay_status', {}).get('overall_verdict')}`",
        f"* OK: `{payload.get('historical_replay_status', {}).get('ok')}`",
        "",
        "## Ops Gates",
        f"* Restart: `{payload.get('restart_status', {}).get('ok')}`",
        f"* Preflight: `{payload.get('preflight_status', {}).get('ok')}`",
        f"* Ops attestation: `{payload.get('ops_attestation_status', {}).get('ok')}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a paper-live profitability gate verdict for an RC1 manifest.")
    parser.add_argument("--manifest-path", "--manifest", dest="manifest_path", required=True)
    parser.add_argument("--shadow-dir", default=None, help="Optional shadow artifact root override.")
    parser.add_argument("--replay-reference-path", default=None)
    parser.add_argument("--restart-drill-path", default=None)
    parser.add_argument("--preflight-path", default=None)
    parser.add_argument("--ops-attestation-path", default=None)
    parser.add_argument("--output-dir", default=None, help="Optional gate artifact root override.")
    args = parser.parse_args()

    payload = build_paper_live_gate(
        manifest_path=args.manifest_path,
        shadow_dir=args.shadow_dir,
        replay_reference_path=args.replay_reference_path,
        restart_drill_path=args.restart_drill_path,
        preflight_path=args.preflight_path,
        ops_attestation_path=args.ops_attestation_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("final_verdict") == "paper_live_profitable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
