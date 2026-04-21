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

from edge_research import BaselineAlphaGate, fit_baseline_alpha_gate, save_baseline_alpha_gate
from evaluate_oos import FEATURE_COLS, _evaluate_policy, _selector_action_provider, load_replay_context
from rule_selector import RuleSelector
from run_logging import configure_run_logging, set_log_context
from selector_manifest import _file_sha256, load_selector_manifest


log = logging.getLogger("alpha_gate_bakeoff")
DEFAULT_MODELS = ("none", "manifest", "logistic_pair", "xgboost_pair", "lightgbm_pair")


def _load_context_for_manifest(manifest_path: Path, *, symbol: str):
    previous_manifest = os.environ.get("EVAL_MANIFEST_PATH")
    previous_symbol = os.environ.get("EVAL_SYMBOL")
    os.environ["EVAL_MANIFEST_PATH"] = str(manifest_path)
    os.environ["EVAL_SYMBOL"] = str(symbol).upper()
    try:
        return load_replay_context(symbol=symbol)
    finally:
        if previous_manifest is None:
            os.environ.pop("EVAL_MANIFEST_PATH", None)
        else:
            os.environ["EVAL_MANIFEST_PATH"] = previous_manifest
        if previous_symbol is None:
            os.environ.pop("EVAL_SYMBOL", None)
        else:
            os.environ["EVAL_SYMBOL"] = previous_symbol


def _score_payload(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(payload.get("metrics", {}) or {})
    trade_log = list(payload.get("trade_log", []) or [])
    long_count = sum(1 for trade in trade_log if float(trade.get("direction", trade.get("long_short", 0)) or 0) > 0)
    short_count = sum(1 for trade in trade_log if float(trade.get("direction", trade.get("long_short", 0)) or 0) < 0)
    return {
        "net_pnl_usd": float(metrics.get("net_pnl_usd", 0.0) or 0.0),
        "gross_pnl_usd": float(metrics.get("gross_pnl_usd", 0.0) or 0.0),
        "total_cost_usd": float(metrics.get("total_transaction_cost_usd", 0.0) or 0.0),
        "profit_factor": float(metrics.get("profit_factor", 0.0) or 0.0),
        "expectancy_usd": float(metrics.get("expectancy_usd", 0.0) or 0.0),
        "trade_count": int(metrics.get("trade_count", 0) or 0),
        "win_rate": float(metrics.get("win_rate", 0.0) or 0.0),
        "timed_sharpe": float(metrics.get("timed_sharpe", 0.0) or 0.0),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0) or 0.0),
        "validation_passed": bool(((metrics.get("validation_status", {}) or {}).get("passed", False))),
        "accounting_gap_detected": bool(metrics.get("accounting_gap_detected", False)),
        "long_count": int(long_count),
        "short_count": int(short_count),
    }


def _gate_fit_summary(gate: BaselineAlphaGate | None) -> dict[str, Any] | None:
    if gate is None:
        return None
    return {
        "model_kind": str(gate.model_kind),
        "probability_threshold": float(gate.probability_threshold),
        "probability_margin": float(gate.probability_margin),
        "min_edge_pips": float(gate.min_edge_pips),
        "fit_trade_count": float(gate.fit_trade_count),
        "fit_long_trade_count": float(gate.fit_long_trade_count),
        "fit_short_trade_count": float(gate.fit_short_trade_count),
        "fit_expectancy_usd": float(gate.fit_expectancy_usd),
        "fit_profit_factor": float(gate.fit_profit_factor),
        "fit_quality_passed": bool(gate.fit_quality_passed),
    }


def _result_sort_key(result: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = dict(result.get("metrics", {}) or {})
    validation_passed = 1.0 if bool(metrics.get("validation_passed", False)) else 0.0
    return (
        validation_passed,
        float(metrics.get("net_pnl_usd", 0.0) or 0.0),
        float(metrics.get("profit_factor", 0.0) or 0.0),
        float(metrics.get("expectancy_usd", 0.0) or 0.0),
    )


def _evaluate_selector_candidate(
    *,
    manifest_path: Path,
    replay_context,
    selector: RuleSelector,
) -> dict[str, Any]:
    payload = _evaluate_policy(
        replay_context=replay_context,
        action_index_provider=partial(_selector_action_provider, selector=selector),
        disable_alpha_gate=True,
    )
    return _score_payload(payload)


def _fit_candidate_gate(
    *,
    replay_context,
    model_name: str,
    probability_threshold: float,
    probability_margin: float,
    min_edge_pips: float,
    horizon_bars: int,
    min_trade_count: int,
) -> BaselineAlphaGate | None:
    return fit_baseline_alpha_gate(
        symbol=str(replay_context.symbol).upper(),
        train_frame=replay_context.trainable_feature_frame,
        feature_cols=list(FEATURE_COLS),
        horizon_bars=int(horizon_bars),
        commission_per_lot=float(replay_context.execution_cost_profile.get("commission_per_lot", 7.0) or 7.0),
        slippage_pips=float(replay_context.execution_cost_profile.get("slippage_pips", 0.25) or 0.25),
        min_edge_pips=float(min_edge_pips),
        probability_threshold=float(probability_threshold),
        probability_margin=float(probability_margin),
        model_preference=model_name,
        min_trade_count=int(min_trade_count),
    )


def build_alpha_gate_bakeoff(
    *,
    manifest_path: str | Path,
    models: list[str],
    probability_threshold: float,
    probability_margin: float,
    min_edge_pips: float,
    horizon_bars: int,
    min_trade_count: int,
    output_dir: str | Path | None = None,
    save_best_gate_artifact: bool = False,
) -> dict[str, Any]:
    resolved_manifest_path = Path(manifest_path).resolve()
    manifest = load_selector_manifest(
        resolved_manifest_path,
        verify_manifest_hash=True,
        strict_manifest_hash=True,
        require_component_hashes=True,
    )
    set_log_context(symbol=manifest.strategy_symbol, event="alpha_gate_bakeoff")
    log.info(
        "Starting AlphaGate bakeoff.",
        extra={
            "manifest_path": str(resolved_manifest_path),
            "manifest_hash": manifest.manifest_hash,
            "models": list(models),
        },
    )
    replay_context = _load_context_for_manifest(
        resolved_manifest_path,
        symbol=manifest.strategy_symbol,
    )
    replay_bars = int(len(replay_context.replay_frame))
    selected_models = [item.strip().lower() for item in models if item.strip()]
    if not selected_models:
        selected_models = list(DEFAULT_MODELS)

    results: list[dict[str, Any]] = []
    refit_gates: dict[str, BaselineAlphaGate] = {}

    if "none" in selected_models:
        selector = RuleSelector(resolved_manifest_path)
        selector.alpha_gate = None
        selector.alpha_gate_threshold_override = None
        selector.alpha_gate_margin_override = None
        metrics = _evaluate_selector_candidate(
            manifest_path=resolved_manifest_path,
            replay_context=replay_context,
            selector=selector,
        )
        result = {
            "name": "rule_only",
            "source": "selector",
            "status": "completed",
            "metrics": metrics,
            "gate_fit": None,
        }
        results.append(result)
        log.info("Evaluated bakeoff candidate.", extra={"candidate": "rule_only", **metrics})

    if "manifest" in selected_models:
        selector = RuleSelector(resolved_manifest_path)
        if selector.alpha_gate is None:
            results.append(
                {
                    "name": "manifest_gate",
                    "source": "manifest",
                    "status": "skipped",
                    "reason": "manifest_alpha_gate_disabled",
                    "metrics": {},
                    "gate_fit": None,
                }
            )
            log.warning("Skipped manifest AlphaGate candidate because manifest gate is disabled.")
        else:
            metrics = _evaluate_selector_candidate(
                manifest_path=resolved_manifest_path,
                replay_context=replay_context,
                selector=selector,
            )
            result = {
                "name": "manifest_gate",
                "source": "manifest",
                "status": "completed",
                "metrics": metrics,
                "gate_fit": _gate_fit_summary(selector.alpha_gate),
            }
            results.append(result)
            log.info("Evaluated bakeoff candidate.", extra={"candidate": "manifest_gate", **metrics})

    for model_name in [item for item in selected_models if item not in {"none", "manifest"}]:
        try:
            gate = _fit_candidate_gate(
                replay_context=replay_context,
                model_name=model_name,
                probability_threshold=probability_threshold,
                probability_margin=probability_margin,
                min_edge_pips=min_edge_pips,
                horizon_bars=horizon_bars,
                min_trade_count=min_trade_count,
            )
        except Exception as exc:
            results.append(
                {
                    "name": model_name,
                    "source": "refit",
                    "status": "error",
                    "reason": str(exc),
                    "metrics": {},
                    "gate_fit": None,
                }
            )
            log.exception("AlphaGate refit failed.", extra={"candidate": model_name})
            continue

        if gate is None:
            results.append(
                {
                    "name": model_name,
                    "source": "refit",
                    "status": "fit_failed",
                    "reason": "insufficient_train_quality",
                    "metrics": {},
                    "gate_fit": None,
                }
            )
            log.warning("AlphaGate refit produced no viable gate.", extra={"candidate": model_name})
            continue

        selector = RuleSelector(resolved_manifest_path)
        selector.alpha_gate = gate
        selector.alpha_gate_threshold_override = float(gate.probability_threshold)
        selector.alpha_gate_margin_override = float(gate.probability_margin)
        refit_gates[model_name] = gate
        metrics = _evaluate_selector_candidate(
            manifest_path=resolved_manifest_path,
            replay_context=replay_context,
            selector=selector,
        )
        result = {
            "name": model_name,
            "source": "refit",
            "status": "completed",
            "metrics": metrics,
            "gate_fit": _gate_fit_summary(gate),
        }
        results.append(result)
        log.info("Evaluated bakeoff candidate.", extra={"candidate": model_name, **metrics})

    completed = [item for item in results if item.get("status") == "completed"]
    ranked = sorted(completed, key=_result_sort_key, reverse=True)
    best_name = str(ranked[0]["name"]) if ranked else None

    artifact_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path("artifacts") / "bakeoff" / manifest.strategy_symbol.upper() / manifest.manifest_hash
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_json = artifact_dir / "alpha_gate_bakeoff.json"
    artifact_md = artifact_dir / "alpha_gate_bakeoff.md"

    saved_gate_payload: dict[str, Any] | None = None
    if save_best_gate_artifact and best_name in refit_gates:
        best_gate_path = artifact_dir / f"{best_name}_best_alpha_gate.joblib"
        save_baseline_alpha_gate(refit_gates[best_name], best_gate_path)
        saved_gate_payload = {
            "name": best_name,
            "path": str(best_gate_path),
            "sha256": _file_sha256(best_gate_path),
        }
        log.info("Saved best refit AlphaGate artifact.", extra={"candidate": best_name, "path": str(best_gate_path)})

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": manifest.strategy_symbol,
        "manifest_path": str(resolved_manifest_path),
        "manifest_hash": manifest.manifest_hash,
        "holdout_start_utc": replay_context.holdout_start_utc,
        "replay_bars": replay_bars,
        "defaults": {
            "probability_threshold": float(probability_threshold),
            "probability_margin": float(probability_margin),
            "min_edge_pips": float(min_edge_pips),
            "horizon_bars": int(horizon_bars),
            "min_trade_count": int(min_trade_count),
            "commission_per_lot": float(replay_context.execution_cost_profile.get("commission_per_lot", 7.0) or 7.0),
            "slippage_pips": float(replay_context.execution_cost_profile.get("slippage_pips", 0.25) or 0.25),
        },
        "results": results,
        "best_candidate": best_name,
        "saved_best_gate": saved_gate_payload,
        "artifact_dir": str(artifact_dir),
    }
    artifact_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    artifact_md.write_text(render_alpha_gate_bakeoff_markdown(payload), encoding="utf-8")
    log.info(
        "Completed AlphaGate bakeoff.",
        extra={
            "best_candidate": best_name,
            "artifact_json": str(artifact_json),
            "artifact_markdown": str(artifact_md),
            "completed_candidates": len(completed),
        },
    )
    return payload


def render_alpha_gate_bakeoff_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# AlphaGate Bakeoff - {payload.get('symbol', 'UNKNOWN')}",
        "",
        f"* Manifest: `{payload.get('manifest_path', '')}`",
        f"* Manifest hash: `{payload.get('manifest_hash', '')}`",
        f"* Holdout start UTC: `{payload.get('holdout_start_utc')}`",
        f"* Replay bars: `{payload.get('replay_bars')}`",
        f"* Best candidate: `{payload.get('best_candidate')}`",
        "",
        "## Defaults",
        f"* Probability threshold: `{payload.get('defaults', {}).get('probability_threshold')}`",
        f"* Probability margin: `{payload.get('defaults', {}).get('probability_margin')}`",
        f"* Min edge pips: `{payload.get('defaults', {}).get('min_edge_pips')}`",
        f"* Horizon bars: `{payload.get('defaults', {}).get('horizon_bars')}`",
        "",
        "## Results",
        "",
        "| Candidate | Source | Status | Net PnL USD | PF | Expectancy USD | Trades | Long | Short | Validation |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in payload.get("results", []):
        metrics = dict(item.get("metrics", {}) or {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("name", "")),
                    str(item.get("source", "")),
                    str(item.get("status", "")),
                    f"{float(metrics.get('net_pnl_usd', 0.0) or 0.0):.2f}",
                    f"{float(metrics.get('profit_factor', 0.0) or 0.0):.3f}",
                    f"{float(metrics.get('expectancy_usd', 0.0) or 0.0):.3f}",
                    str(int(metrics.get("trade_count", 0) or 0)),
                    str(int(metrics.get("long_count", 0) or 0)),
                    str(int(metrics.get("short_count", 0) or 0)),
                    str(bool(metrics.get("validation_passed", False))),
                ]
            )
            + " |"
        )
        gate_fit = dict(item.get("gate_fit", {}) or {})
        if gate_fit:
            lines.append(
                f"Fit: model=`{gate_fit.get('model_kind')}`, threshold=`{gate_fit.get('probability_threshold')}`, "
                f"margin=`{gate_fit.get('probability_margin')}`, fit_pf=`{gate_fit.get('fit_profit_factor')}`, "
                f"fit_trades=`{gate_fit.get('fit_trade_count')}`"
            )
        if item.get("reason"):
            lines.append(f"Reason: `{item['reason']}`")
        lines.append("")
    if payload.get("saved_best_gate"):
        saved = dict(payload["saved_best_gate"] or {})
        lines.extend(
            [
                "## Saved Best Gate",
                f"* Candidate: `{saved.get('name')}`",
                f"* Path: `{saved.get('path')}`",
                f"* SHA256: `{saved.get('sha256')}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Exact-runtime AlphaGate bakeoff for a RULE manifest.")
    parser.add_argument("--manifest-path", "--manifest", dest="manifest_path", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Candidates to evaluate. Supported: none manifest logistic_pair xgboost_pair lightgbm_pair",
    )
    parser.add_argument("--probability-threshold", type=float, default=0.51)
    parser.add_argument("--probability-margin", type=float, default=0.01)
    parser.add_argument("--min-edge-pips", type=float, default=0.0)
    parser.add_argument("--horizon-bars", type=int, default=25)
    parser.add_argument("--min-trade-count", type=int, default=20)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-best-gate-artifact", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path).resolve()
    run_id = manifest_path.parent.name
    configure_run_logging(component="alpha_gate_bakeoff", symbol=None, run_id=run_id, logger_name="alpha_gate_bakeoff")
    payload = build_alpha_gate_bakeoff(
        manifest_path=manifest_path,
        models=list(args.models),
        probability_threshold=float(args.probability_threshold),
        probability_margin=float(args.probability_margin),
        min_edge_pips=float(args.min_edge_pips),
        horizon_bars=int(args.horizon_bars),
        min_trade_count=int(args.min_trade_count),
        output_dir=args.output_dir,
        save_best_gate_artifact=bool(args.save_best_gate_artifact),
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
