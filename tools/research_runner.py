from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from interpreter_guard import ensure_project_venv


ensure_project_venv(project_root=REPO_ROOT, script_path=__file__)

from research.scoring import build_research_decision, compare_against_baseline, compute_composite_score
from research.schema import (
    CURRENT_TRAINING_RUN_PATH,
    Proposal,
    ProposalValidationError,
    append_jsonl_row,
    assert_no_active_training_run,
    build_research_env_overrides,
    ensure_research_layout,
    load_proposal,
    read_jsonl_rows,
    resolve_research_baseline,
    select_baseline_gate_fallback,
)
from validation_metrics import save_json_report


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _python_exe(repo_root: Path) -> Path:
    return repo_root / ".venv" / "Scripts" / "python.exe"


def _result_id_for(proposal: Proposal, *, started_at: datetime) -> str:
    stamp = started_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "__fast" if proposal.fast_mode else ""
    return f"{stamp}__{proposal.experiment_name}__{proposal.symbol.lower()}{suffix}"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _absolute_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).resolve())


def _resolve_report_path(repo_root: Path, raw_path: str | Path | None) -> Path | None:
    if raw_path in (None, ""):
        return None
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _cost_share_of_abs_gross_pnl(metrics: dict[str, Any] | None) -> float | None:
    if not isinstance(metrics, dict):
        return None
    direct_cost_share = metrics.get("cost_share_of_abs_gross_pnl")
    if direct_cost_share not in (None, ""):
        return float(direct_cost_share)
    gross_pnl = float(metrics.get("gross_pnl_usd", 0.0) or 0.0)
    total_cost = float(metrics.get("total_transaction_cost_usd", 0.0) or 0.0)
    if abs(gross_pnl) <= 1e-9:
        return None
    return float(total_cost / abs(gross_pnl))


def _baseline_distance_summary(
    metrics: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    replay_metrics = dict(metrics or {})
    baseline_metrics = dict((baseline or {}).get("metrics", {}) or {})
    trade_count = int(float(replay_metrics.get("trade_count", 0.0) or 0.0))
    baseline_trade_count = int(float(baseline_metrics.get("trade_count", 0.0) or 0.0))
    trade_count_ratio: float | None = None
    if baseline_trade_count > 0:
        trade_count_ratio = float(trade_count / baseline_trade_count)
    expectancy = float(replay_metrics.get("expectancy_usd", replay_metrics.get("expectancy", 0.0)) or 0.0)
    baseline_expectancy = float(baseline_metrics.get("expectancy_usd", 0.0) or 0.0)
    net_pnl = float(replay_metrics.get("net_pnl_usd", 0.0) or 0.0)
    baseline_net_pnl = float(baseline_metrics.get("net_pnl_usd", 0.0) or 0.0)
    return {
        "trade_count_vs_baseline_ratio": trade_count_ratio,
        "net_pnl_delta_vs_baseline": float(net_pnl - baseline_net_pnl),
        "expectancy_delta_vs_baseline": float(expectancy - baseline_expectancy),
        "cost_share": _cost_share_of_abs_gross_pnl(replay_metrics),
    }


def _artifact_pointers(artifacts_dir: Path, symbol: str) -> dict[str, str | None]:
    normalized_symbol = symbol.upper()
    slug = normalized_symbol.lower()
    return {
        "artifacts_dir": _absolute_path(artifacts_dir),
        "model_path": _absolute_path(artifacts_dir / f"model_{slug}_best.zip"),
        "vecnormalize_path": _absolute_path(artifacts_dir / f"model_{slug}_best_vecnormalize.pkl"),
        "scaler_path": _absolute_path(artifacts_dir / f"scaler_{normalized_symbol}.pkl"),
        "manifest_path": _absolute_path(artifacts_dir / f"artifact_manifest_{normalized_symbol}.json"),
        "default_manifest_path": _absolute_path(artifacts_dir / "artifact_manifest.json"),
        "training_diagnostics_path": _absolute_path(artifacts_dir / f"training_diagnostics_{slug}.json"),
        "replay_report_path": _absolute_path(artifacts_dir / f"replay_report_{slug}.json"),
        "deployment_gate_path": _absolute_path(artifacts_dir / f"deployment_gate_{slug}.json"),
    }


def _run_subprocess(
    *,
    name: str,
    command: list[str],
    repo_root: Path,
    env_overrides: dict[str, str],
    log_path: Path,
) -> dict[str, Any]:
    started_at = _utcnow()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(env_overrides)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {name}\n")
        handle.write(f"# cwd: {repo_root}\n")
        handle.write(f"# command: {subprocess.list2cmdline(command)}\n")
        handle.write(json.dumps({"env_overrides": env_overrides}, indent=2, sort_keys=True))
        handle.write("\n\n")
        completed = subprocess.run(
            command,
            cwd=str(repo_root),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    finished_at = _utcnow()
    return {
        "name": name,
        "command": command,
        "command_string": subprocess.list2cmdline(command),
        "cwd": str(repo_root.resolve()),
        "env_overrides": dict(env_overrides),
        "stdout_log_path": str(log_path.resolve()),
        "exit_code": int(completed.returncode),
        "started_at_utc": _isoformat(started_at),
        "completed_at_utc": _isoformat(finished_at),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 6),
    }


def _ledger_entry_from_result(result: dict[str, Any]) -> dict[str, Any]:
    replay_metrics = dict((result.get("evaluation_summary", {}) or {}).get("replay_metrics", {}) or {})
    score_summary = dict(result.get("score_summary", {}) or {})
    return {
        "result_id": result.get("result_id"),
        "experiment_name": (result.get("resolved_proposal", {}) or {}).get("experiment_name"),
        "symbol": (result.get("resolved_proposal", {}) or {}).get("symbol"),
        "fast_mode": bool((result.get("resolved_proposal", {}) or {}).get("fast_mode", False)),
        "status": result.get("run_status"),
        "decision": result.get("decision"),
        "composite_score": score_summary.get("score"),
        "timed_sharpe": replay_metrics.get("timed_sharpe"),
        "profit_factor": replay_metrics.get("profit_factor"),
        "expectancy_usd": replay_metrics.get("expectancy_usd", replay_metrics.get("expectancy")),
        "trade_count": replay_metrics.get("trade_count"),
        "net_pnl_usd": replay_metrics.get("net_pnl_usd"),
        "max_drawdown": replay_metrics.get("max_drawdown"),
        "baseline_gate_passed": (result.get("baseline_gate_status", {}) or {}).get("gate_passed"),
        "reconciliation_passed": (score_summary.get("inputs", {}) or {}).get("reconciliation_passed"),
        "parity_aligned": (score_summary.get("inputs", {}) or {}).get("runtime_parity_aligned"),
        "fragile_under_cost_stress": (score_summary.get("inputs", {}) or {}).get("fragile_under_cost_stress"),
        "dataset_id": (result.get("artifact_manifest", {}) or {}).get("dataset_id"),
        "bar_construction_ticks_per_bar": (result.get("artifact_manifest", {}) or {}).get(
            "bar_construction_ticks_per_bar"
        ),
        "result_json_path": result.get("result_json_path"),
        "completed_at_utc": result.get("completed_at_utc"),
        "baseline_reference": (result.get("baseline_resolution", {}) or {}).get("reference"),
        "baseline_source": (result.get("baseline_resolution", {}) or {}).get("source"),
        "baseline_label": (result.get("baseline_resolution", {}) or {}).get("label"),
        "proposal_path": result.get("proposal_path"),
    }


def _build_validation_summary(proposal: Proposal, env_overrides: dict[str, dict[str, str]]) -> dict[str, Any]:
    return {
        "proposal_path": str(proposal.source_path),
        "resolved_proposal": proposal.to_dict(),
        "resolved_env": env_overrides,
    }


def run_research_proposal(
    proposal_path: str | Path,
    *,
    validate_only: bool = False,
    repo_root: Path = REPO_ROOT,
    current_run_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    layout = ensure_research_layout(repo_root)
    proposal = load_proposal(proposal_path)
    artifacts_probe = layout.results_dir / "probe"
    env_overrides = build_research_env_overrides(proposal, artifacts_probe)

    if validate_only:
        return _build_validation_summary(proposal, env_overrides)

    assert_no_active_training_run(current_run_path or (repo_root / CURRENT_TRAINING_RUN_PATH))
    ledger_rows = read_jsonl_rows(layout.ledger_path)
    if proposal.baseline_reference:
        resolve_research_baseline(
            proposal,
            ledger_rows,
            dataset_id=None,
            bar_construction_ticks_per_bar=None,
        )

    started_at = _utcnow()
    result_id = _result_id_for(proposal, started_at=started_at)
    result_dir = (layout.results_dir / result_id).resolve()
    artifacts_dir = result_dir / "artifacts"
    logs_dir = result_dir / "logs"
    artifacts_dir.mkdir(parents=True, exist_ok=False)
    logs_dir.mkdir(parents=True, exist_ok=True)
    env_overrides = build_research_env_overrides(proposal, artifacts_dir)
    proposal_copy_path = result_dir / "proposal.json"
    proposal_copy_path.write_text(json.dumps(proposal.raw_payload, indent=2), encoding="utf-8")

    normalized_symbol = proposal.symbol.upper()
    slug = normalized_symbol.lower()
    train_command = [str(_python_exe(repo_root)), "-u", "train_agent.py"]
    eval_command = [str(_python_exe(repo_root)), "-u", "evaluate_oos.py"]

    commands: list[dict[str, Any]] = []
    run_status = "completed"
    train_info = _run_subprocess(
        name="train",
        command=train_command,
        repo_root=repo_root,
        env_overrides=env_overrides["train"],
        log_path=logs_dir / "train.stdout.log",
    )
    commands.append(train_info)
    if int(train_info["exit_code"]) != 0:
        run_status = "failed_train"

    if run_status == "completed":
        manifest_path = artifacts_dir / f"artifact_manifest_{normalized_symbol}.json"
        diagnostics_path = artifacts_dir / f"training_diagnostics_{slug}.json"
        if not manifest_path.exists() or not diagnostics_path.exists():
            run_status = "failed_train_outputs"

    if run_status == "completed":
        eval_info = _run_subprocess(
            name="eval",
            command=eval_command,
            repo_root=repo_root,
            env_overrides=env_overrides["eval"],
            log_path=logs_dir / "eval.stdout.log",
        )
        commands.append(eval_info)
        if int(eval_info["exit_code"]) != 0:
            run_status = "failed_eval"

    artifacts = _artifact_pointers(artifacts_dir, normalized_symbol)
    manifest_payload = _load_json(Path(artifacts["manifest_path"])) if artifacts["manifest_path"] else None
    training_diagnostics = (
        _load_json(Path(artifacts["training_diagnostics_path"])) if artifacts["training_diagnostics_path"] else None
    )
    replay_report = _load_json(Path(artifacts["replay_report_path"])) if artifacts["replay_report_path"] else None
    deployment_gate = _load_json(Path(artifacts["deployment_gate_path"])) if artifacts["deployment_gate_path"] else None

    baseline_report_path = _resolve_report_path(
        repo_root,
        (training_diagnostics or {}).get("baseline_report_path"),
    )
    baseline_report = _load_json(baseline_report_path) if baseline_report_path is not None else None
    runtime_parity_verdict = dict(
        ((replay_report or {}).get("runtime_parity_verdict") or (replay_report or {}).get("replay_metrics", {}).get("runtime_parity_verdict") or {})
    )
    replay_metrics = dict((replay_report or {}).get("replay_metrics", {}) or {})
    baseline_gate_status = {
        "gate_passed": bool((baseline_report or {}).get("gate_passed", (training_diagnostics or {}).get("baseline_gate_passed", False))),
        "passing_models": list((baseline_report or {}).get("passing_models", []) or []),
        "baseline_report_path": _absolute_path(baseline_report_path),
    }

    dataset_id = None
    bar_construction_ticks_per_bar = None
    if isinstance(manifest_payload, dict):
        dataset_id = manifest_payload.get("dataset_id")
        raw_bar_spec = manifest_payload.get("bar_construction_ticks_per_bar", manifest_payload.get("ticks_per_bar"))
        if raw_bar_spec not in (None, ""):
            bar_construction_ticks_per_bar = int(raw_bar_spec)

    baseline_resolution = resolve_research_baseline(
        proposal,
        ledger_rows,
        dataset_id=dataset_id,
        bar_construction_ticks_per_bar=bar_construction_ticks_per_bar,
    )
    if baseline_resolution is None:
        baseline_resolution = select_baseline_gate_fallback(baseline_report)

    score_summary = compute_composite_score(
        replay_metrics=replay_metrics,
        training_diagnostics=training_diagnostics,
        runtime_parity_verdict=runtime_parity_verdict,
        baseline_gate_passed=baseline_gate_status["gate_passed"],
    )
    baseline_comparison = compare_against_baseline(
        replay_metrics=replay_metrics,
        score_summary=score_summary,
        baseline=baseline_resolution,
    )
    baseline_distance_summary = _baseline_distance_summary(replay_metrics, baseline_resolution)
    baseline_distance_summary["overtrade_negative_edge_triggered"] = bool(
        (training_diagnostics or {}).get("overtrade_negative_edge_triggered", False)
    )
    decision_summary = build_research_decision(
        run_status=run_status,
        score_summary=score_summary,
        baseline_comparison=baseline_comparison,
    )

    completed_at = _utcnow()
    training_summary = {
        "baseline_gate_passed": bool((training_diagnostics or {}).get("baseline_gate_passed", False)),
        "deploy_ready": bool((training_diagnostics or {}).get("deploy_ready", False)),
        "blockers": list((training_diagnostics or {}).get("blockers", []) or []),
        "validation_metrics": dict((training_diagnostics or {}).get("full_path_validation_metrics", {}) or {}),
        "holdout_metrics": dict((training_diagnostics or {}).get("holdout_metrics", {}) or {}),
        "ppo_diagnostics": {
            "explained_variance": (training_diagnostics or {}).get("explained_variance"),
            "approx_kl": (training_diagnostics or {}).get("approx_kl"),
            "value_loss_mean": (training_diagnostics or {}).get("value_loss_mean"),
        },
        "baseline_report_path": _absolute_path(baseline_report_path),
        "trade_count_vs_baseline_ratio": baseline_distance_summary["trade_count_vs_baseline_ratio"],
        "net_pnl_delta_vs_baseline": baseline_distance_summary["net_pnl_delta_vs_baseline"],
        "expectancy_delta_vs_baseline": baseline_distance_summary["expectancy_delta_vs_baseline"],
        "cost_share": baseline_distance_summary["cost_share"],
        "overtrade_negative_edge_triggered": baseline_distance_summary["overtrade_negative_edge_triggered"],
    }
    evaluation_summary = {
        "replay_metrics": replay_metrics or None,
        "runtime_parity_verdict": runtime_parity_verdict or None,
        "decision_summary": (replay_report or {}).get("decision_summary"),
        "metric_reconciliation": replay_metrics.get(
            "metric_reconciliation",
            replay_metrics.get("metrics_reconciliation"),
        ),
    }

    result_payload = {
        "result_id": result_id,
        "run_status": run_status,
        "started_at_utc": _isoformat(started_at),
        "completed_at_utc": _isoformat(completed_at),
        "duration_seconds": round((completed_at - started_at).total_seconds(), 6),
        "proposal_path": str(proposal.source_path),
        "original_proposal": proposal.raw_payload,
        "resolved_proposal": proposal.to_dict(),
        "resolved_env": env_overrides,
        "commands": commands,
        "artifact_pointers": artifacts,
        "artifact_manifest": manifest_payload,
        "baseline_gate_status": baseline_gate_status,
        "baseline_resolution": baseline_resolution,
        "baseline_comparison": baseline_comparison,
        "baseline_distance_summary": baseline_distance_summary,
        "training_diagnostics_summary": training_summary,
        "evaluation_summary": evaluation_summary,
        "deployment_gate": deployment_gate,
        "score_summary": score_summary,
        "composite_score": score_summary["score"],
        "decision": decision_summary["decision"],
        "decision_summary": decision_summary,
        "result_json_path": str((result_dir / "result.json").resolve()),
    }

    result_json_path = save_json_report(result_payload, result_dir / "result.json")
    result_payload["result_json_path"] = str(result_json_path.resolve())
    ledger_entry = _ledger_entry_from_result(result_payload)
    append_jsonl_row(layout.ledger_path, ledger_entry)
    return result_payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a safe config-first research proposal.")
    parser.add_argument("--proposal", required=True, help="Path to a research proposal JSON file.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and print the resolved proposal without launching training/evaluation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = run_research_proposal(args.proposal, validate_only=bool(args.validate_only))
    except ProposalValidationError as exc:
        print(f"Proposal validation failed: {exc}", file=sys.stderr)
        return 2

    if args.validate_only:
        print(json.dumps(result, indent=2))
        return 0

    print(
        json.dumps(
            {
                "result_id": result["result_id"],
                "run_status": result["run_status"],
                "decision": result["decision"],
                "composite_score": result["composite_score"],
                "result_json_path": result["result_json_path"],
            },
            indent=2,
        )
    )
    return 0 if result["run_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
