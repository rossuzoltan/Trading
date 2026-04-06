from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path to ensure infra modules can be imported from tools/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from interpreter_guard import ensure_project_venv
from trading_config import deployment_paths
from validation_metrics import load_json_report, save_json_report

ensure_project_venv(project_root=Path(__file__).resolve().parent, script_path=__file__)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_execution_audit_summary(path: str | Path) -> dict[str, Any]:
    audit_path = Path(path)
    rows = _load_jsonl(audit_path)
    accepted = [row for row in rows if bool(row.get("accepted"))]
    rejected = [row for row in rows if not bool(row.get("accepted"))]
    deltas = np.asarray([float(row.get("fill_delta_pips", 0.0) or 0.0) for row in accepted], dtype=np.float64)
    retcodes = Counter(str(row.get("retcode")) for row in rows)
    return {
        "path": str(audit_path),
        "symbol": audit_path.stem.split("_")[-1].upper(),
        "sample_count": len(rows),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "retcode_counts": dict(retcodes),
        "mean_fill_delta_pips": float(np.mean(deltas)) if len(deltas) else None,
        "mean_abs_fill_delta_pips": float(np.mean(np.abs(deltas))) if len(deltas) else None,
        "p95_abs_fill_delta_pips": float(np.percentile(np.abs(deltas), 95)) if len(deltas) else None,
    }


def validate_restart_drill_report(
    report: dict[str, Any] | None,
    *,
    min_bars_before_restart: int = 1,
    min_bars_after_restart: int = 1,
) -> list[str]:
    blockers: list[str] = []
    if not report:
        return ["Restart drill evidence missing."]
    if not bool(report.get("startup_reconcile_ok", False)):
        blockers.append("Startup reconcile did not complete cleanly.")
    if not bool(report.get("state_restored_ok", False)):
        blockers.append("Persisted runtime state was not restored after restart.")
    if not bool(report.get("confirmed_position_restored_ok", False)):
        blockers.append("Confirmed broker position did not round-trip across restart.")
    if str(report.get("evidence_mode", "")).strip().lower() != "real_mt5":
        blockers.append("Restart drill evidence is not attestable for live: evidence_mode must be real_mt5.")
    if not bool(report.get("attestable_for_live", False)):
        blockers.append("Restart drill evidence is not marked attestable for live.")
    if int(report.get("bars_processed_before_restart", 0) or 0) < int(min_bars_before_restart):
        blockers.append(f"Restart drill processed too few bars before restart: need {min_bars_before_restart}.")
    if int(report.get("bars_processed_after_restart", 0) or 0) < int(min_bars_after_restart):
        blockers.append(f"Restart drill processed too few bars after restart: need {min_bars_after_restart}.")
    if not report.get("pre_restart_snapshot") or not report.get("post_restart_snapshot"):
        blockers.append("Restart drill snapshots are incomplete.")
    return blockers


def validate_execution_audit_summary(
    summary: dict[str, Any] | None,
    *,
    min_accepted_fills: int = 20,
    max_mean_abs_fill_delta_pips: float = 0.5,
    max_p95_abs_fill_delta_pips: float = 1.5,
) -> list[str]:
    blockers: list[str] = []
    if not summary:
        return ["Execution audit summary missing."]
    accepted_count = int(summary.get("accepted_count", 0) or 0)
    if accepted_count < int(min_accepted_fills):
        blockers.append(f"Execution audit has only {accepted_count} accepted fills; need at least {min_accepted_fills}.")
    mean_abs = summary.get("mean_abs_fill_delta_pips")
    p95_abs = summary.get("p95_abs_fill_delta_pips")
    if mean_abs is None:
        blockers.append("Execution audit does not contain enough accepted fills to estimate mean absolute drift.")
    elif float(mean_abs) > float(max_mean_abs_fill_delta_pips):
        blockers.append(
            f"Mean absolute fill drift {float(mean_abs):.3f} pips exceeds limit {float(max_mean_abs_fill_delta_pips):.3f}."
        )
    if p95_abs is None:
        blockers.append("Execution audit does not contain enough accepted fills to estimate tail drift.")
    elif float(p95_abs) > float(max_p95_abs_fill_delta_pips):
        blockers.append(
            f"95th percentile fill drift {float(p95_abs):.3f} pips exceeds limit {float(max_p95_abs_fill_delta_pips):.3f}."
        )
    return blockers


def build_ops_attestation(
    *,
    symbol: str,
    attested_by: str,
    notes: str,
    shadow_days_completed: int,
    execution_audit_path: str | Path | None = None,
    restart_drill_path: str | Path | None = None,
    output_path: str | Path | None = None,
    model_dir: str | Path = "models",
    min_accepted_fills: int = 20,
    min_shadow_days: int = 14,
    max_mean_abs_fill_delta_pips: float = 0.5,
    max_p95_abs_fill_delta_pips: float = 1.5,
) -> dict[str, Any]:
    normalized_symbol = symbol.upper()
    paths = deployment_paths(normalized_symbol, model_dir=model_dir)
    audit_path = Path(execution_audit_path) if execution_audit_path is not None else paths.execution_audit_path
    drill_path = (
        Path(restart_drill_path)
        if restart_drill_path is not None
        else Path(model_dir) / f"restart_drill_{normalized_symbol.lower()}.json"
    )

    execution_audit_summary = load_execution_audit_summary(audit_path)
    restart_drill_report = load_json_report(drill_path) if drill_path.exists() else None

    blockers: list[str] = []
    warnings: list[str] = []
    if int(shadow_days_completed) < int(min_shadow_days):
        blockers.append(f"Shadow days {int(shadow_days_completed)} < required {int(min_shadow_days)}.")

    execution_blockers = validate_execution_audit_summary(
        execution_audit_summary,
        min_accepted_fills=min_accepted_fills,
        max_mean_abs_fill_delta_pips=max_mean_abs_fill_delta_pips,
        max_p95_abs_fill_delta_pips=max_p95_abs_fill_delta_pips,
    )
    restart_blockers = validate_restart_drill_report(restart_drill_report)
    blockers.extend(execution_blockers)
    blockers.extend(restart_blockers)

    if restart_drill_report is None:
        warnings.append(f"Restart drill evidence missing at {drill_path}.")

    payload = {
        "symbol": normalized_symbol,
        "attested_at_utc": datetime.now(timezone.utc).isoformat(),
        "attested_by": attested_by,
        "shadow_days_completed": int(shadow_days_completed),
        "shadow_days_required": int(min_shadow_days),
        "execution_drift_ok": not execution_blockers,
        "position_reconciliation_ok": not restart_blockers,
        "restart_drill_ok": not restart_blockers,
        "execution_audit_summary": execution_audit_summary,
        "restart_drill_evidence": restart_drill_report,
        "evidence_paths": {
            "execution_audit": str(audit_path),
            "restart_drill": str(drill_path),
        },
        "notes": notes,
        "warnings": warnings,
        "blockers": blockers,
        "approved": not blockers,
    }

    out_path = Path(output_path) if output_path is not None else paths.ops_attestation_path
    save_json_report(payload, out_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ops evidence and write an attestation.")
    parser.add_argument("--symbol", default=os.environ.get("TRADING_SYMBOL", "EURUSD"))
    parser.add_argument("--attested-by", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--shadow-days-completed", type=int, required=True)
    parser.add_argument("--execution-audit-path", default=None)
    parser.add_argument("--restart-drill-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--min-accepted-fills", type=int, default=int(os.environ.get("OPS_MIN_ACCEPTED_FILLS", "20")))
    parser.add_argument("--min-shadow-days", type=int, default=int(os.environ.get("OPS_MIN_SHADOW_DAYS", "14")))
    parser.add_argument(
        "--max-mean-abs-fill-delta-pips",
        type=float,
        default=float(os.environ.get("OPS_MAX_MEAN_ABS_FILL_DELTA_PIPS", "0.5")),
    )
    parser.add_argument(
        "--max-p95-abs-fill-delta-pips",
        type=float,
        default=float(os.environ.get("OPS_MAX_P95_ABS_FILL_DELTA_PIPS", "1.5")),
    )
    args = parser.parse_args()

    payload = build_ops_attestation(
        symbol=args.symbol,
        attested_by=args.attested_by,
        notes=args.notes,
        shadow_days_completed=args.shadow_days_completed,
        execution_audit_path=args.execution_audit_path,
        restart_drill_path=args.restart_drill_path,
        output_path=args.output_path,
        model_dir=args.model_dir,
        min_accepted_fills=args.min_accepted_fills,
        min_shadow_days=args.min_shadow_days,
        max_mean_abs_fill_delta_pips=args.max_mean_abs_fill_delta_pips,
        max_p95_abs_fill_delta_pips=args.max_p95_abs_fill_delta_pips,
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["approved"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


