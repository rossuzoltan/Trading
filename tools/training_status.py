from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path to ensure infra modules can be imported from tools/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from pathlib import Path
from typing import Any

from trading_config import deployment_paths

CURRENT_RUN_CONTEXT_NAME = "current_training_run.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return _load_json(path)
    except Exception:
        return None


def _load_current_run_context(checkpoints_dir: Path) -> dict[str, Any] | None:
    return _maybe_load(checkpoints_dir / CURRENT_RUN_CONTEXT_NAME)


def _heartbeat_from_current_run(current_run: dict[str, Any] | None) -> Path | None:
    if not current_run:
        return None
    heartbeat_path = current_run.get("heartbeat_path")
    if heartbeat_path:
        candidate = Path(str(heartbeat_path))
        if candidate.exists():
            return candidate
    checkpoints_root = current_run.get("checkpoints_root")
    if not checkpoints_root:
        return None
    root = Path(str(checkpoints_root))
    if not root.exists():
        return None
    candidates = list(root.glob("fold_*/training_heartbeat.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _latest_heartbeat(checkpoints_dir: Path, *, current_run: dict[str, Any] | None = None) -> Path | None:
    current_run_heartbeat = _heartbeat_from_current_run(current_run)
    if current_run_heartbeat is not None:
        return current_run_heartbeat
    candidates = list(checkpoints_dir.glob("fold_*/training_heartbeat.json"))
    candidates.extend(checkpoints_dir.glob("run_*/fold_*/training_heartbeat.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_current_run_heartbeat(checkpoints_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    context = _load_current_run_context(checkpoints_dir)
    if context:
        return _latest_heartbeat(checkpoints_dir, current_run=context), context
    return _latest_heartbeat(checkpoints_dir, current_run=None), None


def _parse_timestamp_utc(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def summarize_heartbeat_schema(heartbeat: dict[str, Any] | None) -> dict[str, Any]:
    if not heartbeat:
        return {
            "schema_version": None,
            "schema_state": "missing",
            "freshness_state": "missing",
            "contamination_state": "missing",
            "diagnostic_sample_count": None,
            "n_updates": None,
            "last_distinct_update_seen": None,
            "metrics_fresh": None,
        }

    schema_version = heartbeat.get("schema_version")
    ppo = heartbeat.get("ppo_diagnostics", {}) or {}
    diagnostic_sample_count = ppo.get("diagnostic_sample_count", heartbeat.get("diagnostic_sample_count"))
    n_updates = ppo.get("n_updates", heartbeat.get("n_updates"))
    last_distinct_update_seen = ppo.get("last_distinct_update_seen")
    metrics_fresh = ppo.get("metrics_fresh")

    if schema_version == 2:
        schema_state = "v2"
    elif schema_version is None:
        schema_state = "missing"
    else:
        schema_state = f"legacy_{schema_version}"

    contamination_reasons: list[str] = []
    if schema_state != "v2":
      contamination_reasons.append("schema_mismatch")
    if metrics_fresh is False:
      contamination_reasons.append("stale_metrics")
    if diagnostic_sample_count in (None, 0):
      contamination_reasons.append("no_distinct_samples")

    if schema_state == "v2" and metrics_fresh is True and diagnostic_sample_count not in (None, 0):
        freshness_state = "fresh"
        contamination_state = "clean"
    elif schema_state == "missing":
        freshness_state = "unknown"
        contamination_state = "contaminated"
    elif metrics_fresh is False or diagnostic_sample_count in (None, 0):
        freshness_state = "stale"
        contamination_state = "contaminated"
    else:
        freshness_state = "mixed"
        contamination_state = "contaminated"

    return {
        "schema_version": schema_version,
        "schema_state": schema_state,
        "freshness_state": freshness_state,
        "contamination_state": contamination_state,
        "diagnostic_sample_count": diagnostic_sample_count,
        "n_updates": n_updates,
        "last_distinct_update_seen": last_distinct_update_seen,
        "metrics_fresh": metrics_fresh,
        "contamination_reasons": contamination_reasons,
        "timestamp_utc": _parse_timestamp_utc(heartbeat.get("timestamp_utc")),
        "num_timesteps": heartbeat.get("num_timesteps"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Show latest PPO training + deployment gate status.")
    parser.add_argument("--symbol", default="EURUSD", help="Symbol, e.g. EURUSD")
    parser.add_argument("--checkpoints", default="checkpoints", help="Checkpoints directory")
    args = parser.parse_args()

    symbol = str(args.symbol).strip().upper() or "EURUSD"
    checkpoints_dir = Path(args.checkpoints)
    paths = deployment_paths(symbol)

    heartbeat_path, run_context = resolve_current_run_heartbeat(checkpoints_dir)
    heartbeat = _maybe_load(heartbeat_path) if heartbeat_path is not None else None
    gate = _maybe_load(paths.gate_path)
    diagnostics = _maybe_load(paths.diagnostics_path)
    ops_attestation = _maybe_load(paths.ops_attestation_path)

    print("=" * 80)
    print(f"Training / Gate Status — {symbol}")
    print("=" * 80)

    if run_context is not None:
        print(f"Current run id      : {run_context.get('run_id')}")
        print(f"Current checkpoints : {run_context.get('checkpoints_root')}")
        print(f"Current symbol      : {run_context.get('symbol')}")

    if heartbeat_path is None:
        if run_context is not None:
            print("Heartbeat: (none yet for current run) — training may still be in startup/precompute.")
        else:
            print("Heartbeat: (none found) — training may not be running or heartbeat disabled.")
    else:
        print(f"Heartbeat: {heartbeat_path}")
        if heartbeat:
            schema = summarize_heartbeat_schema(heartbeat)
            ts = heartbeat.get("timestamp_utc")
            steps = heartbeat.get("num_timesteps")
            ppo = heartbeat.get("ppo_diagnostics", {}) or {}
            print(f"  timestamp_utc     : {ts}")
            print(f"  num_timesteps     : {steps}")
            print(f"  schema_version    : {schema.get('schema_version')}")
            print(f"  schema_state      : {schema.get('schema_state')}")
            print(f"  freshness_state   : {schema.get('freshness_state')}")
            print(f"  contamination     : {schema.get('contamination_state')}")
            print(f"  contamination_reasons: {schema.get('contamination_reasons')}")
            print(f"  diagnostic_sample_count: {schema.get('diagnostic_sample_count')}")
            print(f"  n_updates         : {schema.get('n_updates')}")
            print(f"  last_distinct_update_seen: {schema.get('last_distinct_update_seen')}")
            print(f"  metrics_fresh     : {schema.get('metrics_fresh')}")
            print(f"  explained_variance: {ppo.get('explained_variance')}")
            print(f"  approx_kl         : {ppo.get('approx_kl')}")
            print(f"  value_loss_stable : {ppo.get('value_loss_stable')}")
            print(f"  blockers          : {ppo.get('blockers')}")
        else:
            print("  (failed to parse heartbeat JSON)")

    if diagnostics is None:
        print(f"Training diagnostics: missing ({paths.diagnostics_path})")
    else:
        print(f"Training diagnostics: {paths.diagnostics_path}")
        print(f"  gate_passed: {diagnostics.get('gate_passed', diagnostics.get('passes_thresholds'))}")
        print(f"  blockers  : {diagnostics.get('blockers')}")

    if gate is None:
        print(f"Deployment gate: missing ({paths.gate_path})")
    else:
        print(f"Deployment gate: {paths.gate_path}")
        print(f"  approved_for_live: {gate.get('approved_for_live')}")
        blockers = gate.get("blockers") or []
        for blocker in blockers:
            print(f"  BLOCKER: {blocker}")

    if ops_attestation is None:
        print(f"Ops attestation: missing ({paths.ops_attestation_path})")
    else:
        print(f"Ops attestation: {paths.ops_attestation_path}")
        print(f"  shadow_days_completed        : {ops_attestation.get('shadow_days_completed')}")
        print(f"  execution_drift_ok           : {ops_attestation.get('execution_drift_ok')}")
        print(f"  position_reconciliation_ok   : {ops_attestation.get('position_reconciliation_ok')}")

    # Decision-oriented verdict (conservative)
    verdict = "Research more"
    if gate is not None and not bool(gate.get("approved_for_live", False)):
        verdict = "Do not deploy"
    print("-" * 80)
    print(f"Verdict: {verdict}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


