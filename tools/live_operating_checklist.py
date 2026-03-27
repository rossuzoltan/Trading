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
from typing import Any

from artifact_manifest import load_manifest
from ops_attestation_helper import (
    load_execution_audit_summary,
    validate_execution_audit_summary,
    validate_restart_drill_report,
)
from trading_config import deployment_paths, resolve_bar_construction_ticks_per_bar
from validation_metrics import load_json_report


def _item(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def build_operating_checklist(
    *,
    symbol: str,
    ticks_per_bar: int,
    model_dir: str | Path = "models",
    min_accepted_fills: int = 20,
    restart_drill_path: str | Path | None = None,
) -> dict[str, Any]:
    normalized_symbol = symbol.upper()
    model_root = Path(model_dir)
    paths = deployment_paths(normalized_symbol, model_dir=model_dir)
    manifest_path = model_root / f"artifact_manifest_{normalized_symbol}.json"
    drill_path = Path(restart_drill_path) if restart_drill_path is not None else model_root / f"restart_drill_{normalized_symbol.lower()}.json"
    kill_switch_path = Path(os.environ.get("LIVE_KILL_SWITCH_PATH", "live.kill"))

    items: list[dict[str, Any]] = []
    blockers: list[str] = []

    gate = load_json_report(paths.gate_path) if paths.gate_path.exists() else None
    gate_ok = bool(gate and gate.get("approved_for_live", False))
    items.append(
        _item(
            "Deployment gate",
            gate_ok,
            "Approved for live trading." if gate_ok else f"Missing or blocked: {paths.gate_path}",
        )
    )
    if not gate_ok:
        blockers.append("Deployment gate is missing or not approved.")

    preflight = load_json_report(paths.live_preflight_path) if paths.live_preflight_path.exists() else None
    preflight_ok = bool(preflight and preflight.get("approved_for_live_runtime", False))
    items.append(
        _item(
            "Live preflight",
            preflight_ok,
            "Runtime preflight is clean." if preflight_ok else f"Missing or blocked: {paths.live_preflight_path}",
        )
    )
    if not preflight_ok:
        blockers.append("Live preflight is missing or not approved.")

    account_mode_ok = bool(preflight and preflight.get("account_mode_supported", False))
    if preflight is None:
        account_mode_detail = f"Missing preflight account mode evidence: {paths.live_preflight_path}"
    elif account_mode_ok:
        account_mode_detail = "Preflight confirms a netting account."
    else:
        account_mode_detail = "Preflight does not confirm a deployable netting account."
    items.append(_item("Account mode", account_mode_ok, account_mode_detail))
    if not account_mode_ok:
        blockers.append("Account mode is not deployable; a netting account is required.")

    manifest_ok = False
    manifest_detail = f"Missing manifest: {manifest_path}"
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        manifest_ticks = manifest.bar_construction_ticks_per_bar or manifest.ticks_per_bar
        manifest_ok = manifest_ticks is not None and int(manifest_ticks) == int(ticks_per_bar)
        if manifest_ticks is None:
            manifest_detail = "Manifest does not declare bar_construction_ticks_per_bar."
        elif int(manifest_ticks) != int(ticks_per_bar):
            manifest_detail = (
                f"Manifest bar_construction_ticks_per_bar={manifest_ticks} "
                f"differs from live bar_construction_ticks_per_bar={ticks_per_bar}."
            )
        else:
            manifest_detail = f"bar_construction_ticks_per_bar matches {ticks_per_bar}."
    items.append(_item("Artifact manifest parity", manifest_ok, manifest_detail))
    if not manifest_ok:
        blockers.append("Artifact manifest parity is not proven.")

    audit_summary = load_execution_audit_summary(paths.execution_audit_path)
    audit_blockers = validate_execution_audit_summary(audit_summary, min_accepted_fills=min_accepted_fills)
    audit_ok = not audit_blockers
    items.append(
        _item(
            "Execution audit",
            audit_ok,
            f"Accepted fills={audit_summary.get('accepted_count', 0)}; summary written at {paths.execution_audit_path}.",
        )
    )
    blockers.extend(audit_blockers)

    restart_report = load_json_report(drill_path) if drill_path.exists() else None
    restart_blockers = validate_restart_drill_report(restart_report)
    restart_ok = not restart_blockers
    items.append(
        _item(
            "Restart drill",
            restart_ok,
            "Restart evidence is present." if restart_ok else f"Missing or invalid restart evidence: {drill_path}",
        )
    )
    blockers.extend(restart_blockers)

    attestation = load_json_report(paths.ops_attestation_path) if paths.ops_attestation_path.exists() else None
    attestation_ok = bool(attestation and attestation.get("approved", False))
    items.append(
        _item(
            "Ops attestation",
            attestation_ok,
            "Evidence-backed attestation is present." if attestation_ok else f"Missing or blocked: {paths.ops_attestation_path}",
        )
    )
    if not attestation_ok:
        blockers.append("Ops attestation is missing or blocked.")

    kill_switch_ok = not kill_switch_path.exists()
    items.append(
        _item(
            "Manual kill switch",
            kill_switch_ok,
            "No kill-switch file present." if kill_switch_ok else f"Kill switch exists: {kill_switch_path}",
        )
    )
    if not kill_switch_ok:
        blockers.append(f"Manual kill switch exists: {kill_switch_path}")

    return {
        "symbol": normalized_symbol,
        "bar_construction_ticks_per_bar": int(ticks_per_bar),
        "ticks_per_bar": int(ticks_per_bar),
        "approved_for_live": not blockers,
        "blockers": blockers,
        "items": items,
        "evidence_paths": {
            "gate": str(paths.gate_path),
            "preflight": str(paths.live_preflight_path),
            "manifest": str(manifest_path),
            "restart_drill": str(drill_path),
            "ops_attestation": str(paths.ops_attestation_path),
            "execution_audit": str(paths.execution_audit_path),
        },
        "execution_audit_summary": audit_summary,
        "restart_drill": restart_report,
        "ops_attestation": attestation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Print an operator-facing live trading checklist.")
    parser.add_argument("--symbol", default=os.environ.get("TRADING_SYMBOL", "EURUSD"))
    parser.add_argument(
        "--ticks-per-bar",
        type=int,
        default=resolve_bar_construction_ticks_per_bar("BAR_SPEC_TICKS_PER_BAR", "TRADING_TICKS_PER_BAR"),
    )
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--min-accepted-fills", type=int, default=int(os.environ.get("LIVE_MIN_AUDIT_FILLS", "20")))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    checklist = build_operating_checklist(
        symbol=args.symbol,
        ticks_per_bar=args.ticks_per_bar,
        model_dir=args.model_dir,
        min_accepted_fills=args.min_accepted_fills,
    )
    if args.json:
        print(json.dumps(checklist, indent=2))
    else:
        print("=" * 80)
        print(f"Operating Checklist - {checklist['symbol']}")
        print("=" * 80)
        for item in checklist["items"]:
            status = "PASS" if item["ok"] else "FAIL"
            print(f"{status}: {item['name']} — {item['detail']}")
        print("Verdict: " + ("READY" if checklist["approved_for_live"] else "NOT READY"))
    return 0 if checklist["approved_for_live"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


