from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from selector_manifest import compute_selector_manifest_hash, load_selector_manifest, manifest_to_payload


PROFILE_SET_GATE_SWEEP_V1: tuple[dict[str, Any], ...] = (
    {
        "id": "moderate",
        "description": "Small relaxation to test whether mild slope/spread loosening is enough.",
        "manifest_filename": "manifest.shadow_moderate.json",
        "rule_params": {
            "max_spread_z": 0.85,
            "max_abs_ma20_slope": 0.25,
            "max_abs_ma50_slope": 0.10,
        },
    },
    {
        "id": "slope_focus",
        "description": "Relax only the slope filters to isolate whether trend gates are the main blocker.",
        "manifest_filename": "manifest.shadow_slope_focus.json",
        "rule_params": {
            "max_spread_z": 0.50,
            "max_abs_ma20_slope": 0.35,
            "max_abs_ma50_slope": 0.15,
        },
    },
    {
        "id": "aggressive",
        "description": "Upper-bound relaxation to measure signal density once the gates are opened wider.",
        "manifest_filename": "manifest.shadow_aggressive.json",
        "rule_params": {
            "max_spread_z": 1.75,
            "max_abs_ma20_slope": 0.45,
            "max_abs_ma50_slope": 0.20,
        },
    },
)


def _merge_rule_params(base_payload: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    payload = manifest_to_payload(base_payload, include_manifest_hash=True)
    merged_rule_params = dict(payload.get("rule_params") or {})
    merged_rule_params.update(overrides)
    payload["rule_params"] = merged_rule_params

    startup_snapshot = dict(payload.get("startup_truth_snapshot") or {})
    startup_rule_params = dict(startup_snapshot.get("rule_params") or {})
    startup_rule_params.update(overrides)
    startup_snapshot["rule_params"] = startup_rule_params
    payload["startup_truth_snapshot"] = startup_snapshot

    payload["manifest_hash"] = compute_selector_manifest_hash(payload)
    return payload


def build_sweep_manifests(*, base_manifest_path: Path) -> dict[str, Any]:
    raw_payload = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    base_manifest = load_selector_manifest(
        base_manifest_path,
        verify_manifest_hash=True,
        strict_manifest_hash=True,
        require_component_hashes=True,
    )

    generated: list[dict[str, Any]] = []
    for profile in PROFILE_SET_GATE_SWEEP_V1:
        payload = _merge_rule_params(raw_payload, profile["rule_params"])
        manifest_path = base_manifest_path.parent / profile["manifest_filename"]
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        load_selector_manifest(
            manifest_path,
            verify_manifest_hash=True,
            strict_manifest_hash=True,
            require_component_hashes=True,
        )
        generated.append(
            {
                "profile_id": profile["id"],
                "description": profile["description"],
                "manifest_path": str(manifest_path),
                "manifest_hash": str(payload["manifest_hash"]),
                "symbol": str(base_manifest.strategy_symbol),
                "rule_params": payload["rule_params"],
            }
        )

    return {
        "base_manifest_path": str(base_manifest_path),
        "base_manifest_hash": str(base_manifest.manifest_hash),
        "symbol": str(base_manifest.strategy_symbol),
        "generated": generated,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create shadow sweep manifest variants from a base selector manifest.")
    parser.add_argument("--base-manifest", "--manifest", dest="base_manifest", required=True)
    args = parser.parse_args()

    payload = build_sweep_manifests(base_manifest_path=Path(args.base_manifest).resolve())
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
