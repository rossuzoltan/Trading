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


PROFILE_SET_EVIDENCE_LADDER_V1: tuple[dict[str, Any], ...] = (
    {
        "id": "p01_guarded_core",
        "description": "Median live candidate cluster; minimum slope relaxation needed to stop rejecting the observed core setup shape.",
        "manifest_filename": "manifest.shadow_p01_guarded_core.json",
        "rule_params": {
            "max_spread_z": 0.50,
            "max_abs_ma20_slope": 0.30,
            "max_abs_ma50_slope": 0.11,
        },
    },
    {
        "id": "p02_guarded_plus",
        "description": "Hold slopes near the live median but lift spread_z toward the observed live upper quartile (~0.69).",
        "manifest_filename": "manifest.shadow_p02_guarded_plus.json",
        "rule_params": {
            "max_spread_z": 0.70,
            "max_abs_ma20_slope": 0.30,
            "max_abs_ma50_slope": 0.11,
        },
    },
    {
        "id": "p03_centerline",
        "description": "Centered around the observed live p75 envelope for spread and slope without opening the gates aggressively.",
        "manifest_filename": "manifest.shadow_p03_centerline.json",
        "rule_params": {
            "max_spread_z": 0.85,
            "max_abs_ma20_slope": 0.31,
            "max_abs_ma50_slope": 0.12,
        },
    },
    {
        "id": "p04_spread_step",
        "description": "Spread-relaxed step to test whether live under-signaling is still spread-gated once slopes are set near the observed cluster.",
        "manifest_filename": "manifest.shadow_p04_spread_step.json",
        "rule_params": {
            "max_spread_z": 1.00,
            "max_abs_ma20_slope": 0.31,
            "max_abs_ma50_slope": 0.12,
        },
    },
    {
        "id": "p05_slope_step",
        "description": "Slope-relaxed step to isolate trend-guard pressure while keeping spread close to the stricter live median band.",
        "manifest_filename": "manifest.shadow_p05_slope_step.json",
        "rule_params": {
            "max_spread_z": 0.55,
            "max_abs_ma20_slope": 0.34,
            "max_abs_ma50_slope": 0.14,
        },
    },
    {
        "id": "p06_balanced_relaxed",
        "description": "Balanced relaxation near the observed live maxima; intended as a practical mid-ladder comparison point.",
        "manifest_filename": "manifest.shadow_p06_balanced_relaxed.json",
        "rule_params": {
            "max_spread_z": 0.85,
            "max_abs_ma20_slope": 0.35,
            "max_abs_ma50_slope": 0.15,
        },
    },
    {
        "id": "p07_upper_guardrail",
        "description": "Upper guardrail profile that still stays below the earlier aggressive sweep while covering rare higher-slope live bars.",
        "manifest_filename": "manifest.shadow_p07_upper_guardrail.json",
        "rule_params": {
            "max_spread_z": 1.10,
            "max_abs_ma20_slope": 0.38,
            "max_abs_ma50_slope": 0.17,
        },
    },
    {
        "id": "p08_exploratory_ceiling",
        "description": "Ceiling profile for single-process sweep only; broad enough to estimate the top-end signal density without becoming fully unbounded.",
        "manifest_filename": "manifest.shadow_p08_exploratory_ceiling.json",
        "rule_params": {
            "max_spread_z": 1.35,
            "max_abs_ma20_slope": 0.42,
            "max_abs_ma50_slope": 0.19,
        },
    },
)


PROFILE_SETS: dict[str, tuple[dict[str, Any], ...]] = {
    "gate_sweep_v1": PROFILE_SET_GATE_SWEEP_V1,
    "evidence_ladder_v1": PROFILE_SET_EVIDENCE_LADDER_V1,
}


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


def build_sweep_manifests(*, base_manifest_path: Path, profile_set_name: str) -> dict[str, Any]:
    raw_payload = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    base_manifest = load_selector_manifest(
        base_manifest_path,
        verify_manifest_hash=True,
        strict_manifest_hash=True,
        require_component_hashes=True,
    )
    profile_set = PROFILE_SETS[profile_set_name]

    generated: list[dict[str, Any]] = []
    for profile in profile_set:
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
        "profile_set": profile_set_name,
        "symbol": str(base_manifest.strategy_symbol),
        "generated": generated,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create shadow sweep manifest variants from a base selector manifest.")
    parser.add_argument("--base-manifest", "--manifest", dest="base_manifest", required=True)
    parser.add_argument(
        "--profile-set",
        choices=sorted(PROFILE_SETS.keys()),
        default="gate_sweep_v1",
        help="Named profile set to materialize.",
    )
    args = parser.parse_args()

    payload = build_sweep_manifests(
        base_manifest_path=Path(args.base_manifest).resolve(),
        profile_set_name=str(args.profile_set),
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
