from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_live_metrics import resolve_paper_live_gate_paths
from selector_manifest import load_selector_manifest, validate_paper_live_candidate_manifest
from trading_config import deployment_paths, resolve_bar_construction_ticks_per_bar


def build_rc1_preflight(*, manifest_path: str | Path, output_path: str | Path | None = None) -> dict:
    manifest = load_selector_manifest(manifest_path, verify_manifest_hash=True)
    validate_paper_live_candidate_manifest(manifest)
    symbol = manifest.strategy_symbol.upper()
    ticks_per_bar = int(manifest.ticks_per_bar or manifest.bar_construction_ticks_per_bar or 0)
    paths = deployment_paths(symbol)
    gate_paths = resolve_paper_live_gate_paths(symbol=symbol, manifest_hash=manifest.manifest_hash)

    payload = {
        "symbol": symbol,
        "manifest_path": str(Path(manifest_path)),
        "manifest_hash": manifest.manifest_hash,
        "ticks_per_bar": ticks_per_bar,
        "approved_for_live_runtime": False,
        "account_mode_supported": False,
        "notes": [
            "RC1 preflight wrapper for offline evidence chain",
            "Account mode support is intentionally false until a real preflight validates MT5 netting mode and deployment gate parity.",
        ],
        "blockers": [
            "This is a placeholder preflight wrapper; replace with a live MT5 preflight when available.",
        ],
        "paths": {
            "deployment_gate": str(paths.gate_path),
            "ops_attestation": str(paths.ops_attestation_path),
            "restart_drill": str(Path('models') / f'restart_drill_{symbol.lower()}.json'),
            "paper_live_gate": str(gate_paths.json_path),
        },
    }

    out_path = Path(output_path) if output_path is not None else paths.live_preflight_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an RC1-compatible preflight placeholder.")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()
    payload = build_rc1_preflight(manifest_path=args.manifest_path, output_path=args.output_path)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
