from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from selector_manifest import load_selector_manifest
from live_bridge import _resolve_execution_cost_profile


def main() -> int:
    manifest_path = ROOT / 'models' / 'rc1' / 'eurusd_5k_v1_mr_rc1' / 'manifest.json'
    if not manifest_path.exists():
        print(json.dumps({'ok': False, 'reason': f'missing manifest: {manifest_path}'}))
        return 1
    manifest = load_selector_manifest(manifest_path, verify_manifest_hash=True, strict_manifest_hash=True)
    payload = {
        'manifest_path': str(manifest_path),
        'manifest_hash': manifest.manifest_hash,
        'manifest_cost_model': dict(getattr(manifest, 'cost_model', None) or {}),
        'resolved_runtime_cost_profile': _resolve_execution_cost_profile(manifest),
        'threshold_policy': dict(getattr(manifest, 'threshold_policy', None) or {}),
        'alpha_gate': dict(getattr(manifest, 'alpha_gate', None) or {}),
    }
    payload['ok'] = payload['manifest_cost_model'] == payload['resolved_runtime_cost_profile']
    print(json.dumps(payload, indent=2))
    return 0 if payload['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
