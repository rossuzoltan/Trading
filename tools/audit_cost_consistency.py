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
    manifest_cost_model = dict(getattr(manifest, 'cost_model', None) or {})
    payload = {
        'manifest_path': str(manifest_path),
        'manifest_hash': manifest.manifest_hash,
        'manifest_cost_model': manifest_cost_model,
        'resolved_runtime_cost_profile': _resolve_execution_cost_profile(manifest),
        'threshold_policy': dict(getattr(manifest, 'threshold_policy', None) or {}),
        'alpha_gate': dict(getattr(manifest, 'alpha_gate', None) or {}),
    }

    resolved = dict(payload['resolved_runtime_cost_profile'] or {})
    missing_keys = sorted([k for k in resolved.keys() if k not in manifest_cost_model])
    extra_keys = sorted([k for k in manifest_cost_model.keys() if k not in resolved])
    mismatched_keys: list[str] = []
    for k, v in manifest_cost_model.items():
        if k not in resolved:
            continue
        try:
            if float(v) != float(resolved[k]):
                mismatched_keys.append(k)
        except Exception:
            if v != resolved[k]:
                mismatched_keys.append(k)

    implicit_default_expected = {
        'commission_per_lot': 7.0,
        'slippage_pips': 0.25,
        'partial_fill_ratio': 1.0,
    }
    implicit_defaults_ok = all(
        (k in implicit_default_expected and float(resolved.get(k)) == float(implicit_default_expected[k]))
        for k in missing_keys
    )

    payload['implicit_defaults'] = {
        'missing_keys': missing_keys,
        'extra_keys': extra_keys,
        'mismatched_keys': sorted(mismatched_keys),
        'expected_defaults': implicit_default_expected,
        'implicit_defaults_ok': bool(implicit_defaults_ok),
    }

    payload['ok'] = (len(mismatched_keys) == 0) and (len(extra_keys) == 0) and implicit_defaults_ok
    payload['severity'] = 'OK' if payload['ok'] else 'FAIL'
    if payload['ok'] and missing_keys:
        payload['severity'] = 'WARN'
        payload['warning'] = (
            'Manifest cost_model omits keys that are defaulted at runtime. '
            'This is not strategy-behavior drift, but it is an audit/reporting default.'
        )
    print(json.dumps(payload, indent=2))
    return 0 if payload['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
