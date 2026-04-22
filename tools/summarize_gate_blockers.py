from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    gates_root = ROOT / "artifacts" / "gates" / "EURUSD"
    models_root = ROOT / "models"
    print("RC1 Blocker Summary\n===================\n")
    for gate_path in sorted(gates_root.glob("*/paper_live_gate.json")):
        payload = _load(gate_path)
        print(f"Gate: {gate_path.parent.name}")
        print(f"  Final verdict: {payload.get('final_verdict')}")
        print(f"  Reason: {payload.get('verdict_reason')}")
        restart = dict(payload.get("restart_status", {}) or {})
        preflight = dict(payload.get("preflight_status", {}) or {})
        ops = dict(payload.get("ops_attestation_status", {}) or {})
        hist = dict(payload.get("historical_replay_status", {}) or {})
        drift = dict(payload.get("drift_metrics", {}) or {})
        shadow = dict(payload.get("shadow_summary_stats", {}) or {})
        print(f"  Restart: present={restart.get('present')} ok={restart.get('ok')} path={restart.get('path')}")
        print(f"  Preflight: present={preflight.get('present')} ok={preflight.get('ok')} path={preflight.get('path')}")
        print(f"  Ops: present={ops.get('present')} ok={ops.get('ok')} path={ops.get('path')}")
        print(f"  Historical replay: ok={hist.get('ok')} verdict={hist.get('overall_verdict')}")
        print(f"  Drift verdict: {drift.get('verdict')} failures={drift.get('critical_failures') or drift.get('normal_failures')}")
        print(f"  Shadow evidence: days={shadow.get('trading_days')} events={shadow.get('event_count')} actionable={shadow.get('actionable_event_count')} sufficient={shadow.get('evidence_sufficient')}")
        print()
    print("Key artifacts:")
    for path in [
        models_root / "restart_drill_eurusd.json",
        models_root / "live_preflight_eurusd.json",
        models_root / "ops_attestation_eurusd.json",
        models_root / "rc1" / "eurusd_5k_v1_mr_rc1" / "pre_test_gate.json",
        models_root / "rc1" / "eurusd_5k_v1_mr_rc1" / "mt5_historical_replay_report.json",
    ]:
        print(f"- {path.relative_to(ROOT)}: {'present' if path.exists() else 'missing'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
