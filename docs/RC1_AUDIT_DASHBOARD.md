# RC1 Audit Dashboard

## EURUSD RC1

- Manifest: `models/rc1/eurusd_5k_v1_mr_rc1/manifest.json`
- Replay: `models/rc1/eurusd_5k_v1_mr_rc1/mt5_historical_replay_report.json`
- Pre-test gate: `models/rc1/eurusd_5k_v1_mr_rc1/pre_test_gate.json`
- Shadow evidence: `artifacts/shadow/EURUSD/<manifest_hash>/shadow_summary.json`
- Paper-live gate: `artifacts/gates/EURUSD/<manifest_hash>/paper_live_gate.json`
- Ops attestation: `models/ops_attestation_eurusd.json`
- Restart drill: `models/restart_drill_eurusd.json`
- Preflight: `models/live_preflight_eurusd.json`

## Current truth

- The system is **audit-first**, not live-money-ready.
- EURUSD remains the only acceptable focus.
- The gating chain is designed to reject optimism until evidence is real.

## What to watch

- Shadow days completed vs required
- Actionable events vs threshold
- Mean absolute fill drift vs 0.5 pips
- Restart drill evidence acceptance
- Preflight status acceptance
- Paper-live verdict

## When to move forward

Only after all of the following are true:
- shadow validation reaches the threshold
- execution drift is within bounds
- restart drill is accepted
- preflight is accepted
- paper-live gate is not demoted
