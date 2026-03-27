# Operating Checklist

Use this checklist after a training run, restart drill, and shadow session.

## Evidence order

1. Run the restart drill with fake MT5:
   - `.\.venv\Scripts\python.exe .\restart_drill.py --symbol EURUSD --state-path .\models\live_state_eurusd.json --report-path .\models\restart_drill_eurusd.json`
2. Build the ops attestation from the execution audit and restart evidence:
   - `.\.venv\Scripts\python.exe .\ops_attestation_helper.py --symbol EURUSD --attested-by "ops" --shadow-days-completed 14 --notes "shadow run evidence"`
3. Print the operating checklist:
   - `.\.venv\Scripts\python.exe .\live_operating_checklist.py --symbol EURUSD --ticks-per-bar 5000`

## What the checklist enforces

- deployment gate approval
- live preflight approval
- artifact manifest parity for `ticks_per_bar`
- execution audit sample count
- restart drill evidence
- ops attestation approval
- no manual kill-switch file

## Interpretation

- **READY** means the current evidence set is complete enough for the operator gate.
- **NOT READY** means at least one blocker still exists and deployment should remain closed.

This checklist is evidence-driven; it does not change live execution semantics.
