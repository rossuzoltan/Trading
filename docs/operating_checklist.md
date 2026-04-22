# Operating Checklist

Use this checklist after RC1 certification, a restart drill, a shadow session,
and a paper-live gate build.

## Evidence order

1. Re-run RC1 certification if the manifest, evaluator, or rule logic changed:
   - `.\.venv\Scripts\python.exe .\tools\verify_v1_rc.py`
2. Run the restart drill with fake MT5:
   - `.\.venv\Scripts\python.exe .\tools\restart_drill.py --symbol EURUSD --state-path .\models\live_state_eurusd.json --report-path .\models\restart_drill_eurusd.json`
3. Build the paper-live gate verdict:
   - `.\.venv\Scripts\python.exe .\tools\paper_live_gate.py --manifest-path .\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json`
4. Build the ops attestation from the execution audit and restart evidence:
   - `.\.venv\Scripts\python.exe .\tools\ops_attestation_helper.py --symbol EURUSD --attested-by "ops" --shadow-days-completed 14 --notes "shadow run evidence"`
5. Print the operating checklist:
   - `.\.venv\Scripts\python.exe .\tools\live_operating_checklist.py --symbol EURUSD`

## What the checklist enforces

- deployment gate approval
  - includes replay-embedded runtime parity and slippage-stress blockers
- paper-live profitability gate verdict
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
The canonical gate definitions live in `docs/PROFITABILITY_PLAN.md`.
