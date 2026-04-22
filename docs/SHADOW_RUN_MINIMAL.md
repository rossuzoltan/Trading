# Shadow Run Minimal Operator Guide

## Goal

Produce real shadow evidence for the active RC1 symbol.

## Current approved symbol

- EURUSD only

## Minimum working sequence

1. Verify RC1 certification:
   - `..venv\Scripts\python.exe .\tools\verify_v1_rc.py`
2. Build or refresh the pre-test gate:
   - `..venv\Scripts\python.exe .\tools\pre_test_gate.py --manifest-path .\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json`
3. Start the shadow loop using the RC1 manifest:
   - `..venv\Scripts\python.exe .\runtime\shadow_broker.py --manifest-path .\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json`
4. Let it accumulate enough evidence:
   - at least 20 trading days
   - at least 30 actionable events
5. Rebuild the gate artifacts:
   - `..venv\Scripts\python.exe .\tools\paper_live_gate.py --manifest-path .\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json`

## Notes

- Shadow evidence is only meaningful if events actually accumulate in `artifacts/shadow/EURUSD/<manifest_hash>/events.jsonl`.
- Placeholder restart/preflight artifacts are not live-ready evidence.
- If no events appear, the problem is the runtime path, not the summary writer.
