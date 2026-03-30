# Phase 1 Evaluation Accounting Fix

## Root cause

The full-path training evaluation path was mixing two different sources:

1. Top-level replay economics were computed from a trade log extracted via private wrapped runtime access (`_runtime`).
2. Nested `execution_diagnostics` came from `get_training_diagnostics()`, which is cumulative across repeated eval episodes when the eval env is reused.

When wrapped env access failed, the extracted trade log was empty, so top-level fields such as `trade_count`, `gross_pnl_usd`, `net_pnl_usd`, and `profit_factor` serialized as zero. At the same time, `execution_diagnostics` still showed real trades and costs because those counters were coming from the env diagnostics object, and in the full-path callback they could also be polluted by prior eval episodes.

## What changed

- `train_agent.evaluate_model()` now uses the completed episode audit as the primary source for:
  - `trade_log`
  - `execution_log`
  - episode-scoped diagnostics
- Both training full-path eval and `evaluate_oos.py` now build economics through the same `runtime_common.build_evaluation_accounting()` helper.
- Evaluation payloads are validated before being written:
  - `full_path_evaluations.json`
  - heartbeat `latest_eval`
  - `replay_report_<symbol>.json`
- Legacy invalid evaluation entries are skipped when resuming history so a new write does not silently preserve known-bad accounting rows.

## Why the new accounting is more trustworthy

- The trade log and diagnostics now come from the same completed episode boundary.
- Top-level economics are derived from closed trades, not from reward shaping.
- The writer asserts reconciliation between:
  - trade log counts
  - diagnostic counts
  - execution log counts
  - economic totals
- If a payload tries to report zero trades while diagnostics or costs imply otherwise, validation fails instead of silently writing a misleading artifact.
