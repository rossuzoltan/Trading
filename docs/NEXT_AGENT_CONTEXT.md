# Next Agent Context

## Current Goal
- Continue the Bot v1 RC1 hardening path, not the older PPO investigation track.
- Current focus:
  - RC1 artifact generation and certification for the approved paper-live candidates
  - shadow-mode execution tracing for the rule-first selector path

## Approved RC1 Scope
- `EURUSD` at `5000` ticks/bar is approved as an RC1 paper-live candidate.
- `GBPUSD` at `10000` ticks/bar is approved as an RC1 paper-live candidate.
- `USDJPY` is explicitly demoted back to challenger status.

## Current Architecture
- `strategies/rule_logic.py` is the single source of truth for deterministic entry logic.
- `rule_selector.py` is the manifest-driven rule consumer and gate enforcer.
- `selector_manifest.py` is the RC contract for both rule manifests and supervised selector artifacts.
- `tools/generate_v1_rc.py` builds the RC1 artifact packs.
- `tools/verify_v1_rc.py` certifies parity, baseline comparisons, and truth-engine drift.
- `runtime/shadow_broker.py` is the shadow-mode adapter for `RuleSelector` decisions.

## RC1 Safety Contract
- RC1 manifests are Version `4`.
- Mandatory RC1 traceability fields:
  - `release_stage = "paper_live_candidate"`
  - `live_trading_approved = false`
  - `evaluator_hash`
  - `logic_hash`
  - `manifest_hash`
- Do not treat RC1 artifacts as live-trading approved.

## Verified Commands
- Generate and certify RC1 packs:
  - `.\.venv\Scripts\python.exe .\tools\generate_v1_rc.py`
- Re-run RC1 certification only:
  - `.\.venv\Scripts\python.exe .\tools\verify_v1_rc.py`
- Run targeted RC/shadow tests:
  - `.\.venv\Scripts\python.exe -m unittest tests.test_rc1_shadow`
- Run the shadow simulator:
  - `.\tools\run_shadow_simulator.ps1 -ManifestPath models\rc1\eurusd_5k_v1_mr_rc1\manifest.json`

## Current Expectations
- Every RC1 pack should contain:
  - `manifest.json`
  - `release_notes_rc1.md`
  - `baseline_scoreboard_rc1.json`
  - `baseline_scoreboard_rc1.md`
- `tools/verify_v1_rc.py` must fail if `evaluate_oos.py` or `strategies/rule_logic.py` drift from the manifest hashes.
- Latest verified RC1 regeneration succeeded:
  - `eurusd_5k_v1_mr_rc1`: net `+$133.42`, `27` trades
  - `gbpusd_10k_v1_mr_rc1`: net `+$111.81`, `21` trades

## Guardrails
- Do not re-promote `USDJPY` into RC1 without new evidence.
- Do not relax `live_trading_approved: false` in the RC manifest path.
- Do not treat static release notes as truth when the generated scoreboards disagree; the generated scoreboards are the certification artifacts.
