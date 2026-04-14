# Next Agent Context

## Current Goal
- Continue the Bot v1 RC1 hardening path, not the older PPO investigation track.
- Current focus:
  - RC1 artifact generation and certification for the approved paper-live scope
  - Hybrid Gated-Rule (Meta-labeling) search for rule candidates (AlphaGate integration)
  - paper-live profitability gating using shadow evidence, restart evidence, preflight, and ops attestation together once a certified anchor exists

## Canonical Operational Plan
- `docs/PROFITABILITY_PLAN.md` is the canonical operator plan.
- `roadmap.md` is historical context, not the daily source of truth.

## Approved RC1 Scope
- `EURUSD` at `5000` ticks/bar remains approved scope, but the latest RC1 pack is not yet test-ready.
- `GBPUSD` at `10000` ticks/bar remains approved scope, but the latest RC1 pack is not yet test-ready.
- `USDJPY` is explicitly demoted back to challenger status.

## Current Architecture
- `strategies/rule_logic.py` is the single source of truth for deterministic entry logic.
- The current `mean_reversion` logic is price-based (`price_z`) with spread/slope guards, not the older spread-direction proxy.
- **Architecture Pivot (2026-04-10):** Integrating ML-based Meta-labeling (AlphaGate) atop deterministic rules to filter low-probability entries.
- `rule_selector.py` is the manifest-driven rule consumer and gate enforcer.
- `edge_research.py` contains the `BaselineAlphaGate` training logic (`logistic_pair`, `xgboost_pair`, `lightgbm_pair`, `ridge_signed_target`).
- `selector_manifest.py` is the RC contract for both rule manifests and supervised selector artifacts.
- `tools/generate_v1_rc.py` builds the RC1 artifact packs.
- `tools/verify_v1_rc.py` certifies parity, baseline comparisons, and truth-engine drift.
- `tools/pre_test_gate.py` is the fail-fast operator gate before new shadow runs.
- `tools/alpha_gate_bakeoff.py` compares `rule_only`, manifest gate, and AlphaGate challengers exact-runtime on the current holdout.
- `tools/optimize_rules.py` is the research harness, now supporting `AlphaGate` hybrid sweeps.
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
- Run fail-fast pre-test gate:
  - `.\.venv\Scripts\python.exe .\tools\pre_test_gate.py --manifest-path .\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json`
- Run exact-runtime AlphaGate bakeoff:
  - `.\.venv\Scripts\python.exe .\tools\alpha_gate_bakeoff.py --manifest-path .\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json`
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
- Every active anchor shadow run should write:
  - `artifacts/shadow/<SYMBOL>/<MANIFEST_HASH>/events.jsonl`
  - `artifacts/shadow/<SYMBOL>/<MANIFEST_HASH>/shadow_summary.json`
  - `artifacts/shadow/<SYMBOL>/<MANIFEST_HASH>/shadow_summary.md`
- Every active anchor should have a paper-live gate verdict under:
  - `artifacts/gates/<SYMBOL>/<MANIFEST_HASH>/paper_live_gate.json`
  - `artifacts/gates/<SYMBOL>/<MANIFEST_HASH>/paper_live_gate.md`
- `tools/verify_v1_rc.py` must fail if `evaluate_oos.py` or `strategies/rule_logic.py` drift from the manifest hashes.
- Latest verified RC1 regeneration on `2026-04-10` produced structurally valid packs with strict manifest/component hashes:
  - `eurusd_5k_v1_mr_rc1`: net `+$1.89`, `6` trades, `PF 1.186`
  - `gbpusd_10k_v1_mr_rc1`: net `+$5.14`, `4` trades, `PF 9.766`
- Latest pre-test gate on `2026-04-10` blocks both anchors:
  - `EURUSD`: thin replay (`6` trades), stale historical replay hash evidence, `DRIFT_CRITICAL`, Asia opens `1`, Rollover opens `1`, density ratio `2.993x`
  - `GBPUSD`: thin replay (`4` trades), stale historical replay hash evidence, `DRIFT_CRITICAL`, Rollover opens `1`, density ratio `4.454x`
- Latest exact-runtime EURUSD bakeoff on `2026-04-10`:
  - `rule_only`: best current holdout result, net `+$39.47`, `110` trades, `PF 1.172`
  - `manifest_gate`: net `+$24.84`, `44` trades, `PF 1.256`
  - `xgboost_pair`: best refit AlphaGate challenger, net `+$12.57`, `14` trades, `PF 1.483`
  - `lightgbm_pair`: net `+$5.86`, `35` trades, `PF 1.062`
  - `logistic_pair`: collapsed to `0` trades on current holdout
- **Operational verdict**: do not start a new shadow campaign until MT5 historical replay is regenerated against the current manifest hashes and replay density improves.

## Guardrails
- Do not re-promote `USDJPY` into RC1 without new evidence.
- Do not relax `live_trading_approved: false` in the RC manifest path.
- Do not treat static release notes as truth when the generated scoreboards disagree; the generated scoreboards are the certification artifacts.
- Do not treat a positive replay as enough; the anchor path is `candidate` until the paper-live gate is satisfied.
- Do not trust historical replay artifacts whose `manifest_hash`, `logic_hash`, or `evaluator_hash` do not match the current RC manifest.
