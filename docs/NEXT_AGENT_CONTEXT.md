# Next Agent Context

## Current Goal
- Continue the Bot v1 RC1 hardening path, not the older PPO investigation track.
- Current focus:
  - RC1 artifact generation and certification for the approved paper-live scope
  - historical replay and challenger search for rule-first selector variants that survive certification
  - paper-live profitability gating using shadow evidence, restart evidence, preflight, and ops attestation together once a certified anchor exists

## Canonical Operational Plan
- `docs/PROFITABILITY_PLAN.md` is the canonical operator plan.
- `roadmap.md` is historical context, not the daily source of truth.

## Approved RC1 Scope
- `EURUSD` at `5000` ticks/bar remains approved scope, but the latest RC1 pack is currently `demoted`.
- `GBPUSD` at `10000` ticks/bar remains approved scope, but the latest RC1 pack is currently `demoted`.
- `USDJPY` is explicitly demoted back to challenger status.

## Current Architecture
- `strategies/rule_logic.py` is the single source of truth for deterministic entry logic.
- The current `mean_reversion` logic is price-based (`price_z`) with spread/slope guards, not the older spread-direction proxy.
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
- Every active anchor shadow run should write:
  - `artifacts/shadow/<SYMBOL>/<MANIFEST_HASH>/events.jsonl`
  - `artifacts/shadow/<SYMBOL>/<MANIFEST_HASH>/shadow_summary.json`
  - `artifacts/shadow/<SYMBOL>/<MANIFEST_HASH>/shadow_summary.md`
- Every active anchor should have a paper-live gate verdict under:
  - `artifacts/gates/<SYMBOL>/<MANIFEST_HASH>/paper_live_gate.json`
  - `artifacts/gates/<SYMBOL>/<MANIFEST_HASH>/paper_live_gate.md`
- `tools/verify_v1_rc.py` must fail if `evaluate_oos.py` or `strategies/rule_logic.py` drift from the manifest hashes.
- Latest verified RC1 regeneration on `2026-04-08` produced structurally valid packs, but both candidate verdicts are still negative on certification:
  - `eurusd_5k_v1_mr_rc1`: net `-$50.38`, `17` trades, gate `demoted`
  - `gbpusd_10k_v1_mr_rc1`: net `-$27.55`, `18` trades, gate `demoted`
- Latest MT5 historical replay on `2026-04-08` improved live direction balance materially:
  - `EURUSD`: long/short opens `11 / 11`, rollover opens `1`, signal density ratio `2.99x`
  - `GBPUSD`: long/short opens `5 / 5`, rollover opens `1`, signal density ratio `4.45x`
- Latest focused optimizer sweep on `2026-04-08` evaluated 100 new rule configurations (including parameterized `pro_mean_reversion` and a new `macd_trend` family) on the `GBPUSD` `train` stage but failed to find a passing candidate:
  - result: **100/100 candidate sets rejected on train** (due to skewed directionality or extremely low PF/expectancy).
  - report: `artifacts/optimization_report_GBPUSD_train.md`
  - **Verdict regarding GBPUSD Rule-First Challenger Search**: FROZEN. The failure is economic/structural, validating that the 10k horizon with the current v1 feature set has no discoverable edge under the current rule families. **This is a successful falsification.** Do not promote any `GBPUSD` RC challenger and do not launch PPO for this setup. Maintain current demoted anchor track without overrides. Only return to `GBPUSD` optimization if introducing a materially new feature family, horizon, or dataset hypothesis.
- Do not start long shadow evidence collection on `EURUSD` or `GBPUSD` until a certified challenger/retune exists.

## Guardrails
- Do not re-promote `USDJPY` into RC1 without new evidence.
- Do not relax `live_trading_approved: false` in the RC manifest path.
- Do not treat static release notes as truth when the generated scoreboards disagree; the generated scoreboards are the certification artifacts.
- Do not treat a positive replay as enough; the anchor path is `candidate` until the paper-live gate is satisfied.
