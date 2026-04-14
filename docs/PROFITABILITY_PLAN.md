# Profitability Plan

This is the canonical operator-facing plan for the current repo.

The target state is **paper-live profitability**, not live-money release and
not a positive replay in isolation.

## Anchor Scope

- `EURUSD` at `5000` ticks/bar
- `GBPUSD` at `10000` ticks/bar
- `USDJPY` remains challenger-only until new evidence exists

Operational note on `2026-04-10`:

- RC packs were regenerated and strict manifest-hash verification now passes.
- Neither approved anchor is currently test-ready under the new fail-fast gate.
- `EURUSD` regenerated RC replay: `6` trades, net `+$1.89`, still blocked.
- `GBPUSD` regenerated RC replay: `4` trades, net `+$5.14`, still blocked.
- Both anchors are blocked by critical MT5 replay drift and stale historical replay evidence hashes.
- Exact-runtime EURUSD bakeoff currently favors `rule_only`; `xgboost_pair` is the best refit AlphaGate challenger, but it does not beat the ungated rule on current holdout net PnL.

## Gate Model

Anchor releases move through three states:

- `candidate`
- `paper_live_profitable`
- `demoted`

**Hybrid Architecture (2026-04-10):**
The standard for RC1 promotion is now a **Hybrid Gated-Rule** setup. Deterministic rules from `strategies/rule_logic.py` provide the signal, while an **AlphaGate** (ML-based meta-labeling filter) provides the veto.

Promotion to `paper_live_profitable` requires all of the following:

- RC1 certification passes
- fail-fast `pre_test_gate.py` passes
- mandatory baseline comparison passes
- raw anchor baseline comparison passes
- at least `20` trading days of shadow evidence
- at least `30` actionable shadow events
- no critical replay-vs-shadow drift on the aggregated window
- no stale historical replay evidence hash mismatch
- no Asia-session opens in historical replay
- no Rollover opens in historical replay
- no one-sided replay if there are at least `10` replay trades
- restart drill passes
- preflight passes
- ops attestation passes

Demotion happens when any of these are true:

- the RC baseline comparison fails
- two consecutive weekly shadow reviews are critically drifted and each review
  has enough evidence
- restart drill, preflight, or ops attestation fails

## Mandatory Baseline Roles

- `runtime_flat`
- `runtime_always_short`
- `runtime_trend`

The raw family anchor remains separate from the deployed RC release:

- `raw_anchor_baseline`: `runtime_mean_reversion`
- `deployed_anchor_rc`: the certified RC pack under `models/rc1/...`

## Artifact Layout

Shadow evidence lives at:

- `artifacts/shadow/<SYMBOL>/<MANIFEST_HASH>/events.jsonl`
- `artifacts/shadow/<SYMBOL>/<MANIFEST_HASH>/shadow_summary.json`
- `artifacts/shadow/<SYMBOL>/<MANIFEST_HASH>/shadow_summary.md`

Paper-live gate output lives at:

- `artifacts/gates/<SYMBOL>/<MANIFEST_HASH>/paper_live_gate.json`
- `artifacts/gates/<SYMBOL>/<MANIFEST_HASH>/paper_live_gate.md`

## Standard Commands

Healthcheck the current operational surface:

```powershell
.\.venv\Scripts\python.exe .\tools\project_healthcheck.py --mode rc1
```

Generate and certify RC1 packs:

```powershell
.\.venv\Scripts\python.exe .\tools\generate_v1_rc.py
.\.venv\Scripts\python.exe .\tools\verify_v1_rc.py
```

Run the fail-fast pre-test gate:

```powershell
.\.venv\Scripts\python.exe .\tools\pre_test_gate.py --manifest-path .\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json
.\.venv\Scripts\python.exe .\tools\pre_test_gate.py --manifest-path .\models\rc1\gbpusd_10k_v1_mr_rc1\manifest.json
```

Run exact-runtime AlphaGate challenger bakeoff:

```powershell
.\.venv\Scripts\python.exe .\tools\alpha_gate_bakeoff.py --manifest-path .\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json
```

Run the shadow simulator:

```powershell
.\tools\run_shadow_simulator.ps1 -ManifestPath models\rc1\eurusd_5k_v1_mr_rc1\manifest.json
```

Run historical MT5 replay as pre-shadow evidence:

```powershell
.\.venv\Scripts\python.exe .\tools\mt5_historical_replay.py --symbol EURUSD --days 30
```

Build the paper-live gate verdict:

```powershell
.\.venv\Scripts\python.exe .\tools\paper_live_gate.py --manifest-path .\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json
```

Summarize all gate verdict artifacts and generate a compact dashboard:

```powershell
.\.venv\Scripts\python.exe .\tools\summarize_gate_reports.py
```

## Challenger Rule

Do not start new challengers if anchor shadow evidence collection or daily
monitoring would be degraded.

`GBPJPY` and `XAUUSD` are explicitly out of the first profitabilitiy wave.
RL remains research-only until the anchor path is operationally stable.

Regime guards are optional but recommended for challengers:
`min_vol_norm_atr`, `max_abs_log_return`, `max_abs_body_size`, `max_candle_range`.
Keep them manifest-explicit and evidence-driven.

Historical MT5 replay is a pre-shadow accelerator only. It can flag critical
drift early, but it cannot promote an anchor to `paper_live_profitable`
without the full shadow evidence window.

**Operational Milestone (2026-04-10):**
The repo now has strict manifest/component hardening, an exact-runtime AlphaGate bakeoff tool, and a fail-fast pre-test gate. That makes false-positive profitability materially harder to ship, but it also means the current EURUSD and GBPUSD anchors remain blocked until historical replay is regenerated and replay density improves.
