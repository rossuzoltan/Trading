# Profitability Plan

This is the canonical operator-facing plan for the current repo.

The target state is **paper-live profitability**, not live-money release and
not a positive replay in isolation.

## Anchor Scope

- `EURUSD` at `5000` ticks/bar
- `GBPUSD` at `10000` ticks/bar
- `USDJPY` remains challenger-only until new evidence exists

Operational note on `2026-04-08`:

- both approved-scope anchors are currently `demoted`
- historical replay balance improved after the price-based mean-reversion correction
- no symbol is currently eligible for long shadow evidence collection until certification recovers

## Gate Model

Anchor releases move through three states:

- `candidate`
- `paper_live_profitable`
- `demoted`

**Hybrid Architecture (2026-04-10):**
The standard for RC1 promotion is now a **Hybrid Gated-Rule** setup. Deterministic rules from `strategies/rule_logic.py` provide the signal, while an **AlphaGate** (ML-based meta-labeling filter) provides the veto.

Promotion to `paper_live_profitable` requires all of the following:

- RC1 certification passes
- mandatory baseline comparison passes
- raw anchor baseline comparison passes
- at least `20` trading days of shadow evidence
- at least `30` actionable shadow events
- no critical replay-vs-shadow drift on the aggregated window
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

## Challenger Rule

Do not start new challengers if anchor shadow evidence collection or daily
monitoring would be degraded.

`GBPJPY` and `XAUUSD` are explicitly out of the first profitabilitiy wave.
RL remains research-only until the anchor path is operationally stable.

Historical MT5 replay is a pre-shadow accelerator only. It can flag critical
drift early, but it cannot promote an anchor to `paper_live_profitable`
without the full shadow evidence window.

**Operational Milestone (2026-04-10):**
Both `EURUSD` and `GBPUSD` have successfully recovered certification potential. `GBPUSD` achieved a **1.45 Profit Factor** and `EURUSD` achieved a **3.78 Profit Factor** using a hybrid `mean_reversion` + `AlphaGate` configuration, reviving both tracks from previous demoted status.
