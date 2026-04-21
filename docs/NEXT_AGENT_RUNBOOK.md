# Next Agent Runbook

## Start Here
1. Use the project venv, not system Python.
2. Focus on Rule-First candidate generation and certification.
3. The RL/PPO track is legacy/research-only.
4. Do not trust old MT5 replay artifacts after manifest regeneration. `pre_test_gate.py` now hard-fails on stale replay hashes.
5. When evaluating `EURUSD`, confirm the report says `Artifact source: promoted_manifest`, not `checkpoint_fallback`.

## Primary Commands (Rule-First)

### 1. Generate Rule Candidates
```powershell
.\.venv\Scripts\python.exe .\tools\optimize_rules.py --symbol GBPUSD
.\.venv\Scripts\python.exe .\tools\optimize_rules.py --symbol EURUSD
```
Optional AlphaGate backend sweep:
```powershell
.\.venv\Scripts\python.exe .\tools\optimize_rules.py --symbol GBPUSD --use-alpha-gate --alpha-gate-model logistic_pair
.\.venv\Scripts\python.exe .\tools\optimize_rules.py --symbol GBPUSD --use-alpha-gate --alpha-gate-model xgboost_pair
.\.venv\Scripts\python.exe .\tools\optimize_rules.py --symbol GBPUSD --use-alpha-gate --alpha-gate-model lightgbm_pair
```
Optional regime guard sweep:
```powershell
.\.venv\Scripts\python.exe .\tools\optimize_rules.py --symbol GBPUSD --enable-regime-guard-sweep
```

### 2. Evaluate Candidates OOS
```powershell
.\.venv\Scripts\python.exe .\evaluate_oos.py --symbol GBPUSD
.\.venv\Scripts\python.exe .\evaluate_oos.py --symbol EURUSD
```
Expected for the current EURUSD RC1 pack:
- promoted-manifest OOS stays around `net +39.47`, `PF 1.172`, `110` trades
- `fragile_under_cost_stress` remains `True`
- verdict remains `needs_targeted_ablation`

### 3. Generate and Certify RC1 Packs
```powershell
.\.venv\Scripts\python.exe .\tools\generate_v1_rc.py
.\.venv\Scripts\python.exe .\tools\verify_v1_rc.py
```

### 4. Fail-Fast Before Shadow
```powershell
.\.venv\Scripts\python.exe .\tools\pre_test_gate.py --manifest-path .\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json
.\.venv\Scripts\python.exe .\tools\pre_test_gate.py --manifest-path .\models\rc1\gbpusd_10k_v1_mr_rc1\manifest.json
```

### 5. Exact-Runtime AlphaGate Bakeoff
```powershell
.\.venv\Scripts\python.exe .\tools\alpha_gate_bakeoff.py --manifest-path .\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json
```

### 6. Refresh Historical Replay If Pre-Test Fails On Drift Or Stale Evidence
```powershell
.\.venv\Scripts\python.exe .\tools\mt5_historical_replay.py --symbol EURUSD --days 30
.\.venv\Scripts\python.exe .\tools\mt5_historical_replay.py --symbol GBPUSD --days 30
```
Current grounded EURUSD state after the latest replay refresh:
- `375` bars
- `14` opens
- `1` long / `13` short
- overall verdict `DRIFT_CRITICAL`

### 6b. Diagnose Recent Replay Drift
```powershell
.\.venv\Scripts\python.exe .\tools\ablate_recent_replay.py --manifest-path .\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json
```
This writes a research report plus per-variant JSONL event logs under:
- `artifacts/research/recent_replay_ablation/EURUSD/<MANIFEST_HASH>/`

### 7. Run Shadow Certification
```powershell
.\tools\run_shadow_simulator.ps1 -ManifestPath models/rc1/gbpusd_10k_v1_mr_rc1/manifest.json
.\tools\run_shadow_simulator.ps1 -ManifestPath models/rc1/eurusd_5k_v1_mr_rc1/manifest.json
```
Optional: log full feature snapshots per emitted bar (larger JSONL, best for early RC debugging):
```powershell
$env:SHADOW_LOG_FULL_FEATURES='1'
```

Background (recommended for long-running evidence collection):
```powershell
.\tools\start_shadow_simulator_background.ps1 -ManifestPath models/rc1/eurusd_5k_v1_mr_rc1/manifest.json -LogFullFeatures
```

### 8. Build and Summarize Paper-Live Gates
```powershell
.\.venv\Scripts\python.exe .\tools\paper_live_gate.py --manifest-path .\models\rc1\gbpusd_10k_v1_mr_rc1\manifest.json
.\.venv\Scripts\python.exe .\tools\summarize_gate_reports.py
```

## Secondary Commands (Data)

### 1. Confirm data counts
```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; import pandas as pd; files=['data/EURUSD_ticks.parquet','data/GBPUSD_ticks.parquet','data/USDJPY_ticks.parquet']; [print(f'{f}: {len(pd.read_parquet(f, columns=[\"mid\"])):,}') for f in files]"
```

### 2. Re-download thin pairs
```powershell
.\.venv\Scripts\python.exe download_dukascopy.py --pairs EURUSD GBPUSD --days 1095 --force-refresh-pairs EURUSD GBPUSD --max-workers 16
```

## Legacy Commands (RL Research)

### 1. Train RL Agent
```powershell
$env:TRAIN_SYMBOL='EURUSD'; $env:TRAIN_TOTAL_TIMESTEPS='3000000'; .\.venv\Scripts\python.exe train_agent.py
```

## Guardrails
- Do not declare any symbol paper-ready unless certification passes and OOS metrics (PF > 1.15) are stable.
- Every RC1 pack must have a valid `manifest.json` and `release_notes_rc1.md`.
- Current grounded status on `2026-04-21`:
  - `EURUSD` is `pre_test_gate` ready but still fails paper-live readiness
  - the promoted-manifest RC1 OOS remains positive
  - the latest MT5 replay window is `DRIFT_CRITICAL` and strongly short-heavy
- `EURUSD` exact-runtime bakeoff currently favors `rule_only`; `xgboost_pair` is the best refit AlphaGate challenger but does not beat the ungated rule on net PnL.
- `restart_drill.py` and `mt5_live_preflight.py` now support the RC1 selector manifest via `--manifest-path`, but ops evidence still requires a `real_mt5` restart drill plus acceptable execution drift.
