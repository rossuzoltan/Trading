# Next Agent Runbook

## Start Here
1. Use the project venv, not system Python.
2. Focus on Rule-First candidate generation and certification.
3. The RL/PPO track is legacy/research-only.

## Primary Commands (Rule-First)

### 1. Generate Rule Candidates
```powershell
.\.venv\Scripts\python.exe .\tools\optimize_rules.py --symbol GBPUSD
.\.venv\Scripts\python.exe .\tools\optimize_rules.py --symbol EURUSD
```

### 2. Evaluate Candidates OOS
```powershell
.\.venv\Scripts\python.exe .\evaluate_oos.py --symbol GBPUSD
.\.venv\Scripts\python.exe .\evaluate_oos.py --symbol EURUSD
```

### 3. Generate and Certify RC1 Packs
```powershell
.\.venv\Scripts\python.exe .\tools\generate_v1_rc.py
.\.venv\Scripts\python.exe .\tools\verify_v1_rc.py
```

### 4. Run Shadow Certification
```powershell
.\tools\run_shadow_simulator.ps1 -ManifestPath models/rc1/gbpusd_10k_v1_mr_rc1/manifest.json
.\tools\run_shadow_simulator.ps1 -ManifestPath models/rc1/eurusd_5k_v1_mr_rc1/manifest.json
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
