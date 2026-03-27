# Next Agent Runbook

## Start Here
1. Use the project venv, not system Python.
2. Do **not** re-audit the architecture unless a failing test forces it.
3. Focus on data repair, per-symbol training, and per-symbol OOS evaluation.

## Commands

### 1. Confirm data counts
```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; import pandas as pd; files=['data/EURUSD_ticks.parquet','data/GBPUSD_ticks.parquet','data/USDJPY_ticks.parquet']; [print(f'{f}: {len(pd.read_parquet(f, columns=[\"mid\"])):,}') for f in files]"
```

### 2. Re-download thin pairs
```powershell
.\.venv\Scripts\python.exe download_dukascopy.py --pairs EURUSD GBPUSD --days 1095 --bar-volume 2000 --force-refresh-pairs EURUSD GBPUSD --max-workers 16
```

### 3. Rebuild consolidated bars
```powershell
.\.venv\Scripts\python.exe build_volume_bars.py --ticks-per-bar 2000
```

### 4. Train one symbol at a time
```powershell
$env:TRAIN_SYMBOL='EURUSD'; $env:TRAIN_TOTAL_TIMESTEPS='3000000'; .\.venv\Scripts\python.exe train_agent.py
$env:TRAIN_SYMBOL='GBPUSD'; $env:TRAIN_TOTAL_TIMESTEPS='3000000'; .\.venv\Scripts\python.exe train_agent.py
$env:TRAIN_SYMBOL='USDJPY'; $env:TRAIN_TOTAL_TIMESTEPS='3000000'; .\.venv\Scripts\python.exe train_agent.py
```

### 5. Evaluate one symbol at a time
```powershell
$env:EVAL_SYMBOL='EURUSD'; .\.venv\Scripts\python.exe evaluate_oos.py
$env:EVAL_SYMBOL='GBPUSD'; .\.venv\Scripts\python.exe evaluate_oos.py
$env:EVAL_SYMBOL='USDJPY'; .\.venv\Scripts\python.exe evaluate_oos.py
```

### 6. Regression tests
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```

## Decision Rules
- If EURUSD/GBPUSD tick counts stay near ~2M after refresh, treat the data source step as still broken.
- Do not declare any symbol paper-ready unless:
  - tests pass
  - training artifacts match manifests
  - symbol-specific replay completes cleanly
  - OOS metrics are acceptable

## Token-Saving Notes
- Read only these first:
  - `docs/NEXT_AGENT_CONTEXT.md`
  - `docs/NEXT_AGENT_FILE_MAP.md`
  - `docs/NEXT_AGENT_RUNBOOK.md`
- Then inspect only:
  - `train_agent.py`
  - `evaluate_oos.py`
  - `download_dukascopy.py`
  - `build_volume_bars.py`
- Avoid reopening `event_pipeline.py` unless tests or replay point to a core-runtime regression.
