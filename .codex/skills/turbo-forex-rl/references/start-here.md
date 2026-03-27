# Start Here

## First Reads

Read these in order before opening core code:

1. `docs/NEXT_AGENT_CONTEXT.md`
2. `docs/NEXT_AGENT_FILE_MAP.md`
3. `docs/NEXT_AGENT_RUNBOOK.md`

Use those handoff docs to avoid re-reading the entire repository.

## Default Commands

Use the project virtualenv explicitly:

```powershell
.\.venv\Scripts\python.exe .\tools\project_healthcheck.py
.\.venv\Scripts\python.exe -m unittest discover tests
```

For current symbol-scoped work:

```powershell
$env:TRAIN_SYMBOL='EURUSD'; $env:TRAIN_TOTAL_TIMESTEPS='3000000'; .\.venv\Scripts\python.exe train_agent.py
$env:EVAL_SYMBOL='EURUSD'; .\.venv\Scripts\python.exe evaluate_oos.py
```

For data repair:

```powershell
.\.venv\Scripts\python.exe download_dukascopy.py --pairs EURUSD GBPUSD --days 1095 --bar-volume 2000 --force-refresh-pairs EURUSD GBPUSD --max-workers 16
.\.venv\Scripts\python.exe build_volume_bars.py --ticks-per-bar 2000
```

## Current Practical Focus

- Continue hardening or operating the supported runtime, not redesigning the architecture from scratch.
- Prefer per-symbol training and per-symbol OOS evaluation.
- Treat EURUSD and GBPUSD as potentially under-covered until fresh downloads prove otherwise.

## Guardrails

- Use `MaskablePPO + RuntimeGymEnv + volume bars` as the supported training path.
- Do not treat legacy `RecurrentPPO` references as current architecture.
- Do not reopen `event_pipeline.py` unless evidence from tests, replay, or the requested change points there.
