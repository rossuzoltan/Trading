# Current Usage Guide

This is the practical operator guide for the current supported stack:
`MaskablePPO + RuntimeGymEnv + volume bars`.

Use this document when you want to run the project now without re-learning the
entire repo from code or old handoff notes.

## 1. Environment

- Work from the repo root: `c:\dev\trading`
- Use the project venv explicitly:
  `.\.venv\Scripts\python.exe`
- Core entrypoints will re-exec into the venv automatically, but explicit venv
  usage is still the cleanest path.

## 2. First Check

Run:

```powershell
.\.venv\Scripts\python.exe .\tools\project_healthcheck.py
```

What this tells you:

- dataset exists and matches the expected bar spec
- runtime artifacts are structurally valid when present
- required packages and files are available

Use strict runtime asset validation only when you expect trained artifacts:

```powershell
.\.venv\Scripts\python.exe .\tools\project_healthcheck.py --strict-runtime-assets
```

## 3. Data Workflow

Refresh raw tick data if needed:

```powershell
.\.venv\Scripts\python.exe .\download_dukascopy.py --pairs EURUSD GBPUSD --days 1095 --force-refresh-pairs EURUSD GBPUSD --max-workers 16
```

Rebuild the consolidated volume-bar dataset:

```powershell
.\.venv\Scripts\python.exe .\build_volume_bars.py --ticks-per-bar 2000
```

Re-run healthcheck after a rebuild:

```powershell
.\.venv\Scripts\python.exe .\tools\project_healthcheck.py
```

## 4. Normal Training

Train one symbol at a time:

```powershell
$env:TRAIN_SYMBOL='EURUSD'
$env:TRAIN_TOTAL_TIMESTEPS='3000000'
.\.venv\Scripts\python.exe .\train_agent.py
```

Important current behavior:

- training is symbol-scoped
- the baseline gate runs before RL training
- if the baseline gate fails, training aborts unless debug bypass is enabled
- runtime training is the supported path; legacy envs are compatibility only

Useful status command while training:

```powershell
.\.venv\Scripts\python.exe .\training_status.py --symbol EURUSD
```

## 5. Current Experiment Profiles

The training script now supports reproducible experiment presets via
`TRAIN_EXPERIMENT_PROFILE`.

Available profiles:

- `reward_strip`
  - removes secondary reward shaping
  - keeps base reward
  - disables participation bonus
- `reward_strip_hard_churn`
  - `reward_strip` plus hard churn controls
  - sets `TRAIN_CHURN_MIN_HOLD_BARS=5`
  - sets `TRAIN_CHURN_ACTION_COOLDOWN=3`
- `reward_strip_hard_churn_alpha_gate`
  - `reward_strip_hard_churn` plus baseline-driven alpha gate

Example:

```powershell
$env:TRAIN_EXPERIMENT_PROFILE='reward_strip_hard_churn_alpha_gate'
$env:TRAIN_SYMBOL='EURUSD'
$env:TRAIN_TOTAL_TIMESTEPS='120000'
.\.venv\Scripts\python.exe .\train_agent.py
```

Manual overrides still win over the profile. If you set an env var explicitly,
the profile will not overwrite it.

## 6. Windowed Observations

The runtime path now supports a real observation window via `TRAIN_WINDOW_SIZE`.

Default:

```powershell
$env:TRAIN_WINDOW_SIZE='1'
```

Example ablation:

```powershell
$env:TRAIN_EXPERIMENT_PROFILE='reward_strip_hard_churn_alpha_gate'
$env:TRAIN_WINDOW_SIZE='8'
$env:TRAIN_SYMBOL='EURUSD'
$env:TRAIN_TOTAL_TIMESTEPS='120000'
.\.venv\Scripts\python.exe .\train_agent.py
```

Notes:

- larger windows change the model observation shape
- that shape is now persisted in manifests and respected by replay/live validation

## 7. Evaluation

Run OOS evaluation for one symbol:

```powershell
$env:EVAL_SYMBOL='EURUSD'
.\.venv\Scripts\python.exe .\evaluate_oos.py
```

Current behavior:

- replay uses the current runtime path
- if training used a non-default window size, OOS replay honors it
- if training diagnostics recorded alpha gate usage, OOS replay rebuilds and applies the same gate
- use `.\.venv\Scripts\python.exe .\compare_oos_baselines.py --symbol EURUSD` to compare the RL replay against simple baselines; it now falls back to checkpoint artifacts and can recompute the RL replay when no saved replay report exists

If you need a quick postmortem after a losing replay or a closed-trade audit:

```powershell
.\.venv\Scripts\python.exe .\tools\diagnose_losses.py --symbol EURUSD
```

The helper now tolerates summary-only replay reports when a closed-trade log is not available.

## 8. Deployment And Live Readiness

Before treating a model as live-ready:

1. Run tests
2. Run symbol-specific OOS evaluation
3. Run MT5 preflight
4. Check ops evidence and audit outputs

Commands:

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
$env:EVAL_SYMBOL='EURUSD'; .\.venv\Scripts\python.exe .\evaluate_oos.py
.\.venv\Scripts\python.exe .\compare_oos_baselines.py --symbol EURUSD
.\.venv\Scripts\python.exe .\mt5_live_preflight.py --symbol EURUSD --ticks-per-bar 5000
```

## 9. Main Decision Rules

- If the baseline gate fails, do not keep pushing PPO on that feature set.
- Prefer per-symbol training and per-symbol evaluation.
- Treat reward-ablation and churn-control as the first corrective experiments when PPO degenerates into microtrading.
- Do not treat old `RecurrentPPO` or H1-only notes as the active architecture.
- Do not declare live readiness from training curves alone; use holdout, preflight, and audit evidence.

## 10. Good Default Workflows

Standard training pass:

```powershell
.\.venv\Scripts\python.exe .\tools\project_healthcheck.py
$env:TRAIN_SYMBOL='EURUSD'
$env:TRAIN_TOTAL_TIMESTEPS='3000000'
.\.venv\Scripts\python.exe .\train_agent.py
$env:EVAL_SYMBOL='EURUSD'
.\.venv\Scripts\python.exe .\evaluate_oos.py
```

Fast corrective run after a bad PPO diagnosis:

```powershell
$env:TRAIN_EXPERIMENT_PROFILE='reward_strip_hard_churn_alpha_gate'
$env:TRAIN_WINDOW_SIZE='8'
$env:TRAIN_SYMBOL='EURUSD'
$env:TRAIN_TOTAL_TIMESTEPS='120000'
.\.venv\Scripts\python.exe .\train_agent.py
```

## 11. Files That Matter Most

- `train_agent.py`
- `runtime_gym_env.py`
- `runtime/runtime_engine.py`
- `edge_research.py`
- `feature_engine.py`
- `evaluate_oos.py`
- `trading_config.py`
- `tools/project_healthcheck.py`

## 12. Read Next

- `docs/NEXT_AGENT_CONTEXT.md`
- `docs/NEXT_AGENT_FILE_MAP.md`
- `docs/NEXT_AGENT_RUNBOOK.md`
- `docs/operating_checklist.md`
