# Current Usage Guide

This is the practical operator guide for the current supported stack.

The repo still contains RL tooling, but the primary operational path is now:
`rule-first RC1 + shadow evidence + paper-live gate`.

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
.\.venv\Scripts\python.exe .\tools\project_healthcheck.py --mode rc1
```

What this tells you:

- approved RC1 packs exist and are structurally valid
- dataset metadata exists and matches the active anchor bar spec
- required packages and files are available

Use RL mode only when you are explicitly working the legacy/research runtime path:

```powershell
.\.venv\Scripts\python.exe .\tools\project_healthcheck.py --mode rl --strict-runtime-assets
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

## 4. Rule-First Anchor Workflow

The primary path is no longer "train until the replay looks good". The anchor
workflow is:

1. Healthcheck in `rc1` mode
2. Generate and certify RC1 packs
3. Run historical replay as a pre-shadow screen
4. Only if certification is not demoted, run shadow mode with the RC1 manifest
5. Build the paper-live gate verdict

Commands:

```powershell
.\.venv\Scripts\python.exe .\tools\project_healthcheck.py --mode rc1
.\.venv\Scripts\python.exe .\tools\generate_v1_rc.py
.\.venv\Scripts\python.exe .\tools\mt5_historical_replay.py --symbol EURUSD --days 30
.\tools\run_shadow_simulator.ps1 -ManifestPath models\rc1\eurusd_5k_v1_mr_rc1\manifest.json
.\.venv\Scripts\python.exe .\tools\paper_live_gate.py --manifest-path .\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json
```

Shadow artifacts are written under:

- `artifacts/shadow/<SYMBOL>/<MANIFEST_HASH>/events.jsonl`
- `artifacts/shadow/<SYMBOL>/<MANIFEST_HASH>/shadow_summary.json`
- `artifacts/shadow/<SYMBOL>/<MANIFEST_HASH>/shadow_summary.md`

Gate artifacts are written under:

- `artifacts/gates/<SYMBOL>/<MANIFEST_HASH>/paper_live_gate.json`
- `artifacts/gates/<SYMBOL>/<MANIFEST_HASH>/paper_live_gate.md`

Current status as of `2026-04-08`:

- both approved-scope RC1 packs are structurally valid
- both current gate verdicts are still `demoted`
- historical MT5 replay is directionally healthier than before, but not yet promotion-ready
- do not treat `EURUSD` or `GBPUSD` as active shadow anchors until certification recovers

Historical replay writes under the RC1 pack directory:

- `mt5_historical_replay_report.md`
- `mt5_historical_replay_report.json`
- `mt5_historical_replay_report.audit.jsonl`
- `mt5_historical_replay_report.bars.jsonl`

## 5. Normal Training

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
.\.venv\Scripts\python.exe .\tools\training_status.py --symbol EURUSD
```

## 6. Current Experiment Profiles

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
- `reward_strip_rehab_safer_alpha_gate`
  - stricter corrective profile for churn rehab
  - sets `TRAIN_CHURN_MIN_HOLD_BARS=8`
  - sets `TRAIN_CHURN_ACTION_COOLDOWN=5`
  - sets `TRAIN_ENTRY_SPREAD_Z_LIMIT=0.75`
  - enables the baseline-driven alpha gate

Example:

```powershell
$env:TRAIN_EXPERIMENT_PROFILE='reward_strip_hard_churn_alpha_gate'
$env:TRAIN_SYMBOL='EURUSD'
$env:TRAIN_TOTAL_TIMESTEPS='120000'
.\.venv\Scripts\python.exe .\train_agent.py
```

Manual overrides still win over the profile. If you set an env var explicitly,
the profile will not overwrite it.

## 7. Windowed Observations

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

## 8. Evaluation

Run OOS evaluation for one symbol:

```powershell
$env:EVAL_SYMBOL='EURUSD'
.\.venv\Scripts\python.exe .\evaluate_oos.py
```

Current behavior:

- replay uses the current runtime path
- if training used a non-default window size, OOS replay honors it
- if training diagnostics recorded churn min-hold, cooldown, or entry-spread guard settings, OOS replay applies the same execution-time masks
- if training diagnostics recorded alpha gate usage, OOS replay rebuilds and applies the same gate
- replay reports now include an authoritative `runtime_parity_verdict`; the deployment gate reads that verdict directly instead of relying on a separately-run comparison file
- use `.\.venv\Scripts\python.exe .\tools\compare_oos_baselines.py --symbol EURUSD` to compare the RL replay against simple baselines; it now falls back to checkpoint artifacts and can recompute the RL replay when no saved replay report exists

If you need a quick postmortem after a losing replay or a closed-trade audit:

```powershell
.\.venv\Scripts\python.exe .\tools\diagnose_losses.py --symbol EURUSD
```

The helper now tolerates summary-only replay reports when a closed-trade log is not available.

## 9. Paper-Live Gate And Live Readiness

Before treating anything as live-ready:

1. Run tests
2. Generate or re-verify the RC1 pack
3. Collect shadow evidence
4. Build the paper-live gate verdict
5. Run MT5 preflight
6. Check ops evidence and audit outputs

Commands:

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
.\.venv\Scripts\python.exe .\tools\verify_v1_rc.py
.\.venv\Scripts\python.exe .\tools\paper_live_gate.py --manifest-path .\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json
.\.venv\Scripts\python.exe .\mt5_live_preflight.py --symbol EURUSD --ticks-per-bar 5000
```

## 10. Main Decision Rules

- If the baseline gate fails, do not keep pushing PPO on that feature set.
- Do not treat a positive replay as sufficient; `paper_live_gate.json` is the operational verdict.
- Prefer per-symbol training and per-symbol evaluation.
- Treat reward-ablation and churn-control as the first corrective experiments when PPO degenerates into microtrading.
- Do not treat old `RecurrentPPO` or H1-only notes as the active architecture.
- Do not declare live readiness from training curves alone; use holdout, preflight, and audit evidence.

## 11. Good Default Workflows

Standard anchor pass:

```powershell
.\.venv\Scripts\python.exe .\tools\project_healthcheck.py --mode rc1
.\.venv\Scripts\python.exe .\tools\generate_v1_rc.py
.\.venv\Scripts\python.exe .\tools\mt5_historical_replay.py --symbol EURUSD --days 30
.\tools\run_shadow_simulator.ps1 -ManifestPath models\rc1\eurusd_5k_v1_mr_rc1\manifest.json
.\.venv\Scripts\python.exe .\tools\paper_live_gate.py --manifest-path .\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json
```

Fast corrective run after a bad PPO diagnosis:

```powershell
$env:TRAIN_EXPERIMENT_PROFILE='reward_strip_rehab_safer_alpha_gate'
$env:TRAIN_WINDOW_SIZE='8'
$env:TRAIN_SYMBOL='EURUSD'
$env:TRAIN_TOTAL_TIMESTEPS='120000'
.\.venv\Scripts\python.exe .\train_agent.py
```

## 12. Files That Matter Most

- `docs/PROFITABILITY_PLAN.md`
- `rule_selector.py`
- `runtime/shadow_broker.py`
- `tools/paper_live_gate.py`
- `train_agent.py`
- `runtime_gym_env.py`
- `runtime/runtime_engine.py`
- `edge_research.py`
- `feature_engine.py`
- `evaluate_oos.py`
- `trading_config.py`
- `tools/project_healthcheck.py`

## 13. Read Next

- `docs/PROFITABILITY_PLAN.md`
- `docs/NEXT_AGENT_CONTEXT.md`
- `docs/NEXT_AGENT_FILE_MAP.md`
- `docs/NEXT_AGENT_RUNBOOK.md`
- `docs/operating_checklist.md`
