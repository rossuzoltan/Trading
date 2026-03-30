# AI Handoff Context

## Workspace

- Primary repo: `C:\dev\trading`
- Safe dev worktree: `C:\dev\trading-dev`
- This handoff bundle: `C:\dev\trading\ai_handoff\run_20260327T185007Z_20260328_0105`

## Project Summary

Python forex RL project with the currently supported stack:

- `MaskablePPO`
- `RuntimeGymEnv`
- volume bars
- symbol-scoped training/evaluation artifacts

The live / runtime architecture is considered stabilized. The current blocker is strategy and evidence quality, not the core event-driven runtime.

## Current Training Run

- Run ID: `20260327T185007Z`
- Symbol: `EURUSD`
- Current fold: `0`
- State: `training`
- Current progress at bundle creation: `2,424,834 / 3,000,000` steps
- Progress fraction: `0.808278`
- Speed: about `131-133 steps/sec`

Latest heartbeat takeaways:

- `explained_variance`: about `0.70`
- `approx_kl`: about `0.0022`
- `value_loss_stable`: `true`
- latest eval `final_equity`: about `843.29`
- latest eval `timed_sharpe`: about `-0.116`
- latest eval `max_drawdown`: about `15.67%`

## High-Level Interpretation

Technically, the run is healthy:

- dataset integrity passed
- no current crash in stderr
- PPO internal diagnostics look much healthier than earlier in the run

Economically, the run is still not convincing:

- eval Sharpe remains negative
- eval final equity remains below starting capital
- drawdown worsened in the later part of the run
- `approx_kl` remains below the desired threshold window

Important caveat:

- Evaluation reporting appears internally inconsistent.
- In `latest_eval`, top-level summary metrics like `trade_count`, `net_pnl_usd`, and `profit_factor` are often zero.
- But inside `execution_diagnostics`, there are clearly many executed / closed trades and large transaction costs.
- Any downstream analysis should treat the execution diagnostics as more trustworthy than the top-level zeroed trade summary until the reporting path is fixed.

## Baseline Gate Context

Baseline diagnostics are included. Current baseline gate status:

- `gate_passed = false`
- `passing_models = []`

This means the RL setup is not yet justified by the repo's own baseline policy.

## Supervisor / Auto-Restart Context

The workspace has an auto-supervisor setup that:

- watches progress
- restarts on no-progress / failure
- can switch to a repair profile after a bad run at the configured threshold

There has already been one automatic restart earlier in this run lifecycle due to no step progress.

## Recommended Read Order For Another AI

1. `CONTEXT.md`
2. `FILE_MANIFEST.json`
3. `docs/NEXT_AGENT_CONTEXT.md`
4. `docs/NEXT_AGENT_FILE_MAP.md`
5. `docs/NEXT_AGENT_RUNBOOK.md`
6. `docs/research-policy.md`
7. `checkpoints/training_heartbeat.json`
8. `checkpoints/baseline_diagnostics_EURUSD.json`
9. `logs/train_run.log`
10. `checkpoints/full_path_evaluations.json`

## Suggested Focus Areas

1. Diagnose the mismatch between eval summary metrics and execution diagnostics.
2. Judge the run on economic results, not only PPO internal learning quality.
3. Compare the RL behavior against the included baselines and the repo's research policy.
4. Avoid changing the live training workspace while this run is still active.
