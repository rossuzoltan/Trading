# Research Runner

`tools/research_runner.py` adds a safe, repo-native experimentation layer on top
of the existing training and evaluation entrypoints.

It is intentionally narrower than Karpathy-style `autoresearch`:

- no arbitrary code editing
- no live or MT5 mutation
- no manifest or baseline-gate bypass
- no promotion into canonical `models/`

The runner is proposal-driven, config-first, and fully namespaced under
`research/`.

## Architecture

One proposal produces one research result bundle.

1. Validate a JSON proposal from `research/proposals/`
2. Refuse to run if shared checkpoint state shows active training
3. Pin safe env vars and allow only explicit override knobs
4. Launch `train_agent.py` in a research-only artifact directory
5. Launch `evaluate_oos.py` against the research manifest and artifact bundle
6. Read authoritative training/eval outputs already produced by the repo
7. Compute a documented composite research score
8. Compare against a deterministic baseline
9. Classify the result as `reject`, `keep`, or `promote_candidate`
10. Write `research/results/<result_id>/result.json`
11. Append one JSONL summary row to `research/ledger/experiments.jsonl`

The runner uses subprocess orchestration on purpose. v1 stays close to the
current repo entrypoints instead of refactoring training or replay internals.

## Safety Model

The runner forces the supported stack and keeps artifacts isolated.

Forced training env vars:

- `TRAIN_ENV_MODE=runtime`
- `TRAIN_SYMBOL=<proposal symbol>`
- `TRAIN_TOTAL_TIMESTEPS=<proposal timesteps>`
- `TRAIN_MODEL_DIR=<research result>/artifacts`
- `TRAIN_EXPORT_BEST_FOLD=1`
- `TRAIN_RESUME_LATEST=0`
- `TRAIN_DEBUG_ALLOW_BASELINE_BYPASS=0`

Forced evaluation env vars:

- `EVAL_SYMBOL=<proposal symbol>`
- `EVAL_MANIFEST_PATH=<research result>/artifacts/artifact_manifest_<SYMBOL>.json`
- `EVAL_OUTPUT_DIR=<research result>/artifacts`

Fast mode adds:

- `EVAL_SKIP_PLOT=1`

Research output never targets canonical `models/`. The training/eval scripts may
still use shared `checkpoints/` and `logs/` internally, but promoted runtime
artifacts remain untouched because `TRAIN_MODEL_DIR` is pinned to the
research-only namespace.

## Proposal Schema

Each v1 proposal is strict JSON with only these top-level fields:

- `experiment_name`: required, slug-safe lowercase name matching `[a-z0-9][a-z0-9_-]*`
- `symbol`: required, uppercased by the loader
- `timesteps`: required integer
- `fast_mode`: optional boolean, default `false`
- `baseline_reference`: optional string
- `rationale`: required string
- `overrides`: required object
- `tags`: optional list of strings
- `parent_experiment`: optional string

Example:

```json
{
  "experiment_name": "eurusd_reward_strip_window8",
  "symbol": "EURUSD",
  "timesteps": 300000,
  "rationale": "Window-size and safer reward-shaping ablation for EURUSD.",
  "overrides": {
    "TRAIN_EXPERIMENT_PROFILE": "reward_strip_hard_churn_alpha_gate",
    "TRAIN_WINDOW_SIZE": 8,
    "TRAIN_PPO_LEARNING_RATE": 0.0003
  },
  "tags": ["ablation", "window"],
  "parent_experiment": "manual_baseline_20260406"
}
```

## Allowed Search Space

Only these repo-safe training knobs may appear in `overrides`:

- `TRAIN_EXPERIMENT_PROFILE`
- `TRAIN_WINDOW_SIZE`
- `TRAIN_CHURN_MIN_HOLD_BARS`
- `TRAIN_CHURN_ACTION_COOLDOWN`
- `TRAIN_CHURN_PENALTY_USD`
- `TRAIN_ENTRY_SPREAD_Z_LIMIT`
- `TRAIN_REWARD_DOWNSIDE_RISK_COEF`
- `TRAIN_REWARD_TURNOVER_COEF`
- `TRAIN_REWARD_NET_RETURN_COEF`
- `TRAIN_REWARD_SCALE`
- `TRAIN_REWARD_CLIP_LOW`
- `TRAIN_REWARD_CLIP_HIGH`
- `TRAIN_ALPHA_GATE_ENABLED`
- `TRAIN_ALPHA_GATE_MODEL`
- `TRAIN_ALPHA_GATE_WARMUP_STEPS`
- `TRAIN_ALPHA_GATE_WARMUP_THRESHOLD_DELTA`
- `TRAIN_ALPHA_GATE_WARMUP_MARGIN_SCALE`
- `TRAIN_ADAPTIVE_KL_MAX_LR`
- `TRAIN_ADAPTIVE_KL_LOW`
- `TRAIN_ADAPTIVE_KL_UP_MULT`
- `TRAIN_FAIL_FAST_ENABLED`
- `TRAIN_FAIL_FAST_WARMUP_STEPS`
- `TRAIN_FAIL_FAST_CONSECUTIVE`
- `TRAIN_FAIL_FAST_SPARSE_ALPHA_GATE_BLOCK_RATE`
- `TRAIN_FAIL_FAST_APPROX_KL_MAX`
- `TRAIN_FAIL_FAST_EXPLAINED_VARIANCE_MAX`
- `TRAIN_FAIL_FAST_MAX_TRADE_COUNT`
- `TRAIN_PPO_LEARNING_RATE`
- `TRAIN_PPO_N_STEPS`
- `TRAIN_PPO_BATCH_SIZE`
- `TRAIN_PPO_N_EPOCHS`
- `TRAIN_PPO_ENT_COEF`
- `TRAIN_PPO_TARGET_KL`

Anything else is rejected before execution.

## Fast Mode

Fast mode is a cheap ablation preset, not a different evaluation regime.

- still symbol-scoped
- still runs the normal baseline gate
- still runs `evaluate_oos.py`
- still writes a full research result bundle
- skips only the replay plot to save time
- requires `timesteps <= 120000`

Fast mode exists so proposals stay comparable while remaining much cheaper than
full runs.

## Baseline Resolution

The baseline is deterministic:

1. If `baseline_reference` is set, resolve it from the ledger by exact
   `result_id`, then by exact `experiment_name`
2. Otherwise use the best prior comparable research result for the same symbol
   and `fast_mode` bucket
3. If there is no prior comparable research result, fall back to the current
   experiment's baseline-gate holdout winner from `edge_research.py`

Comparable prior results also require matching dataset identity and bar spec
when those fields are available.

## Scoring

The composite score is holdout-first and intentionally simple.

Positive terms:

- `timed_sharpe * 2.0`
- `(clamped_profit_factor - 1.0) * 3.0`
- `expectancy_usd * 0.5`
- `trade_count_credit` up to `+1.5`
- `+0.5` if training diagnostics report `deploy_ready=true`

Negative terms:

- `max_drawdown * 8.0`
- low-trade penalty up to `-3.0` when trade count is below `20`
- `-8.0` if the baseline gate failed
- `-10.0` if replay accounting reconciliation failed
- `-8.0` if runtime parity is not aligned with the research baseline gate
- `-4.0` if the replay is fragile under slippage stress

PPO internals are recorded in the result but are not primary score inputs.

Decision floors:

- `promote_candidate`: score `>= 2.5`, no critical gate failures, and materially
  better than the resolved baseline
- `keep`: score `>= 0.5` without critical gate failures, but promotion criteria
  not fully met
- `reject`: everything else

For prior research baselines, "materially better" means composite score delta
`>= 0.5`.

For baseline-gate fallbacks, "materially better" means the replay clears the
baseline floor on shared economic fields:

- better net PnL
- no worse profit factor
- no worse expectancy
- at least `20` trades

## Result Schema

Each completed run writes:

- `research/results/<result_id>/proposal.json`
- `research/results/<result_id>/result.json`
- `research/results/<result_id>/logs/train.stdout.log`
- `research/results/<result_id>/logs/eval.stdout.log`
- `research/results/<result_id>/artifacts/...`

`result.json` includes:

- proposal metadata
- resolved env vars
- exact commands executed
- subprocess exit status and durations
- artifact and log pointers
- baseline gate status
- training validation and holdout summaries
- replay metrics and runtime parity verdict
- deployment gate payload when present
- composite score, baseline comparison, and final decision

Example shape:

```json
{
  "result_id": "20260406T120000Z__eurusd_reward_strip_window8__eurusd",
  "run_status": "completed",
  "resolved_proposal": {
    "experiment_name": "eurusd_reward_strip_window8",
    "symbol": "EURUSD",
    "timesteps": 300000,
    "fast_mode": false
  },
  "composite_score": 3.42,
  "decision": "promote_candidate",
  "baseline_resolution": {
    "source": "research_result",
    "reference": "20260401T090000Z__eurusd_baseline__eurusd"
  },
  "artifact_pointers": {
    "artifacts_dir": "C:/dev/trading/research/results/.../artifacts",
    "manifest_path": "C:/dev/trading/research/results/.../artifacts/artifact_manifest_EURUSD.json"
  }
}
```

The ledger appends one compact row per completed experiment at
`research/ledger/experiments.jsonl`.

## Commands

Validate a proposal without running:

```powershell
.\.venv\Scripts\python.exe .\tools\research_runner.py --proposal .\research\proposals\eurusd_fast_reward_strip_window8.json --validate-only
```

Run a proposal end to end:

```powershell
.\.venv\Scripts\python.exe .\tools\research_runner.py --proposal .\research\proposals\eurusd_reward_strip_window8.json
```

## Why v1 Uses Config Mutation

v1 is deliberately proposal-driven instead of code-writing autonomy because the
repo already has meaningful safety and accounting guardrails:

- baseline gate
- purged walk-forward validation
- disjoint holdout replay
- runtime parity checks
- manifests
- MT5 preflight and diagnostics

Changing code automatically would weaken auditability and make it much easier to
confuse training diagnostics with economic truth. v1 only mutates safe training
configuration via an explicit allowlist.

## Out Of Scope For v1

- arbitrary Python or module rewriting
- live trading or MT5 execution changes
- broker wiring changes
- manifest bypasses
- baseline gate bypasses
- automatic promotion into canonical deploy artifacts
- replacing `optimize_hparams.py`

`optimize_hparams.py` remains separate and non-authoritative for promotion
decisions.

## Future Evolution

If this grows into a more autonomous system later, the next safe step is not
arbitrary code editing. The next step would be multi-proposal campaign support:

- generate constrained proposals from prior ledger history
- batch-run fast mode before full runs
- require explicit human approval before any broader search or code change

That keeps auditability and safety intact while increasing automation.
