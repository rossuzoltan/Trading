# Next Agent File Map

## Canonical Docs
- `docs/README.md`
  - Documentation index and reading order.
- `docs/NEXT_AGENT_CONTEXT.md`
  - Current repo/data snapshot and latest rebuild notes.
- `docs/NEXT_AGENT_RUNBOOK.md`
  - Verified commands for data rebuild, training, and evaluation.
- `docs/operating_checklist.md`
  - Live-readiness evidence gate after restart drill and shadow trading.
- `docs/h1_data.md`
  - Optional H1 dataset builder; not the default runtime training path.

## Core Runtime
- `event_pipeline.py`
  - Shared event-driven core used by replay/paper/live.
- `live_bridge.py`
  - MT5 live bridge, artifact startup, broker reconciliation.
- `mt5_live_preflight.py`
  - Mandatory live-readiness gate before real MT5 sessions.
- `artifact_manifest.py`
  - Manifest validation and safe model/scaler/VecNormalize loading.
- `runtime_common.py`
  - Shared action-map/state helpers.
- `symbol_utils.py`
  - Symbol-aware pip size, pip value, contract size, conversion logic.

## Training / Evaluation
- `train_agent.py`
  - Now supports `TRAIN_SYMBOL=<PAIR>`.
  - Saves `models/model_<pair>_best.zip`.
  - Saves `models/model_<pair>_best_vecnormalize.pkl`.
  - Writes `models/artifact_manifest_<PAIR>.json`.
- `runtime_gym_env.py`
  - Supported runtime training environment for the current stack.
- `evaluate_oos.py`
  - Now supports `EVAL_SYMBOL=<PAIR>`.
  - Loads the symbol-specific manifest.
- `trading_env.py`
  - Compatibility or legacy environment path for older experiments.
- `feature_engine.py`
  - Incremental feature pipeline used by replay/live.
- `trading_config.py`
  - Shared guardrails, thresholds, and deployment gates.

## Data
- `download_dukascopy.py`
  - Dukascopy downloader.
  - Added retries/backoff.
  - Added `--force-refresh-pairs`.
  - Added `--max-workers`.
- `build_volume_bars.py`
  - Rebuilds `data/DATA_CLEAN_VOLUME.csv` from local tick parquet/csv files.
- `build_h1_dataset.py`
  - Optional H1 dataset builder for separate research datasets.

## Tests / Ops
- `tests/test_runtime_refactor.py`
  - Startup, ingestion, artifact validation, parity, recovery, JPY logic, broker truth.
- `tools/project_healthcheck.py`
  - Environment and dataset integrity check.
- `training_status.py`
  - PPO and heartbeat diagnostics.
- `summarize_execution_audit.py`
  - Live execution drift summaries.
