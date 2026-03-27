# Next Agent File Map

## Core Runtime
- `event_pipeline.py`
  - Shared event-driven core used by replay/paper/live.
- `live_bridge.py`
  - MT5 live bridge, artifact startup, broker reconciliation.
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
- `evaluate_oos.py`
  - Now supports `EVAL_SYMBOL=<PAIR>`.
  - Loads the symbol-specific manifest.
- `trading_env.py`
  - Training env, risk/equity mechanics, sizing hooks.
- `feature_engine.py`
  - Incremental feature pipeline used by replay/live.

## Data
- `download_dukascopy.py`
  - Dukascopy downloader.
  - Added retries/backoff.
  - Added `--force-refresh-pairs`.
  - Added `--max-workers`.
- `build_volume_bars.py`
  - Rebuilds `data/DATA_CLEAN_VOLUME.csv` from local tick parquet/csv files.

## Tests
- `tests/test_runtime_refactor.py`
  - Startup, ingestion, artifact validation, parity, recovery, JPY logic, broker truth.

## Existing Docs
- `docs/FIN-PRO_HANDOVER_ULTIMATE.md`
- `docs/DATA_SOURCING_GUIDE.md`
- `docs/LIVE_TRADING_GUIDE.md`
- These may contain older context; prefer this handoff first.
