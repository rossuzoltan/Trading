# Trading Project

This repo is a Python-based Forex reinforcement-learning pipeline with four main pieces:

- `download_dukascopy.py` downloads raw tick data and can build initial volume-bar datasets.
- `build_volume_bars.py` consolidates per-pair tick data into the training dataset.
- `train_agent.py` trains a `MaskablePPO` agent on engineered Forex features.
- `evaluate_oos.py` and `live_bridge.py` handle out-of-sample validation and live or simulated execution.

## Current Architecture

See [REPOSITORY_MAP.md](file:///c:/dev/trading/REPOSITORY_MAP.md) for a complete visual file tree and detailed component descriptions.

- Supported training environment: `RuntimeGymEnv` on volume bars
- Compatibility fallback: legacy `trading_env.py` path for older experiments only
- Features: engineered in `feature_engine.py`
- Model: `sb3-contrib` `MaskablePPO`
- Primary data format: volume bars, with compatibility fallback to `FOREX_MULTI_SET.csv`

## Quick Start

1. Activate the repo virtualenv or call it explicitly: `.\.venv\Scripts\python.exe`
2. Run `.\.venv\Scripts\python.exe .\tools\project_healthcheck.py`
3. If needed, repair or recreate `.venv`
4. Install anything missing from `Requirements.txt` and `requirements.project.txt`
5. Download data with `.\.venv\Scripts\python.exe .\download_dukascopy.py`
6. Build the combined dataset with `.\.venv\Scripts\python.exe .\build_volume_bars.py`
7. Train with `.\.venv\Scripts\python.exe .\train_agent.py`
8. Evaluate with `.\.venv\Scripts\python.exe .\evaluate_oos.py`

Core entrypoints (`train_agent.py`, `evaluate_oos.py`, `live_bridge.py`) now re-exec into the
project `.venv` automatically when launched from the wrong interpreter, but explicit `.venv`
usage is still the least ambiguous path.

## Training Defaults And Guardrails

- `train_agent.py` now defaults to `TRAIN_ENV_MODE=runtime`. The supported stack is `MaskablePPO + RuntimeGymEnv + volume bars`.
- `TRAIN_NUM_ENVS=1` and `TRAIN_FORCE_DUMMY_VEC=1` are treated as debug or profiling settings and emit warnings during training.
- Training now fails closed when a symbol cannot satisfy the minimum train, validation, and holdout bar requirements. Defaults:
  - `TRAIN_MIN_TRAIN_BARS=5000`
  - `TRAIN_MIN_VAL_BARS=200`
  - `TRAIN_MIN_HOLDOUT_BARS=500`

## Bar Spec

- `BAR_SPEC_TICKS_PER_BAR` is the explicit volume-bar construction setting used for train/live/runtime parity.
- `download_dukascopy.py` and `build_volume_bars.py` now default to that same resolved bar spec, so build/train/live stay aligned unless you explicitly override them.
- `TRAIN_BAR_TICKS`, `TRAIN_TICKS_PER_BAR`, and `TRADING_TICKS_PER_BAR` are still accepted as fallbacks for compatibility.
- This setting describes how the volume bars were built. It is not an inner-loop training speed knob in `train_agent.py`.
- Training throughput is primarily driven by dataset size, `TRAIN_TOTAL_TIMESTEPS`, `TRAIN_PPO_N_STEPS`, `TRAIN_NUM_ENVS`, CPU/GPU availability, and whether `SubprocVecEnv` can be used.

## H1 Data Builder

- Use `build_h1_dataset.py` when you need 5-10 years of closed H1 candles for research or alternate experiments.
- It prefers existing `data/*_ticks.parquet` files, then the Dukascopy downloader already in this repo, then MT5, then `yfinance`.
- Example: `.\.venv\Scripts\python.exe .\build_h1_dataset.py --years 7`
- Add `--strict-coverage --min-years 5 --max-gap-hours 72` when you want the builder to reject sparse history.
- Details and output paths: `docs/h1_data.md`

## Training Feedback

- `train_agent.py` writes a lightweight progress heartbeat to `checkpoints/fold_*/training_heartbeat.json`.
- Use `.\.venv\Scripts\python.exe .\tools\training_status.py --symbol EURUSD` to see the latest PPO diagnostics + deployment gate blockers.

## MT5 Live Readiness

- Run `.\.venv\Scripts\python.exe .\mt5_live_preflight.py --symbol EURUSD` before any live MT5 session.
- The preflight writes `models/live_preflight_eurusd.json` and fails closed when gate approval, MT5 connectivity, bar-spec parity, or ops evidence is missing.
- Live orders now append a structured execution audit to `models/execution_audit_eurusd.jsonl`.
- Summarize real fill drift with `.\.venv\Scripts\python.exe .\tools\summarize_execution_audit.py --symbol EURUSD`.

## Important Notes

- The checked-in `.venv` currently appears unhealthy in this workspace, so verify it before trusting runtime behavior.
- The repo previously mixed older H1-bar and RecurrentPPO docs with the current volume-bar and MaskablePPO stack. Treat `RecurrentPPO` references as legacy history, not the supported training architecture.
- Live trading requires a trained model plus scaler files in `models/`.
