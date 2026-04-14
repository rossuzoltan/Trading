# Trading Project

This repo is a Rule-First Forex Trading System with AlphaGate meta-labeling. It focuses on deterministic rule candidates certified against exact-runtime parity, with optional ML filters to improve precision.

- `tools/optimize_rules.py` generates and validates rule candidates over historical data.
- `rule_selector.py` is the manifest-driven runtime loop for signals and execution.
- `runtime/shadow_broker.py` provides pure shadow-mode operation for certification.
- `evaluate_oos.py` provides authoritative out-of-sample validation.
- `train_agent.py` (Legacy) remains available for research-only RL exploration.

## Current Status

As of `2026-04-10`:

- RC packs were regenerated and now pass strict manifest-hash verification.
- `tools/verify_v1_rc.py` currently certifies structurally, but both anchors still fail the new fail-fast pre-test gate.
- `EURUSD` RC replay is positive but too thin for testing: `6` trades, net `+$1.89`.
- `GBPUSD` RC replay is also too thin: `4` trades, net `+$5.14`.
- Both anchors are blocked by stale historical replay evidence plus critical replay/live drift from the last MT5 replay artifacts.
- Exact-runtime EURUSD bakeoff currently says the best holdout result is still `rule_only`; `xgboost_pair` is the strongest refit challenger, but it does not beat the ungated rule on current holdout.

## Documentation

- [docs/README.md](docs/README.md) is the current documentation index.
- [docs/CURRENT_USAGE_GUIDE.md](docs/CURRENT_USAGE_GUIDE.md) is the practical how-to guide for the current workflow.
- [docs/evaluation_accounting.md](docs/evaluation_accounting.md) explains which evaluation/accounting metrics are authoritative and how baselines are compared.
- [docs/NEXT_AGENT_CONTEXT.md](docs/NEXT_AGENT_CONTEXT.md) captures the current repo and data state.
- [docs/NEXT_AGENT_FILE_MAP.md](docs/NEXT_AGENT_FILE_MAP.md) points to the files that matter for each task.
- [docs/NEXT_AGENT_RUNBOOK.md](docs/NEXT_AGENT_RUNBOOK.md) keeps the verified commands in one place.

## Current Architecture

- Primary signal engine: deterministic rule families in `strategies/rule_logic.py`
- Optional veto layer: `BaselineAlphaGate` (`logistic_pair`, `xgboost_pair`, `lightgbm_pair`, `ridge_signed_target`)
- Runtime contract: manifest-driven `rule_selector.py` with exact-runtime replay parity checks
- Safety path: `runtime/shadow_broker.py` + `tools/paper_live_gate.py` before any live-money discussion
- Legacy RL remains available for research continuity only (`train_agent.py`, `RuntimeGymEnv`)

## Quick Start

1. Activate the repo virtualenv: `.\.venv\Scripts\python.exe`
2. Run project healthcheck: `.\.venv\Scripts\python.exe .\tools\project_healthcheck.py --mode rc1`
3. Generate rule candidates: `.\.venv\Scripts\python.exe .\tools\optimize_rules.py --symbol GBPUSD`
4. Evaluate candidates OOS: `.\.venv\Scripts\python.exe .\evaluate_oos.py --symbol GBPUSD`
5. Certify an RC pack: `.\.venv\Scripts\python.exe .\tools\generate_v1_rc.py`
6. Re-run strict certification: `.\.venv\Scripts\python.exe .\tools\verify_v1_rc.py`
7. Run fail-fast pre-test gate: `.\.venv\Scripts\python.exe .\tools\pre_test_gate.py --manifest-path .\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json`
8. Compare AlphaGate challengers exact-runtime: `.\.venv\Scripts\python.exe .\tools\alpha_gate_bakeoff.py --manifest-path .\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json`
9. Run shadow certification only after pre-test passes: `.\tools\run_shadow_simulator.ps1 -ManifestPath models/rc1/gbpusd_10k_v1_mr_rc1/manifest.json`

## Optimization and Certification

The primary research workflow is manifest-driven using `tools/optimize_rules.py`. Rule families are defined in `strategies/rule_logic.py`, and the optimizer performs an exact-runtime sweep to identify candidates that satisfy strict Profit Factor (PF > 1.15) and stability constraints.

Certified RC (Release Candidate) packs are built and verified using:
- `.\.venv\Scripts\python.exe .\tools\generate_v1_rc.py`
- `.\.venv\Scripts\python.exe .\tools\verify_v1_rc.py`
- `.\.venv\Scripts\python.exe .\tools\pre_test_gate.py --manifest-path .\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json`

Exact-runtime AlphaGate challenger bakeoff:
- `.\.venv\Scripts\python.exe .\tools\alpha_gate_bakeoff.py --manifest-path .\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json`

Optional hybrid sweeps:
- `.\.venv\Scripts\python.exe .\tools\optimize_rules.py --symbol EURUSD --use-alpha-gate --alpha-gate-model logistic_pair`
- `.\.venv\Scripts\python.exe .\tools\optimize_rules.py --symbol EURUSD --use-alpha-gate --alpha-gate-model xgboost_pair`
- `.\.venv\Scripts\python.exe .\tools\optimize_rules.py --symbol EURUSD --use-alpha-gate --alpha-gate-model lightgbm_pair`
- Add `--enable-regime-guard-sweep` to evaluate optional low-volatility/news-spike filters.

## Legacy RL Pipeline

The previous reinforcement learning pipeline (`train_agent.py` and `MaskablePPO`) is still available for research purposes but is no longer the primary path for production signals.
- To monitor legacy training: `.\.venv\Scripts\python.exe .\tools\training_status.py --symbol EURUSD`

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
- Use `.\.venv\Scripts\python.exe .\training_status.py --symbol EURUSD` to see the latest PPO diagnostics + deployment gate blockers.
- Use `.\.venv\Scripts\python.exe .\compare_oos_baselines.py --symbol EURUSD` to compare RL replay against simple baselines under the same replay cost model.

## MT5 Live Readiness

- Run `.\.venv\Scripts\python.exe .\mt5_live_preflight.py --symbol EURUSD` before any live MT5 session.
- The preflight writes `models/live_preflight_eurusd.json` and fails closed when gate approval, MT5 connectivity, bar-spec parity, or ops evidence is missing.
- Live orders now append a structured execution audit to `models/execution_audit_eurusd.jsonl`.
- Summarize real fill drift with `.\.venv\Scripts\python.exe .\summarize_execution_audit.py --symbol EURUSD`.

## Baseline Comparison

- Use `.\.venv\Scripts\python.exe .\compare_oos_baselines.py --symbol EURUSD` to compare the latest RL replay against runtime-rule and research baselines under the same replay context.
- The comparison now falls back to checkpoint artifacts when no promoted manifest is available, so it remains usable during failed or in-progress training runs.

## Important Notes

- The repo previously mixed older H1-bar and RecurrentPPO docs with the current volume-bar and MaskablePPO stack. Treat `RecurrentPPO` references as legacy history, not the supported training architecture.
- Live trading requires a trained model plus scaler files in `models/`.
