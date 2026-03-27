# File Map

## Open These First For Most Tasks

- `train_agent.py`: primary training entrypoint and current model workflow
- `evaluate_oos.py`: symbol-scoped OOS evaluation
- `download_dukascopy.py`: raw tick ingestion and refresh logic
- `build_volume_bars.py`: training dataset construction

## Open These Only When The Task Requires Them

- `feature_engine.py`: engineered features and any feature/schema changes
- `runtime_gym_env.py`: supported runtime training environment
- `live_bridge.py`: MT5 execution behavior
- `mt5_live_preflight.py`: mandatory live-readiness gate
- `artifact_manifest.py`: artifact compatibility and manifest validation
- `project_paths.py`: repo path resolution and artifact locations

## Treat As Compatibility Or Legacy Unless Proven Needed

- `trading_env.py`: older environment path
- `legacy/`: archived experiments and superseded scripts
- H1-specific flows: alternate research path, not the default RL runtime

## Tests And Ops

- `tests/`: regression coverage
- `tools/project_healthcheck.py`: environment and dependency check
- `tools/training_status.py`: PPO and heartbeat diagnostics
- `tools/summarize_execution_audit.py`: live execution drift summaries
