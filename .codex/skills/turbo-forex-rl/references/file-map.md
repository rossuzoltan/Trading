# File Map

## Open These First For Most Tasks

- `train_agent.py`: primary training entrypoint and current model workflow
- `evaluate_oos.py`: symbol-scoped OOS evaluation that loads the symbol manifest first
- `download_dukascopy.py`: raw tick ingestion and refresh logic
- `build_volume_bars.py`: training dataset construction

## Open These Only When The Task Requires Them

- `feature_engine.py`: engineered features and any feature/schema changes
- `runtime_gym_env.py`: supported runtime training environment
- `live_bridge.py`: MT5 execution behavior
- `mt5_live_preflight.py`: mandatory live-readiness gate
- `artifact_manifest.py`: artifact compatibility, checksums, and manifest validation
- `project_paths.py`: repo path resolution and symbol-scoped artifact locations

## Treat As Compatibility Or Legacy Unless Proven Needed

- `trading_env.py`: older environment path
- `legacy/`: archived experiments and superseded scripts
- H1-specific flows: alternate research path, not the default RL runtime

## Tests And Ops

- `tests/`: regression coverage
- `tools/project_healthcheck.py`: dataset integrity, bar-spec, and runtime-artifact smoke check
- `training_status.py`: PPO and heartbeat diagnostics
- `summarize_execution_audit.py`: live execution drift summaries
