# Documentation Guide

This folder contains the canonical repo documentation. The supported production
path is `Rule-First + AlphaGate + manifest-driven shadow gating`.
Legacy PPO/RL docs are kept only for research continuity.

## Read In This Order

1. [`../README.md`](../README.md)
2. [`NEXT_AGENT_CONTEXT.md`](NEXT_AGENT_CONTEXT.md)
3. [`NEXT_AGENT_FILE_MAP.md`](NEXT_AGENT_FILE_MAP.md)
4. [`NEXT_AGENT_RUNBOOK.md`](NEXT_AGENT_RUNBOOK.md)

## Keep Handy

- [`CURRENT_USAGE_GUIDE.md`](CURRENT_USAGE_GUIDE.md): practical day-to-day usage, current training/eval flow, and experiment profiles.
- [`PROFITABILITY_PLAN.md`](PROFITABILITY_PLAN.md): canonical paper-live profitability gates, shadow evidence layout, and anchor status model.
- [`evaluation_accounting.md`](evaluation_accounting.md): what the evaluation metrics mean and which accounting source is authoritative.
- [`research_runner.md`](research_runner.md): safe proposal-driven experimentation, research ledgering, scoring, and result storage.
- [`operating_checklist.md`](operating_checklist.md): live-readiness evidence order and operator gate.
- [`h1_data.md`](h1_data.md): optional H1 dataset builder for side research. This is separate from the default volume-bar training path.

## Main Code Surfaces

- Data: `download_dukascopy.py`, `build_volume_bars.py`, `build_h1_dataset.py`
- Training and eval: `train_agent.py`, `runtime_gym_env.py`, `evaluate_oos.py`, `feature_engine.py`, `trading_config.py`
- Live and ops: `live_bridge.py`, `mt5_live_preflight.py`, `restart_drill.py`, `ops_attestation_helper.py`, `live_operating_checklist.py`, `training_status.py`, `summarize_execution_audit.py`
