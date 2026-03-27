---
name: turbo-forex-rl
description: Repository-specific operating guide for the trading project at c:\dev\trading. Use when Codex needs to modify, debug, test, train, evaluate, review, or operate this Python forex reinforcement-learning stack, especially for Dukascopy data download and repair, volume-bar construction, symbol-scoped MaskablePPO training, out-of-sample evaluation, MT5 live trading, or project runbooks. Also use when continuing prior work in this repository and the fastest safe path is to start from the handoff docs instead of re-reading the whole codebase.
---

# Turbo Forex RL

## Start Here

- Work from the repository root: `c:\dev\trading`.
- Use `.\.venv\Scripts\python.exe` for Python commands unless you are only reading files.
- Read [references/start-here.md](references/start-here.md) first.
- Read [references/file-map.md](references/file-map.md) after you identify the subsystem you need.
- Read [references/research-policy.md](references/research-policy.md) when the task involves strategy quality, deployment readiness, or whether RL is justified.

## Follow This Workflow

1. Classify the request before opening code.
   - Data ingest or repair: start with `download_dukascopy.py` and `build_volume_bars.py`.
   - Training or evaluation: start with `train_agent.py`, `evaluate_oos.py`, and related manifests under `models/`.
   - Live trading or ops: start with `live_bridge.py`, `mt5_live_preflight.py`, and ops helpers under `tools/`.
   - Review or research judgement: read [references/research-policy.md](references/research-policy.md) before making claims about readiness or model quality.
2. Load the minimum context.
   - For current repo state, verified commands, and known data issues, read [references/start-here.md](references/start-here.md).
   - For file targeting, read [references/file-map.md](references/file-map.md).
   - Open only the files that match the classified task unless evidence forces a wider audit.
3. Preserve the supported architecture.
   - Treat `MaskablePPO + RuntimeGymEnv + volume bars + symbol-scoped artifacts` as the supported path.
   - Treat `RecurrentPPO`, H1 experiments, and older `trading_env.py` flows as legacy or compatibility paths unless the current task explicitly requires them.
   - Do not re-audit `event_pipeline.py` or the live runtime unless a failing test, replay mismatch, or the user request forces it.
4. Keep the edit surface small.
   - For data-repair or training tasks, inspect `download_dukascopy.py`, `build_volume_bars.py`, `train_agent.py`, `evaluate_oos.py`, and directly related tests or tools first.
   - If you change features or model artifacts, verify the matching scaler, manifest, and model outputs under `models/`.
5. Verify with project defaults.
   - Use `.\.venv\Scripts\python.exe .\tools\project_healthcheck.py` when runtime health matters.
   - Use `.\.venv\Scripts\python.exe -m unittest discover tests` for repo regression coverage.
   - Use `.\.venv\Scripts\python.exe .\mt5_live_preflight.py --symbol EURUSD --ticks-per-bar 5000` before treating MT5 live changes as ready.
6. Apply the project decision rules.
   - Optimize for correctness, parity, auditability, and cost realism before speed or model novelty.
   - Treat thin EURUSD or GBPUSD data coverage as a likely data-quality issue, not a reason to loosen validation.
   - Assume edge claims are false positives until they survive out-of-sample checks, realistic costs, and live/backtest parity.
   - Prefer simpler baselines over RL unless the evidence clearly justifies RL.

## Operating Rules

- Avoid broad repo re-analysis when the handoff docs already narrow the scope.
- Prefer symbol-scoped training and evaluation flows when touching current model work.
- If the task touches live trading behavior, confirm artifact parity, manifests, preflight output, and audit trails before declaring success.
- If the request is about large architectural redesign, justify that expansion from current evidence before touching unrelated subsystems.

## Reference Pack

- [references/start-here.md](references/start-here.md): fastest onboarding path, verified commands, and current data caveats.
- [references/file-map.md](references/file-map.md): which files matter for data, training, evaluation, runtime, and ops.
- [references/research-policy.md](references/research-policy.md): evaluation, do-not-deploy gates, and when AI or RL is not justified.
