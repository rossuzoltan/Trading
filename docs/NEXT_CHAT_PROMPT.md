Use the `turbo-forex-rl` skill first.

Repository: `C:\dev\trading-dev`

Important context:
- A training run is actively running in the original workspace at `C:\dev\trading`.
- Do not modify files in `C:\dev\trading`.
- Treat `C:\dev\trading-dev` as the safe development worktree.
- The current blocker is strategy/data-quality evidence, not runtime architecture.

Read first:
1. `C:\dev\trading-dev\docs\NEXT_AGENT_CONTEXT.md`
2. `C:\dev\trading-dev\docs\NEXT_AGENT_FILE_MAP.md`
3. `C:\dev\trading-dev\docs\NEXT_AGENT_RUNBOOK.md`
4. `C:\dev\trading-dev\.codex\skills\turbo-forex-rl\references\research-policy.md`

Current objective:
- Improve the project in ways that are safe to do while the live training continues elsewhere.

Priority tasks:
1. Audit and fix evaluation/reporting inconsistencies, especially any mismatch between summary metrics (`trade_count`, `win_rate`, `profit_factor`, `net_pnl`) and execution diagnostics in training/eval outputs.
2. Add simple baseline experiments on the same cost model so RL can be compared against rule-based / linear / small tree baselines.
3. Add lightweight experiment-analysis helpers that summarize heartbeats, evals, and failure reasons without touching the training hot path.
4. Only propose AI sidecar integrations if they stay outside the live trading decision loop.

Guardrails:
- Do not redesign the runtime.
- Do not touch MT5/live execution unless a direct dependency forces it.
- Prefer small, verifiable changes with tests.
- Run focused tests after changes.

Definition of success:
- Clearer, trustworthy evaluation numbers.
- At least one simpler baseline path added or scaffolded.
- No interference with the active training workspace.
