# Evaluation And Accounting

This repo treats closed-trade economics as the authoritative source for profit
claims.

## Authoritative Metrics

- `evaluate_oos.py` and training full-path eval both build economics through
  `runtime_common.build_evaluation_accounting()`.
- `trade_count`, `gross_pnl_usd`, `net_pnl_usd`, `profit_factor`,
  `expectancy_usd`, and transaction-cost totals all come from the closed trade
  log plus reconciliation against execution diagnostics.
- Reward shaping is not an economic metric. Use it to understand PPO training,
  not to judge deployability.

## What To Trust

- Trust `latest_eval` and replay reports only when
  `metric_reconciliation.passed=true`.
- Trust `execution_diagnostics` for action distribution, closed-trade counts,
  cost totals, and reward-component decomposition.
- Treat PPO internals such as `explained_variance` and `approx_kl` as training
  diagnostics only. They do not prove profitability.

## Reporting Notes

- The best-checkpoint snapshot now records PPO diagnostics at the moment a new
  best eval is saved, so best-eval economics are not mixed with end-of-training
  PPO metrics.
- `tools/training_status.py` falls back to the latest checkpoint diagnostics
  when promoted `models/training_diagnostics_<symbol>.json` is absent.
- `evaluate_oos.py` now writes an authoritative `runtime_parity_verdict`
  inside the replay report. That verdict contains the best runtime baseline
  under replay costs, the best research baseline summary for reference, and
  slippage-stress results at stricter execution assumptions.
- `compare_oos_baselines.py` compares RL replay against both research baselines
  and simple runtime-rule baselines under the same replay cost model, but it is
  now secondary to the replay-embedded verdict.

## Practical Workflow

1. Run `.\.venv\Scripts\python.exe .\training_status.py --symbol EURUSD`
2. Check `latest_eval` net PnL, transaction costs, and the best holdout
   baseline in the status output.
3. Run `.\.venv\Scripts\python.exe .\evaluate_oos.py` and inspect
   `runtime_parity_verdict`, reject-fast cost diagnostics, and reconciliation.
4. Run `.\.venv\Scripts\python.exe .\compare_oos_baselines.py --symbol EURUSD`
   when you want a standalone comparison report.
5. Reject deployment when reconciliation fails, runtime-parity baselines do not
   support the research gate, or profitability fails under stricter slippage.
