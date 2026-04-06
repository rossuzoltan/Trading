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
- `compare_oos_baselines.py` compares RL replay against both research baselines
  and simple runtime-rule baselines under the same replay cost model.

## Practical Workflow

1. Run `.\.venv\Scripts\python.exe .\training_status.py --symbol EURUSD`
2. Check `latest_eval` net PnL, transaction costs, and the best holdout
   baseline in the status output.
3. Run `.\.venv\Scripts\python.exe .\compare_oos_baselines.py --symbol EURUSD`
   after producing replay artifacts.
4. Reject deployment when reconciliation fails, costs dominate gross PnL, or
   simple baselines beat the RL replay.
