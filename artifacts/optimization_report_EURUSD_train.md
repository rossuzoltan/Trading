# Automated Rule Candidate Generation — EURUSD
**Stage:** `train`
**Manifest Path:** `C:\dev\trading\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json`
**Evaluator Hash (`evaluate_oos.py`):** `3434c70a`
**Rule Logic Hash (`strategies/rule_logic.py`):** `cc035d63`
**Manifest Hash:** `2577d31e`

**Objective:** Maximize Net PnL & Expectancy subject to strict stability constraints.
**Method:** Exact-runtime evaluation over parameter grid.

## Passed Candidates (Ranked)
| Rank | Rule Family | Params | Net PnL | PF | Expectancy | Trades | Win% | MaxDD | Acc.Valid | Rollover | L/S Mix |
|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|

## Top 5 Rejected Constraints Example
| Rule Family | Params | Net PnL | PF | Trades | Rejection Reason |
|---|---|---:|---:|---:|---|
| mean_reversion | `threshold=1.0, max_spread_z=0.5, max_time_delta_z=2.0, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $-54.46 | 0.00 | 5 | Too few trades (5 < 10) | Low PF (0.00 < 1.25) | Negative Expectancy ($-10.89) |
| mean_reversion | `threshold=1.0, max_spread_z=0.5, max_time_delta_z=2.0, max_abs_ma20_slope=0.2, max_abs_ma50_slope=0.1` | $-51.97 | 0.00 | 5 | Too few trades (5 < 10) | Low PF (0.00 < 1.25) | Negative Expectancy ($-10.39) |
| mean_reversion | `threshold=1.0, max_spread_z=0.75, max_time_delta_z=2.0, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $-54.46 | 0.00 | 5 | Too few trades (5 < 10) | Low PF (0.00 < 1.25) | Negative Expectancy ($-10.89) |
| mean_reversion | `threshold=1.0, max_spread_z=0.75, max_time_delta_z=2.0, max_abs_ma20_slope=0.2, max_abs_ma50_slope=0.1` | $-51.97 | 0.00 | 5 | Too few trades (5 < 10) | Low PF (0.00 < 1.25) | Negative Expectancy ($-10.39) |
| mean_reversion | `threshold=1.25, max_spread_z=0.5, max_time_delta_z=2.0, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $-50.48 | 0.82 | 38 | Low PF (0.82 < 1.25) | Negative Expectancy ($-1.33) | Direction skewed (Max 100.0% > 85%) |