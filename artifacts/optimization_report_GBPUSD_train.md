# Automated Rule Candidate Generation — GBPUSD
**Stage:** `train`
**Manifest Path:** `C:\dev\trading\models\rc1\gbpusd_10k_v1_mr_rc1\manifest.json`
**Evaluator Hash (`evaluate_oos.py`):** `3434c70a`
**Rule Logic Hash (`strategies/rule_logic.py`):** `cc035d63`
**Manifest Hash:** `eb86f2f1`

**Objective:** Maximize Net PnL & Expectancy subject to strict stability constraints.
**Method:** Exact-runtime evaluation over parameter grid.

## Passed Candidates (Ranked)
| Rank | Rule Family | Params | Net PnL | PF | Expectancy | Trades | Win% | MaxDD | Acc.Valid | Rollover | L/S Mix |
|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|

## Top 5 Rejected Constraints Example
| Rule Family | Params | Net PnL | PF | Trades | Rejection Reason |
|---|---|---:|---:|---:|---|
| mean_reversion | `threshold=1.0, max_spread_z=0.5, max_time_delta_z=2.0, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $-55.71 | 0.49 | 14 | Low PF (0.49 < 1.25) | Negative Expectancy ($-3.98) | Direction skewed (Max 100.0% > 85%) |
| mean_reversion | `threshold=1.0, max_spread_z=0.5, max_time_delta_z=2.0, max_abs_ma20_slope=0.2, max_abs_ma50_slope=0.1` | $-57.16 | 0.75 | 33 | Low PF (0.75 < 1.25) | Negative Expectancy ($-1.73) | Direction skewed (Max 100.0% > 85%) |
| mean_reversion | `threshold=1.0, max_spread_z=0.75, max_time_delta_z=2.0, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $-55.71 | 0.49 | 14 | Low PF (0.49 < 1.25) | Negative Expectancy ($-3.98) | Direction skewed (Max 100.0% > 85%) |
| mean_reversion | `threshold=1.0, max_spread_z=0.75, max_time_delta_z=2.0, max_abs_ma20_slope=0.2, max_abs_ma50_slope=0.1` | $-53.39 | 0.69 | 24 | Low PF (0.69 < 1.25) | Negative Expectancy ($-2.22) | Direction skewed (Max 100.0% > 85%) |
| mean_reversion | `threshold=1.25, max_spread_z=0.5, max_time_delta_z=2.0, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $-54.41 | 0.25 | 8 | Too few trades (8 < 10) | Low PF (0.25 < 1.25) | Negative Expectancy ($-6.80) |