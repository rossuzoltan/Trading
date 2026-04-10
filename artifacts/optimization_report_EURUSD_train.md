# Automated Rule Candidate Generation — EURUSD
**Stage:** `train`
**Manifest Path:** `C:\dev\trading\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json`
**Evaluator Hash (`evaluate_oos.py`):** `bf42759b`
**Rule Logic Hash (`strategies/rule_logic.py`):** `56edcaee`
**Manifest Hash:** `2577d31e`

**Objective:** Maximize Net PnL & Expectancy subject to strict stability constraints.
**Method:** Exact-runtime evaluation over parameter grid.

## Passed Candidates (Ranked)
| Rank | Rule Family | Params | Net PnL | PF | Expectancy | Trades (L/S) | Signal (L/S) | Win% | MaxDD | Acc.Valid | L/S Mix | Confidence |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| 1 | volatility_breakout | `mean_revert=True, threshold_up=0.8, threshold_down=0.1` | $29.89 | 4.51 | $4.27 | 7 (7/0) | 1357/2459 | 85.7% | 1.0% | ✔️ | 100%/0% | exploratory |
| 2 | volatility_breakout | `mean_revert=True, threshold_up=0.9, threshold_down=0.1` | $29.89 | 4.51 | $4.27 | 7 (7/0) | 1357/1398 | 85.7% | 1.0% | ✔️ | 100%/0% | exploratory |
| 3 | volatility_breakout | `mean_revert=True, threshold_up=1.0, threshold_down=0.1` | $29.89 | 4.51 | $4.27 | 7 (7/0) | 1357/668 | 85.7% | 1.0% | ✔️ | 100%/0% | exploratory |
| 4 | volatility_breakout | `mean_revert=True, threshold_up=1.1, threshold_down=0.1` | $29.89 | 4.51 | $4.27 | 7 (7/0) | 1357/224 | 85.7% | 1.0% | ✔️ | 100%/0% | exploratory |
| 5 | volatility_breakout | `mean_revert=True, threshold_up=0.8, threshold_down=0.0` | $26.47 | 8.32 | $5.29 | 5 (5/0) | 615/2459 | 80.0% | 0.7% | ✔️ | 100%/0% | exploratory |
| 6 | volatility_breakout | `mean_revert=True, threshold_up=0.9, threshold_down=0.0` | $26.47 | 8.32 | $5.29 | 5 (5/0) | 615/1398 | 80.0% | 0.7% | ✔️ | 100%/0% | exploratory |
| 7 | volatility_breakout | `mean_revert=True, threshold_up=1.0, threshold_down=0.0` | $26.47 | 8.32 | $5.29 | 5 (5/0) | 615/668 | 80.0% | 0.7% | ✔️ | 100%/0% | exploratory |
| 8 | volatility_breakout | `mean_revert=True, threshold_up=1.1, threshold_down=0.0` | $26.47 | 8.32 | $5.29 | 5 (5/0) | 615/224 | 80.0% | 0.7% | ✔️ | 100%/0% | exploratory |
| 9 | volatility_breakout | `mean_revert=True, threshold_up=0.8, threshold_down=-0.1` | $24.16 | 5.07 | $4.83 | 5 (5/0) | 215/2459 | 80.0% | 0.7% | ✔️ | 100%/0% | exploratory |
| 10 | volatility_breakout | `mean_revert=True, threshold_up=0.9, threshold_down=-0.1` | $24.16 | 5.07 | $4.83 | 5 (5/0) | 215/1398 | 80.0% | 0.7% | ✔️ | 100%/0% | exploratory |
| 11 | volatility_breakout | `mean_revert=True, threshold_up=1.0, threshold_down=-0.1` | $24.16 | 5.07 | $4.83 | 5 (5/0) | 215/668 | 80.0% | 0.7% | ✔️ | 100%/0% | exploratory |
| 12 | volatility_breakout | `mean_revert=True, threshold_up=1.1, threshold_down=-0.1` | $24.16 | 5.07 | $4.83 | 5 (5/0) | 215/224 | 80.0% | 0.7% | ✔️ | 100%/0% | exploratory |
| 13 | mean_reversion | `long_threshold=-1.0, short_threshold=1.25, max_spread_z=0.5, max_time_delta_z=2.0, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $8.90 | 3.01 | $1.78 | 5 (5/0) | 1068/773 | 80.0% | 0.6% | ✔️ | 100%/0% | exploratory |
| 14 | mean_reversion | `long_threshold=-1.0, short_threshold=1.25, max_spread_z=0.5, max_time_delta_z=2.0, max_abs_ma20_slope=0.2, max_abs_ma50_slope=0.1` | $8.90 | 3.01 | $1.78 | 5 (5/0) | 1536/1128 | 80.0% | 0.6% | ✔️ | 100%/0% | exploratory |
| 15 | mean_reversion | `long_threshold=-1.0, short_threshold=1.25, max_spread_z=0.75, max_time_delta_z=2.0, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $8.90 | 3.01 | $1.78 | 5 (5/0) | 1105/792 | 80.0% | 0.6% | ✔️ | 100%/0% | exploratory |
| 16 | mean_reversion | `long_threshold=-1.0, short_threshold=1.25, max_spread_z=0.75, max_time_delta_z=2.0, max_abs_ma20_slope=0.2, max_abs_ma50_slope=0.1` | $8.90 | 3.01 | $1.78 | 5 (5/0) | 1580/1160 | 80.0% | 0.6% | ✔️ | 100%/0% | exploratory |
| 17 | mean_reversion | `long_threshold=-1.0, short_threshold=1.5, max_spread_z=0.5, max_time_delta_z=2.0, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $8.90 | 3.01 | $1.78 | 5 (5/0) | 1068/530 | 80.0% | 0.6% | ✔️ | 100%/0% | exploratory |
| 18 | mean_reversion | `long_threshold=-1.0, short_threshold=1.5, max_spread_z=0.5, max_time_delta_z=2.0, max_abs_ma20_slope=0.2, max_abs_ma50_slope=0.1` | $8.90 | 3.01 | $1.78 | 5 (5/0) | 1536/793 | 80.0% | 0.6% | ✔️ | 100%/0% | exploratory |
| 19 | mean_reversion | `long_threshold=-1.0, short_threshold=1.5, max_spread_z=0.75, max_time_delta_z=2.0, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $8.90 | 3.01 | $1.78 | 5 (5/0) | 1105/543 | 80.0% | 0.6% | ✔️ | 100%/0% | exploratory |
| 20 | mean_reversion | `long_threshold=-1.0, short_threshold=1.5, max_spread_z=0.75, max_time_delta_z=2.0, max_abs_ma20_slope=0.2, max_abs_ma50_slope=0.1` | $8.90 | 3.01 | $1.78 | 5 (5/0) | 1580/817 | 80.0% | 0.6% | ✔️ | 100%/0% | exploratory |
| 21 | mean_reversion | `long_threshold=-1.0, short_threshold=1.75, max_spread_z=0.5, max_time_delta_z=2.0, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $8.90 | 3.01 | $1.78 | 5 (5/0) | 1068/327 | 80.0% | 0.6% | ✔️ | 100%/0% | exploratory |
| 22 | mean_reversion | `long_threshold=-1.0, short_threshold=1.75, max_spread_z=0.5, max_time_delta_z=2.0, max_abs_ma20_slope=0.2, max_abs_ma50_slope=0.1` | $8.90 | 3.01 | $1.78 | 5 (5/0) | 1536/512 | 80.0% | 0.6% | ✔️ | 100%/0% | exploratory |
| 23 | mean_reversion | `long_threshold=-1.0, short_threshold=1.75, max_spread_z=0.75, max_time_delta_z=2.0, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $8.90 | 3.01 | $1.78 | 5 (5/0) | 1105/333 | 80.0% | 0.6% | ✔️ | 100%/0% | exploratory |
| 24 | mean_reversion | `long_threshold=-1.0, short_threshold=1.75, max_spread_z=0.75, max_time_delta_z=2.0, max_abs_ma20_slope=0.2, max_abs_ma50_slope=0.1` | $8.90 | 3.01 | $1.78 | 5 (5/0) | 1580/525 | 80.0% | 0.6% | ✔️ | 100%/0% | exploratory |

## Top 5 Rejected Constraints Example
| Rule Family | Params | Net PnL | PF | Trades | Rejection Reason |
|---|---|---:|---:|---:|---|
| mean_reversion | `long_threshold=-1.25, short_threshold=1.25, max_spread_z=0.5, max_time_delta_z=2.0, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $0.45 | 1.10 | 3 | Too few trades (3 < 5) | Low PF (1.10 < 1.25) |
| mean_reversion | `long_threshold=-1.25, short_threshold=1.25, max_spread_z=0.5, max_time_delta_z=2.0, max_abs_ma20_slope=0.2, max_abs_ma50_slope=0.1` | $0.45 | 1.10 | 3 | Too few trades (3 < 5) | Low PF (1.10 < 1.25) |
| mean_reversion | `long_threshold=-1.25, short_threshold=1.25, max_spread_z=0.75, max_time_delta_z=2.0, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $0.45 | 1.10 | 3 | Too few trades (3 < 5) | Low PF (1.10 < 1.25) |
| mean_reversion | `long_threshold=-1.25, short_threshold=1.25, max_spread_z=0.75, max_time_delta_z=2.0, max_abs_ma20_slope=0.2, max_abs_ma50_slope=0.1` | $0.45 | 1.10 | 3 | Too few trades (3 < 5) | Low PF (1.10 < 1.25) |
| mean_reversion | `long_threshold=-1.25, short_threshold=1.5, max_spread_z=0.5, max_time_delta_z=2.0, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $0.45 | 1.10 | 3 | Too few trades (3 < 5) | Low PF (1.10 < 1.25) |