# Automated Rule Candidate Generation — EURUSD
**Stage:** `train`
**Manifest Path:** `C:\dev\trading\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json`
**Evaluator Hash (`evaluate_oos.py`):** `110dfff3`
**Rule Logic Hash (`strategies/rule_logic.py`):** `e1c9bf0f`
**Manifest Hash:** `fadd635a`

**Objective:** Maximize Net PnL & Expectancy subject to strict stability constraints.
**Method:** Exact-runtime evaluation over parameter grid.

## Passed Candidates (Ranked)
| Rank | Rule Family | Params | Net PnL | PF | Expectancy | Trades (L/S) | Signal (L/S) | Win% | MaxDD | Acc.Valid | L/S Mix | Confidence |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| 1 | mean_reversion | `long_threshold=-0.5, short_threshold=0.5, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $17.34 | 1.96 | $1.93 | 9 (9/0) | 3382/3260 | 77.8% | 1.0% | ✔️ | 100%/0% | exploratory |
| 2 | mean_reversion | `long_threshold=-0.5, short_threshold=0.75, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $17.34 | 1.96 | $1.93 | 9 (9/0) | 3382/2708 | 77.8% | 1.0% | ✔️ | 100%/0% | exploratory |
| 3 | mean_reversion | `long_threshold=-0.5, short_threshold=1.0, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $17.34 | 1.96 | $1.93 | 9 (9/0) | 3382/2157 | 77.8% | 1.0% | ✔️ | 100%/0% | exploratory |
| 4 | mean_reversion | `long_threshold=-0.5, short_threshold=1.25, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $17.34 | 1.96 | $1.93 | 9 (9/0) | 3382/1642 | 77.8% | 1.0% | ✔️ | 100%/0% | exploratory |
| 5 | mean_reversion | `long_threshold=-0.5, short_threshold=1.5, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $17.34 | 1.96 | $1.93 | 9 (9/0) | 3382/1163 | 77.8% | 1.0% | ✔️ | 100%/0% | exploratory |
| 6 | mean_reversion | `long_threshold=-0.5, short_threshold=1.75, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $17.34 | 1.96 | $1.93 | 9 (9/0) | 3382/769 | 77.8% | 1.0% | ✔️ | 100%/0% | exploratory |
| 7 | mean_reversion | `long_threshold=-0.5, short_threshold=2.0, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $17.34 | 1.96 | $1.93 | 9 (9/0) | 3382/482 | 77.8% | 1.0% | ✔️ | 100%/0% | exploratory |
| 8 | mean_reversion | `long_threshold=-0.75, short_threshold=0.5, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $14.30 | 1.79 | $1.79 | 8 (8/0) | 2784/3260 | 75.0% | 1.0% | ✔️ | 100%/0% | exploratory |
| 9 | mean_reversion | `long_threshold=-0.75, short_threshold=0.75, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $14.30 | 1.79 | $1.79 | 8 (8/0) | 2784/2708 | 75.0% | 1.0% | ✔️ | 100%/0% | exploratory |
| 10 | mean_reversion | `long_threshold=-0.75, short_threshold=1.0, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $14.30 | 1.79 | $1.79 | 8 (8/0) | 2784/2157 | 75.0% | 1.0% | ✔️ | 100%/0% | exploratory |
| 11 | mean_reversion | `long_threshold=-0.75, short_threshold=1.25, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $14.30 | 1.79 | $1.79 | 8 (8/0) | 2784/1642 | 75.0% | 1.0% | ✔️ | 100%/0% | exploratory |
| 12 | mean_reversion | `long_threshold=-0.75, short_threshold=1.5, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $14.30 | 1.79 | $1.79 | 8 (8/0) | 2784/1163 | 75.0% | 1.0% | ✔️ | 100%/0% | exploratory |
| 13 | mean_reversion | `long_threshold=-0.75, short_threshold=1.75, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $14.30 | 1.79 | $1.79 | 8 (8/0) | 2784/769 | 75.0% | 1.0% | ✔️ | 100%/0% | exploratory |
| 14 | mean_reversion | `long_threshold=-0.75, short_threshold=2.0, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $14.30 | 1.79 | $1.79 | 8 (8/0) | 2784/482 | 75.0% | 1.0% | ✔️ | 100%/0% | exploratory |
| 15 | mean_reversion | `long_threshold=-0.75, short_threshold=0.5, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $9.15 | 1.39 | $1.02 | 9 (9/0) | 2866/3340 | 66.7% | 1.1% | ✔️ | 100%/0% | exploratory |
| 16 | mean_reversion | `long_threshold=-0.75, short_threshold=0.75, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $9.15 | 1.39 | $1.02 | 9 (9/0) | 2866/2775 | 66.7% | 1.1% | ✔️ | 100%/0% | exploratory |
| 17 | mean_reversion | `long_threshold=-0.75, short_threshold=1.0, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $9.15 | 1.39 | $1.02 | 9 (9/0) | 2866/2210 | 66.7% | 1.1% | ✔️ | 100%/0% | exploratory |
| 18 | mean_reversion | `long_threshold=-0.75, short_threshold=1.25, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $9.15 | 1.39 | $1.02 | 9 (9/0) | 2866/1688 | 66.7% | 1.1% | ✔️ | 100%/0% | exploratory |
| 19 | mean_reversion | `long_threshold=-0.75, short_threshold=1.5, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $9.15 | 1.39 | $1.02 | 9 (9/0) | 2866/1199 | 66.7% | 1.1% | ✔️ | 100%/0% | exploratory |
| 20 | mean_reversion | `long_threshold=-0.75, short_threshold=1.75, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $9.15 | 1.39 | $1.02 | 9 (9/0) | 2866/794 | 66.7% | 1.1% | ✔️ | 100%/0% | exploratory |
| 21 | mean_reversion | `long_threshold=-0.75, short_threshold=2.0, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $9.15 | 1.39 | $1.02 | 9 (9/0) | 2866/502 | 66.7% | 1.1% | ✔️ | 100%/0% | exploratory |
| 22 | mean_reversion | `long_threshold=-0.75, short_threshold=0.5, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $5.99 | 1.43 | $1.00 | 6 (6/0) | 1449/1725 | 66.7% | 1.0% | ✔️ | 100%/0% | exploratory |
| 23 | mean_reversion | `long_threshold=-0.75, short_threshold=0.75, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $5.99 | 1.43 | $1.00 | 6 (6/0) | 1449/1389 | 66.7% | 1.0% | ✔️ | 100%/0% | exploratory |
| 24 | mean_reversion | `long_threshold=-0.75, short_threshold=1.0, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $5.99 | 1.43 | $1.00 | 6 (6/0) | 1449/1075 | 66.7% | 1.0% | ✔️ | 100%/0% | exploratory |
| 25 | mean_reversion | `long_threshold=-0.75, short_threshold=1.25, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $5.99 | 1.43 | $1.00 | 6 (6/0) | 1449/792 | 66.7% | 1.0% | ✔️ | 100%/0% | exploratory |
| 26 | mean_reversion | `long_threshold=-0.75, short_threshold=1.5, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $5.99 | 1.43 | $1.00 | 6 (6/0) | 1449/543 | 66.7% | 1.0% | ✔️ | 100%/0% | exploratory |
| 27 | mean_reversion | `long_threshold=-0.75, short_threshold=1.75, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $5.99 | 1.43 | $1.00 | 6 (6/0) | 1449/333 | 66.7% | 1.0% | ✔️ | 100%/0% | exploratory |
| 28 | mean_reversion | `long_threshold=-0.75, short_threshold=2.0, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $5.99 | 1.43 | $1.00 | 6 (6/0) | 1449/210 | 66.7% | 1.0% | ✔️ | 100%/0% | exploratory |

## Top 5 Rejected Constraints Example
| Rule Family | Params | Net PnL | PF | Trades | Rejection Reason |
|---|---|---:|---:|---:|---|
| mean_reversion | `long_threshold=-0.5, short_threshold=0.5, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $-4.75 | 0.69 | 5 | Low PF (0.69 < 1.25) | Negative Expectancy ($-0.95) |
| mean_reversion | `long_threshold=-0.5, short_threshold=0.5, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.5, max_abs_ma50_slope=0.3` | $35.43 | 2.47 | 14 | Direction skewed (Max 100.0% > 85%) |
| mean_reversion | `long_threshold=-0.5, short_threshold=0.5, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $-9.56 | 0.52 | 6 | Low PF (0.52 < 1.25) | Negative Expectancy ($-1.59) |
| mean_reversion | `long_threshold=-0.5, short_threshold=0.5, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $12.18 | 1.52 | 10 | Direction skewed (Max 100.0% > 85%) |
| mean_reversion | `long_threshold=-0.5, short_threshold=0.5, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.5, max_abs_ma50_slope=0.3` | $27.92 | 1.77 | 15 | Direction skewed (Max 100.0% > 85%) |