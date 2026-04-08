# Automated Rule Candidate Generation — EURUSD
**Objective:** Maximize Net PnL & Expectancy subject to strict stability constraints.
**Method:** Exact-runtime evaluation over parameter grid.

## Passed Candidates (Ranked)
| Rank | Rule Family | Params | Net PnL | PF | Expectancy | Trades | Win% | MaxDD | L/S Mix |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | price_mr_spread_filter | `threshold=3.0` | $44.38 | 1.81 | $4.44 | 10 | 50.0% | 3.2% | 0%/100% |

## Top 5 Rejected Constraints Example
| Rule Family | Params | Net PnL | PF | Trades | Rejection Reason |
|---|---|---:|---:|---:|---|
| mean_reversion | `threshold=0.8` | $-51.53 | 0.68 | 22 | Low PF (0.68 < 1.25) | Negative Expectancy ($-2.34) | Direction skewed (Max 100.0% > 85%) |
| mean_reversion | `threshold=1.0` | $-52.26 | 0.85 | 49 | Low PF (0.85 < 1.25) | Negative Expectancy ($-1.07) | Direction skewed (Max 100.0% > 85%) |
| mean_reversion | `threshold=1.2` | $-26.82 | 0.94 | 67 | Low PF (0.94 < 1.25) | Negative Expectancy ($-0.40) | Direction skewed (Max 100.0% > 85%) |
| mean_reversion | `threshold=1.5` | $-7.19 | 0.00 | 1 | Too few trades (1 < 10) | Low PF (0.00 < 1.25) | Negative Expectancy ($-7.19) |
| mean_reversion | `threshold=2.0` | $19.43 | 1.93 | 4 | Too few trades (4 < 10) |