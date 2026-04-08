# Automated Rule Candidate Generation — EURUSD
**Stage:** `train`
**Manifest Path:** `C:\dev\trading\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json`
**Evaluator Hash (`evaluate_oos.py`):** `3434c70a`
**Rule Logic Hash (`strategies/rule_logic.py`):** `5ae99788`
**Manifest Hash:** `2c52c411`

**Objective:** Maximize Net PnL & Expectancy subject to strict stability constraints.
**Method:** Exact-runtime evaluation over parameter grid.

## Passed Candidates (Ranked)
| Rank | Rule Family | Params | Net PnL | PF | Expectancy | Trades | Win% | MaxDD | Acc.Valid | Rollover | L/S Mix |
|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|

## Top 5 Rejected Constraints Example
| Rule Family | Params | Net PnL | PF | Trades | Rejection Reason |
|---|---|---:|---:|---:|---|
| mean_reversion | `threshold=0.8` | $-51.53 | 0.68 | 22 | Low PF (0.68 < 1.25) | Negative Expectancy ($-2.34) | Direction skewed (Max 100.0% > 85%) |
| mean_reversion | `threshold=1.0` | $-52.26 | 0.85 | 49 | Low PF (0.85 < 1.25) | Negative Expectancy ($-1.07) | Direction skewed (Max 100.0% > 85%) |
| mean_reversion | `threshold=1.2` | $-26.82 | 0.94 | 67 | Low PF (0.94 < 1.25) | Negative Expectancy ($-0.40) | Direction skewed (Max 100.0% > 85%) |
| mean_reversion | `threshold=1.5` | $-7.19 | 0.00 | 1 | Too few trades (1 < 10) | Low PF (0.00 < 1.25) | Negative Expectancy ($-7.19) |
| mean_reversion | `threshold=2.0` | $19.43 | 1.93 | 4 | Too few trades (4 < 10) |