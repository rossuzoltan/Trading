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
| 1 | mean_reversion | `long_threshold=-0.5, short_threshold=0.5, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $275.73 | 1.40 | $1.25 | 220 (100/120) | 3484/3340 | 63.2% | 13.1% | ✔️ | 45%/55% | stable |
| 2 | mean_reversion | `long_threshold=-0.5, short_threshold=0.5, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.5, max_abs_ma50_slope=0.3` | $247.25 | 1.26 | $0.88 | 280 (133/147) | 4046/4072 | 63.6% | 12.3% | ✔️ | 48%/52% | stable |
| 3 | mean_reversion | `long_threshold=-0.5, short_threshold=1.0, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $233.27 | 1.43 | $1.16 | 201 (100/101) | 3484/2210 | 65.7% | 11.0% | ✔️ | 50%/50% | stable |
| 4 | mean_reversion | `long_threshold=-0.5, short_threshold=1.0, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.5, max_abs_ma50_slope=0.3` | $229.79 | 1.29 | $0.88 | 260 (133/127) | 4046/2787 | 65.8% | 9.1% | ✔️ | 51%/49% | stable |
| 5 | mean_reversion | `long_threshold=-0.5, short_threshold=0.75, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $213.43 | 1.34 | $1.01 | 211 (100/111) | 3484/2775 | 63.5% | 11.9% | ✔️ | 47%/53% | stable |
| 6 | mean_reversion | `long_threshold=-0.5, short_threshold=1.0, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.5, max_abs_ma50_slope=0.3` | $201.84 | 1.27 | $0.80 | 252 (127/125) | 3927/2719 | 65.5% | 8.5% | ✔️ | 50%/50% | stable |
| 7 | mean_reversion | `long_threshold=-0.5, short_threshold=2.0, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.5, max_abs_ma50_slope=0.3` | $198.70 | 1.40 | $1.18 | 169 (133/36) | 4046/613 | 64.5% | 6.4% | ✔️ | 79%/21% | stable |
| 8 | mean_reversion | `long_threshold=-0.5, short_threshold=2.0, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.5, max_abs_ma50_slope=0.3` | $188.70 | 1.40 | $1.16 | 162 (127/35) | 3927/591 | 63.6% | 6.3% | ✔️ | 78%/22% | stable |
| 9 | mean_reversion | `long_threshold=-0.5, short_threshold=1.5, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.5, max_abs_ma50_slope=0.3` | $186.45 | 1.29 | $0.86 | 216 (133/83) | 4046/1503 | 64.4% | 5.6% | ✔️ | 62%/38% | stable |
| 10 | mean_reversion | `long_threshold=-0.5, short_threshold=1.5, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.5, max_abs_ma50_slope=0.3` | $179.99 | 1.30 | $0.86 | 209 (127/82) | 3927/1463 | 63.6% | 5.6% | ✔️ | 61%/39% | stable |
| 11 | mean_reversion | `long_threshold=-0.5, short_threshold=1.0, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $179.16 | 1.35 | $0.93 | 192 (93/99) | 3382/2157 | 64.6% | 10.8% | ✔️ | 48%/52% | stable |
| 12 | mean_reversion | `long_threshold=-0.5, short_threshold=2.0, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $175.23 | 1.53 | $1.36 | 129 (100/29) | 3484/502 | 65.1% | 7.7% | ✔️ | 78%/22% | stable |
| 13 | mean_reversion | `long_threshold=-0.5, short_threshold=1.75, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.5, max_abs_ma50_slope=0.3` | $167.26 | 1.30 | $0.89 | 187 (133/54) | 4046/983 | 62.6% | 7.0% | ✔️ | 71%/29% | stable |
| 14 | mean_reversion | `long_threshold=-0.5, short_threshold=1.75, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.5, max_abs_ma50_slope=0.3` | $158.90 | 1.30 | $0.88 | 180 (127/53) | 3927/955 | 61.7% | 6.9% | ✔️ | 71%/29% | stable |
| 15 | mean_reversion | `long_threshold=-0.5, short_threshold=1.5, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $153.98 | 1.35 | $0.93 | 166 (100/66) | 3484/1199 | 64.5% | 8.3% | ✔️ | 60%/40% | stable |
| 16 | mean_reversion | `long_threshold=-0.5, short_threshold=1.75, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $151.91 | 1.40 | $1.07 | 142 (100/42) | 3484/794 | 62.7% | 7.9% | ✔️ | 70%/30% | stable |
| 17 | mean_reversion | `long_threshold=-0.5, short_threshold=2.0, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $145.92 | 1.45 | $1.21 | 121 (93/28) | 3382/482 | 63.6% | 7.8% | ✔️ | 77%/23% | stable |
| 18 | mean_reversion | `long_threshold=-0.5, short_threshold=1.5, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $136.63 | 1.33 | $0.86 | 158 (93/65) | 3382/1163 | 63.3% | 8.4% | ✔️ | 59%/41% | stable |
| 19 | mean_reversion | `long_threshold=-0.75, short_threshold=2.0, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.5, max_abs_ma50_slope=0.3` | $132.68 | 1.30 | $0.86 | 154 (118/36) | 3389/613 | 62.3% | 5.0% | ✔️ | 77%/23% | stable |
| 20 | mean_reversion | `long_threshold=-0.75, short_threshold=2.0, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.5, max_abs_ma50_slope=0.3` | $132.08 | 1.32 | $0.90 | 147 (112/35) | 3290/591 | 61.9% | 4.9% | ✔️ | 76%/24% | stable |
| 21 | mean_reversion | `long_threshold=-0.5, short_threshold=1.75, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $126.93 | 1.35 | $0.95 | 134 (93/41) | 3382/769 | 61.2% | 8.0% | ✔️ | 69%/31% | stable |
| 22 | mean_reversion | `long_threshold=-0.75, short_threshold=2.0, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $125.70 | 1.45 | $1.09 | 115 (86/29) | 2866/502 | 63.5% | 5.7% | ✔️ | 75%/25% | stable |
| 23 | mean_reversion | `long_threshold=-0.75, short_threshold=2.0, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $111.23 | 1.42 | $1.04 | 107 (79/28) | 2784/482 | 61.7% | 5.7% | ✔️ | 74%/26% | stable |
| 24 | mean_reversion | `long_threshold=-0.75, short_threshold=1.75, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $108.87 | 1.35 | $0.85 | 128 (86/42) | 2866/794 | 60.9% | 5.8% | ✔️ | 67%/33% | stable |
| 25 | mean_reversion | `long_threshold=-0.75, short_threshold=1.75, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $96.65 | 1.86 | $1.79 | 54 (44/10) | 1449/333 | 59.3% | 3.0% | ✔️ | 81%/19% | stable |
| 26 | mean_reversion | `long_threshold=-0.5, short_threshold=0.75, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $96.29 | 1.42 | $1.04 | 93 (56/37) | 1864/1423 | 55.9% | 7.3% | ✔️ | 60%/40% | stable |
| 27 | mean_reversion | `long_threshold=-0.5, short_threshold=1.75, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $93.62 | 1.66 | $1.53 | 61 (51/10) | 1810/333 | 57.4% | 5.4% | ✔️ | 84%/16% | stable |
| 28 | mean_reversion | `long_threshold=-0.5, short_threshold=1.75, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $92.20 | 1.60 | $1.40 | 66 (56/10) | 1864/344 | 56.1% | 5.6% | ✔️ | 85%/15% | stable |
| 29 | mean_reversion | `long_threshold=-0.75, short_threshold=1.75, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $89.56 | 1.30 | $0.75 | 120 (79/41) | 2784/769 | 59.2% | 5.7% | ✔️ | 66%/34% | stable |
| 30 | mean_reversion | `long_threshold=-0.75, short_threshold=1.75, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $89.23 | 1.73 | $1.51 | 59 (49/10) | 1490/344 | 55.9% | 3.0% | ✔️ | 83%/17% | stable |
| 31 | mean_reversion | `long_threshold=-0.5, short_threshold=1.5, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $86.23 | 1.52 | $1.25 | 69 (51/18) | 1810/543 | 56.5% | 5.9% | ✔️ | 74%/26% | stable |
| 32 | mean_reversion | `long_threshold=-0.75, short_threshold=1.5, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $83.67 | 1.60 | $1.35 | 62 (44/18) | 1449/543 | 58.1% | 5.1% | ✔️ | 71%/29% | stable |
| 33 | mean_reversion | `long_threshold=-0.5, short_threshold=1.5, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $82.63 | 1.46 | $1.12 | 74 (56/18) | 1864/559 | 55.4% | 6.2% | ✔️ | 76%/24% | stable |
| 34 | mean_reversion | `long_threshold=-0.75, short_threshold=1.5, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $76.25 | 1.50 | $1.14 | 67 (49/18) | 1490/559 | 55.2% | 5.1% | ✔️ | 73%/27% | stable |
| 35 | mean_reversion | `long_threshold=-1.0, short_threshold=1.75, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $68.03 | 1.76 | $1.48 | 46 (36/10) | 1105/333 | 56.5% | 3.4% | ✔️ | 78%/22% | stable |
| 36 | mean_reversion | `long_threshold=-0.5, short_threshold=1.25, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $61.86 | 1.31 | $0.77 | 80 (56/24) | 1864/814 | 52.5% | 7.5% | ✔️ | 70%/30% | stable |
| 37 | mean_reversion | `long_threshold=-1.0, short_threshold=1.5, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $58.47 | 1.51 | $1.08 | 54 (36/18) | 1105/543 | 55.6% | 4.1% | ✔️ | 67%/33% | stable |
| 38 | mean_reversion | `long_threshold=-1.0, short_threshold=1.75, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $54.02 | 1.54 | $1.13 | 48 (38/10) | 1139/344 | 54.2% | 3.4% | ✔️ | 79%/21% | stable |
| 39 | mean_reversion | `long_threshold=-1.0, short_threshold=1.5, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $47.49 | 1.38 | $0.85 | 56 (38/18) | 1139/559 | 53.6% | 4.1% | ✔️ | 68%/32% | stable |

## Top 5 Rejected Constraints Example
| Rule Family | Params | Net PnL | PF | Trades | Rejection Reason |
|---|---|---:|---:|---:|---|
| mean_reversion | `long_threshold=-0.5, short_threshold=0.5, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $-50.47 | 0.60 | 35 | Low PF (0.60 < 1.25) | Negative Expectancy ($-1.44) |
| mean_reversion | `long_threshold=-0.5, short_threshold=0.5, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.25, max_abs_ma50_slope=0.15` | $-54.28 | 0.88 | 111 | Low PF (0.88 < 1.25) | Negative Expectancy ($-0.49) |
| mean_reversion | `long_threshold=-0.5, short_threshold=0.5, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.5, max_abs_ma50_slope=0.3` | $208.52 | 1.23 | 274 | Low PF (1.23 < 1.25) |
| mean_reversion | `long_threshold=-0.5, short_threshold=0.5, max_spread_z=1.25, max_time_delta_z=2.5, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $-51.70 | 0.60 | 35 | Low PF (0.60 < 1.25) | Negative Expectancy ($-1.48) |
| mean_reversion | `long_threshold=-0.5, short_threshold=0.75, max_spread_z=0.75, max_time_delta_z=2.5, max_abs_ma20_slope=0.15, max_abs_ma50_slope=0.08` | $-51.43 | 0.67 | 47 | Low PF (0.67 < 1.25) | Negative Expectancy ($-1.09) |