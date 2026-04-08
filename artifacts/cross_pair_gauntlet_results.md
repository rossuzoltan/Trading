# Experiment G: Cross-Pair Raw Rule Gauntlet (10k ticks/bar)

Scoreboard side-by-side comparison for Mean Reversion vs Anchors.

## EURUSD Performance
| Baseline | Trades | Net USD | Gross Profit | Gross Loss | Costs | PF | Expct | WinRate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| runtime_flat | 0 | $0.00 | $0.00 | $0.00 | $0.00 | 0.00 | $0.00 | 0.0% |
| runtime_always_long | 0 | $0.00 | $0.00 | $0.00 | $0.00 | 0.00 | $0.00 | 0.0% |
| runtime_always_short | 11 | $-50.68 | $36.13 | $86.81 | $0.00 | 0.42 | $-4.61 | 18.2% |
| runtime_trend | 16 | $-53.10 | $68.25 | $121.35 | $0.00 | 0.56 | $-3.32 | 25.0% |
| runtime_mean_reversion | 21 | **$75.62** | $191.25 | $115.63 | $0.00 | **1.65** | $3.60 | 47.6% |

## GBPUSD Performance
| Baseline | Trades | Net USD | Gross Profit | Gross Loss | Costs | PF | Expct | WinRate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| runtime_flat | 0 | $0.00 | $0.00 | $0.00 | $0.00 | 0.00 | $0.00 | 0.0% |
| runtime_always_long | 0 | $0.00 | $0.00 | $0.00 | $0.00 | 0.00 | $0.00 | 0.0% |
| runtime_always_short | 23 | $-51.03 | $112.35 | $163.39 | $0.00 | 0.69 | $-2.22 | 26.1% |
| runtime_trend | 20 | $-50.55 | $90.27 | $140.82 | $0.00 | 0.64 | $-2.53 | 25.0% |
| runtime_mean_reversion | 21 | **$111.81** | $219.25 | $107.44 | $0.00 | **2.04** | $5.32 | 52.4% |

## USDJPY Performance
| Baseline | Trades | Net USD | Gross Profit | Gross Loss | Costs | PF | Expct | WinRate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| runtime_flat | 0 | $0.00 | $0.00 | $0.00 | $0.00 | 0.00 | $0.00 | 0.0% |
| runtime_always_long | 0 | $0.00 | $0.00 | $0.00 | $0.00 | 0.00 | $0.00 | 0.0% |
| runtime_always_short | 46 | **$26.68** | $351.35 | $324.67 | $0.00 | **1.08** | $0.58 | 39.1% |
| runtime_trend | 27 | **$41.70** | $219.89 | $178.20 | $0.00 | **1.23** | $1.54 | 40.7% |
| runtime_mean_reversion | 13 | $-55.56 | $56.07 | $111.63 | $0.00 | 0.50 | $-4.27 | 23.1% |
