# Experiment G: Cross-Pair Raw Rule Gauntlet (5000 ticks/bar)

Scoreboard side-by-side comparison for Mean Reversion vs Anchors.

## EURUSD Performance
| Baseline | Trades | Net USD | Gross Profit | Gross Loss | Costs | PF | Expct | WinRate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| runtime_flat | 0 | $0.00 | $0.00 | $0.00 | $0.00 | 0.00 | $0.00 | 0.0% |
| runtime_always_long | 0 | $0.00 | $0.00 | $0.00 | $0.00 | 0.00 | $0.00 | 0.0% |
| runtime_always_short | 5 | $-50.45 | $0.00 | $50.45 | $0.00 | 0.00 | $-10.09 | 0.0% |
| runtime_trend | 5 | $-50.68 | $0.00 | $50.68 | $0.00 | 0.00 | $-10.14 | 0.0% |
| runtime_mean_reversion | 27 | **$133.42** | $274.46 | $141.04 | $0.00 | **1.95** | $4.94 | 55.6% |

## GBPUSD Performance
| Baseline | Trades | Net USD | Gross Profit | Gross Loss | Costs | PF | Expct | WinRate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| runtime_flat | 0 | $0.00 | $0.00 | $0.00 | $0.00 | 0.00 | $0.00 | 0.0% |
| runtime_always_long | 0 | $0.00 | $0.00 | $0.00 | $0.00 | 0.00 | $0.00 | 0.0% |
| runtime_always_short | 53 | $-50.12 | $338.33 | $388.45 | $0.00 | 0.87 | $-0.95 | 32.1% |
| runtime_trend | 30 | $-50.29 | $174.16 | $224.45 | $0.00 | 0.78 | $-1.68 | 30.0% |
| runtime_mean_reversion | 20 | $-5.05 | $133.54 | $138.59 | $0.00 | 0.96 | $-0.25 | 35.0% |

## USDJPY Performance
| Baseline | Trades | Net USD | Gross Profit | Gross Loss | Costs | PF | Expct | WinRate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| runtime_flat | 0 | $0.00 | $0.00 | $0.00 | $0.00 | 0.00 | $0.00 | 0.0% |
| runtime_always_long | 0 | $0.00 | $0.00 | $0.00 | $0.00 | 0.00 | $0.00 | 0.0% |
| runtime_always_short | 101 | **$234.74** | $988.87 | $754.14 | $0.00 | **1.31** | $2.32 | 42.6% |
| runtime_trend | 53 | **$36.74** | $398.80 | $362.05 | $0.00 | **1.10** | $0.69 | 35.8% |
| runtime_mean_reversion | 16 | **$50.17** | $145.57 | $95.40 | $0.00 | **1.53** | $3.14 | 43.8% |
