# EURUSD AlphaGate Model Comparison (train)

| Model | Evaluated | Passed | Rejected | Errors | Best Net PnL | Best PF | Best Expectancy | Best Trades |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_pair | 120 | 28 | 92 | 0 | $17.34 | 1.96 | $1.93 | 9 |
| xgboost_pair | 120 | 39 | 81 | 0 | $275.73 | 1.40 | $1.25 | 220 |
| lightgbm_pair | 120 | 58 | 62 | 0 | $482.62 | 1.40 | $1.19 | 405 |

## Best Candidate Params
- `logistic_pair`: `{"long_threshold": -0.5, "max_abs_ma20_slope": 0.25, "max_abs_ma50_slope": 0.15, "max_spread_z": 0.75, "max_time_delta_z": 2.5, "short_threshold": 0.5}`
- `xgboost_pair`: `{"long_threshold": -0.5, "max_abs_ma20_slope": 0.25, "max_abs_ma50_slope": 0.15, "max_spread_z": 1.25, "max_time_delta_z": 2.5, "short_threshold": 0.5}`
- `lightgbm_pair`: `{"long_threshold": -0.75, "max_abs_ma20_slope": 0.5, "max_abs_ma50_slope": 0.3, "max_spread_z": 1.25, "max_time_delta_z": 2.5, "short_threshold": 1.5}`
