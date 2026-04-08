# Experiment F: Meta-label on runtime_mean_reversion — EURUSD 10k

> **Raw rule baseline:** 21 trades | $75.62 net | PF 1.65

> ✅ = beats raw rule (net > raw and trades >= 5)

## Filtered Selectors

| Policy | Trades | Net USD | PF | Expct | WinRate | Allowed | Rejected | Flat (no rule) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_mean_rev_rule | 21 | **$75.62** | 1.65 | $3.60 | 47.6% | 0 | 0 | 0 |
| logistic_p50 | 16 | **$37.77** | 1.40 | $2.36 | 43.8% | 50 | 49 | 905 |
| logistic_p70 | 10 | $-17.36 | 0.76 | $-1.74 | 30.0% | 30 | 69 | 905 |
| logistic_p80 | 7 | $-14.41 | 0.71 | $-2.06 | 28.6% | 20 | 79 | 905 |
| logistic_p90 | 5 | **$4.59** | 1.15 | $0.92 | 40.0% | 10 | 89 | 905 |
| hgb_clf_p50 | 10 | **$65.27** | 2.60 | $6.53 | 60.0% | 50 | 49 | 905 |
| hgb_clf_p70 | 5 | **$98.87 ✅** | inf | $19.77 | 100.0% | 30 | 69 | 905 |
| hgb_clf_p80 | 3 | **$58.10** | inf | $19.37 | 100.0% | 20 | 79 | 905 |
| hgb_clf_p90 | 1 | **$17.39** | inf | $17.39 | 100.0% | 10 | 89 | 905 |

## Anchor Baselines

| Policy | Trades | Net USD | PF | Expct | WinRate | Allowed | Rejected | Flat (no rule) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| runtime_flat | 0 | $0.00 | 0.00 | $0.00 | 0.0% | - | - | - |
| runtime_always_long | 0 | $0.00 | 0.00 | $0.00 | 0.0% | - | - | - |
| runtime_always_short | 30 | $-52.24 | 0.76 | $-1.74 | 30.0% | - | - | - |
| runtime_mean_reversion | 21 | **$75.62** | 1.65 | $3.60 | 47.6% | - | - | - |
| runtime_trend | 16 | $-53.10 | 0.56 | $-3.32 | 25.0% | - | - | - |