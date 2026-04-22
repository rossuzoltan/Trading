# AlphaGate Bakeoff - EURUSD

* Manifest: `C:\dev\trading\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json`
* Manifest hash: `a1ceb43cbaafe196e991742ac33c7d9d593dd496371d2583b423c27463e2d2d1`
* Holdout start UTC: `2025-09-10T07:06:02+00:00`
* Replay bars: `2016`
* Best candidate: `rule_only`

## Defaults
* Probability threshold: `0.51`
* Probability margin: `0.01`
* Min edge pips: `0.0`
* Horizon bars: `25`

## Results

| Candidate | Source | Status | Net PnL USD | PF | Expectancy USD | Trades | Long | Short | Validation |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| rule_only | selector | completed | 39.47 | 1.172 | 0.359 | 110 | 59 | 51 | True |

| manifest_gate | manifest | skipped | 0.00 | 0.000 | 0.000 | 0 | 0 | 0 | False |
Reason: `manifest_alpha_gate_disabled`

| logistic_pair | refit | completed | 0.00 | 0.000 | 0.000 | 0 | 0 | 0 | True |
Fit: model=`logistic_pair`, threshold=`0.61`, margin=`0.01`, fit_pf=`2.5672494829516115`, fit_trades=`13.0`

| xgboost_pair | refit | completed | 12.57 | 1.483 | 0.898 | 14 | 11 | 3 | True |
Fit: model=`xgboost_pair`, threshold=`0.61`, margin=`0.01`, fit_pf=`5.869638005050957`, fit_trades=`313.0`

| lightgbm_pair | refit | completed | 5.86 | 1.062 | 0.168 | 35 | 20 | 15 | True |
Fit: model=`lightgbm_pair`, threshold=`0.61`, margin=`0.01`, fit_pf=`17.574903591966656`, fit_trades=`419.0`

