# AlphaGate Bakeoff - EURUSD

* Manifest: `C:\dev\trading\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json`
* Manifest hash: `7a0d30967c898826dae2b2a3ffd3636a0848ad6b32a6143724b4dcb1a6992477`
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

| manifest_gate | manifest | completed | 24.84 | 1.256 | 0.565 | 44 | 25 | 19 | True |
Fit: model=`xgboost_pair`, threshold=`0.63`, margin=`0.03`, fit_pf=`7.207309196764081`, fit_trades=`263.0`

| logistic_pair | refit | completed | 0.00 | 0.000 | 0.000 | 0 | 0 | 0 | True |
Fit: model=`logistic_pair`, threshold=`0.61`, margin=`0.01`, fit_pf=`2.5672494829516115`, fit_trades=`13.0`

| xgboost_pair | refit | completed | 12.57 | 1.483 | 0.898 | 14 | 11 | 3 | True |
Fit: model=`xgboost_pair`, threshold=`0.61`, margin=`0.01`, fit_pf=`5.869638005050957`, fit_trades=`313.0`

| lightgbm_pair | refit | completed | 5.86 | 1.062 | 0.168 | 35 | 20 | 15 | True |
Fit: model=`lightgbm_pair`, threshold=`0.61`, margin=`0.01`, fit_pf=`17.574903591966656`, fit_trades=`419.0`

