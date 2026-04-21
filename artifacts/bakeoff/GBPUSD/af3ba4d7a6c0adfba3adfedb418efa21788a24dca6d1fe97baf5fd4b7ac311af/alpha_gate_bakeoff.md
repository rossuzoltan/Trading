# AlphaGate Bakeoff - GBPUSD

* Manifest: `C:\dev\trading\models\rc1\gbpusd_10k_v1_mr_rc1\manifest.json`
* Manifest hash: `af3ba4d7a6c0adfba3adfedb418efa21788a24dca6d1fe97baf5fd4b7ac311af`
* Holdout start UTC: `2025-10-13T15:26:57+00:00`
* Replay bars: `1054`
* Best candidate: `xgboost_pair`

## Defaults
* Probability threshold: `0.51`
* Probability margin: `0.01`
* Min edge pips: `0.0`
* Horizon bars: `25`

## Results

| Candidate | Source | Status | Net PnL USD | PF | Expectancy USD | Trades | Long | Short | Validation |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| rule_only | selector | completed | -55.46 | 0.310 | -3.081 | 18 | 10 | 8 | True |

| manifest_gate | manifest | completed | -26.12 | 0.541 | -1.136 | 23 | 9 | 14 | True |
Fit: model=`logistic_pair`, threshold=`0.5800000000000001`, margin=`0.03`, fit_pf=`1.8835311603085498`, fit_trades=`128.0`

| logistic_pair | refit | completed | 0.00 | 0.000 | 0.000 | 0 | 0 | 0 | True |
Fit: model=`logistic_pair`, threshold=`0.61`, margin=`0.01`, fit_pf=`1.5874938041374147`, fit_trades=`61.0`

| xgboost_pair | refit | completed | 4.09 | 2.286 | 1.024 | 4 | 2 | 2 | True |
Fit: model=`xgboost_pair`, threshold=`0.61`, margin=`0.01`, fit_pf=`6.718638511350614`, fit_trades=`196.0`

| lightgbm_pair | refit | completed | -52.91 | 0.182 | -3.307 | 16 | 6 | 10 | True |
Fit: model=`lightgbm_pair`, threshold=`0.56`, margin=`0.01`, fit_pf=`34.10910072489303`, fit_trades=`233.0`

