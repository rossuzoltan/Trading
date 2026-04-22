# AlphaGate Bakeoff - GBPUSD

* Manifest: `C:\dev\trading\models\rc1\gbpusd_10k_v1_mr_rc1\manifest.json`
* Manifest hash: `814777123cf53e7a58180d973352aaaff9db3a4b5f09c6b0939d1361af72ad9a`
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

| manifest_gate | manifest | completed | 4.09 | 2.286 | 1.024 | 4 | 2 | 2 | True |
Fit: model=`xgboost_pair`, threshold=`0.71`, margin=`0.01`, fit_pf=`39.34995812650966`, fit_trades=`92.0`

| logistic_pair | refit | completed | 0.00 | 0.000 | 0.000 | 0 | 0 | 0 | True |
Fit: model=`logistic_pair`, threshold=`0.61`, margin=`0.01`, fit_pf=`1.5874938041374147`, fit_trades=`61.0`

| xgboost_pair | refit | completed | 4.76 | 1.836 | 0.794 | 6 | 3 | 3 | True |
Fit: model=`xgboost_pair`, threshold=`0.61`, margin=`0.01`, fit_pf=`5.316795962329176`, fit_trades=`195.0`

| lightgbm_pair | refit | completed | -21.51 | 0.597 | -1.132 | 19 | 6 | 13 | True |
Fit: model=`lightgbm_pair`, threshold=`0.56`, margin=`0.01`, fit_pf=`34.286305205081035`, fit_trades=`234.0`

