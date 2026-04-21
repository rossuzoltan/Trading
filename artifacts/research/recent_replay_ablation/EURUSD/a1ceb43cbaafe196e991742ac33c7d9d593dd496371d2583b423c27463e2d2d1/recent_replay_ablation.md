# Recent Replay Ablation - EURUSD

* Manifest: `C:\dev\trading\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json`
* Bars source: `C:\dev\trading\models\rc1\eurusd_5k_v1_mr_rc1\mt5_historical_replay_report.bars.jsonl`

| Variant | Holdout Net | Holdout PF | Holdout Trades | Replay Opens | Replay Long | Replay Short | Replay L/S |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rc1_baseline | 39.47 | 1.172 | 110 | 15 | 2 | 13 | 0.15 |
| asym_short_1.80 | -50.73 | 0.652 | 48 | 8 | 2 | 6 | 0.33 |
| asym_long_-1.20 | 13.24 | 1.043 | 133 | 19 | 6 | 13 | 0.46 |
| asym_long_-1.20_short_1.80 | -52.29 | 0.735 | 68 | 12 | 6 | 6 | 1.00 |
| relax_ma50_0.10 | -51.27 | 0.671 | 51 | 17 | 4 | 13 | 0.31 |
| relax_ma50_0.15 | -50.07 | 0.656 | 51 | 19 | 5 | 14 | 0.36 |
| relax_ma20_0.20_ma50_0.10 | -56.12 | 0.632 | 54 | 23 | 5 | 18 | 0.28 |
| relax_ma20_0.25_ma50_0.15 | -51.06 | 0.536 | 33 | 26 | 8 | 18 | 0.44 |
| relax_spread_0.75_ma20_0.20_ma50_0.10 | -50.01 | 0.697 | 59 | 25 | 5 | 20 | 0.25 |
| price_only_no_guards | -50.77 | 0.779 | 82 | 33 | 7 | 26 | 0.27 |

## Detail
### rc1_baseline
* Holdout net: `39.47` | PF: `1.172` | Trades: `110`
* Replay opens: `15` | long `2` | short `13`
* Guard failures: `spread=57`, `time_delta=4`, `ma20=122`, `ma50=170`
* Signal breakdown: `raw_long=38`, `raw_short=71`, `guarded_long=6`, `guarded_short=19`

### asym_short_1.80
* Holdout net: `-50.73` | PF: `0.652` | Trades: `48`
* Replay opens: `8` | long `2` | short `6`
* Guard failures: `spread=57`, `time_delta=4`, `ma20=122`, `ma50=170`
* Signal breakdown: `raw_long=38`, `raw_short=40`, `guarded_long=6`, `guarded_short=9`

### asym_long_-1.20
* Holdout net: `13.24` | PF: `1.043` | Trades: `133`
* Replay opens: `19` | long `6` | short `13`
* Guard failures: `spread=57`, `time_delta=4`, `ma20=122`, `ma50=170`
* Signal breakdown: `raw_long=62`, `raw_short=71`, `guarded_long=14`, `guarded_short=19`

### asym_long_-1.20_short_1.80
* Holdout net: `-52.29` | PF: `0.735` | Trades: `68`
* Replay opens: `12` | long `6` | short `6`
* Guard failures: `spread=57`, `time_delta=4`, `ma20=122`, `ma50=170`
* Signal breakdown: `raw_long=62`, `raw_short=40`, `guarded_long=14`, `guarded_short=9`

### relax_ma50_0.10
* Holdout net: `-51.27` | PF: `0.671` | Trades: `51`
* Replay opens: `17` | long `4` | short `13`
* Guard failures: `spread=57`, `time_delta=4`, `ma20=122`, `ma50=126`
* Signal breakdown: `raw_long=38`, `raw_short=71`, `guarded_long=12`, `guarded_short=20`

### relax_ma50_0.15
* Holdout net: `-50.07` | PF: `0.656` | Trades: `51`
* Replay opens: `19` | long `5` | short `14`
* Guard failures: `spread=57`, `time_delta=4`, `ma20=122`, `ma50=48`
* Signal breakdown: `raw_long=38`, `raw_short=71`, `guarded_long=16`, `guarded_short=23`

### relax_ma20_0.20_ma50_0.10
* Holdout net: `-56.12` | PF: `0.632` | Trades: `54`
* Replay opens: `23` | long `5` | short `18`
* Guard failures: `spread=57`, `time_delta=4`, `ma20=82`, `ma50=126`
* Signal breakdown: `raw_long=38`, `raw_short=71`, `guarded_long=15`, `guarded_short=26`

### relax_ma20_0.25_ma50_0.15
* Holdout net: `-51.06` | PF: `0.536` | Trades: `33`
* Replay opens: `26` | long `8` | short `18`
* Guard failures: `spread=57`, `time_delta=4`, `ma20=48`, `ma50=48`
* Signal breakdown: `raw_long=38`, `raw_short=71`, `guarded_long=26`, `guarded_short=43`

### relax_spread_0.75_ma20_0.20_ma50_0.10
* Holdout net: `-50.01` | PF: `0.697` | Trades: `59`
* Replay opens: `25` | long `5` | short `20`
* Guard failures: `spread=38`, `time_delta=4`, `ma20=82`, `ma50=126`
* Signal breakdown: `raw_long=38`, `raw_short=71`, `guarded_long=16`, `guarded_short=32`

### price_only_no_guards
* Holdout net: `-50.77` | PF: `0.779` | Trades: `82`
* Replay opens: `33` | long `7` | short `26`
* Guard failures: `spread=0`, `time_delta=0`, `ma20=0`, `ma50=0`
* Signal breakdown: `raw_long=38`, `raw_short=71`, `guarded_long=38`, `guarded_short=71`

