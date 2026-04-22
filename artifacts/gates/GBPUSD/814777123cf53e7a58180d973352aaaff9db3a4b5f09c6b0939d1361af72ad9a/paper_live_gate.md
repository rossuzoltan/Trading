# Paper-Live Gate - GBPUSD

* Manifest hash: `814777123cf53e7a58180d973352aaaff9db3a4b5f09c6b0939d1361af72ad9a`
* Verdict: `demoted`
* Anchor status: `demoted`
* Reason: shadow evidence below threshold: need 20 trading days and 30 actionable events; restart drill failed or missing; preflight failed or missing; ops attestation failed or missing; historical MT5 replay shows critical drift

## Replay
* Net PnL USD: `4.09420261576463`
* Profit factor: `2.28556749370435`
* Expectancy USD: `1.0235506539411574`
* Trade count: `4`

## Baseline Comparison
* Mandatory baseline pass: `True`
* Raw anchor baseline pass: `True`
* Same logic as raw anchor: `False`

## Shadow Window
* Start: `None`
* End: `None`
* Evidence sufficient: `False`
* Trading days: `0`
* Actionable events: `0`

## Drift
* Verdict: `critical`
* Critical: `True`
* Signal density ratio: `None`
* Would-open density ratio: `None`
* Spread rejection delta pp: `0.0`
* Session rejection delta pp: `0.0`
* Directional occupancy delta pp: `44.44444444444444`

## Historical Replay
* Present: `True`
* Verdict: `DRIFT_CRITICAL`
* OK: `False`

## Ops Gates
* Restart: `False`
* Preflight: `False`
* Ops attestation: `False`
