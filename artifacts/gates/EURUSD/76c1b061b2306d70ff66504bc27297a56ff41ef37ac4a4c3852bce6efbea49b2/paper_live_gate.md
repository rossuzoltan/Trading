# Paper-Live Gate - EURUSD

* Manifest hash: `76c1b061b2306d70ff66504bc27297a56ff41ef37ac4a4c3852bce6efbea49b2`
* Verdict: `demoted`
* Anchor status: `demoted`
* Reason: shadow evidence below threshold: need 20 trading days and 30 actionable events; restart drill failed or missing; preflight failed or missing; ops attestation failed or missing; historical MT5 replay shows critical drift

## Replay
* Net PnL USD: `133.42165756711938`
* Profit factor: `1.945964546599123`
* Expectancy USD: `4.941542872856274`
* Trade count: `27`

## Baseline Comparison
* Mandatory baseline pass: `True`
* Raw anchor baseline pass: `True`
* Same logic as raw anchor: `True`

## Shadow Window
* Start: `None`
* End: `None`
* Evidence sufficient: `False`
* Trading days: `0`
* Actionable events: `0`

## Drift
* Verdict: `critical`
* Critical: `True`
* Signal density ratio: `0.0`
* Would-open density ratio: `0.0`
* Spread rejection delta pp: `0.0`
* Session rejection delta pp: `0.0`
* Directional occupancy delta pp: `2.0618556701030926`

## Historical Replay
* Present: `True`
* Verdict: `DRIFT_CRITICAL`
* OK: `False`

## Ops Gates
* Restart: `False`
* Preflight: `False`
* Ops attestation: `False`
