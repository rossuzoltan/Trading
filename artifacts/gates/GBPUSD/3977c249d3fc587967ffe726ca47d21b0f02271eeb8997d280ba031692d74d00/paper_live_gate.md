# Paper-Live Gate - GBPUSD

* Manifest hash: `3977c249d3fc587967ffe726ca47d21b0f02271eeb8997d280ba031692d74d00`
* Verdict: `demoted`
* Anchor status: `demoted`
* Reason: shadow evidence below threshold: need 20 trading days and 30 actionable events; restart drill failed or missing; preflight failed or missing; ops attestation failed or missing; historical MT5 replay shows critical drift

## Replay
* Net PnL USD: `111.80844910541714`
* Profit factor: `2.0406701316532927`
* Expectancy USD: `5.324211862162721`
* Trade count: `21`

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
* Verdict: `aligned`
* Critical: `False`
* Signal density ratio: `None`
* Would-open density ratio: `None`
* Spread rejection delta pp: `0.0`
* Session rejection delta pp: `0.0`
* Directional occupancy delta pp: `0.0`

## Historical Replay
* Present: `True`
* Verdict: `DRIFT_CRITICAL`
* OK: `False`

## Ops Gates
* Restart: `False`
* Preflight: `False`
* Ops attestation: `False`
