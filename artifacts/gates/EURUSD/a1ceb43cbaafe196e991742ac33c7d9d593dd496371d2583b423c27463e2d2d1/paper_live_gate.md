# Paper-Live Gate - EURUSD

* Manifest hash: `a1ceb43cbaafe196e991742ac33c7d9d593dd496371d2583b423c27463e2d2d1`
* Verdict: `demoted`
* Anchor status: `demoted`
* Reason: shadow evidence below threshold: need 20 trading days and 30 actionable events; ops attestation failed or missing; historical MT5 replay shows critical drift

## Replay
* Net PnL USD: `39.46980566077127`
* Profit factor: `1.1715556524562543`
* Expectancy USD: `0.35881641509792067`
* Trade count: `110`

## Baseline Comparison
* Mandatory baseline pass: `True`
* Raw anchor baseline pass: `True`
* Same logic as raw anchor: `False`

## Shadow Window
* Start: `2026-04-14T08:14:14.062000+00:00`
* End: `2026-04-21T12:43:13.656000+00:00`
* Evidence sufficient: `False`
* Trading days: `2`
* Actionable events: `0`

## Drift
* Verdict: `insufficient`
* Critical: `False`
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
* Restart: `True`
* Preflight: `True`
* Ops attestation: `False`
