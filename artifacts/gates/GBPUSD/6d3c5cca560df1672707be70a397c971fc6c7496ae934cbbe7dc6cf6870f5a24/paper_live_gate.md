# Paper-Live Gate - GBPUSD

* Manifest hash: `6d3c5cca560df1672707be70a397c971fc6c7496ae934cbbe7dc6cf6870f5a24`
* Verdict: `demoted`
* Anchor status: `demoted`
* Reason: shadow evidence below threshold: need 20 trading days and 30 actionable events; restart drill failed or missing; preflight failed or missing; ops attestation failed or missing; historical MT5 replay shows critical drift

## Replay
* Net PnL USD: `4.764736852046099`
* Profit factor: `1.8362181779877433`
* Expectancy USD: `0.7941228086743498`
* Trade count: `6`

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
