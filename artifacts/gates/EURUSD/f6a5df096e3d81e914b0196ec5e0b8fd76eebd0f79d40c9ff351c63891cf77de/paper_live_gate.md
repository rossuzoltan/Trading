# Paper-Live Gate - EURUSD

* Manifest hash: `f6a5df096e3d81e914b0196ec5e0b8fd76eebd0f79d40c9ff351c63891cf77de`
* Verdict: `demoted`
* Anchor status: `demoted`
* Reason: deployed anchor underperformed a mandatory baseline; deployed anchor underperformed the raw anchor baseline on PF or expectancy; shadow evidence below threshold: need 20 trading days and 30 actionable events; restart drill failed or missing; preflight failed or missing; ops attestation failed or missing; historical MT5 replay shows critical drift

## Replay
* Net PnL USD: `-50.37801269977899`
* Profit factor: `0.6050648499249752`
* Expectancy USD: `-2.9634125117517054`
* Trade count: `17`

## Baseline Comparison
* Mandatory baseline pass: `False`
* Raw anchor baseline pass: `False`
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
