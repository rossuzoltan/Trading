# Paper-Live Gate - EURUSD

* Manifest hash: `4c4463fcfac0d4e89db4cb5ebae5824112d6d81acbf636350da468a37ad18abd`
* Verdict: `demoted`
* Anchor status: `demoted`
* Reason: deployed anchor underperformed the raw anchor baseline on PF or expectancy; shadow evidence below threshold: need 20 trading days and 30 actionable events; ops attestation failed or missing; historical MT5 replay shows critical drift

## Replay
* Net PnL USD: `39.46980566077127`
* Profit factor: `1.1715556524562543`
* Expectancy USD: `0.35881641509792067`
* Trade count: `110`

## Baseline Comparison
* Mandatory baseline pass: `True`
* Raw anchor baseline pass: `False`
* Same logic as raw anchor: `False`

## Context
* Enabled: `True`
* OK: `True`
* Calendar path: `models\rc1\eurusd_5k_v1_mr_rc1\macro_calendar.json`
* Expected SHA256: `26d685ad1d703f17e7ea7b42bf9ef74230355259d1dda10e542704100688c0ba`
* Actual SHA256: `26d685ad1d703f17e7ea7b42bf9ef74230355259d1dda10e542704100688c0ba`
* Error: `None`
* Relevant tier-1 events: `0`

## Shadow Window
* Start: `None`
* End: `None`
* Evidence sufficient: `False`
* Trading days: `0`
* Actionable events: `0`

## Drift
* Verdict: `no_data`
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
