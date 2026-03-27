# Next Agent Context

## Goal
- Continue the trading-bot hardening work with minimal re-analysis.
- Current focus: train and evaluate **per-symbol** models, then repair the thin EURUSD/GBPUSD raw data if needed.

## Do Not Re-Audit
- The shared event-driven runtime refactor is already in place.
- The regression suite for the runtime passes in the project venv.
- The current blocker is strategy/data quality, not the live-bridge architecture.

## Current Repo State
- Runtime/test stack is stabilized.
- `train_agent.py` is now **symbol-scoped** via `TRAIN_SYMBOL`.
- `evaluate_oos.py` is now **symbol-scoped** via `EVAL_SYMBOL`.
- `download_dukascopy.py` now supports retries and `--force-refresh-pairs`.

## Verified Commands
- Tests:
  - `.\.venv\Scripts\python.exe -m unittest discover tests`
- Result:
  - `Ran 9 tests ... OK`

## Current Data Snapshot
- `data/EURUSD_ticks.parquet`: `67,477,967` ticks
- `data/GBPUSD_ticks.parquet`: `70,793,556` ticks
- `data/USDJPY_ticks.parquet`: `31,399,697` ticks
- `data/DATA_CLEAN_VOLUME.csv` symbol rows:
  - `USDJPY`: `15,700`
  - `GBPUSD`: `35,397`
  - `EURUSD`: `33,739`

## Important Interpretation
- The active combined dataset is now a clean `2000` ticks/bar build across `EURUSD`, `GBPUSD`, and `USDJPY`.
- EURUSD and GBPUSD are no longer thin; the older low-count handoff numbers are obsolete.
- USDJPY now has fewer bars than the prior mixed-spec dataset because it has been rebuilt at the correct `2000` ticks/bar spec.
- Use `data/dataset_build_info.json` and `tools/project_healthcheck.py` as the source of truth before a large run.

## Latest Rebuild
```powershell
.\.venv\Scripts\python.exe download_dukascopy.py --pairs EURUSD GBPUSD --days 1095 --bar-volume 2000 --force-refresh-pairs EURUSD GBPUSD --max-workers 16
.\.venv\Scripts\python.exe build_volume_bars.py --ticks-per-bar 2000
.\.venv\Scripts\python.exe .\tools\project_healthcheck.py
```
- Current healthcheck status: dataset integrity OK for `EURUSD`, `GBPUSD`, `USDJPY`.

## User Intent
- Optimize for correctness first.
- If needed, download better market data from good sources.
- Keep the next pass efficient and avoid wasting tokens on re-reading the whole repo.
