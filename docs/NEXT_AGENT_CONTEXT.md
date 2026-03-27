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
- `data/EURUSD_ticks.parquet`: `1,923,534` ticks
- `data/GBPUSD_ticks.parquet`: `2,095,665` ticks
- `data/USDJPY_ticks.parquet`: `31,399,697` ticks
- `data/DATA_CLEAN_VOLUME.csv` symbol rows:
  - `USDJPY`: `15,700`
  - `GBPUSD`: `1,048`
  - `EURUSD`: `962`

## Important Interpretation
- EURUSD and GBPUSD are still severely under-covered versus USDJPY.
- The interrupted refresh did **not** change those counts.
- Per-symbol training is now possible, but EURUSD/GBPUSD data likely still needs a real re-download before their results are trustworthy.

## Last Interrupted Command
```powershell
.\.venv\Scripts\python.exe download_dukascopy.py --pairs EURUSD GBPUSD --days 1095 --bar-volume 2000 --force-refresh-pairs EURUSD GBPUSD --max-workers 16
```
- User interrupted it.
- Re-check file counts after re-running.

## User Intent
- Optimize for correctness first.
- If needed, download better market data from good sources.
- Keep the next pass efficient and avoid wasting tokens on re-reading the whole repo.
