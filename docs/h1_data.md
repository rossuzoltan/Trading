# H1 Forex Dataset Builder

`build_h1_dataset.py` creates a long-format H1 dataset from the best source available:

1. Local tick parquet / CSV cache in `data/`
2. Existing Dukascopy tick pipeline in `download_dukascopy.py`
3. MT5 H1 history, if `MetaTrader5` is installed and connected
4. `yfinance` as a last-resort fallback

## Recommended usage

- Build roughly 7 years of hourly data:
  - `.\.venv\Scripts\python.exe .\build_h1_dataset.py --years 7`
- Force Dukascopy cache refresh for selected pairs:
  - `.\.venv\Scripts\python.exe .\build_h1_dataset.py --source dukascopy --pairs EURUSD GBPUSD --years 10 --force-refresh`
- Use local tick files only:
  - `.\.venv\Scripts\python.exe .\build_h1_dataset.py --source local --pairs EURUSD`
- Fail the build if a pair does not really cover enough history:
  - `.\.venv\Scripts\python.exe .\build_h1_dataset.py --years 10 --min-years 5 --max-gap-hours 72 --strict-coverage`

## Outputs

- Combined dataset: `data/FOREX_H1_MULTI_SET.csv`
- Per-pair files: `data/EURUSD_h1.csv`, `data/GBPUSD_h1.csv`, etc.
- Manifest: `data/FOREX_H1_MULTI_SET.manifest.json`

The manifest now includes `coverage_by_symbol`, including:

- covered years
- expected vs actual H1 row count
- coverage ratio
- largest observed gap in hours
- count of gaps larger than the configured threshold

## Notes

- The script keeps H1 data separate from the current volume-bar training dataset.
- If you want to experiment with H1 features later, point downstream code at the generated H1 CSV instead of the volume-bar file.
- `yfinance` is only a fallback; for 5-10 years, Dukascopy or broker history is the preferred path.
- `--strict-coverage` is useful when you want the build to fail fast instead of silently accepting a sparse or hole-ridden H1 series.
