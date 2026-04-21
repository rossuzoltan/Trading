from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_engine import FEATURE_COLS, FeatureEngine, _compute_raw


def main() -> int:
    dataset = Path('data/DATA_CLEAN_VOLUME_5000.csv')
    if not dataset.exists():
        print(json.dumps({'ok': False, 'reason': f'missing dataset: {dataset}'}))
        return 1

    frame = pd.read_csv(dataset, low_memory=False, parse_dates=['Gmt time'])
    frame['Gmt time'] = pd.to_datetime(frame['Gmt time'], utc=True, errors='coerce')
    frame = frame.dropna(subset=['Gmt time']).set_index('Gmt time').sort_index()
    frame = frame.tail(220).copy()

    engine = FeatureEngine()
    # Warm the engine on the same effective window used for parity comparison.
    engine.warm_up(frame)
    raw = _compute_raw(frame, latest_only_hurst=True, fast_mode=True)
    raw = engine._drop_invalid_feature_rows(raw)
    if raw.empty:
        print(json.dumps({'ok': False, 'reason': 'no valid feature rows after compute_raw'}))
        return 1

    last = raw.iloc[-1]
    hot = engine._get_obs_hot_path()
    hot_raw = engine.latest_features_raw
    reference = np.array([float(last.get(col, 0.0)) for col in FEATURE_COLS], dtype=np.float32)
    diff = np.abs(reference - hot_raw)

    payload = {
        'ok': bool(np.all(diff < 1e-3)),
        'max_abs_diff': float(np.max(diff)) if len(diff) else 0.0,
        'feature_diffs': {col: float(val) for col, val in zip(FEATURE_COLS, diff)},
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
