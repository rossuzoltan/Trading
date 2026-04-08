import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge_research import run_edge_baseline_research
from feature_engine import FEATURE_COLS, _compute_raw
from train_agent import _split_holdout
from compare_oos_baselines import _build_folds

def main():
    symbol = "EURUSD"
    cost_profile = {
        "commission_per_lot": 7.0,
        "slippage_pips": 0.25,
    }
    
    # Analyze the currently active dataset
    dataset_path = Path("data") / "DATA_CLEAN_VOLUME.csv"
    if not dataset_path.exists():
        print(f"Error: dataset not found at {dataset_path}")
        return
        
    print(f"Loading {dataset_path}...")
    raw = pd.read_csv(dataset_path, low_memory=False, parse_dates=["Gmt time"])
    raw = raw.loc[raw["Symbol"].astype(str).str.upper() == symbol].copy()
    raw["Gmt time"] = pd.to_datetime(raw["Gmt time"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["Gmt time"]).set_index("Gmt time").sort_index()
    
    # Infer ticks from first bar's volume (usually a volume bar is fairly constant)
    estimated_ticks = int(raw["Volume"].median())
    print(f"Estimated current dataset ticks per bar: {estimated_ticks}")
    
    print("Computing features...")
    featured = _compute_raw(raw).dropna(subset=list(FEATURE_COLS))
    trainable_feature_frame, holdout_feature_frame = _split_holdout(featured, 0.15)
    
    if trainable_feature_frame.empty or holdout_feature_frame.empty:
        print("Empty feature frames.")
        return
        
    folds = _build_folds(trainable_feature_frame, validation_frac=0.15)
    out_path = Path("models") / f"horizon_falsifier_{estimated_ticks}.json"
    
    print("Running baseline research...")
    baseline_report = run_edge_baseline_research(
        symbol=symbol,
        trainable_frame=trainable_feature_frame,
        holdout_frame=holdout_feature_frame,
        folds=folds,
        feature_cols=list(FEATURE_COLS),
        out_path=out_path,
        horizon_bars=10,
        commission_per_lot=cost_profile["commission_per_lot"],
        slippage_pips=cost_profile["slippage_pips"],
        min_edge_pips=0.0,
        probability_threshold=0.55,
        probability_margin=0.05,
        min_trade_count=20,
    )
    
    models = baseline_report.get("holdout_metrics", {}).get("models", {})
    
    print("\nScoreboard (Research Path):")
    print(f"{'Model':<25} | {'Trades':<8} | {'PnL (USD)':<10} | {'Profit Factor':<15}")
    print("-" * 65)
    for name, data in models.items():
        metrics = data.get("metrics", {})
        trades = metrics.get('trade_count', 0)
        pnl = metrics.get('net_pnl_usd', 0)
        pf = metrics.get('profit_factor', 0)
        print(f"{name:<25} | {trades:<8.0f} | ${pnl:<9.2f} | {pf:<15.2f}")

if __name__ == "__main__":
    main()
