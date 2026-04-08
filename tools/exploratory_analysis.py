import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_engine import FeatureEngine
from symbol_utils import price_to_pips

def analyze_edge(symbol="EURUSD", target_horizon=10):
    dataset_path = ROOT / "data" / "DATA_CLEAN_VOLUME_5000.csv"
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    print("Loading data...")
    df = pd.read_csv(dataset_path)
    df = df[df["Symbol"] == symbol].copy()
    df["Gmt time"] = pd.to_datetime(df["Gmt time"])
    df.sort_values("Gmt time", inplace=True)
    df.drop_duplicates(subset=["Gmt time"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print("Computing features...")
    fe = FeatureEngine()
    # Use _compute_raw to get all features including legacy/challenger ones
    from feature_engine import _compute_raw
    features_df = _compute_raw(df)
    
    # Forward return (using Close price)
    print(f"Computing forward returns (horizon {target_horizon} bars)...")
    features_df["fwd_ret"] = features_df["Close"].shift(-target_horizon) - features_df["Close"]
    
    # Convert exactly to pips
    features_df["fwd_ret_pips"] = features_df["fwd_ret"].apply(lambda x: float(price_to_pips(symbol, x)))
    
    # Drop NaNs
    analysis_df = features_df.dropna(subset=["fwd_ret_pips", "price_z", "spread_z", "time_delta_z", "ma20_slope", "vol_norm_atr"])
    
    print("\n--- Correlation with Forward Return (Pips) ---")
    corr_cols = [
        "log_return", "body_size", "candle_range", "ma20_slope", "ma50_slope",
        "vol_norm_atr", "spread_z", "time_delta_z", "price_z", "adx"
    ]
    
    # We want to see if extreme values of features predict direction.
    # So we compute spearman correlation
    correlations = analysis_df[corr_cols + ["fwd_ret_pips"]].corr(method='spearman')["fwd_ret_pips"].sort_values()
    print(correlations)
    
    # Deep dive into spread_z 
    print("\n--- Spread Z Signal Analysis ---")
    long_signals = analysis_df[analysis_df["spread_z"] <= -1.0]
    short_signals = analysis_df[analysis_df["spread_z"] >= 1.0]
    print(f"Spread Z Longs (spread_z <= -1.0): {len(long_signals)} bars, Avg Fwd Pips: {long_signals['fwd_ret_pips'].mean():.2f}")
    print(f"Spread Z Shorts (spread_z >= 1.0): {len(short_signals)} bars, Avg Fwd Pips: {short_signals['fwd_ret_pips'].mean():.2f}")

    # Deep dive into price_z
    print("\n--- Price Z Signal Analysis ---")
    for thresh in [1.0, 1.5, 2.0, 2.5]:
        p_longs = analysis_df[analysis_df["price_z"] <= -thresh]
        p_shorts = analysis_df[analysis_df["price_z"] >= thresh]
        print(f"Price Z Thresh={thresh}")
        print(f"  Longs: {len(p_longs)} bars, Avg Fwd Pips: {p_longs['fwd_ret_pips'].mean():.2f}")
        print(f"  Shorts: {len(p_shorts)} bars, Avg Fwd Pips: {p_shorts['fwd_ret_pips'].mean():.2f}")
        
    print("\n--- Combined Signal Analysis (Price Z + Time Delta Z) ---")
    # Hypothesis: Exhaustion (High volume/fast ticks = low time_delta_z) + Price Extension
    combo_longs = analysis_df[(analysis_df["price_z"] <= -2.0) & (analysis_df["time_delta_z"] <= -1.0)]
    combo_shorts = analysis_df[(analysis_df["price_z"] >= 2.0) & (analysis_df["time_delta_z"] <= -1.0)]
    print(f"Combo Longs: {len(combo_longs)} bars, Avg Fwd Pips: {combo_longs['fwd_ret_pips'].mean():.2f}")
    print(f"Combo Shorts: {len(combo_shorts)} bars, Avg Fwd Pips: {combo_shorts['fwd_ret_pips'].mean():.2f}")

if __name__ == "__main__":
    analyze_edge()
