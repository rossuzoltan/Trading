import sys
import os
import pandas as pd
from dataclasses import replace

# Add project root to sys.path
sys.path.append(os.getcwd())

import evaluate_oos
from tools.optimize_rules import _run_single_variant

def main():
    symbol = "EURUSD"
    print(f"--- Canary Verification for {symbol} ---")
    
    # Load context
    ctx = evaluate_oos.load_replay_context(symbol)
    
    # Target a mean_reversion variant known to have signals
    config = {
        "rule_family": "mean_reversion",
        "params": {
            "long_threshold": -1.0,  # Should fire Long
            "short_threshold": 1.25, # Should fire Short
            "max_spread_z": 1.0,
            "max_time_delta_z": 5.0,
            "max_abs_ma20_slope": 1.0,
            "max_abs_ma50_slope": 1.0,
        }
    }
    
    # Force 'train' stage alignment
    ctx = replace(ctx,
        replay_feature_frame=ctx.trainable_feature_frame,
        replay_frame=ctx.trainable_feature_frame
    )
    
    print("Running single variant replay...")
    res = _run_single_variant(ctx, config)
    
    print("\nOutcome:")
    print(f"Status: {res.get('status')}")
    print(f"Total Trades: {res.get('trades')}")
    print(f"Long Trades: {res.get('long_trades')}")
    print(f"Short Trades: {res.get('short_trades')}")
    print(f"Signals (L/S): {res.get('signal_longs')}/{res.get('signal_shorts')}")
    
    if res.get('signal_longs', 0) > 0:
        print("\nSUCCESS: Long signals detected in the optimizer context!")
    else:
        print("\nFAILURE: Still no long signals detected.")

if __name__ == "__main__":
    main()
