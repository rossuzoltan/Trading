import sys, os
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
os.environ['EVAL_MANIFEST_PATH'] = 'models/rc1/eurusd_5k_v1_mr_rc1/manifest.json'

import evaluate_oos
import pandas as pd
import numpy as np
from strategies.rule_logic import compute_rule_direction

print("=== RULE FIRING DIAGNOSTIC (EURUSD TRAIN) ===")
ctx = evaluate_oos.load_replay_context('EURUSD')
df = ctx.trainable_feature_frame

# Rule variants to check
test_configs = [
    {"rule_family": "mean_reversion", "params": {"threshold": 1.0, "max_spread_z": 0.5, "max_abs_ma20_slope": 0.15}},
    {"rule_family": "pro_mean_reversion", "params": {"adx_threshold": 25.0, "rsi_oversold": 35.0, "price_z_threshold": 1.5}},
    {"rule_family": "microstructure_bounce", "params": {"td_threshold": -0.2, "price_z_threshold": 1.5, "spread_max_z": 1.0}}
]

for cfg in test_configs:
    family = cfg["rule_family"]
    params = cfg["params"]
    print(f"\n--- Testing {family} with params {params} ---")
    
    long_count = 0
    short_count = 0
    
    # Track reasons for non-firing
    blocks = {
        "adx": 0,
        "spread_z": 0,
        "time_delta_z": 0,
        "ma20_slope": 0,
        "ma50_slope": 0,
        "rsi": 0,
        "price_z_not_reached": 0
    }

    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        res = compute_rule_direction(family, row, params)
        
        if res == 1: long_count += 1
        elif res == -1: short_count += 1
        else:
            # Manual check why it didn't fire (simulating rule logic)
            price_z = float(row.get("price_z", 0.0))
            
            if family == "mean_reversion":
                spread_z = float(row.get("spread_z", 0.0))
                ma20_slope = float(row.get("ma20_slope", 0.0))
                if spread_z > params.get("max_spread_z", 0.5): blocks["spread_z"] += 1
                elif abs(ma20_slope) > params.get("max_abs_ma20_slope", 0.15): blocks["ma20_slope"] += 1
                elif price_z > -params.get("threshold", 1.0) and price_z < params.get("threshold", 1.0):
                    blocks["price_z_not_reached"] += 1

            elif family == "pro_mean_reversion":
                adx = float(row.get("adx", 0.0))
                rsi = float(row.get("rsi_14", 50.0))
                if adx > params.get("adx_threshold", 25.0): blocks["adx"] += 1
                elif price_z > -params.get("price_z_threshold", 1.5) and price_z < params.get("price_z_threshold", 1.5):
                    blocks["price_z_not_reached"] += 1
                elif rsi >= params.get("rsi_oversold", 35.0) and rsi <= (100-params.get("rsi_oversold", 35.0)):
                    blocks["rsi"] += 1

    print(f"Fired: Long={long_count}, Short={short_count}")
    print(f"Blocks: {blocks}")
