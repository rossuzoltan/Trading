"""
Test for Microstructure Bounce logic to ensure symmetry and expected behavior.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategies.rule_logic import compute_microstructure_bounce

params = {
    "td_threshold": -2.0,
    "price_z_threshold": 1.5,
    "spread_max_z": 1.0
}

# 1. Validation for LONG: High speed (-2.5), Price stretched down (-1.6), normal spread (0.5)
long_features = {
    "time_delta_z": -2.5,
    "price_z": -1.6,
    "spread_z": 0.5
}

# 2. Validation for SHORT: High speed (-2.5), Price stretched up (+1.6), normal spread (0.5)
short_features = {
    "time_delta_z": -2.5,
    "price_z": 1.6,
    "spread_z": 0.5
}

# 3. Validation for REJECT (Too slow): low speed (-1.0 > -2.0)
slow_features = {
    "time_delta_z": -1.0,
    "price_z": 1.6,
    "spread_z": 0.5
}

# 4. Validation for REJECT (Spread too high): 
high_spread = {
    "time_delta_z": -2.5,
    "price_z": 1.6,
    "spread_z": 1.5
}

def test_rule():
    l_sig = compute_microstructure_bounce(long_features, params)
    s_sig = compute_microstructure_bounce(short_features, params)
    slow_sig = compute_microstructure_bounce(slow_features, params)
    spread_sig = compute_microstructure_bounce(high_spread, params)

    print("--- MICROSCRTUCTURE BOUNCE SYMMETRY TEST ---")
    print(f"LONG  signal (want 1): {l_sig}")
    print(f"SHORT signal (want -1): {s_sig}")
    print(f"SLOW  signal (want 0): {slow_sig}")
    print(f"WIDE_SPREAD signal (want 0): {spread_sig}")
    
    if l_sig == 1 and s_sig == -1 and slow_sig == 0 and spread_sig == 0:
        print("[OK] Test Passed: Perfectly symmetric and handles guards.")
        sys.exit(0)
    else:
        print("[FAIL] Logic is incorrect!")
        sys.exit(1)

if __name__ == "__main__":
    test_rule()
