import sys, os
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluate_oos import load_replay_context
from project_paths import resolve_dataset_path

print("Loading EURUSD 5000 context...")
ctx = load_replay_context("EURUSD")
df = ctx.trainable_feature_frame

td_z = df["time_delta_z"].dropna()
pz = df["price_z"].dropna()
sz = df["spread_z"].dropna()

print("\n--- Feature Distributions on Train Set ---")

def stats(name, series):
    print(f"\n{name} Stat:")
    print(f"  Min:  {series.min():.2f}")
    print(f"  Max:  {series.max():.2f}")
    print(f"  Mean: {series.mean():.2f}")
    print(f"  P01:  {np.percentile(series, 1):.2f}")
    print(f"  P05:  {np.percentile(series, 5):.2f}")
    print(f"  P10:  {np.percentile(series, 10):.2f}")
    print(f"  P90:  {np.percentile(series, 90):.2f}")
    print(f"  P95:  {np.percentile(series, 95):.2f}")
    print(f"  P99:  {np.percentile(series, 99):.2f}")

stats("time_delta_z", td_z)
stats("price_z", pz)
stats("spread_z", sz)

# Joint condition test
print("\n--- Joint Co-occurrence ---")
fast_bars = (td_z <= -1.5)
spread_ok = (sz <= 1.0)
extended_long = (pz <= -1.25)
extended_short = (pz >= 1.25)

print(f"Total Bars: {len(df)}")
print(f"Bars with td_z <= -1.5: {fast_bars.sum()}")
print(f"Bars with spread_z <= 1.0: {spread_ok.sum()}")
print(f"Bars with price_z <= -1.25: {extended_long.sum()}")
print(f"Bars with price_z >= 1.25: {extended_short.sum()}")

long_triggers = (fast_bars & spread_ok & extended_long).sum()
short_triggers = (fast_bars & spread_ok & extended_short).sum()

print(f"\nLONG Triggers (td<=-1.5, spread<=1.0, pz<=-1.25): {long_triggers}")
print(f"SHORT Triggers (td<=-1.5, spread<=1.0, pz>=1.25): {short_triggers}")
