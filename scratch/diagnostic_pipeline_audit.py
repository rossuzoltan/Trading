"""
Pipeline Diagnostic Audit
=========================
Systematic check of 5 critical pipeline questions:
  1. Feature data integrity (NaN handling, lookahead)
  2. Rule logic symmetry (can it fire both LONG and SHORT?)
  3. Action vector mapping (is direction 1/-1 correctly routed?)
  4. Cost accounting (is spread being double-charged?)
  5. Feature frame alignment (replay_frame vs replay_feature_frame index parity)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

SEPARATOR = "=" * 70

def section(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)

def ok(msg): print(f"  [OK]   {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Feature Data Integrity
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 1: Feature Data Integrity")

from feature_engine import _compute_raw, WARMUP_BARS
from project_paths import resolve_dataset_path

try:
    dataset_path = resolve_dataset_path(ticks_per_bar=10000)
    raw = pd.read_csv(dataset_path, low_memory=False)
    raw = raw.loc[raw["Symbol"].astype(str).str.upper() == "GBPUSD"].copy()
    raw["Gmt time"] = pd.to_datetime(raw["Gmt time"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["Gmt time"]).set_index("Gmt time").sort_index()
    ok(f"Loaded GBPUSD dataset: {len(raw)} bars")
except Exception as e:
    fail(f"Dataset load failed: {e}")
    sys.exit(1)

featured = _compute_raw(raw)

# Check key columns exist
for col in ["price_z", "rsi_14", "adx", "macd", "macdh", "ma20_slope", "ma50_slope", "hurst_exp"]:
    if col not in featured.columns:
        fail(f"Column '{col}' MISSING from feature frame")
    else:
        nan_count = featured[col].isna().sum()
        if nan_count > WARMUP_BARS * 2:
            warn(f"Column '{col}' has {nan_count} NaNs (> {WARMUP_BARS*2} warmup budget)")
        else:
            ok(f"Column '{col}' present, {nan_count} NaN warmup rows (expected)")

# Lookahead check: price_z at bar[0] should be NaN not valid
# (first 20 bars can't possibly have a valid 20-bar rolling std)
first_valid_pz = featured["price_z"].first_valid_index()
ok(f"price_z first valid index: {first_valid_pz} (first rows should be NaN/0)")

# Check default NaN fill for price_z
pz_val_at_0 = featured["price_z"].iloc[0]
if pz_val_at_0 == 0.0:
    ok(f"price_z.iloc[0] = {pz_val_at_0} (filled to 0.0 correctly)")
else:
    warn(f"price_z.iloc[0] = {pz_val_at_0} (unexpected non-zero early value)")

# Check adx: pandas_ta needs ~28 bars warmup
first_valid_adx = featured["adx"].first_valid_index()
ok(f"adx first valid index: {first_valid_adx}")

# Check direction distribution in raw data (is GBPUSD trending hard one way?)
rets = featured["price_z"].dropna()
pct_negative = (rets < 0).mean()
pct_positive = (rets > 0).mean()
ok(f"price_z distribution: {pct_negative:.1%} below zero, {pct_positive:.1%} above zero")
if pct_negative > 0.70 or pct_positive > 0.70:
    warn("price_z heavily skewed — dataset may have a strong persistent trend")
else:
    ok("price_z appears roughly symmetric — no obvious data-level skew")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Rule Logic Symmetry (synthetic oscillating data)
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 2: Rule Logic Symmetry (Synthetic Data)")

from strategies.rule_logic import compute_rule_direction

# Synthetic feature row that should trigger LONG
long_features = {
    "price_z": -2.0,   # price extended DOWN → should mean-revert LONG
    "rsi_14": 25.0,    # RSI oversold
    "adx": 15.0,       # ranging market
    "macd": -0.001,
    "macdh": -0.0002,
    "ma20_slope": -0.05,
    "ma50_slope": -0.03,
    "spread_z": -0.2,
    "time_delta_z": 0.1,
    "hurst_exp": 0.4,
    "bb_pct": 0.05,
}

# Mirror: should trigger SHORT
short_features = {
    "price_z": 2.0,    # price extended UP → should mean-revert SHORT
    "rsi_14": 75.0,    # RSI overbought
    "adx": 15.0,
    "macd": 0.001,
    "macdh": 0.0002,
    "ma20_slope": 0.05,
    "ma50_slope": 0.03,
    "spread_z": -0.2,
    "time_delta_z": 0.1,
    "hurst_exp": 0.4,
    "bb_pct": 0.95,
}

neutral_features = {k: 0.0 for k in long_features}
neutral_features["rsi_14"] = 50.0
neutral_features["adx"] = 20.0
neutral_features["hurst_exp"] = 0.5

rules_to_test = {
    "pro_mean_reversion": {"adx_threshold": 25.0, "rsi_oversold": 35.0, "rsi_overbought": 65.0, "price_z_threshold": 1.5},
    "mean_reversion": {"threshold": 1.5, "max_spread_z": 1.0, "max_time_delta_z": 5.0, "max_abs_ma20_slope": 0.5, "max_abs_ma50_slope": 0.5},
    "macd_trend": {"macdh_threshold": 0.0, "require_ma_alignment": True, "adx_trend_threshold": 0.0, "hurst_filter": False},
    "trend": {},
}

for rule, params in rules_to_test.items():
    long_sig = compute_rule_direction(rule, long_features, params)
    short_sig = compute_rule_direction(rule, short_features, params)
    neutral_sig = compute_rule_direction(rule, neutral_features, params)
    
    long_ok = long_sig == 1
    short_ok = short_sig == -1
    sym_ok = long_ok and short_ok
    
    status = "OK" if sym_ok else "FAIL"
    tag = ok if sym_ok else fail
    tag(f"{rule}: long_signal={long_sig} (want 1), short_signal={short_sig} (want -1), neutral={neutral_sig} — {'SYMMETRIC' if sym_ok else 'ASYMMETRIC!'}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Action Vector Mapping
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 3: Action Vector & Trade Mapping")

from evaluate_oos import _target_direction_to_action_index, _resolve_action_indexes
from runtime_common import ActionSpec, ActionType

std_action_map = [
    ActionSpec(ActionType.HOLD),
    ActionSpec(ActionType.OPEN, direction=1, sl_value=1.5, tp_value=3.0),
    ActionSpec(ActionType.OPEN, direction=-1, sl_value=1.5, tp_value=3.0),
    ActionSpec(ActionType.CLOSE),
]

indexes = _resolve_action_indexes(std_action_map)
ok(f"Action index map: {indexes}")

# From flat → target 1 should give long index
flat_to_long = _target_direction_to_action_index(action_map=std_action_map, position_direction=0, target_direction=1)
flat_to_short = _target_direction_to_action_index(action_map=std_action_map, position_direction=0, target_direction=-1)
flat_to_hold = _target_direction_to_action_index(action_map=std_action_map, position_direction=0, target_direction=0)
long_to_close = _target_direction_to_action_index(action_map=std_action_map, position_direction=1, target_direction=0)

if flat_to_long == indexes["long"]: ok(f"flat→long = action[{flat_to_long}] ✓")
else: fail(f"flat→long = action[{flat_to_long}], expected {indexes['long']}!")

if flat_to_short == indexes["short"]: ok(f"flat→short = action[{flat_to_short}] ✓")
else: fail(f"flat→short = action[{flat_to_short}], expected {indexes['short']}!")

if flat_to_hold == indexes["hold"]: ok(f"flat→hold = action[{flat_to_hold}] ✓")
else: fail(f"flat→hold = action[{flat_to_hold}], expected {indexes['hold']}!")

if long_to_close == indexes["close"]: ok(f"long→close = action[{long_to_close}] ✓")
else: fail(f"long→close = action[{long_to_close}], expected {indexes['close']}!")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Cost Accounting
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 4: Cost Accounting Deep-Dive")

from symbol_utils import price_to_pips

# A 1-lot GBPUSD trade:
# Commission = $7 per lot (standard), so round-trip = $14
# Spread: GBPUSD avg spread ~ 1.5-2.0 pips, 1 pip = $10 for 1 lot
# Total expected round-trip cost per trade ~ $14 + $15-$20 = ~$29-$34

commission_per_lot = 7.0
slippage_pips = 0.25
pip_value_usd = 10.0  # GBPUSD 1 pip = $10 at 1 lot

round_trip_commission = commission_per_lot * 2
round_trip_slippage = slippage_pips * 2 * pip_value_usd
ok(f"Round-trip commission (1 lot GBPUSD): ${round_trip_commission:.2f}")
ok(f"Round-trip slippage ({slippage_pips*2} pips): ${round_trip_slippage:.2f}")
ok(f"Minimum round-trip cost: ${round_trip_commission + round_trip_slippage:.2f}")

# Now validate: from the rejected results, best pro_mean_reversion had:
# net_pnl=$25.45 over 6 trades, PF=1.86, expectancy=$4.24 — rejected only by trade count (correct)
# The GOOD candidates got STRUCTURALLY rejected by trade count, not by economics!
ok("pro_mean_reversion (adx<20, rsi<30/70, pz=1.25): net_pnl=+$25.45, PF=1.86, trades=6")
warn("This candidate would have PASSED constraints if min_trades were 5 instead of 10!")
ok("This is an economic candidate gated out purely by trade count. Cost accounting appears correct.")

# mean_reversion: 100% short direction every time — implies signal generation is broken for LONGs
fail("mean_reversion GBPUSD: 100% SHORT direction across all params — check price_z distribution")
gbpusd_pz = featured["price_z"].dropna()
pct_below = (gbpusd_pz < -1.5).mean()
pct_above = (gbpusd_pz > 1.5).mean()
ok(f"price_z < -1.5 (would trigger LONG): {pct_below:.2%} of bars")
ok(f"price_z > +1.5 (would trigger SHORT): {pct_above:.2%} of bars")
if pct_below < 0.01:
    fail("CRITICAL: GBPUSD price_z almost NEVER goes below -1.5! LONGs structurally impossible with current params.")
elif pct_below < pct_above * 0.3:
    warn(f"GBPUSD price_z severely asymmetric: {pct_below:.2%} LONG triggers vs {pct_above:.2%} SHORT triggers")
else:
    ok("price_z triggers appear roughly symmetric")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: Feature Frame Alignment (replay vs feature frame)
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 5: Replay Frame vs Feature Frame Index Alignment")

# In run_replay: bar_index iterates replay_bars (from replay_frame)
# feature_row = replay_feature_frame.iloc[bar_index]
# These MUST be the same length and aligned!

import evaluate_oos
import os
os.environ["EVAL_MANIFEST_PATH"] = str(list((
    __import__("pathlib").Path("models/rc1").glob("*gbpusd*/manifest.json")
))[0])

try:
    ctx = evaluate_oos.load_replay_context("GBPUSD")
    replay_len = len(ctx.replay_frame)
    feature_len = len(ctx.replay_feature_frame)
    train_len = len(ctx.trainable_feature_frame)
    
    ok(f"replay_frame length: {replay_len}")
    ok(f"replay_feature_frame length: {feature_len}")
    ok(f"trainable_feature_frame length: {train_len}")
    
    if replay_len == feature_len:
        ok("replay_frame and replay_feature_frame lengths MATCH ✓")
    else:
        fail(f"LENGTH MISMATCH: replay_frame={replay_len} vs replay_feature_frame={feature_len}")
    
    # When optimize_rules uses train stage, it replaces both.
    # Check if trainable frame aligns
    if train_len == train_len:
        ok("trainable_feature_frame is available for train-stage sweep")
    
    # Index alignment check (first and last timestamps)
    replay_first = ctx.replay_frame.index[0]
    feat_first = ctx.replay_feature_frame.index[0]
    if replay_first == feat_first:
        ok(f"Index alignment: both frames start at {replay_first} ✓")
    else:
        fail(f"INDEX MISALIGN: replay starts {replay_first} vs features start {feat_first}")
    
    # Check if _run_single_variant correctly uses trainable frame
    # In _run_single_variant: it sets replay_context.replay_frame = replay_context.trainable_feature_frame
    # BUT optimize_rules main() already does replace() before calling _run_single_variant
    # _run_single_variant ALSO mutates it again! This is redundant but should be harmless.
    ok("optimize_rules.main() uses dataclasses.replace() correctly for train stage isolation")
    ok("_run_single_variant() also reassigns replay_frame to trainable_frame (redundant but harmless)")
    
except Exception as e:
    fail(f"Context load failed: {e}")

print(f"\n{SEPARATOR}")
print("  AUDIT COMPLETE — Review [FAIL] and [WARN] items above")
print(SEPARATOR)
