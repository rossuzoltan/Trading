import sys, os
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
os.environ['EVAL_MANIFEST_PATH'] = 'models/rc1/eurusd_5k_v1_mr_rc1/manifest.json'

import evaluate_oos

print('=== WHAT THE OPTIMIZER ACTUALLY SEES ===')
ctx = evaluate_oos.load_replay_context('EURUSD')

print(f'replay_frame (holdout):   {len(ctx.replay_frame)} bars  {ctx.replay_frame.index[0]} --> {ctx.replay_frame.index[-1]}')
print(f'trainable_feature_frame:  {len(ctx.trainable_feature_frame)} bars  {ctx.trainable_feature_frame.index[0]} --> {ctx.trainable_feature_frame.index[-1]}')
print(f'holdout_feature_frame:    {len(ctx.holdout_feature_frame)} bars  {ctx.holdout_feature_frame.index[0]} --> {ctx.holdout_feature_frame.index[-1]}')
print()

pz_train = ctx.trainable_feature_frame['price_z'].dropna()
print(f'=== price_z IN TRAINABLE_FEATURE_FRAME (what optimize_rules actually evaluates) ===')
print(f'Bars: {len(pz_train)}')
print(f'min={pz_train.min():.3f}  max={pz_train.max():.3f}  mean={pz_train.mean():.3f}')
print(f'price_z below -1.0: {(pz_train < -1.0).sum()} bars ({(pz_train < -1.0).mean()*100:.1f}%)')
print(f'price_z below -1.5: {(pz_train < -1.5).sum()} bars ({(pz_train < -1.5).mean()*100:.1f}%)')
print(f'price_z above +1.0: {(pz_train > 1.0).sum()} bars ({(pz_train > 1.0).mean()*100:.1f}%)')
print(f'price_z above +1.5: {(pz_train > 1.5).sum()} bars ({(pz_train > 1.5).mean()*100:.1f}%)')
print()

# RSI distribution - is RSI ever oversold enough?
rsi = ctx.trainable_feature_frame['rsi_14'].dropna()
print(f'=== RSI_14 IN TRAINABLE FRAME ===')
print(f'min={rsi.min():.1f}  max={rsi.max():.1f}  mean={rsi.mean():.1f}')
print(f'RSI < 30 (oversold for long): {(rsi < 30).sum()} bars ({(rsi < 30).mean()*100:.1f}%)')
print(f'RSI > 70 (overbought for short): {(rsi > 70).sum()} bars ({(rsi > 70).mean()*100:.1f}%)')
print()

# How many bars meet BOTH conditions for a long signal (price_z AND rsi)
both_long = ((pz_train.reindex(rsi.index) < -1.0) & (rsi < 30)).sum()
both_short = ((pz_train.reindex(rsi.index) > 1.0) & (rsi > 70)).sum()
print(f'=== pro_mean_reversion SIGNAL CONDITIONS ===')
print(f'Bars meeting price_z<-1.0 AND rsi<30 (LONG): {both_long}')
print(f'Bars meeting price_z>+1.0 AND rsi>70 (SHORT): {both_short}')

# What is the price doing in trainable window?
from project_paths import resolve_dataset_path
import pandas as pd
dataset_path = resolve_dataset_path(ticks_per_bar=5000)
raw = pd.read_csv(dataset_path, low_memory=False)
raw = raw.loc[raw['Symbol'].astype(str).str.upper() == 'EURUSD'].copy()
raw['Gmt time'] = pd.to_datetime(raw['Gmt time'], utc=True, errors='coerce')
raw = raw.dropna(subset=['Gmt time']).set_index('Gmt time').sort_index()

train_start = ctx.trainable_feature_frame.index[0]
train_end = ctx.trainable_feature_frame.index[-1]
raw_train = raw.loc[(raw.index >= train_start) & (raw.index <= train_end)]
print()
print(f'=== RAW CLOSE IN TRAINABLE WINDOW ===')
print(f'Period: {train_start} --> {train_end}')
if len(raw_train) > 0:
    start_p = raw_train['Close'].iloc[0]
    end_p = raw_train['Close'].iloc[-1]
    print(f'Close: start={start_p:.5f}  end={end_p:.5f}  net={end_p-start_p:+.5f}')
