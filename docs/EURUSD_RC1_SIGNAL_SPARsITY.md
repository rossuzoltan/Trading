# EURUSD RC1 Signal Sparsity Notes

## Observation

The current shadow sample is not being blocked by session, spread, or risk filters.
It is simply producing `no signal` on the observed bars.

## Why this happens

The active `mean_reversion` rule requires a conjunction of conditions:

- `price_z <= -1.5` for long OR `price_z >= 1.5` for short
- `spread_z <= 0.5`
- `abs(time_delta_z) <= 2.0`
- `abs(ma20_slope) <= 0.15`
- `abs(ma50_slope) <= 0.08`
- allowed session only (`London`, `London/NY`, `NY`)

This is a fairly strict gate stack, so live-like shadow windows may legitimately produce many `no signal` bars.

## Implication

Before changing thresholds, prove one of these:

1. The live-like feed eventually accumulates enough actionable events with time.
2. The replay path materially overestimates trigger frequency.
3. The rule thresholds are too strict for the intended market regime.

## Safe next steps

- Accumulate more real shadow bars for the active manifest.
- Compare shadow signal density against replay density.
- Only tune thresholds after the sparsity is evidenced, not guessed.
