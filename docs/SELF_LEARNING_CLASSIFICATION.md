# Self-Learning Classification

## Safe to use (offline research / falsification)

- `tools/horizon_falsifier.py`
- `tools/analyze_shadow_drift.py`
- `tools/alpha_gate_bakeoff.py`
- `tools/compare_oos_baselines.py`
- `tools/analyze_holdout_diagnostics.py`

These are useful for falsifying fragile ideas and understanding failure modes.

## Use with caution (easy to overfit)

- `tools/calibrate_sparsity.py`
- `tools/optimize_rules.py`
- `tools/research_runner.py`
- `tools/experiment_f_meta_label.py`

These can generate useful challengers, but also create false progress if used without strict out-of-sample and shadow discipline.

## Do not trust for live promotion by themselves

- any training or optimizer path that outputs a "best" candidate without live-like evidence
- especially `optimize_rules.py`, `research_runner.py`, and raw training loops

## Recommended operating order

1. Falsify first
   - horizon falsifier
   - baseline comparison
   - shadow drift analysis
2. Calibrate sparsity only after falsification
3. Generate challengers only after the baseline path is honest
4. Never promote automatically; every challenger must still pass replay, drift, shadow, ops, and manifest/hash checks
