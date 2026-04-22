# Current Usage Guide

This is the practical operator guide for the current supported stack.

The primary operational path is:
`rule-first RC1 + fail-fast pre-test gate + shadow evidence + paper-live gate`.

## 0. Grounded Status

As of `2026-04-10` the repo is structurally healthier, but neither anchor is test-ready yet.

- `EURUSD` regenerated RC: `6` replay trades, net `+$1.89`, blocked by stale/critical historical replay evidence.
- `GBPUSD` regenerated RC: `4` replay trades, net `+$5.14`, blocked by stale/critical historical replay evidence.
- Exact-runtime EURUSD bakeoff result: `rule_only` is currently better than the manifest AlphaGate and the refit challengers on holdout. `xgboost_pair` is the best refit challenger, but still weaker than the ungated rule on net PnL.

## 1. Environment

- Work from the repo root: `c:\dev\trading`
- Use the project venv explicitly: `.\.venv\Scripts\python.exe`

## 2. Healthcheck

Run:
```powershell
.\.venv\Scripts\python.exe .\tools\project_healthcheck.py --mode rc1
```
This confirms that approved RC1 packs exist and are structurally valid.

## 3. Data Workflow

Refresh raw tick data if needed:
```powershell
.\.venv\Scripts\python.exe .\download_dukascopy.py --pairs EURUSD GBPUSD --days 1095 --force-refresh-pairs EURUSD GBPUSD --max-workers 16
```

## 4. Rule-First / AlphaGate Workflow

The anchor workflow is manifest-driven:

1. **Optimize Rules**: Find the best rule parameters for a symbol.
   ```powershell
   .\.venv\Scripts\python.exe .\tools\optimize_rules.py --symbol GBPUSD
   ```
   Optional AlphaGate backends:
   ```powershell
   .\.venv\Scripts\python.exe .\tools\optimize_rules.py --symbol GBPUSD --use-alpha-gate --alpha-gate-model logistic_pair
   .\.venv\Scripts\python.exe .\tools\optimize_rules.py --symbol GBPUSD --use-alpha-gate --alpha-gate-model xgboost_pair
   .\.venv\Scripts\python.exe .\tools\optimize_rules.py --symbol GBPUSD --use-alpha-gate --alpha-gate-model lightgbm_pair
   ```
   Optional regime-guard sweep:
   ```powershell
   .\.venv\Scripts\python.exe .\tools\optimize_rules.py --symbol GBPUSD --enable-regime-guard-sweep
   ```
2. **Evaluate OOS**: Verify the candidates against out-of-sample data.
   ```powershell
   .\.venv\Scripts\python.exe .\evaluate_oos.py --symbol GBPUSD
   ```
3. **Generate RC1 Pack**: Create the release candidate artifact.
   ```powershell
   .\.venv\Scripts\python.exe .\tools\generate_v1_rc.py
   ```
4. **Verify RC1 Pack**: Re-run parity and baseline certification on the generated packs.
   ```powershell
   .\.venv\Scripts\python.exe .\tools\verify_v1_rc.py
   ```
5. **Fail-Fast Pre-Test Gate**: Do not start a new shadow run until this passes.
   ```powershell
   .\.venv\Scripts\python.exe .\tools\pre_test_gate.py --manifest-path .\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json
   ```
6. **Exact-Runtime AlphaGate Bakeoff**: Compare `rule_only`, manifest gate, and refit challengers on the same holdout.
   ```powershell
   .\.venv\Scripts\python.exe .\tools\alpha_gate_bakeoff.py --manifest-path .\models\rc1\eurusd_5k_v1_mr_rc1\manifest.json
   ```
7. **Historical Replay Refresh**: If pre-test fails on stale or drifted MT5 replay evidence, regenerate the report before shadow.
   ```powershell
   .\.venv\Scripts\python.exe .\tools\mt5_historical_replay.py --symbol EURUSD --days 30
   ```
8. **Shadow Simulation**: Verify the logic against live-like conditions.
   ```powershell
   .\tools\run_shadow_simulator.ps1 -ManifestPath models\rc1\gbpusd_10k_v1_mr_rc1\manifest.json
   ```
9. **Paper-Live Gate**: Build the final operational verdict.
   ```powershell
   .\.venv\Scripts\python.exe .\tools\paper_live_gate.py --manifest-path .\models\rc1\gbpusd_10k_v1_mr_rc1\manifest.json
   ```

## 5. Decision Rules

- **PF > 1.15**: Rule candidates must satisfy minimum Profit Factor on both train and OOS.
- **AlphaGate Veto**: The AlphaGate meta-filter is mandatory for RC1 certification; `logistic_pair` is the default and `xgboost_pair` / `lightgbm_pair` are optional challengers.
- **Pre-Test Gate First**: `pre_test_gate.py` must pass before a new shadow campaign. It hard-fails on thin replay, one-sided replay, stale historical replay hashes, Asia/Rollover opens, and critical density drift.
- **Shadow Parity**: Shadow evidence must align with replay outcomes before live promotion.
- **Regime Filter (Optional)**: Rule params can include `min_vol_norm_atr`, `max_abs_log_return`, `max_abs_body_size`, and `max_candle_range` to suppress sideways and spike-like regimes.
- **Legacy RL**: Treat `train_agent.py` and `MaskablePPO` as research-only. Do not use them for primary signal generation.

## 6. Automated Gate Reporting

Generate a consolidated gate summary + visualization:
```powershell
.\.venv\Scripts\python.exe .\tools\summarize_gate_reports.py
```

## 7. Files That Matter Most

- `strategies/rule_logic.py`: Single source of truth for signal logic.
- `tools/optimize_rules.py`: The research and optimization harness.
- `rule_selector.py`: The manifest-driven signal consumer.
- `runtime/shadow_broker.py`: Shadow-mode implementation.
- `evaluate_oos.py`: The authoritative evaluator.
- `tools/pre_test_gate.py`: The fail-fast test-readiness gate.
- `tools/alpha_gate_bakeoff.py`: Exact-runtime AlphaGate challenger comparison.
- `docs/PROFITABILITY_PLAN.md`: The canonical operator plan.
