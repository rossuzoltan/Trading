# Current Usage Guide

This is the practical operator guide for the current supported stack.

The primary operational path is:
`rule-first RC1 + AlphaGate + shadow evidence + paper-live gate`.

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
2. **Evaluate OOS**: Verify the candidates against out-of-sample data.
   ```powershell
   .\.venv\Scripts\python.exe .\evaluate_oos.py --symbol GBPUSD
   ```
3. **Generate RC1 Pack**: Create the release candidate artifact.
   ```powershell
   .\.venv\Scripts\python.exe .\tools\generate_v1_rc.py
   ```
4. **Shadow Simulation**: Verify the logic against live-like conditions.
   ```powershell
   .\tools\run_shadow_simulator.ps1 -ManifestPath models\rc1\gbpusd_10k_v1_mr_rc1\manifest.json
   ```
5. **Paper-Live Gate**: Build the final operational verdict.
   ```powershell
   .\.venv\Scripts\python.exe .\tools\paper_live_gate.py --manifest-path .\models\rc1\gbpusd_10k_v1_mr_rc1\manifest.json
   ```

## 5. Decision Rules

- **PF > 1.15**: Rule candidates must satisfy minimum Profit Factor on both train and OOS.
- **AlphaGate Veto**: The AlphaGate (Logistic Regression meta-filter) is mandatory for RC1 certification.
- **Shadow Parity**: Shadow evidence must align with replay outcomes before live promotion.
- **Legacy RL**: Treat `train_agent.py` and `MaskablePPO` as research-only. Do not use them for primary signal generation.

## 6. Files That Matter Most

- `strategies/rule_logic.py`: Single source of truth for signal logic.
- `tools/optimize_rules.py`: The research and optimization harness.
- `rule_selector.py`: The manifest-driven signal consumer.
- `runtime/shadow_broker.py`: Shadow-mode implementation.
- `evaluate_oos.py`: The authoritative evaluator.
- `docs/PROFITABILITY_PLAN.md`: The canonical operator plan.
