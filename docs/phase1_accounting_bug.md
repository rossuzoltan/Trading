# Phase 1 — Evaluation Accounting Bug: Root Cause and Fix

**Date:** 2026-03-29  
**Affects:** `train_agent.py` `evaluate_model()` + `FullPathEvalCallback`  
**Does NOT affect:** `evaluate_oos.py` (that path was correct)

---

## The Bug

The training eval callback (`FullPathEvalCallback._on_step`) reported
`trade_count = 0`, `net_pnl_usd = 0`, `profit_factor = 0` in the
heartbeat's `latest_eval` field — while the heartbeat's `execution_diagnostics`
**simultaneously showed thousands of closed trades and large transaction costs**.

This appeared to be contradictory evidence. It was.

### Two independent sources, not one

The heartbeat combined two unrelated data streams into the same JSON blob:

| Field in heartbeat | Source | Meaning |
|---|---|---|
| `latest_eval.trade_count` | `compute_trade_metrics(trade_log)` in `evaluate_model()` | Count from the **eval replay** trade log |
| `execution_diagnostics.closed_trade_count` | `TrainingDiagnostics.snapshot()` from training envs | Count accumulated from **all training steps** since last reset |

They counted different things. They do not need to match.

### Why `latest_eval.trade_count` was always 0

`evaluate_model()` called `_extract_eval_trade_log(eval_env)`, which attempted:

```python
runtimes = eval_env.get_attr("_runtime")  # returns the RuntimeEngine object
broker = runtimes[0].broker
return broker.trade_log
```

When using `SubprocVecEnv` (multiple workers), `get_attr("_runtime")` serializes
the `RuntimeEngine` object across a process boundary. `RuntimeEngine` contains
`ReplayBroker`, `FeatureEngine`, and `RiskEngine` objects — none of which are
`pickle`-serializable in this state. The `get_attr` call either raised an exception
(caught silently) or returned `None`.

**Result:** `_extract_eval_trade_log` returned `[]`. `compute_trade_metrics([])` returned
all-zero metrics. No warning was emitted. The zeros were written to the heartbeat as if
they were real results.

### Why this did not surface in unit tests

The `DummyEvalEnv` in `test_train_rehab.py` has no `_runtime` attribute and
`get_attr` raises `AssertionError`. The old code caught that exception and returned
`[]` — which meant the unit test was silently testing the broken empty-log path
the whole time.

---

## The Fix

### 1. Expose logs via `env_method()` — `runtime_gym_env.py`

Two new public methods were added to `RuntimeGymEnv`:

```python
def get_trade_log(self) -> list[dict[str, Any]]:
    ...  # returns broker.trade_log

def get_execution_log(self) -> list[dict[str, Any]]:
    ...  # returns broker.execution_log
```

`env_method("get_trade_log")` serializes the **return value** (a plain Python list
of dicts), not the broker object. This works correctly across `SubprocVecEnv`
process boundaries.

### 2. 3-strategy fallback chain — `train_agent.py`

`_extract_eval_trade_log()` now attempts:
1. `eval_env.env_method("get_trade_log")` — primary path, works via SubprocVecEnv
2. `eval_env.get_attr("_runtime") → broker.trade_log` — works in DummyVecEnv
3. `eval_env.get_attr("trade_log")` — ForexTradingEnv / legacy fallback

If all three fail, it **emits a WARNING** and returns `[]`. It does not raise.
The warning is logged under `train_agent`.

### 3. Reconciliation in `evaluate_model()` output

`evaluate_model()` now includes:
- `metric_reconciliation`: the full output of `build_trade_metric_reconciliation()`
- `accounting_gap_detected: bool`: `True` if `trade_log` is empty but
  `execution_diagnostics` reports >0 closed trades

When `accounting_gap_detected` is `True`, a WARNING is logged explaining the mismatch.

---

## Why the New Accounting is Trustworthy

1. **`evaluate_oos.py` was always correct** — it reads `broker.trade_log` directly
   from the `ReplayBroker` instance (no serialization needed). This is unchanged.

2. **The fix uses the same data structure** as `evaluate_oos.py`: `broker.trade_log`
   is a `list[dict]` populated by `ReplayBroker._close_position()`. The path is now
   `env_method("get_trade_log")` → `broker.trade_log`, matching what OOS eval uses.

3. **`compute_trade_metrics()` is the single source of truth** for all of:
   `trade_count`, `gross_pnl_usd`, `net_pnl_usd`, `total_transaction_cost_usd`,
   `commission`, `spread_slippage_cost`, `profit_factor`, `win_rate`, `expectancy`,
   `avg_holding_bars`, `forced_close_count`, `win_loss_asymmetry`.

4. **`build_trade_metric_reconciliation()`** cross-checks the metrics dict against
   the `TrainingDiagnostics` counters. If they agree: `passed: true`. If they
   disagree: `mismatch_fields` lists the exact fields. This check runs in both
   `evaluate_model()` and `evaluate_oos.py`.

5. **Regression tests in `tests/test_eval_accounting.py`** prove that:
   - 10 known trades cannot return `trade_count=0`
   - The 3-strategy chain is exercised individually
   - `accounting_gap_detected=True` fires when the original bug conditions occur
   - `build_trade_metric_reconciliation` detects deliberate count mismatches

---

## What Is NOT Changed

- `evaluate_oos.py` — already correct, no changes
- `ReplayBroker` and `event_pipeline.py` — runtime/execution logic unchanged
- `TrainingDiagnostics` — training step counters unchanged (they count training steps,
  not eval steps; this is correct and expected behaviour)
- Live runtime path (`live_bridge.py`) — not touched

---

## Metrics Now Trustworthy From

| Source | Primary economic metrics | Notes |
|---|---|---|
| `evaluate_oos.py` | `trade_metrics.*` from `compute_trade_metrics(broker.trade_log)` | Always was correct |
| `evaluate_model()` in training | `trade_metrics.*` from `compute_trade_metrics(broker.trade_log)` | Fixed by this PR |
| `FullPathEvalCallback.latest_metrics` | Same as `evaluate_model()` output | Fixed by this PR |
| `TrainingHeartbeatCallback.latest_eval` | Copies `evaluate_model()` output | Fixed transitively |
| `execution_diagnostics` / `trade_diagnostics` | Training step counters | These are for training monitoring ONLY, not economic reporting |
