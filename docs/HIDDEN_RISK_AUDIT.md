# Hidden Risk Audit

## Confirmed hidden risks so far

### 1. Feature parity risk between replay and live/shadow paths

The feature engine has two different computation paths:

- `_compute_raw(...)` for replay/training-style feature generation
- `_get_obs_hot_path()` for fast runtime generation

Even small semantic differences can cause hidden replay/live mismatch.

Current example found and patched:
- `price_z` in the hot path did not match `_compute_raw` semantics exactly.

Additional finding:
- An apparent runtime/replay feature mismatch was largely a rolling-context parity issue.
- When the parity check used different warmup/history windows, `time_delta_z` drift appeared significant.
- When the same effective rolling context was used, feature parity became effectively exact.
- Operational rule: parity checks must compare on the same warmup/history window.

### 2. Evidence-standard mismatch

- `pre_test_gate.py` treats presence of evidence files as enough for test readiness.
- `paper_live_gate.py` requires those artifacts to be truly valid/approved.

This can create false confidence if read casually.

### 3. Manifest-specific evidence trap

Shadow evidence is tied to the manifest hash.
Evidence for an older EURUSD manifest does not prove anything about the latest one.

### 4. Runtime cost-model source mismatch

`live_bridge.py` originally read execution costs from `manifest.execution_cost_profile`,
while the selector manifest stores them under `cost_model`.

Risk:
- runtime/live costs can silently default instead of using the certified manifest values
- profitability realism can be overstated or distorted

This was patched to prefer `cost_model` with backward-compatible fallback.

### 5. Warmup fallback can silently degrade parity

If `LIVE_ALLOW_WARMUP_DATASET_FALLBACK=1` is enabled, runtime warmup may use a generic dataset
instead of exact symbol/bar-spec warmup bars.

Risk:
- subtle feature-context mismatch at runtime
- false confidence in live parity

### 6. Restart-state equivalence can be judged too weakly

The restart drill originally considered state restoration acceptable with a minimal subset of fields.
That can hide loss of meaningful in-progress bar/risk/equity state.

This was hardened so restart checks now require much closer snapshot equivalence.

### 7. Startup reconcile can look okay without proving enough

A weak startup reconcile check can treat the runtime as reconciled even if broker/runtime position linkage
is not strongly evidenced.

This was hardened to require stronger position evidence when not flat.

### 8. Cost assumptions are duplicated across paths

Costs currently live in multiple layers:
- selector manifest `cost_model`
- training/evaluation `execution_cost_profile`
- replay broker ctor defaults
- runtime reward estimation via broker attributes
- various tools with hardcoded defaults

Risk:
- one path becomes cheaper or more expensive than another without obvious visibility
- profitability realism drifts silently

A cost-audit helper was added to make this easier to inspect.

## Operating rule

Never treat green-looking summaries as sufficient without checking:
- manifest hash
- logic hash
- evaluator hash
- feature parity
- actual shadow event accumulation for the active manifest
- whether a self-learning/optimizer tool is being used for falsification or merely for candidate generation
