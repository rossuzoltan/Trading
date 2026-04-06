# Common Pitfalls

## Path and environment mistakes

- Treating `c:\dev\trading` as mandatory instead of using the actual checkout path in the running environment
- Using the wrong Python interpreter instead of the project virtualenv

## Stale-doc mistakes

- Trusting older handoff prose when current code, `data/dataset_build_info.json`, or `tools/project_healthcheck.py` says otherwise
- Repeating a historical caveat after a dataset rebuild or artifact refresh already removed it

## Artifact-contract mistakes

- Assuming a bare model zip is enough when the live or evaluation path expects a valid symbol manifest
- Forgetting that feature-shape changes can invalidate scaler, VecNormalize, or manifest expectations
- Declaring symbol readiness without checking the symbol-scoped files under `models/`

## Wrong-subsystem mistakes

- Editing `trading_env.py` when the supported runtime path is `runtime_gym_env.py`
- Reopening `event_pipeline.py` even though the evidence does not point there
- Treating H1 or `RecurrentPPO` flows as the default path for current work

## Verification mistakes

- Calling a training-path change successful without running even a targeted check
- Claiming out-of-sample improvement from code inspection or in-sample behavior alone
- Claiming live readiness without manifest parity, preflight evidence, and operational rollback confidence

## Research mistakes

- Assuming RL is justified without comparing against simpler baselines
- Treating visually attractive charts or backtests as enough evidence for deployment
- Optimizing for novelty instead of falsifiability, auditability, and post-cost realism
