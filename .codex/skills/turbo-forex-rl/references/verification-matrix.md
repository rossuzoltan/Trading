# Verification Matrix

## Table of contents

- [Claim ladder](#claim-ladder)
- [Minimum checks by task type](#minimum-checks-by-task-type)
- [Report missing verification explicitly](#report-missing-verification-explicitly)

## Claim ladder

Use the weakest accurate claim.

- `inspected`: read docs or code, but did not execute a verifying command
- `modified`: changed files, but did not run a relevant check yet
- `smoke-tested`: ran a targeted command or test that exercises the changed path
- `regression-checked`: ran broader regression coverage relevant to the task
- `ready for manual validation`: code and targeted checks look sound, but required end-to-end or environment-dependent validation is still outstanding
- `live-ready`: only use when the live/ops gate below is satisfied

Never collapse these levels into a stronger claim.

## Minimum checks by task type

### Docs-only triage or handoff continuation

Minimum standard:

- read the latest `docs/NEXT_AGENT_*.md` files if present
- confirm whether any claim is based on docs only versus fresh runtime evidence

Allowed claim level:

- `inspected`

### Data ingest, refresh, or repair

Minimum standard before saying the data path is fixed or healthy:

- inspect the affected logic in `download_dukascopy.py` and or `build_volume_bars.py`
- run the cheapest relevant validation available, usually `tools/project_healthcheck.py` or a targeted data-path test
- confirm the implied bar spec and symbol coverage against `data/dataset_build_info.json`

Do not claim:

- that the rebuilt dataset is complete if no rebuild or validation ran
- that the active bar spec changed unless metadata proves it

### Feature or schema change

Minimum standard before saying the change is safe:

- inspect the feature producer and every immediate consumer that depends on shape, order, or names
- run relevant tests or a smoke path that touches the changed schema
- state whether scaler, VecNormalize, manifest, and existing model bundles remain compatible

Do not claim:

- backward compatibility unless you checked it
- training or live compatibility from code inspection alone

### Training-path change

Minimum standard before saying training is fixed or improved:

- inspect `train_agent.py` plus directly affected runtime or feature files
- run at least targeted tests or a smoke validation that exercises the training path if the repository supports one
- state clearly whether full training was not run

Do not claim:

- improved alpha, reward quality, or profitability without fresh evaluation evidence
- a model was retrained unless a training run actually happened

### Evaluation-path change

Minimum standard before saying evaluation is fixed or trustworthy:

- inspect `evaluate_oos.py` and the relevant manifest-loading logic
- verify the target symbol bundle exists and matches the expected manifest contract
- run the evaluation path when the environment and artifacts are available, otherwise say it remains unverified

Do not claim:

- out-of-sample improvement from code changes alone
- bundle compatibility from filename presence alone

### Live, MT5, or readiness work

Minimum standard before saying a live path is ready:

- confirm artifact parity through manifests and current bundle files
- run `tools/project_healthcheck.py` when runtime health matters
- run `mt5_live_preflight.py` with the symbol and active ticks-per-bar from current metadata or task context
- call out recovery, reconciliation, kill-switch, or audit limitations if they were not checked

Allowed strongest claim:

- `live-ready` only when the required preflight and parity evidence exists

### Research judgment or whether RL is justified

Minimum standard before recommending RL:

- apply the rules in `references/research-policy.md`
- compare against simpler baselines conceptually or empirically
- separate research hypothesis from deployable evidence

Do not claim:

- that RL is warranted by default
- that a backtest result alone justifies deployment

## Report missing verification explicitly

When a needed check could not be run, say exactly which check is missing and why. Prefer wording like:

- `I updated the manifest-loading path and ran unit tests, but I did not run OOS evaluation because the target artifact bundle was not present.`
- `The code change looks internally consistent, but live readiness is still unverified because MT5 preflight was not run.`
