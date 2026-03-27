# Validation and Do-Not-Deploy Gates

## Validation Philosophy
Backtesting is an adversarial test environment, not a marketing tool.
Any validation process that does not aggressively search for leakage, overfitting, and implementation bias is insufficient.

## Mandatory Validation Topics
The agent must explicitly evaluate:
- target leakage
- feature leakage
- cross-sectional leakage
- overlapping labels
- multiple testing / selection bias
- hyperparameter tuning bias
- backtest engine defects
- unrealistic fill assumptions
- incomplete cost models
- unrealistic execution models
- train/test contamination
- live/backtest drift

## Required Methods (when applicable)
The agent should favor:
- walk-forward evaluation,
- purged / embargoed cross-validation where overlap risk exists,
- nested tuning where hyperparameter search is material,
- sensitivity analysis,
- ablation analysis,
- regime-split validation,
- cost-stress testing,
- capacity stress testing.

## Automatic “Do Not Deploy” Conditions
Unless explicitly rebutted, classify as **Do Not Deploy** if any of the following is true:
- point-in-time correctness is unverified,
- data leakage is unresolved,
- fill logic is clearly unrealistic,
- transaction costs are materially understated or omitted,
- backtest/live code paths differ in important ways,
- no kill switch or hard risk limit exists,
- reconciliation is absent,
- operational recovery after restart is undefined,
- paper/shadow validation has not been completed for an execution-sensitive system,
- model complexity materially exceeds evidence quality.

## Validation Interpretation Rule
A statistically significant result is not enough.
A robust result must also be:
- economically meaningful,
- implementable after costs,
- stable under perturbation,
- explainable enough to debug,
- operationally supportable.
