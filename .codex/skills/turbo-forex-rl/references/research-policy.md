# Research Policy

## Evaluation Standard

Judge every meaningful change by whether it improves the probability of robust live post-cost performance, not whether it makes a backtest look better.

Check at least:

- data integrity and point-in-time correctness
- live/backtest parity
- transaction cost realism
- regime robustness
- operational recovery and auditability

## Do-Not-Deploy Gates

Treat the system as not deployable when any of these remain unresolved:

- leakage or contamination risk
- unrealistic fill or cost assumptions
- important divergence between backtest and live paths
- missing kill switch, recovery path, reconciliation, or rollback confidence
- model complexity that exceeds the quality of the evidence

## AI And RL Policy

- Do not justify AI by default.
- Compare against simpler baselines first: rule-based, linear, then modest tree models.
- Treat RL as high false-positive risk unless it shows clear incremental value after realistic validation and costs.
- Ask what the RL system learns that a simpler baseline does not, and whether that gain survives walk-forward or regime-split evaluation.

## Practical Bias

- Prefer simpler systems when the economic value of added complexity is unclear.
- Reject visually attractive results that are operationally fragile or hard to falsify.
- Optimize for debugability, rollback safety, and evidence quality.
