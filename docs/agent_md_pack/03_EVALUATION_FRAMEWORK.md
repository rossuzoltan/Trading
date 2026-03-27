# Evaluation Framework

## Master Question
For every component, ask:

> Does this increase the probability of robust, live, post-cost performance — or does it mainly improve the backtest?

## Mandatory Evaluation Dimensions
Evaluate important components along these dimensions:
- Direct impact on live PnL
- Robustness across regimes
- Implementation difficulty
- Error risk
- Scalability / capacity
- Maintainability

## Mandatory Rating Labels
Every major component should be classified as one of:
- Must-have
- Strongly recommended
- Situational
- Usually overrated
- Dangerous / high false-positive risk

## Core Buckets
The agent must explicitly examine:
- alpha source quality
- edge decay / crowding / capacity / turnover / half-life
- data integrity and point-in-time correctness
- feature engineering vs model complexity
- regime sensitivity / non-stationarity / drift
- transaction costs and execution realism
- portfolio construction and position sizing
- risk controls and drawdown containment
- live/backtest parity
- monitoring, recovery, auditability, rollback

## Critical Interpretive Rule
A system is not “good” because it has:
- high Sharpe in backtest,
- many features,
- a sophisticated model,
- a clean research notebook,
- a profitable paper-trading period.

A system is closer to good only if:
- its edge remains after realistic costs,
- assumptions survive adversarial validation,
- execution is implementable,
- operational controls are present,
- live behavior is inspectable and reversible.

## Default Skepticism Toward Complexity
Prefer the simpler system unless the more complex version demonstrates:
- clear incremental economic value,
- better stability,
- acceptable maintenance burden,
- no obvious validation contamination.
