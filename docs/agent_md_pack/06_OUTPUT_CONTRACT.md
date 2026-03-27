# Output Contract

## Purpose
The agent's final answer should be decision-oriented, not descriptive fluff.

## Required Output Behavior
The answer should:
- separate scope assumptions from conclusions,
- distinguish theory vs backtest vs live monetization,
- identify must-have vs optional components,
- state evidence strength,
- make failure modes explicit,
- identify what is missing before deployment,
- end with a clear verdict.

## Tone
Use a technical, critical, non-marketing style.
Avoid:
- generic motivational advice,
- unearned optimism,
- fashionable AI praise,
- pretending uncertainty does not exist.

## Final Decision Modes
The final recommendation should be one of:
- Build now
- Research more
- Do not deploy

## Decision Logic
Choose **Build now** only when evidence, controls, execution realism, and operational readiness are all strong.
Choose **Research more** when the thesis may be viable but critical uncertainties remain unresolved.
Choose **Do not deploy** when validation, execution, data integrity, or operations are too weak to support live trading.

## Golden Question
The final document should continuously answer this question:

> Which factors actually increase the probability of live, post-cost, regime-robust success — and which factors only look valuable in backtests?
