# AI / ML / RL Decision Policy

## Default Position
AI is **not justified by default**.
Use it only when it can show incremental value over simpler baselines under realistic validation and execution assumptions.

## Required Baseline Order
The agent should mentally compare:
1. Rule-based baseline
2. Linear / generalized linear baseline
3. Tree-based or modest supervised ML baseline
4. Deep learning, only if justified
5. Reinforcement learning, only with especially strong justification

## Rule-Based Systems
Treat as strong candidates when:
- sample sizes are limited,
- interpretability matters,
- latency/control requirements are strict,
- the domain logic is stable and explicit.

## Supervised ML
Most justified when:
- there is enough data,
- labels are well-defined,
- nonlinearity adds measurable value,
- costs and turnover are modeled,
- performance survives regime splits.

## Deep Learning
Demand stronger evidence because it typically requires:
- more data,
- more tuning,
- more engineering,
- more monitoring,
- more failure analysis,
- more retraining discipline.

## Reinforcement Learning
Assume high false-positive risk unless proven otherwise.
Common reasons for failure:
- reward misspecification,
- simulation-to-live gap,
- cost ignorance,
- sparse reward instability,
- unstable policy learning,
- non-stationary environment mismatch,
- weak offline evaluation.

## Required AI Questions
Before endorsing AI, the agent must ask:
- What does the complex model learn that a simpler one does not?
- Is the gain still present after realistic costs?
- Does the gain survive walk-forward and regime splits?
- Is the additional maintenance burden justified?
- Can failures be diagnosed?
- Can the model be safely rolled back?

## “AI Not Justified” Rule
Conclude **AI not justified** when most of the performance claim comes from:
- weak out-of-sample evidence,
- fragile feature sets,
- unstable training,
- heavy hyperparameter search,
- simulated assumptions that do not match live reality,
- marginal gains over simpler baselines.
