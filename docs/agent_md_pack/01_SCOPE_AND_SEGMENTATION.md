# Scope and Segmentation Rules

## Do Not Generalize Across Incompatible Contexts
The agent must not collapse all trading systems into one universal template.
Requirements differ materially across:

- asset class,
- time horizon,
- execution sensitivity,
- market structure,
- capacity constraints.

## Required Segmentation
Whenever conclusions depend on context, split analysis explicitly by:

### Asset Class
- Equities
- Futures
- FX
- Crypto

### Time Horizon
- HFT
- Intraday
- Swing
- Daily+

### Execution Sensitivity
- Microstructure-sensitive systems
- Lower-frequency / lower-execution-sensitivity systems

## Rule for Claims
Every non-trivial claim should be tagged mentally as one of:
- broadly generalizable,
- valid mainly for high-frequency / microstructure-sensitive systems,
- valid mainly for medium-frequency intraday systems,
- valid mainly for swing / daily+ systems,
- valid mainly for a specific asset class.

If uncertain, say so explicitly instead of over-generalizing.

## Examples
- Queue position, matching rules, and adverse selection are critical for HFT and some intraday systems, but much less central for daily-rebalanced strategies.
- Corporate actions and point-in-time fundamentals matter much more in equities than in many futures strategies.
- Order book features are often more informative in microstructure-sensitive contexts than in lower-frequency macro systems.

## Anti-Confusion Rule
Never present a requirement that is essential in one regime as universally essential in all regimes without qualification.
