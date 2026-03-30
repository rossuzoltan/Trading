# Response Contract

## Default structure for substantive repo work

Use this order unless the user asks for something different:

1. `Outcome`: one sentence on what happened
2. `Why`: diagnosis or rationale
3. `Files`: the key files read or changed
4. `Verification`: commands run, what they showed, and what remains unverified
5. `Risks`: residual uncertainty, blocked checks, or rollout cautions
6. `Next step`: the single highest-value follow-up

## Writing rules

- Distinguish clearly between `observed`, `changed`, `ran`, `recommended`, and `did not verify`.
- Name exact files and commands when they matter.
- Prefer a precise limitation over a vague confidence statement.
- Use the weakest accurate completion claim from `verification-matrix.md`.
- Do not present profitability, robustness, or deployment readiness as implied by code cleanup alone.

## Patterns by task type

### Diagnosis with no code change

Use:

- outcome
- diagnosis
- files inspected
- suggested fix or next diagnostic step
- verification gap, if any

### Code change

Use:

- outcome
- files changed and what changed materially
- verification run
- explicit note on checks not run
- rollout or retrain implications, if any

### Readiness or deployment judgment

Use:

- verdict
- supporting evidence
- failing gate or missing gate
- concrete action to close the gap

## Examples of disciplined phrasing

Prefer:

- `I changed X and ran Y.`
- `I inspected X, but I did not run Y.`
- `This improves internal consistency, not proven live performance.`
- `The bundle looks incomplete because the manifest contract is not satisfied.`

Avoid:

- `Everything looks good.`
- `This should be production-ready.`
- `The strategy is profitable now.`
- `The bug is definitely fixed.`
