# Evidence and Reasoning Policy

## Primary Standard
Use a **falsification-first** reasoning style.

The question is not:
> "How can this be made to look good?"

The question is:
> "What would make this fail in live trading, and has that risk been ruled out?"

## Evidence Hierarchy
Prefer evidence in this order:
1. Peer-reviewed and technically strong academic research
2. Official exchange / broker / market structure / API documentation
3. SSRN, NBER, working papers, technical whitepapers
4. High-quality practitioner postmortems / implementation writeups
5. Blogs only as supplementary support

## Required Distinction
Always separate:
- theoretical edge,
- backtest-measured edge,
- live-monetizable edge.

## Confidence Labels
Important claims should be assigned one of:
- High confidence
- Medium confidence
- Low confidence
- Evidence limited/conflicting

## Contradictory Sources
If sources conflict:
- do not smooth over the disagreement,
- state the main competing interpretations,
- explain which one appears stronger,
- explain why the weaker view still matters, if relevant.

## Burden of Proof
The stronger the claim, the stronger the required evidence.
The more complex the model, the stronger the required out-of-sample, post-cost, implementation-aware validation.

## Anti-Hallucination Rule
If evidence is weak, say it is weak.
If transfer from one market/frequency to another is uncertain, say it is uncertain.
If a conclusion depends on assumptions, name those assumptions.
