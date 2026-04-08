# Trading Repository Implementation Roadmap

## Rule-First, Exact-Runtime, Shadow-Mode Hardening Plan

## Strategic Context

The repository has now moved beyond the earlier PPO-centered research phase.

The current highest-value direction is:

* **Rule-First**
* **Exact-runtime parity as truth**
* **Manifest-driven deployment**
* **Paper-live / shadow-mode validation before any live-money path**
* **Controlled reduction of legacy RL technical debt**

This roadmap is designed to preserve current wins while reducing architectural ambiguity, configuration fragility, and technical debt.

It is intentionally **staged**, because the repository already contains working, high-value paths that must not be destabilized.

---

## Core Principles

1. **Do not break the current Rule-First / exact-runtime path**
2. **Do not rewrite blindly**
3. **Preserve parity and truth engines**
4. **Reduce ambiguity before increasing sophistication**
5. **Favor manifest-driven, auditable behavior**
6. **Treat PPO as legacy / research-only unless explicitly re-justified**
7. **No live-money execution without explicit production gates**

---

# Phase P0 — Safe Immediate Cleanup

## Goal

Remove obvious clutter and ambiguity from the repository without touching working logic.

## Scope

* Duplicate root-level tool scripts
* Temporary logs and junk files
* Legacy folder removal from active scope
* Deprecation marker for obsolete environment path

## Actions

* Remove duplicate utility scripts from the repository root when canonical versions already exist under `tools/`
* Remove temporary `tmp_*` logs and similar clutter
* Archive or remove `legacy/` from the active working surface
* Mark `trading_env.py` as deprecated and explicitly point to `runtime_gym_env.py`

## Deliverables

* Cleaner repository root
* Lower operator confusion
* Explicit signal about canonical environment paths

## Success Criteria

* No impact on exact-runtime replay
* No impact on RC1 generation / verification
* No impact on Rule-First logic or shadow-mode preparation

## Status

**Completed**

---

# Phase P1 — Configuration Stabilization

## Goal

Replace scattered raw environment-variable handling with a typed, centralized configuration façade while preserving existing launch behavior.

## Why This Matters

The current training / RL shell still depends on a large number of `os.environ.get(...)` calls spread across the codebase. This creates drift risk, hidden defaults, and startup ambiguity.

## Scope

* Typed `TrainingConfig` or `TrainConfig`
* Centralized `from_env()` loader
* Startup materialized config logging
* Run-local config truth snapshot

## Actions

* Create a typed config module, for example:

  * `train_config.py`
  * or `config/training.py`
* Introduce:

  * `TrainingConfig.from_env()`
* Move critical training parameters into the config object:

  * reward mode
  * alpha gate
  * horizon
  * worker count
  * fail-fast toggles
  * evaluation settings
  * baseline requirements
  * RL-specific switches
* Log the fully materialized config at startup
* Persist a run-local config snapshot for future audits

## Deliverables

* Single typed source for training config
* Stronger startup truth visibility
* Reduced config drift risk
* Easier debugging and reproducibility

## Success Criteria

* Existing launch scripts still work
* Training behavior does not change unintentionally
* Critical config no longer depends on scattered inline env reads
* Run startup tells us exactly what was active

## Status

**Completed**

---

# Phase P2 — Controlled Modularization of `train_agent.py`

## Goal

Reduce the size, ambiguity, and blast radius of the RL training shell without changing behavior.

## Why This Matters

`train_agent.py` remains a God Object that mixes:

* env parsing
* hyperparameter mapping
* curriculum logic
* callbacks
* diagnostics
* fail-fast logic
* training orchestration

This is high-risk technical debt.

## Scope

Behavior-preserving extraction into focused modules.

## Suggested Module Split

* `train_config.py`
* `train_callbacks.py`
* `train_curriculum.py`
* `train_diagnostics.py`
* `train_failfast.py`
* `train_bootstrap.py`
* leaner `train_agent.py` orchestrator

## Actions

* Extract one concern at a time
* Avoid semantic changes during extraction
* Keep all current startup / artifact behavior intact
* Add smoke tests after each extraction

## Deliverables

* Smaller, easier-to-reason-about training entrypoint
* Lower refactor risk in future work
* Clearer ownership boundaries

## Success Criteria

* Same training behavior
* Same artifacts
* Same startup truth snapshot
* Same exact-runtime compatibility

## Status

**After P1**

---

# Phase P3 — Environment Consolidation

## Goal

Eliminate ambiguity between old and current RL environment paths.

## Why This Matters

The coexistence of `trading_env.py` and `runtime_gym_env.py` increases confusion and creates hidden migration debt.

## Scope

* Reference tracing
* Import audit
* Test audit
* Safe deprecation / unlinking plan

## Actions

* Audit all references to `trading_env.py`
* Audit all branches inside `train_agent.py` and related modules
* Confirm whether `runtime_gym_env.py` is the sole canonical RL Gym wrapper
* Remove or isolate obsolete references when proven safe
* Keep deprecation markers until full unlink is verified

## Deliverables

* One clear canonical RL environment path
* Fewer hidden legacy branches
* Lower maintenance confusion

## Success Criteria

* No active train/replay path relies on deprecated env code
* Tests and entrypoints are aligned with the canonical environment

## Status

**After P1, or in parallel with late P2**

---

# Phase P4 — Repository and Tooling Normalization

## Goal

Make the repository easier to navigate, operate, and maintain.

## Scope

* Root folder cleanup
* Tool canonicalization
* Documentation cleanup
* Removal of stale task / walkthrough fragments

## Actions

* Enforce a rule that tool scripts live under `tools/`
* Keep only true core entrypoints at repository root
* Clean stale documentation
* Update:

  * `docs/NEXT_AGENT_CONTEXT.md`
  * `docs/NEXT_AGENT_FILE_MAP.md`
  * RC1 status notes
  * Rule-First canonical architecture notes
* Normalize artifact and report placement

## Deliverables

* Less chaotic repository structure
* Lower onboarding and operator confusion
* Better doc/code alignment

## Success Criteria

* Root no longer acts as a dumping ground
* Canonical tools are obvious
* Docs reflect actual architecture

## Status

**After or alongside P2/P3**

---

# Phase P5 — Rule-First / RC1 Hardening

## Goal

Strengthen the current manifest-driven Rule-First architecture as the primary working path.

## Scope

* Stronger manifest contract
* Better audit logging
* Baseline scoreboard automation
* Parity hash enforcement
* RC artifact standardization

## Actions

* Continue strengthening `selector_manifest.py`
* Expand `rule_selector.py` audit output
* Generate both:

  * human-readable baseline scoreboard
  * machine-readable baseline scoreboard
* Include in each RC pack:

  * manifest
  * parity report
  * baseline scoreboard
  * release notes
  * evaluator and logic hashes
  * dataset fingerprint
* Maintain exact-runtime parity discipline

## Deliverables

* Stronger RC packs
* Better drift resistance
* Better comparability across releases

## Success Criteria

* Every RC pack fully explains:

  * what logic ran
  * on what data
  * with what costs
  * against which baselines
  * with which evaluator version

## Status

**Largely completed, may continue incrementally**

---

# Phase P6 — Shadow Mode Operationalization

## Goal

Move RC candidates into a true shadow-mode operating loop with live-like decision logging but no live-money execution.

## Scope

* `shadow_broker.py`
* MT5 shadow integration
* structured no-trade / would-trade audit logging
* daily summaries

## Actions

* Run selector decisions through shadow mode only
* Log:

  * signal direction
  * no-trade reason
  * spread
  * session filter result
  * risk filter result
  * would_open
  * would_close
  * would_hold
  * active position state
  * manifest fingerprint
* Ensure the shadow path consumes the same manifest-driven rule logic

## Deliverables

* Paper-live candidate operating loop
* Actionable live-behavior evidence
* Live decision audit trail

## Success Criteria

* No real-money execution
* Full decision transparency
* Manifest-driven behavior confirmed
* Live decision flow can be reviewed after the fact

## Status

**Next production-like milestone**

---

# Phase P7 — Paper-Live Validation Window

## Goal

Observe whether RC1 behaves sensibly over real-time shadow operation, beyond replay evidence.

## Scope

* Multi-day or multi-week shadow run
* Daily / weekly audit summaries
* Replay-vs-live drift review

## Actions

* Record shadow decision streams
* Review:

  * signal density
  * no-trade reasons
  * spread conditions
  * session behavior
  * drift between replay assumptions and shadow reality
* Compare shadow activity to expected replay behavior

## Deliverables

* First real operational confidence layer
* Evidence of whether the rule behaves sensibly outside replay

## Success Criteria

* No chaotic or pathological live behavior
* No-trade audit reasons are meaningful
* Spread/session guards matter in practice
* Shadow behavior broadly matches replay expectations

## Status

**After P6**

---

# Phase P8 — Challenger Enhancements

## Goal

Only after RC1 shadow stability is established, test improvements around the Rule-First anchor.

## Possible Directions

* ML meta-filter challenger
* Pair-specific threshold optimization
* Regime-aware filters
* Dynamic spread-aware gating
* Exit refinement
* USDJPY challenger work

## Important Rule

All challengers must be measured against the raw Rule-First anchor, not against a hypothetical PPO comeback.

## Deliverables

* Clearly isolated challenger experiments
* Honest comparison vs raw rule baseline

## Success Criteria

* No challenger is accepted unless it improves exact-runtime outcomes honestly
* No return to opaque overfit-heavy complexity without proof

## Status

**Later / optional**

---

# Phase P9 — RL Legacy Containment

## Goal

Officially reduce the legacy PPO / RL system to a contained, clearly labeled research-only layer.

## Scope

* Labeling
* docs
* optional directory moves later
* reduction of accidental strategic ambiguity

## Actions

* Mark RL training as:

  * legacy
  * research-only
  * not default signal-learning path
* Update docs and entrypoints accordingly
* Consider later relocation into:

  * `research/rl/`
  * or `legacy_rl/`

## Deliverables

* Clear repo narrative
* Lower risk of unintentional strategic backsliding

## Success Criteria

* New contributors understand the primary path immediately
* RL is no longer accidentally treated as the default strategy engine

## Status

**Medium-term**

---

# Phase P10 — Production Readiness Gate

## Goal

Ensure that nothing reaches live-money execution without explicit, enforced approval.

## Scope

* hard production gate
* release-stage enforcement
* manifest safety checks

## Actions

* Enforce:

  * `live_trading_approved == true`
  * before any real-money order path is allowed
* Require:

  * parity evidence
  * baseline scoreboard
  * shadow-mode evidence
  * reviewed release stage
* Reject ambiguous manifests

## Deliverables

* Strong live-trading safety barrier
* Lower operational risk

## Success Criteria

* Impossible to accidentally trade live with a paper-live artifact
* Clear signoff path before live-money approval

## Status

**Mandatory before any live-money path**

---

# Recommended Execution Order

## Completed or largely completed

* P0 — Safe Immediate Cleanup
* P5 — Rule-First / RC1 Hardening (partially)
* Rule-First pivot
* RC1 generation
* Cross-pair rule gauntlet foundation

## Recommended next order

1. **P1 — Configuration Stabilization**
2. **P6 — Shadow Mode Operationalization**
3. **P7 — Paper-Live Validation Window**
4. **P3 — Environment Consolidation**
5. **P2 — Controlled `train_agent.py` modularization**
6. **P4 — Repository and Tooling Normalization**
7. **P8 — Challenger Enhancements**
8. **P9 — RL Legacy Containment**
9. **P10 — Production Readiness Gate**

---

# Final Summary

The repository is no longer primarily an RL research repo.

It is now evolving into a:

* **Rule-First**
* **Exact-runtime**
* **Manifest-driven**
* **Paper-live hardened**
* **Home-runnable**
  trading system, with legacy RL infrastructure still present but no longer strategically central.

The most important phases now are:

* **P1** — config truth
* **P6** — shadow mode
* **P7** — paper-live validation

These three phases will determine whether the current Rule-First path is only replay-positive, or actually operationally credible.
