# Goals

Use this file to describe what you want to achieve in this repository.

## Problem Statement

As implemented today, the optimizer uses a finite-horizon dynamic-programming formulation whose value tables are indexed by:

- fraction / time-to-go
- accumulated delivered dose
- current overlap volume

In that sense, the implemented DP is effectively 3-dimensional. If time is treated separately from the Markov state, then the decision state is effectively 2-dimensional: accumulated dose plus current observed overlap.

The overlap-distribution parameters are not explicit parts of the state. At the start of each call to `adaptive_fractionation_core(...)`, the code derives:

- `mean` from `volumes.mean()`
- `std` from `std_calc(volumes, alpha, beta)`

Those values influence the transition model for future overlap, but they remain fixed during the backward recursion and are not represented as state coordinates in the value or policy tables.

The goal is to expand the state representation so the policy can depend explicitly on the current distribution of the sparing factor / overlap process, starting with:

- overlap mean
- overlap standard deviation

With that change, the finite-horizon formulation would move from roughly `(time, accumulated_dose, overlap)` to `(time, accumulated_dose, overlap, mean, std)`. The practical motivation is to let the optimizer distinguish between patients or treatment stages that have the same current overlap but different expected future behavior or uncertainty.

This work will likely require changes to:

- state discretization
- transition modeling
- value/policy storage
- computational performance strategy
- tests and documentation

## Constraints

The human developer should remain continuously aware of what code is changing, why it is changing, and how the new formulation differs from the current one. The work should increase the developer's understanding of the codebase, not reduce it behind opaque refactors or hidden abstractions.

In practice, that means:

- changes should be incremental and reviewable rather than large, hard-to-follow rewrites
- algorithmic changes should be explained in plain terms, especially changes to state definition, transition logic, penalty calculation, and DP table shape
- documentation should be updated alongside the code so the developer can trace how inputs, state variables, and outputs relate
- tests should make the new behavior legible, not just verify it mechanically
- new abstractions are acceptable only if they make the MDP easier to reason about; they should not hide the core optimization flow
- at any point in the work, the developer should be able to answer:
  - what the state is
  - what the action is
  - what the transition model depends on
  - what reward or penalty is being optimized
  - how the proposed extension changes runtime, memory, and behavior

## Success Criteria

The change in behavior should be observable, testable, and explainable. It is not enough to introduce extra state dimensions; we should be able to show that the new formulation changes decisions in cases where it ought to, and preserves behavior in cases where it should not.

Success means:

- the new behavior is covered by unit tests that isolate the effect of adding `mean` and `std` as explicit state variables
- there are comparison tests between the current implementation and the new one, so behavior changes are measured rather than guessed
  - this comparison does not require the production code to support both formulations forever
  - one acceptable strategy is to capture baseline outputs or fixtures from the current implementation before the refactor
  - those baselines can then be used as regression references in tests, so the old behavior remains inspectable even if the runtime code no longer exposes the old lower-dimensional solver
- before implementation or during design, we state a concrete prediction about what the richer state should change
  - for example, two scenarios with the same current overlap and accumulated dose but different estimated future mean or uncertainty should be allowed to produce different recommended actions
- after implementation, tests or analysis confirm that this predicted impact actually appears in controlled examples
- the effect of the new state variables is understandable from the outputs
  - policy differences should be traceable to differences in mean/std, not to unrelated code changes
- the previous behavior remains reproducible when the richer state collapses to the old assumptions or when the new feature is disabled, if backward compatibility is part of the chosen design
- runtime and memory costs are measured well enough to understand the practical cost of the higher-dimensional state
- the updated documentation explains:
  - what changed in the state definition
  - why the new state should improve decision quality
  - which tests demonstrate that improvement or behavioral difference


## Notes

Add any background, references, or examples here.
