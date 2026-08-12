# Robust Conditional-Mode Newton and BFGS Line Search

## How to Use
Design document for the task. See Approval for sign-off status.

## Metadata
- Task ID: 0003
- Title: Stabilize the inner conditional-mode Newton solver and the outer BFGS
  line search in `playground/logistic-regression.cpp` so trial parameter vectors
  with a large `sigma` cannot drive the iterates to overflow / NaN.
- Requested by: User
- Owner: Copilot
- Date: 2026-08-12
- Status: Done
- Estimated effort: 1 implementation pass + revalidation
- Related documents:
  - tasks/0001_mixed-effects-logistic-regression-laplace.md
  - tasks/0002_taylor-series-math-approximations.md
- Branch: logistic-regression

## Problem Statement
- What is the request? During the BFGS fit the inner `ConditionalMode` Newton
  iteration explodes:
  ```text
  ConditionalModes iter 1: u = 0.000e+00
  ConditionalModes iter 2: u = 4.142e+05
  ConditionalModes iter 3: u = -2.455e+123
  ConditionalModes iter 4: u = 1.929e+123   (then alternates / diverges)
  ```
  The failure reproduces with both the custom Taylor math and the standard
  library, so it is an optimizer-robustness problem, not an approximation bug.
- Why now? The estimator cannot fit (assertions never reached) because a single
  bad trial parameter set poisons the objective and the optimizer never recovers.

## Root Cause
The outer BFGS line search probes a trial parameter vector with a large `s`
(hence large `sigma`). With a nearly flat Gaussian prior (`1/sigma2 -> ~0`) and
(quasi-)separable per-group binary data, the conditional mode `u*` is genuinely
far from 0. The **undamped** Newton step (`u += gradient / curvature`) overshoots
into the region where every `p` saturates to 0/1, `p(1-p)` underflows to 0, the
curvature collapses to the tiny `1/sigma2`, and successive steps diverge and
oscillate to `+/-1e123` and then to non-finite values. The outer line search then
both evaluates and (on stall) *accepts* these poisoned objective values.

## Goals
- Goal 1: Damp/bound the inner Newton step so it cannot launch `u` to overflow.
- Goal 2: Guard the iterate and the objective against non-finite values so bad
  trials produce `+inf` (rejected by the minimizer) instead of NaN propagation.
- Goal 3: Make the outer line search reject non-finite / stalled trials instead
  of accepting them.
- Goal 4: Keep the recovered estimates correct (assertions pass) on the default
  standard-library path.

## Non-Goals
- Non-goal 1: Changing the statistical model or the Laplace/BFGS approach.
- Non-goal 2: Reworking the Taylor approximations (task 0002).
- Non-goal 3: Adding bounds/reparameterization of `sigma` in the model.

## Scope
- In scope: `ConditionalMode`, `GroupLaplaceLogLik`/`NegMarginalLogLik` finiteness
  guarding, the `MinimizeBFGS` line search, and removal of the temporary debug
  prints that were added while diagnosing this issue (they flood output and make
  validation impractical).
- Out of scope: other files, other math functions, CMake/build changes.

## Alternatives
### Option A: Damp/cap the inner Newton step (recommended)
Pros:
- Minimal, local change; preserves fast Newton convergence for normal steps.
- Bounds per-iteration movement so overflow is impossible.
Cons:
- A fixed cap is a heuristic; extreme legitimate modes need more iterations.

### Option B: Inner backtracking line search on the conditional log-likelihood
Pros:
- More principled; guarantees monotone improvement of the inner objective.
Cons:
- More code and evaluations for little practical gain here.

### Recommendation
Option A (step cap) plus finiteness guards, and reject-on-stall in the outer line
search. This is the smallest change that removes the divergence while keeping the
good-case behavior intact.

## Implementation Plan
1. `ConditionalMode`: compute the Newton step, clamp its magnitude to a fixed cap,
   apply it, and break out if the iterate becomes non-finite.
2. `NegMarginalLogLik`: return `+inf` when the accumulated objective is not finite
   so the minimizer rejects the trial.
3. `MinimizeBFGS` line search: accept a step only when `fx_new` is finite and
   satisfies Armijo; on stall, restore the current point (reject) instead of
   accepting the poisoned trial.
4. Remove the temporary per-iteration / per-observation debug prints added during
   diagnosis; keep the concise `[BFGS] iter` summary line.
5. Rebuild and run (default std path); confirm the fit converges and assertions
   pass.

## Risks and Mitigations
- Risk: Step cap too small slows legitimate convergence.
  - Mitigation: Cap only bounds per-iteration movement; 100 inner iterations still
    allow reaching moderate modes; normal steps are well under the cap.
- Risk: Reject-on-stall could stop the outer optimizer prematurely.
  - Mitigation: With finite objectives restored by the step cap, the line search
    finds a decreasing step, so stall-rejection is a safety net, not the norm.

## Validation Plan
- Build with the default (std) path and run the demo.
- Expected: no `1e123`/NaN iterates; BFGS converges; all `main` assertions pass;
  estimates close to ground truth.
- Spot-check the Taylor path still builds.

## Definition of Done
- Inner Newton step damped + finiteness guard; objective returns `+inf` on
  non-finite; line search rejects non-finite/stalled trials.
- Debug scaffolding removed.
- Program builds and passes assertions on the default path.

## Approval
- [x] Task document reviewed
- [x] Approved to implement
- Approver: User
- Approval date: 2026-08-12

## Change Log
- 2026-08-12: Initial draft created and approved (user requested doc + apply).
- 2026-08-12: Implemented step cap + finiteness guard in `ConditionalMode`, `+inf`
  rejection in `NegMarginalLogLik`, finite/Armijo-only acceptance with
  reject-on-stall in the `MinimizeBFGS` line search, and removed the temporary
  debug prints. Validated on the default std path: BFGS converges in 28 iterations
  to marginal log-likelihood -3246.8579, no non-finite iterates, all `main`
  assertions pass, estimates close to ground truth. Status -> Done.
