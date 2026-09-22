# Task 0016 — Consistent iteration budgets across the fourteen model fits

## Metadata
- Task ID: 0016
- Title: Consistent iteration budgets across the fourteen model fits
- Requested by: Adam Godel
- Owner: Adam Godel
- Date: 2026-09-22
- Status: Done — changes 1 and 2 applied and verified; change 3 tested and reverted; `kMaxSeriesTerms` measured and rejected. Two fits remain poorly converged, cause unidentified.
- Estimated effort: Small (three constants) + one long verification run
- Related documents: [tasks/0015](0015_unify-duplicated-playground-operators.md) (where the
  non-convergence was found), [tasks/0009](0009_mpc-analysis-pipeline.md) (the fourteen fits),
  [tasks/0014](0014_production-range-convergence-test.md) (convergence over production ranges)
- Branch: `logistic-regression`

## Problem Statement

Seven of the fourteen fits stop because they run out of iterations, not because they converge.
All six 5a/5b fits exhaust `kGlmmBfgsIterations = 12` with gradient norms between **96 and 190** —
truncated mid-descent, not fitted. Their reported coefficients are wherever BFGS happened to be at
iteration 12.

This was found during task 0015 and is **not** caused by it: the same six fail identically in the
pre-0015 baseline. 0015 is where it became visible, because that task built the first
before/after comparison of all 75 `RESULT` lines.

### Why the models disagree with each other

| family | BFGS dim (p+1) | cap | cap / dim | outcome |
| --- | --- | --- | --- | --- |
| 2a / 2b | 3 | 12 | **4.0x** | converge at 9-12 |
| 5a / 5b | 7 (5b UMass: 6) | 12 | **1.7x** | all six hit the cap, \\|grad\\| 96-190 |
| 6a / 6b | n/a (IRLS) | 8 trips | fixed | fine |

One flat cap applied to models of different size. BFGS starts from an identity inverse-Hessian and
applies one rank-2 update per iteration, so it needs roughly `dim` iterations before it has any
usable curvature at all. At 12 the 3-parameter models got four times their dimension; the
7-parameter models got less than twice, leaving about five iterations of real quasi-Newton
behaviour on designs conditioned at 1e4.

**The inconsistency is the fixed cap meeting models of different dimension.**

## Goals
- Every fit stops for a reason other than the iteration cap.
- The iteration budget is consistent *relative to what each model needs*, not equal in absolute
  terms.
- No change outside iteration counts.

## Non-Goals
- Anything touching series terms, precision, or the number format. `kMaxSeriesTerms` remains open
  from 0015 and is explicitly not part of this task.
- Fixing the near-collinearity of the 5a/5b designs. Raising the budget lets BFGS reach the
  optimum; it does not make the optimum well determined, and the "close to singular" note stays.
- Reworking the optimiser. `MinimizeBFGS`'s line search has no iteration count to raise -- it
  halves `alpha` until it drops below `1e-5`, which is a floor, not a trip count.

## Changes

| # | constant | location | from | to | status |
| --- | --- | --- | --- | --- | --- |
| 1 | `kGlmmBfgsIterations` | `library/regression.h` | 12 | **60** | applied |
| 2 | `kFlatConditionalModeNewtonIterations` | `library/regression.h` | 3 | **5** | applied |
| 3 | `kIrlsIterations` | `library/regression.h` | 8 | 12 | **tested, reverted** |

### 1 — BFGS budget 12 -> 60

60 is ~8.6x dim for 5a/5b and 20x for 2a/2b. **Raising a cap costs nothing where it does not
bind:** 2a/2b exit early on a failed line search at 9-12 iterations and are unaffected, so only the
six fits that were being truncated get slower.

### 2 — inner conditional-mode Newton 3 -> 5

Not optional if change 1 is made. The Laplace objective **and its analytic gradient** are both
evaluated at the conditional mode this inner loop finds. An under-solved mode biases the function
BFGS is minimising, so spending 60 iterations on it would converge to a stationary point of the
wrong objective and report success — strictly worse than the honest non-convergence it replaces,
because the output would look trustworthy.

Five is what `library/regression.h`'s own `mixedeffects::ConditionalModeBatched` uses, via
`kNewtonIterations`, for the identical solve.

### 3 — IRLS trips 8 -> 12: REVERTED, no effect

Tested with the per-model runner on both 6a and 6b:

```
./mpc-analysis -S models -N 6a -D runs-verify/ctrl -ra 549 -rb 568
```

Built two binaries differing only in `kIrlsIterations` and diffed their whole output. The **only**
difference in either model is the line reporting the iteration count:

```
15c15
<   iters    8 (converged)
---
>   iters    12 (converged)
```

Every coefficient, standard error, z statistic, p-value, odds ratio and confidence interval is
bit-identical. IRLS has fully converged by trip 8, so the extra four are pure cost. **Reverted to
8**, per the instruction not to keep a change with no real effect.

## Deliberately not changed, with reasons

| constant | value | why not |
| --- | --- | --- |
| `kRecipNewtonSteps` | 3 | Reaches **1.4e-10** relative error; one ULP at precision 16 is 1.5e-5. No headroom. |
| `kRsqrtNewtonSteps` | 2 | Reaches **1.1e-6**. Same. The header documents both figures. |
| `kRecipUnitSteps`, `kRecipLogSteps` | 3, 2 | **Now dead.** They served root's `Sigmoid`/`Log`, which task 0015's merge replaced with the library's. Removable as cleanup. |
| `kMatrixInverseIterations` | 14 | `library/`-side; not on the pipeline path since 0015 reverted `logistic::Covariance`. |
| `kLineSearchSteps` | 17 | `library/`'s batched line search; the pipeline uses the `std::vector<AV>` `MinimizeBFGS`. |
| `kNewtonIterations` | 5 | `library/`'s own conditional-mode solver, not the pipeline's. |
| `kMaxSeriesTerms` | 3 | A series-term count, not an iteration count. Open from 0015. |

The first two are the useful negative result: **the primitive-level Newton iterations are already
converged below the number format's resolution**, so there is nothing to gain by raising them. All
the available headroom is in the two optimiser counts.

## Acceptance test

**No fit stops because of the iteration cap.** Today 7 of 14 do. Secondary: 6a/6b must be unchanged,
since neither change 1 nor change 2 touches IRLS.

If 5a/5b still reach 60, raise again — 60 is a measurement point, not a proven value.

## Method note

A per-model runner already exists and was used for change 3: `-N 2a|2b|5a|5b|6a|6b[:population]`
runs one fit instead of all fourteen, and its output carries a fuller table than the aggregate run
(z, Pr>|z|, odds ratio, 95% CI). Its coefficients match the full pipeline's exactly. Testing a
single-model change does not need an 18-minute whole-pipeline run.

**A correction worth recording:** the first attempt at the change-3 comparison diffed
`grep "^RESULT"` of the two outputs and reported "identical". The `-N` runner emits a formatted
table and **no `RESULT` lines at all**, so that diff compared two empty files. The conclusion
happened to be right, the evidence was worthless. The verified comparison diffs the whole output.

## How to read `|grad|`, and what to track instead

### What the number is

`MinimizeBFGS` prints `max_grad = max_i |g_i|` -- the **infinity norm** of the gradient, not the
L2 norm:

```cpp
max_grad = std::max(max_grad, std::abs(static_cast<double>(opened_g[i]) / scale));
```

Units: the objective is the negative log-likelihood in **nats**, and the parameters are log-odds
coefficients plus `s = log(sigma)`, all dimensionless. So `|grad|` is **nats per unit change in the
worst-behaved coefficient**.

### Why it is a poor accuracy metric on its own

It is scale-dependent in two ways that break comparison:

1. **It scales with n.** The objective is a sum over rows, so a 1117-row fit carries larger
   gradients than a 493-row fit at the same statistical quality.
2. **It depends on the objective being differentiated.** Change 2 altered the conditional-mode
   solve, so the before and after gradients in this task are gradients of *different functions*.
   That is why 2b's numbers got 34-62x "worse" while its estimates barely moved.

### What to track: coefficient movement in standard-error units

`|delta beta| / SE(beta)` is dimensionless, comparable across models, and is the scale on which a
coefficient is statistically meaningful -- a shift of 0.5 SE moves a z statistic by 0.5.

Measured across this task's before/after, worst term per fit:

| model | \|grad\| before | \|grad\| after | **max \|d est\| / SE** |
| --- | --- | --- | --- |
| 2a any | 3.58e+0 | 3.74e-1 | 0.061 |
| 2a non-UMass | 3.58e-1 | 1.98e-1 | 0.003 |
| 2a UMass | 1.50e+0 | 1.82e+0 | 0.004 |
| 2b any | 5.22e-2 | 1.80e+0 | 0.038 |
| 2b non-UMass | 2.21e-1 | 1.69e-1 | 0.006 |
| 2b UMass | 1.67e-2 | 1.03e+0 | 0.055 |
| **5a any** | 1.41e+2 | 7.48e+1 | **0.306** |
| **5a non-UMass** | 1.64e+2 | 1.08e+1 | **0.742** |
| **5a UMass** | 9.65e+1 | 7.95e+0 | **0.536** |
| **5b any** | 1.14e+2 | 5.19e+1 | **0.707** |
| **5b non-UMass** | 1.48e+2 | 2.26e+0 | 0.328 |
| **5b UMass** | 1.90e+2 | 2.58e+0 | **0.907** |
| 6a / 6b | n/a | n/a | **0.000** (IRLS control) |

The split is clean. Every fit that was **already** below \|grad\| ~4 moved by **under 0.07 SE** --
statistically nothing, including the three whose gradient norms nominally regressed. Every fit that
had been truncated at \|grad\| ~100-190 moved by **0.31 to 0.91 SE** -- up to nearly a full standard
error, which is inference-changing. 6a/6b moved exactly 0.000, confirming the control.

### Suggested thresholds

| \|d est\| / SE | meaning |
| --- | --- |
| < 0.01 | converged for any purpose |
| < 0.10 | fine for reporting |
| > 0.25 | changes inference; the fit is not done |

### Consequence for the two stragglers

`5a [all systems]` and `5b [all systems]` still sit at \|grad\| 74.8 and 51.9. Everything else in
that gradient range moved 0.3-0.9 SE when it was allowed to converge further, so **their current
estimates should be assumed to carry a few tenths of a standard error of optimisation error** --
on top of the "close to singular" warning they already print. They are the two fits whose
coefficients should not be quoted without a caveat.

### Making this cheap to check

The per-model runner does it in minutes rather than 40:

```
./mpc-analysis -S models -N 5a -D <data> -ra 549 -rb 568
```

Run the fit at two iteration budgets, diff the estimates, divide by the printed SE. The `-N` output
prints Estimate, StdErr, z and Pr>|z| per term directly, so no extra tooling is needed.

A normalised criterion -- the usual relative-gradient form `max_i |g_i * beta_i| / |f|` -- would be
comparable across models and is not currently printed. Adding it would make the BFGS trace
self-interpreting; that is a small change to the optimiser, not part of this task.

## Risks and Mitigations
- **Risk:** 60 iterations still is not enough for 5a/5b.
  - Mitigation: the acceptance test names the cap explicitly; raise and re-measure.
- **Risk:** the fits converge but to a badly determined optimum, and the standard errors are still
  not trustworthy.
  - Mitigation: this is expected and already reported — the observed-information condition
    estimate and the "close to singular" note are printed per fit. Convergence is necessary, not
    sufficient.
- **Risk:** runtime grows beyond what is practical.
  - Mitigation: only the six binding fits scale with the cap. Measured below.

## Results

Full pipeline, blinky, `PROTOCOL=1`, `ctrl` dataset, `-ra 549 -rb 568`. 75 `RESULT` lines.
Comparison point is `runs-verify/S3-models.txt`, task 0015's verified final state.

### Acceptance test: PASSED

**No mixed fit stops at the iteration cap any more.** All twelve now stop on a failed line search.
(6a/6b show a fixed trip count of 8 by design -- IRLS has no cap to hit.)

| model | before: iters, \|grad\|, stop | after: iters, \|grad\|, stop |
| --- | --- | --- |
| 2a UMass | 11, 1.50e+0, line search | 11, 1.82e+0, line search |
| 2a all | 9, 3.58e+0, line search | 7, 3.74e-1, line search |
| 2a non-UMass | 12, 3.58e-1, **CAP** | 12, 1.98e-1, line search |
| 2b UMass | 10, 1.67e-2, line search | 10, 1.03e+0, line search |
| 2b all | 10, 5.22e-2, line search | 13, 1.80e+0, line search |
| 2b non-UMass | 12, 2.21e-1, line search | 13, 1.69e-1, line search |
| **5a UMass** | 12, **9.65e+1**, **CAP** | 21, **7.95e+0**, line search |
| **5a all** | 12, **1.41e+2**, **CAP** | 24, **7.48e+1**, line search |
| **5a non-UMass** | 12, **1.64e+2**, **CAP** | 33, **1.08e+1**, line search |
| **5b UMass** | 12, **1.90e+2**, **CAP** | 33, **2.58e+0**, line search |
| **5b all** | 12, **1.14e+2**, **CAP** | 34, **5.19e+1**, line search |
| **5b non-UMass** | 12, **1.48e+2**, **CAP** | 18, **2.26e+0**, line search |
| 6a all | 8, n/a, fixed | 8, n/a, fixed (unchanged) |
| 6b all | 8, n/a, fixed | 8, n/a, fixed (unchanged) |

Four of the six previously-truncated fits improve by **12x to 74x** and land at gradient norms of
2.3 to 10.8, in the same range as the 2a/2b family. The budget was the binding constraint for
those four, exactly as diagnosed.

6a/6b are untouched, which is the control: neither change touches IRLS.

### Two stragglers, and they are not an iteration problem

`5a [all systems]` (7.48e+1) and `5b [all systems]` (5.19e+1) improved only ~2x and are still an
order of magnitude worse than their own UMass and non-UMass siblings. Both are the largest scope of
their specification. They no longer run out of budget -- they stall because the line search cannot
find a descent direction, which is the signature of the objective's **fixed-point noise floor**,
not of conditioning or of too few iterations.

### Cost

| | time |
| --- | --- |
| task 0015 final (cap 12) | 18m47s |
| this task (cap 60, inner Newton 5) | **39m42s** |

2.11x, concentrated in the six fits that now actually use their budget. Against the pre-0015
baseline of 4m13.9s the pipeline is now 9.4x, of which 4.33x came from 0015's boolean division.

## Follow-on measurement: `kMaxSeriesTerms` 3 -> 5

Prompted by the two stragglers. Tested with the per-model runner, one variant binary, source
restored afterwards.

| scope | 3 terms | 5 terms | \|grad\| change |
| --- | --- | --- | --- |
| **5a all systems** | 24 iters, **7.48e+1** | 22 iters, **7.51e+0** | **10x better** |
| 5a UMass | 21 iters, 7.95e+0 | 33 iters, 1.04e+1 | 1.3x worse |
| 5a non-UMass | 33 iters, 1.08e+1 | 21 iters, 1.39e+0 | 7.8x better |

The pathological case is fixed: `5a [all systems]` drops into line with its siblings. This confirms
the noise-floor diagnosis above. The other two scopes move in opposite directions, so where the line
search gives up is partly stochastic once near the floor -- 5 terms is not uniformly better, it
removes a floor that was binding in one case.

**Coefficients, `5a [all systems]`:** `visit_num`, the estimand, moves **0.8%** (0.18044 ->
0.18190). The large movements are confined to weakly determined nuisance covariates --
`gender=female` 29%, `newage` 11% -- which are the small-magnitude terms sitting near the ULP floor
anyway. Standard errors move under 1% throughout. The 5-term fit is ~10x closer to a stationary
point, so where they differ it is the better estimate.

**Cost: none, in this case.** One `5a [all systems]` fit took **4m45s at 5 terms against 5m17s at
3** -- it needed 22 iterations instead of 24, and that saving outweighed the extra series work.
That will not hold where 5 terms needs *more* iterations (5a UMass went 21 -> 33).

### 5b tested next: the pattern did NOT hold

`5a [all systems]` improving 10x suggested `kMaxSeriesTerms` was a general fix for the stragglers.
Testing the other straggler refuted that:

| straggler | 3 terms | 5 terms | improvement |
| --- | --- | --- | --- |
| 5a [all systems] | 7.48e+1 | **7.51e+0** | **10x** |
| 5b [all systems] | 5.19e+1 | **2.99e+1** | **1.7x** |

`5b [all systems]` stays an order of magnitude worse than its own UMass (2.58) and non-UMass (2.26)
siblings. Whatever limits it is not the series truncation. **The two stragglers do not share a
cause**, despite both being the "all systems" scope of a covariate model.

Across the four fits measured: one large win, one modest, one small win, one small regression. That
is not a floor-removal that generalises.

### DECISION: not applied

`kMaxSeriesTerms` stays at 3. The evidence does not support a `library/` code change: it fixes one
fit out of two it was meant to fix, and is non-monotonic across scopes of the same model.

### It also resolves the 0015 "sign flip" -- in the opposite direction from what was predicted

Task 0015 reported `5b [all systems] gender=female` flipping from +0.0537 to -0.0129 and flagged it
as the headline accuracy concern. Tracking that coefficient as the fit becomes progressively better
converged:

| configuration | \|grad\| | `gender=female` |
| --- | --- | --- |
| pre-0015 baseline (cap 12) | 1.14e+2 **(CAP)** | **+0.05374** |
| 0015 final (cap 12) | 1.14e+2 **(CAP)** | -0.01294 |
| 0016 (cap 60, inner Newton 5) | 5.19e+1 | -0.06255 |
| 0016 + 5 series terms | 2.99e+1 | -0.06818 |

The estimate moves **monotonically** toward about -0.068 as the gradient falls from 114 to 30.
**The baseline's +0.0537 was the artefact**, read off a fit truncated at \|grad\| = 114. The
coefficient is genuinely negative, and 0015's framing of this as "a sign flip introduced by the
3-term series" was wrong: the series change moved the number, but the sign was never trustworthy in
the baseline. Better convergence settles it, not better arithmetic.

A prediction made mid-investigation -- that 5 series terms would "undo the sign flip" -- was also
wrong, and in the same direction: it pushes further negative, because it converges further.

## Is everything converging better? Not uniformly.

| model | before | after | ratio | |
| --- | --- | --- | --- | --- |
| 2a UMass | 1.501e+0 | 1.822e+0 | 1.21x | worse |
| 2a all | 3.581e+0 | 3.741e-1 | 0.10x | better |
| 2a non-UMass | 3.576e-1 | 1.978e-1 | 0.55x | better |
| **2b UMass** | **1.671e-2** | **1.029e+0** | **61.6x** | **worse** |
| **2b all** | **5.222e-2** | **1.797e+0** | **34.4x** | **worse** |
| 2b non-UMass | 2.211e-1 | 1.689e-1 | 0.76x | better |
| 5a UMass | 9.649e+1 | 7.945e+0 | 0.08x | better |
| 5a all | 1.410e+2 | 7.478e+1 | 0.53x | better |
| 5a non-UMass | 1.638e+2 | 1.080e+1 | 0.07x | better |
| 5b UMass | 1.897e+2 | 2.578e+0 | 0.01x | better |
| 5b all | 1.140e+2 | 5.192e+1 | 0.46x | better |
| 5b non-UMass | 1.483e+2 | 2.264e+0 | 0.02x | better |

Nine better, three worse. The two large regressions are the 2b family, which were previously the
**best** converged fits in the whole pipeline.

### Why, and why it does not matter in practice

Change 1 cannot cause it: 2b used 10 and 13 iterations, under both the old and new cap. It is
change 2. Raising the inner Newton count **changes the function BFGS is minimising** -- a more
accurately solved conditional mode, but reached through more MPC operations and therefore carrying
more accumulated fixed-point noise. The line search stalls earlier on the noisier surface.

So the before and after gradients are **gradients of different functions** and are not strictly
comparable. What is comparable is the estimates, and they barely move:

| fit | largest coefficient move |
| --- | --- |
| 2b UMass | 1.84% (`random_intercept_variance`) |
| 2b all | 0.44% (`fu_month`) |
| 2a UMass | 0.08% (`visit_num`) |

A 61x worse gradient norm that shifts the estimates by under 2% is a change in the objective's
noise floor, not a loss of accuracy. Against that, the six previously-truncated fits improved by
12x to 74x and their estimates moved by far more.

### Honest summary

- **Structurally fixed:** all twelve mixed fits now stop on a failed line search rather than the
  iteration cap. That was the acceptance test and it passed.
- **Materially fixed:** four of the six truncated fits now sit at \|grad\| 2.3-10.8, the same range
  as the well-behaved 2a/2b family.
- **Still broken:** `5a [all systems]` (74.8) and `5b [all systems]` (51.9) remain an order of
  magnitude worse than their siblings, for reasons that are not iteration count and, per the series
  test above, not a single shared cause either.
- **Nominally regressed, practically unchanged:** three 2b/2a fits report worse gradient norms on a
  noisier objective while their estimates move under 2%.

## Rollback and Recovery
- Rollback: three constants in one file; revert the commit.
- The pre-change values are 12, 3 and 8.
