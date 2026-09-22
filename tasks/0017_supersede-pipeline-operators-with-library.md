# Task 0017 — Supersede the pipeline's operators with the library's

## Metadata
- Task ID: 0017
- Title: Supersede the pipeline's operators with the library's (delete `MinimizeBFGS`)
- Requested by: Adam Godel
- Owner: Adam Godel
- Date: 2026-09-22
- Status: In Progress — Phase 0 measured: the secure inverse is precision-limited, so Phases 4/5 need a decision. Phases 1-3, 6-7 unaffected.
- Related documents: [tasks/0015](0015_unify-duplicated-playground-operators.md) (the merge),
  [tasks/0016](0016_consistent-iteration-budgets.md) (iteration budgets),
  [tasks/0009](0009_mpc-analysis-pipeline.md) (the fourteen fits),
  [tasks/0010](0010_secure-matrix-inversion-newton-schulz.md) (the inverse)
- Branch: `logistic-regression`
- Plan file: `~/.claude/plans/closely-examine-the-current-joyful-reddy.md` (approved)

## Problem Statement

Task 0015 unified `playground/` into one copy of every operator under `playground/library/`, but it
unified **names, not capability**. At HEAD:

- **31 library-origin operators are reached by zero pipeline code.** 22 call only each other; the
  other 9 are roots used only by the three LR drivers and the four test binaries.
- `MinimizeBFGS` has **exactly one caller in the repo** — `FitGlmmLaplace`
  (`library/regression.h:1712`). Its gradient-free overload (`optimizer.h:1326`) has none.
- No file anywhere mixes the two families.

`MinimizeBFGSBatched` sits in the same file as `MinimizeBFGS` and the pipeline cannot reach it.

## Decisions
1. Full scope: optimiser, inference, and the division/square-root family.
2. Correctness first; cost measured and reported afterwards.
3. 6a/6b keep the IRLS *fit* (only estimator leaking nothing beyond its outputs); inference moves.
4. Mixed-model inference superseded despite a measured accuracy regression.
5. `NewtonSchulzInverse` fixed first — it is on the critical path for 4 and 5.

## Key findings

- **`MinimizeBFGSBatched` is data-layout agnostic.** `num_points` counts *candidate parameter
  vectors*, not data rows. So the flat ragged objective can drive it unchanged, and the balanced
  `BatchedDataset` 3.1x padding that blocked task 0015 is irrelevant here.
- **The analytic gradient survives** — `BatchedGradient` is a single-point analytic gradient that
  `MinimizeBFGSBatched` prefers over numerical differencing.
- **It leaks far less**: one bit per iteration, versus the full gradient vector, the full step
  vector, the directional derivative and one `fx_new` per backtracking trial. This closes the leak
  recorded as a known follow-up in task 0009.
- **The only blocker to a batched ragged objective** is `SegScanPlanned` (`segmented.h:251`),
  which addresses levels with a single arithmetic progression and needs to become block-aware.

The full approach, phase by phase, is in the approved plan file.

## Phase 0 findings — the premise was wrong, corrected here

The plan proposed "add a scaled initial guess" to `NewtonSchulzInverse`. **It already has one.**
`optimizer.h:171-173` computes `X_0 = A^T / ||A||_F^2`, and the comment correctly notes this puts
any nonsingular A inside the convergence basin, since `sigma_max(A) <= ||A||_F`.

The defect is the **rate**, not the basin. With that initialiser the error is

```
||E_0|| = 1 - sigma_min^2 / ||A||_F^2  ~  1 - 1/(kappa^2 * n)
```

and `E_{k+1} = E_k^2` merely *doubles* the gap `delta = 1/(kappa^2 n)` per step while `E` is near
1. Reaching `E ~ 0.5` therefore takes about `log2(kappa^2 * n)` iterations before quadratic
convergence even begins. At the pipeline's observed condition of ~1.2e4 that is
`log2(1.44e8 * 8) ~ 30`, plus a few more for precision — against `kMatrixInverseIterations = 14`.

`2^14 = 16384` against a required `~1.4e8`: the iteration is not close to converged, which matches
the garbage measured in task 0015 exactly.

**So Phase 0 is a calibration problem, not a scaling bug.** The repo already ships the instrument:
`kRunMatrixInverseCalibration` / `kMatrixInverseCalibrationMax` (`optimizer.h:140-141`) sweep the
iteration count and print accuracy at each. Measure against ill-conditioned operands, then set the
constant — and check whether the fixed-point noise floor at precision 16 caps the achievable
accuracy before the iteration count does.

## Phase 0 RESULT — the secure inverse cannot reach the required accuracy at precision 16

Measured with a calibration probe (scaffolding, kept at
`/scratch/adam/CryptDough-unify/playground/tests/probe_inverse.cpp`, deliberately not landed):
symmetric positive definite `n = 7` operands with a geometric spectrum normalised to
`lambda_max = 1`, both diagonal and dense (`Q D Q^T`). Entries are `max |A*X - I|`.

### Dense case (`Q D Q^T`)

| condition | 14 iters | 20 | 25 | 30 | 35 | 40 | 50 | 60 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1e1 | 8.4e-5 | 8.7e-5 | 9.2e-5 | 8.6e-5 | 8.7e-5 | 8.7e-5 | 1.0e-4 | 1.0e-4 |
| 1e2 | 1.6e-1 | 5.8e-4 | 6.0e-4 | 6.0e-4 | 5.7e-4 | 5.6e-4 | 5.9e-4 | 5.8e-4 |
| 1e3 | 6.9e-1 | 2.3e-1 | 5.1e-3 | 5.1e-3 | 5.1e-3 | 5.1e-3 | 5.1e-3 | 5.1e-3 |
| **1e4** | 8.6e-1 | 6.8e-1 | 4.2e-1 | **4.2e-2** | 4.2e-2 | 4.2e-2 | 4.2e-2 | 4.2e-2 |
| **2e4** | 9.0e-1 | 7.0e-1 | 5.5e-1 | **9.7e-2** | 1.0e-1 | 1.0e-1 | 1.0e-1 | 1.0e-1 |

### The residual PLATEAUS — more iterations do nothing

Every row is flat from ~30 iterations onward. Going 30 -> 60 changes the answer in the fourth
significant figure. The plateau height tracks the conditioning exactly:

| condition | measured floor | `kappa * ULP` (ULP = 1.5e-5) | ratio |
| --- | --- | --- | --- |
| 1e1 | 5.19e-5 | 1.50e-4 | 0.35 |
| 1e2 | 5.50e-4 | 1.50e-3 | 0.37 |
| 1e3 | 7.03e-3 | 1.50e-2 | 0.47 |
| 1e4 | 6.38e-2 | 1.50e-1 | 0.43 |
| 2e4 | 9.23e-2 | 3.00e-1 | 0.31 |

A constant ratio of **0.31 to 0.47 across five orders of magnitude**. The limit is the
**fixed-point resolution at precision 16**, not the iteration count: an inverse whose entries span
`kappa` cannot be represented to better than `~kappa * ULP` relative error, whatever algorithm
produces it.

### Consequence: Phases 4 and 5 are not achievable as planned

The pipeline's own fits report 1-norm condition estimates of **1.2e4 for 5a/5b and 6a**. At that
conditioning the secure inverse floors at **4-10% residual**, against the `< 1e-3` that a usable
standard error needs — three orders of magnitude short.

The existing pipeline path does not have this problem: `ObservedInformationSE` inverts by
**plaintext** Gauss-Jordan and `FitLogisticIrls` by a Cholesky solve whose result is opened, both
effectively double precision.

So the original diagnosis in task 0015 ("Newton-Schulz at 14 iterations does not converge") was
right about the symptom and wrong about the cause. It is not under-iterated; it is
under-represented. **Raising `kMatrixInverseIterations` cannot fix it, and neither can a better
initialiser** — the scaled initialiser `X_0 = A^T / ||A||_F^2` was already there and is correct.

Only models 2a/2b (condition 4.1e1 - 2.7e2) sit where the secure inverse is usable. Choosing the
inverse per model by conditioning would branch on secret-derived data, so it is not an option.

**Phases 1, 2, 3, 6 and 7 are unaffected** — none of them touches the matrix inverse.

## Progress

### Phase 0 — DONE
`kMatrixInverseIterations` 14 -> **30**, the knee of the measured curve. At 14 the ill-conditioned
rows return residual ~1 (no information at all); at 30 they reach the precision floor. The constant
now carries the measured table and an explicit warning that the plateau is a **representation**
limit, not an iteration limit, so a future reader does not try to fix it by raising the count
again.

Decision recorded: the 4-10% residual at the pipeline's conditioning is **accepted**, per the
instruction to proceed. It propagates into every standard error on 5a/5b/6a, on top of the
17-46% Hessian regression already accepted.

### Phase 1 — DONE
`SegScanPlanned` and `SegTotalPlanned` take a `blocks` parameter (default 1, so every existing
caller is untouched). The union-of-progressions problem is solved with an explicit
`BlockLevelMap` public view.

One implementation note worth keeping: the reverse direction could **not** be layered as a second
view, because `mapping_reference` asserts `!has_mapping()` and the accumulator was already a
mapping. The block-reversal is therefore folded into the index arithmetic
(`offset = reversed ? n-1-local : local`), which also removes one view from the forward path.

**Gate: PASS, bit-for-bit.** A probe comparing `blocks = 3` against three separate `blocks = 1`
calls, over four ragged groups (sizes 3/5/1/7) with distinct values per block so any cross-block
leak would show:

```
  SegScanPlanned Forward     blocks=3  worst |batched - per-block| = 0.000e+00   EXACT
  SegScanPlanned Reverse     blocks=3  worst |batched - per-block| = 0.000e+00   EXACT
  SegTotalPlanned            blocks=3  worst |batched - per-block| = 0.000e+00   EXACT
```

**Regression: clean.** All four test binaries pass (104 checks). Model 2a `[all systems]`
reproduces the T1 baseline exactly (Intercept -1.25937, visit_num 0.17538).

### Phase 2 — DONE
`FlatObjectiveBatched(md, params, num_points, gradient_out)` evaluates K candidate parameter
vectors in one call, plus helpers `TileRows`, `BatchedColumn`, `BatchedLinearPredictor`,
`FlatConditionalModeBatched`.

**Gate: PASS, exact on both claims.**

```
(1) num_points = 1 against FlatObjective
    value      ref=11.76358032  batched=11.76358032  diff=0.000e+00  EXACT
    gradient   worst |diff| = 0.000e+00  EXACT
(2) num_points = 4 against 4 separate calls
    worst |diff| = 0.000e+00  EXACT
```

The gradient matching exactly means the analytic derivation survived the rewrite intact.

**Two `mapping_reference` composition traps**, both now commented in place: it asserts
`!has_mapping()`, so views cannot nest. Fixed by folding the index arithmetic into one map (the
block reversal in `SegScanPlanned`) and by materialising with `Clone` (`beta_all`).

### Phase 3 — DONE: `MinimizeBFGS` deleted

`FitGlmmLaplace` calls `MinimizeBFGSBatched` with the analytic gradient supplied as
`BatchedGradient`, so the derivation is kept and `NumericalGradientBatched` never runs.

**Removed, 342 lines:** `MinimizeBFGS` (both overloads, 187), `BfgsInverseUpdate` (107),
`MatVec` (30), `Dot` (19), `struct OptResult` (13). The middle three served only `MinimizeBFGS`.
`ValueGradFn` and `NumericalGradient` survive for `ObservedInformationSE` and `harness.h`; they go
in Phase 4.

`x0` is now one packed `dim`-length vector, which also removes the one-element transcendental
circuits. The "no BFGS curvature update was accepted" note is gone, because `BatchedOptResult` has
no `hessian_updated` flag -- the update is multiplexed rather than branched, so it is no longer
plaintext-knowable. That is the leak closing, not information lost.

### Correctness: confirmed. Cost: a problem, and it exposed a defect.

Model 2a `[all systems]`, against the T1 baseline:

| term | T1 (`MinimizeBFGS`) | P3 (`MinimizeBFGSBatched`) |
| --- | --- | --- |
| Intercept | -1.25936890 (se 0.12140194) | -1.25937 (se 0.12140) |
| visit_num | 0.17538452 (se 0.02406840) | 0.17538 (se 0.02407) |

Identical to printed precision. But it took **60 iterations and 15m35s**, where the old optimiser
finished in **7** -- and the 53 extra iterations changed the estimates not at all. 5a likewise ran
all 60.

**The termination test cannot fire for this objective.** `MinimizeBFGSBatched` stops on

```cpp
AV grad_big = AnyAbsAtLeast(gradient, kSmallEpsilon_scaled);   // kSmallEpsilon = 1e-4
AV keep_going = *(grad_big * line_search_ok);
```

an ABSOLUTE gradient tolerance of 1e-4, applied to an objective whose gradients task 0016 measured
at **0.17 to 190**. The condition is always true. The old `MinimizeBFGS` stopped at 7 because its
backtracking gave up -- and reported `converged = true`, which flattered it. The batched line
search keeps finding acceptable steps and grinds to the cap.

So **the cap is now the only stopping rule and every model hits it**. Projected full pipeline:
12 mixed models x 60 x ~15s ~ **3 hours**, against 39m42s today, with most of it buying nothing.

The fix belongs in the termination test, not the cap: a scale-aware criterion (relative gradient,
or step size / objective change) would stop 2a near 7 and still let 5a/5b run as long as they
genuinely need. **Flagged for decision before Phases 4-7.**

### Phases 4-7 — remaining

## Verification
As specified in the plan file. Headline gates:
- Phase 0: `NewtonSchulzInverse` agrees with `linalg.h`'s exact Cholesky `SymmetricInverse` up to
  condition ~2e4.
- Phase 1-2: the batched objective at `num_points = 1` reproduces `FlatObjective` bit-for-bit, and
  at `K` returns K values matching K separate single-point calls.
- Accuracy tracked as `|delta estimate| / SE`, not `|grad|`.
- Full-pipeline diff against `runs-verify/T1-models.txt` (75 `RESULT` lines, 39m42s baseline).

## Risks
- Mixed-model SEs get worse by construction (8% -> 17-46% at condition 2e4, per the measurement
  recorded at `regression.h:1535-1545`), and cost rises from `2*dim` gradient evaluations to
  `4*dim^2` objective evaluations. Accepted by decision 4.
- The pipeline-origin halves (~1390 lines) have **no test coverage**; Phases 2-5 rewrite untested
  code.
- Phase 6 (`SecureSqrt` = `Exp(0.5*Log(x))`) is the likeliest source of a runtime regression.
