# 0009 — MPC analysis pipeline (`playground/mpc-analysis.cpp` + headers)

## Metadata
- Task ID: 0009
- Title: Full MPC port of the SISA acute-care analysis pipeline
- Requested by: Adam Godel
- Owner: Adam Godel
- Date: 2026-09-10
- Status: Done
- Estimated effort: Large (~1,300 new lines on top of the existing ~1,700)
- Target completion date: TBD
- Related issue or PR: —
- Related documents: <https://cs-people.bu.edu/liagos/pilot/mpc_analysis_lineage.html> (lineage map
  for `MPC_analysis_Mar26-2.sas`); upstream ETL lineage at `pcc_sisa_lineage.html`; tasks 0001–0006
  (the logistic-regression / fixed-point lineage this builds on)
- Branch: `orchestration`

## Problem Statement
- **What is the request?** Implement, under MPC, the complete analysis pipeline documented by the
  lineage map above: three input encounter tables in, and **19 terminal outputs** out — 2 printed
  descriptive reports, 3 one-row aggregate tables, and 14 fitted regression models (6 specifications
  × 3 populations, minus the 4 interaction fits that can only run on the pooled table).
- **Why does this matter now?** It is the first end-to-end, real-world analysis for CryptDough rather
  than a single-kernel benchmark, and it is the workload the cross-org 3PC deployment (task 0008) was
  built to carry. It also converts `secure-logistic-regression.cpp` from a numerical-accuracy harness
  into a usable analysis program.

## Goals
- Goal 1: Reproduce all 19 lineage nodes faithfully, including the source script's known faults
  (§ "Faithfulness" below), with a printed warning beside each affected output.
- Goal 2: Make the 12 mixed models work on **ragged** clusters (patients have 1..k encounters), which
  the current balanced `ClusterGroup`/`Dataset` code cannot express.
- Goal 3: Add the missing numeric primitives — `Recip`, `Rsqrt`/`Sqrt`, `CholeskySolve` — and remove
  the boolean divider from the hot path. **Done, including from inside `Sigmoid` and `Log`, which
  the first pass missed.**
- Goal 5 (added): bring the cost of a mixed-model fit down far enough to be usable. **Done: 11.1x
  end to end, 64x on the gradient.**
- Goal 4: Keep the program runnable standalone (synthetic data, self-validating) *and* against real
  per-party CSVs.

## Non-Goals
- Non-goal 1: GEE / robust standard errors (the lineage page offers them as a correction to step 6;
  out of scope).
- Non-goal 2: Residual pseudo-likelihood (RSPL). Model 6b's missing `method=laplace` is flagged, not
  emulated.
- Non-goal 3: `LSMEANS ... / diff` adjusted-mean contrasts.
- Non-goal 4: Promoting the fixed-point kernels into `include/` as library headers. Deferred.
- Non-goal 5: A secure `erf`. See the decision below.

## Scope
- In scope: `playground/secure-logistic-regression.cpp` → `playground/mpc-analysis.cpp` (rename plus
  substantial growth); `scripts/testing/validate_mpc_analysis.py` (new — the plaintext oracle that
  makes the verification reproducible rather than a one-off); this task document; the
  `tasks/tasks.md` ledger line.
- Out of scope: any change to `include/`, `CMakeLists.txt` (the `playground/*.cpp` glob at
  `CMakeLists.txt:269` already covers the new name), or `playground/logistic-regression.cpp`.

## Impact Assessment
- **User impact:** `secure-logistic-regression` disappears as a build target and is replaced by
  `mpc-analysis`. Any script or note referencing the old target name must be updated.
- **Performance impact:** Large, and in both directions. Replacing `BSharedVector::operator/` with a
  Newton reciprocal on the four hot call sites is an estimated ~1000× reduction in the dominant cost;
  against that, the pipeline runs 14 model fits rather than one.
- **Security and privacy impact:** The pipeline reveals its 19 published outputs, plus two things
  worth stating plainly. (a) Each table has a designated owner that holds it in the clear, so the
  sort by `subject_id` and the categorical coding happen locally in plaintext — this matches
  `examples/ex6_three_party_private_input.cpp` and `docker/DEPLOYMENT.md`, and reveals nothing the
  owner does not already have. A deployment where *no* party holds `any_system` would need an
  oblivious sort and merge, which is out of scope here. (b) `MinimizeBFGS` opens scalars to steer its
  line search — inherited from the existing code, not introduced by this task, and worth a follow-up.
  The quantile search opens one comparison bit per step, which discloses exactly the published
  quantile and nothing further.
- **Backward compatibility impact:** The rename is a breaking change to the target name; no API or
  header is touched.

## Context
- Relevant files and modules:
  - `playground/secure-logistic-regression.cpp` — the file being renamed and grown. Existing kernels:
    `ClampAbs` (l.102), `Exp` (l.123), `Log` (l.182), `Log1p` (l.287), `Sigmoid` (l.308),
    `LogOnePlusExp` (l.350), `Sum` (l.372), `ConditionalMode` (l.386), `GroupLaplaceLogLik` (l.470),
    `NumericalGradient` (l.585), `MinimizeBFGS` (l.833).
  - `include/core/operators/aggregation.h:162` — the Brent-Kung segmented `aggregate`.
  - `include/core/operators/circuits.h:39` — the non-restoring boolean divider (the thing to avoid).
  - `include/core/containers/a_shared_vector.h:338,367` — `dot_product`,
    `matrixRightMultiplyWithColumnMatrixVectorized`.
  - `include/core/containers/tabular/encoded_table.h:492` — `inputCSVTableData`.
- Dependencies: none new.
- Constraints and assumptions: fixed point at `precision = 16` on `int64_t`; 3PC replicated is the
  default protocol; all loop trip counts must be public for obliviousness.

## Alternatives

### Option A: New self-contained `mpc-analysis.cpp` alongside the existing file
Pros:
- Smallest diff; `secure-logistic-regression` keeps working as an accuracy harness.
Cons:
- Duplicates ~300 lines of kernels that will immediately drift, and the kernels are exactly what this
  task modifies (`Recip` rewiring touches all of them).

### Option B: Extract a shared `playground/fixed_point_math.h`
Pros:
- No duplication; both programs stay.
Cons:
- Two files to keep in sync at the point where the second program adds little; the accuracy harness
  in `main()` is what we want to *extend*, not fork.

### Option C: Rename and grow the existing file (chosen)
Pros:
- Zero duplication; the existing per-kernel plaintext-oracle harness is inherited and extended rather
  than re-created; `Recip`/`Rsqrt` land in exactly one place.
Cons:
- Loses `secure-logistic-regression` as a build target; a larger single file (~3,000 lines).

### Recommendation
- Recommended option: **C**, explicitly confirmed by the requester (required, since `agents.md`
  forbids renaming files without confirmation).
- Why this option is preferred: the pipeline's models *are* the logistic regression this file already
  implements; splitting them apart creates two copies of the same numerics.
- Open questions needing confirmation: whether a `--public-clusters` fast path (revealing subject
  count and group sizes, in exchange for an estimated ~250× speedup) is acceptable in a real
  deployment. Not implemented; the decision can wait until the cost is a problem in practice.

## Key decisions

| Question | Decision | Rationale |
|---|---|---|
| Ingestion | Synthetic by default, `--data-dir` for per-party CSV | Binary stays runnable and self-validating with no data files |
| Cluster representation | Flat table sorted by `subject_id` + segmented prefix scan | Ragged clusters at O(n log n); padding to max-K would blow up on a right-skewed visit distribution and leak K |
| p-values | Reveal `(estimate, SE)`, compute z/p/OR-CI in plaintext | p is a deterministic public function of two values the pipeline already publishes; a secure `erf` would protect nothing and would floor at p ≈ 1e-4 at precision 16 |
| Steps 6a/6b fitting | IRLS/Fisher scoring, not BFGS | Analytic Hessian, quadratic convergence, and the converged `XᵀWX` **is** the covariance matrix, so SEs are exact |
| Order statistics | Publish `n_patients` | It is already the first reported column of `d1a`/`d1b`; treating it as secret buys nothing and turns each quantile into an oblivious O(n) select |

## Faithfulness: source-script faults to reproduce and flag

Documented on the lineage page; they are part of the specification, not bugs to fix silently.

1. Steps 6a/6b have **no random effect** — `subject_id` is declared in `CLASS` and never used, so SAS
   fits an ordinary fixed-effects logistic regression on the one comparison the script exists to make.
2. Model 6b omits `method=laplace`, silently falling back to pseudo-likelihood.
3. The 6a `ESTIMATE` statement names `data_source*fu_month` in a model containing
   `data_source*visit_num`; SAS rejects it. The answer survives as the interaction coefficient.
4. `fu_month` enters as a continuous linear term despite taking only `{0,1,3,6}`.
5. `fu_month` is NULL outside the windows; those rows drop from every `b` model and from the step 3/4
   denominators without appearing as an exclusion.
6. `sisa_pct` denominators count patients *observed* in the window — a conditional prevalence, not
   cumulative incidence.
7. `hispanic` is dropped from model 5b/UMass only, so the two 5b odds-ratio sets are not comparable.
8. `fu_month_{umass,nonumass}` inherits an upstream ETL bug (guarded on the global `index_visit`
   rather than the per-system one).

## Implementation Plan
1. Process: this document + `tasks/tasks.md` ledger line.
2. `git mv playground/secure-logistic-regression.cpp playground/mpc-analysis.cpp`; update the run
   recipe comment on l.8.
3. Kernels: add `ClampRange`, `Abs`, `Recip` (Newton, public range hint); rewire `Sigmoid`, `Log`,
   the inner Newton step and `1/σ²` off `BSharedVector::operator/`; add `Rsqrt`/`Sqrt` (own
   even-exponent range reduction, degree-2 minimax seed, **one** Newton step).
4. Segmented helpers: `SegTotal` (forward + reverse over the *original* column, then
   `pre + suf - in`), `LastOfGroup`, `CountDistinct`, `DummyCode`.
5. Linear algebra: `Gram` (XᵀWX via the row-major/column-major identity — no transpose needed),
   `CholeskyFactor`, `CholeskySolve`, `CovDiag`, with a mandatory ridge on the diagonal.
6. Data model: `struct AnalysisTable`; ragged synthetic cohort generator; CSV loader.
7. Descriptive nodes `d1a`, `d1b`; aggregate nodes `sisa_perct_cnt` ×3.
8. Models: design-matrix builder per (step × dataset); `FitLogisticIrls` for 6a/6b; flat vectorized
   `FitGlmmLaplaceFlat` for 2a/2b/5a/5b.
9. Reporting: coefficient tables, plaintext z/p/OR-CI, per-node fault footers.
10. `main()`: CLI, ingestion, all 19 nodes, extended accuracy harness.

## Risks and Mitigations
- **Risk:** Cost. A naive port is ~500 GB/party per GLMM fit.
  - Mitigation: `Recip` (step 3) is non-negotiable and is estimated at ~1000×. Two further levers held
    in reserve: caching the Brent-Kung group bits across the ~360 objective evaluations, and the
    `--public-clusters` fast path (`prefix_sum` is local and free) worth an estimated further ~250×.
- **Risk:** the fully oblivious flat design computes per-cluster nonlinearities on every row to use
  one row per cluster.
  - Mitigation: accepted for now; the `--public-clusters` alternative is documented but deliberately
    not implemented, so no leakage decision is being made implicitly.
- **Risk:** Quasi-separation with p = 8 and an interaction term. At precision 16, `W = p(1−p)`
  truncates to 0 for |η| ≥ 12, making `XᵀWX` singular and handing `Rsqrt` a non-positive diagonal.
  - Mitigation: ridge `λ = 1e-3` on the diagonal, mandatory and reported.
- **Risk:** Silent int64 wraparound in `Rsqrt` from a bad seed (`g·(3 − x·g²)` reaches ~7e22 in fixed
  point if `g₀` hits `Exp`'s saturation value `e¹⁰`).
  - Mitigation: public clamps on both `x` and `g₀`, so `h = x·g² ∈ (0,3)` is guaranteed by construction.
- **Risk:** The `SegTotal` forward-then-reverse trap — running Reverse on an already-scanned column
  yields a suffix-of-prefixes, which is silently wrong rather than an error.
  - Mitigation: both passes read the *original* column; dedicated unit check against a plaintext
    `std::map` group-by (validation step 2).
- **Risk:** SAS `CLASS` makes the **last** sorted level the reference; patsy/lme4 use the first.
  Getting it wrong flips coefficient signs silently.
  - Mitigation: `DummyCode` implements the SAS convention explicitly; cross-checked in validation
    step 4.

## Rollback and Recovery
- Rollback plan: `git revert` the commit. The rename is the only externally visible change, and
  nothing else in the tree depends on the old target name.
- Recovery steps: none beyond a fresh `cmake ..` (the playground glob is evaluated at configure time,
  so a rename requires re-configuring, not just `make`).
- Monitoring and alerts after release: n/a — this is a research binary, not a service.

## Validation Plan
- Automated checks:
  - Extend the existing per-kernel plaintext-oracle harness in `main()` to cover `Recip`, `Rsqrt`,
    `Sqrt`, `CholeskySolve` (residual ‖Ax − b‖∞) and `SymmetricInverse` (‖AA⁻¹ − I‖∞).
  - `SegTotal` unit check on a ragged key column against a plaintext `std::map` group-by.
  - Run under `-p 1` (plaintext 1PC) for a fast correctness loop.
- Manual verification:
  - End-to-end synthetic recovery of a known `(β, σ)` for all 14 fits.
  - Cross-check the coefficients against `pymer4`/`statsmodels` on the same cohort dumped to CSV,
    exactly as the lineage page's own Python does.
  - Cross-check `d1a`, `d1b` and the three `sisa_perct_cnt` tables cell by cell against SQLite, using
    the lineage page's SQL verbatim.
- Expected success criteria: coefficients agree to ~1e-3 (the fixed-point floor at precision 16 makes
  anything tighter meaningless); descriptive and count nodes agree exactly, modulo the documented
  `PCTLDEF=4` vs `PCTLDEF=5` quantile convention.

## Definition of Done
- All 19 lineage nodes implemented and printed, each labelled with its node id.
- Both ingestion paths work; the binary runs with no data files present.
- The eight source-script faults are reproduced and flagged in the output.
- `BSharedVector::operator/` no longer appears on any hot path.
- Tests updated or added: the in-binary accuracy and segmented-scan harnesses above. No new file
  under `tests/` — this is a playground program.
- Documentation updated: this document, the `tasks/tasks.md` ledger line, and the run recipe comment
  at the top of the renamed file.

## Approval
- [x] Task document reviewed
- [x] Approved to implement
- Approver: Adam Godel
- Approval date: 2026-09-10

## Implementation notes (what changed against the plan)

Recorded as the work went, because several of these were only visible once the
code ran.

1. **`Exp` had a live bug, and it mattered more than anything else here.**
   `div_const_a` truncates toward zero rather than flooring, so the round-to-
   nearest in the range reduction (`+0.5` then shift) rounded the wrong way for
   negative arguments and let `r` escape `[-ln2/2, ln2/2]`. `exp(-1)` returned
   exactly 1/3 -- the 3-term series evaluated at `r = -1`. It propagated into
   `Sigmoid`, which was off by **0.019 at eta = +/-1**, the densest part of the
   logit range and the value every model in this pipeline depends on. Fixed with
   a public bias before the shift. Measured `Sigmoid` error went from 1.9e-2 to
   3.0e-5, about 600x. The series lengths were then raised from 3 to 5 terms,
   which is what made the truncation error the binding constraint rather than the
   bug.

2. **`Rsqrt` needs two Newton steps, not one.** The plan's one-step estimate
   assumed a seed accurate to 1.5e-3. The actual degree-2 minimax seed on
   `[0.5, 2)` is 2.4e-2, so one step leaves 8.7e-4 -- about 57 ulp at precision
   16, far too coarse for a standard error. Two steps reach 1.1e-6, under the
   floor. Seed coefficients were derived by Remez exchange and checked
   numerically before being written in.

3. **The row-major / column-major transpose identity does not apply to `X^T W X`.**
   The plan claimed the Gram matrix needed no transpose because X row-major is
   X^T column-major. That identity is true, but `matrixRightMultiplyWithColumn-
   MatrixVectorized` needs lhs row-major AND rhs column-major, and only one of
   the two operands can be satisfied that way. Implemented instead as a single
   chunked `dot_product` over operands assembled by `mapping_reference` (a free
   public gather), which lays the `p(p+1)/2` column pairs end to end. The
   contraction still happens locally, so only `p(p+1)/2` ring elements cross the
   wire, independent of n -- the intended property, by a different route.

4. **Quantiles by CDF binary search, not by sorting.** Both descriptive nodes run
   on bounded small integers with publicly known ranges, so an order statistic is
   a binary search over the CDF. This is cheaper than an oblivious sort AND leaks
   strictly less than the plan's "publish n and take a public index": the search
   path is a function of the returned quantile alone, so it discloses exactly the
   values the pipeline already publishes and nothing about the rest of the
   distribution.

5. **`AV a = b` is a SHALLOW copy.** The copy constructor shares the underlying
   buffer while `operator=` is a deep element-wise copy -- which is why `Clone`
   exists. Copying a cohort column and then mutating it in place corrupted the
   cohort for every later use; it showed up as the 3-month and 6-month follow-up
   windows returning zero counts. Every copy-then-mutate site now goes through
   `Clone`.

6. **`aggregators::Direction` cannot be named directly.** `common.h:14` declares
   `enum class Direction { ... } Direction;` -- the trailing name is a variable
   that shadows the type, so an elaborated specifier is required. `aggregate()`
   itself works around this the same way.

7. **BFGS can return without a usable covariance.** If no curvature update ever
   passes the `y^T s > 0` test, `h_inv` is still the identity and its diagonal is
   not a variance. This showed as a standard error of exactly 1.00000 on one fit.
   `OptResult` now carries `hessian_updated`, and such fits report `n/a` rather
   than a fabricated standard error.

8. **The BFGS inverse Hessian is not usable as a covariance matrix.** The plan
   accepted `h_inv` for the mixed-model standard errors, noting it as an
   approximation. In practice it is worse than approximate: with twelve
   iterations and up to eight parameters it leaves whole directions untouched,
   and the reported standard error is then the identity it was initialised with
   -- literally 1.00000 on one fit, and 1.00043 on another. Replaced by a
   finite-difference observed-information matrix at the optimum, inverted in
   plaintext. That costs `1 + 2*dim + dim(dim-1)/2` extra objective evaluations
   (36 at `dim = 7`, about two and a half gradients) and is consistent with the
   decision already taken for p-values: the curvature of the log-likelihood at
   the optimum is precisely what a published standard error discloses.

9. **Standard errors on near-collinear terms are unreliable at precision 16, and
   the program now says so.** A second difference divides by `h^2`, so it
   amplifies the objective's ~1e-4 fixed-point noise by `1/h^2`; where the
   Hessian is also near-singular, inverting it amplifies that without bound.
   `hispanic` is about 90% ones and therefore nearly the intercept, which is
   exactly such a direction. Measured against the oracle, the **time
   coefficient's** standard error -- the pipeline's headline output -- agrees to
   at worst 6.8%, and to under 4% in eleven of twelve fits; the intercept and the
   sparse dummies reach 34%. Both implementations independently return no
   standard error for the same `newage` directions, so they agree that the
   Hessian is indefinite there. The fits now carry a 1-norm condition estimate
   and print an explicit warning above 1e4, and the validation script holds the
   two groups of terms to different tolerances rather than one blunt bound.

10. **Initial measured cost.** Under `-DPROTOCOL=1` (plaintext 1PC) on blinky, one mixed
   model on a 400-patient / ~1000-encounter cohort takes about 2m45s, so the full
   set of fourteen is roughly 40 minutes. The dominant term is BFGS with
   central-difference numerical gradients: `2*dim` objective evaluations per
   gradient, `dim` up to 8. The three levers named in the plan -- analytic or
   forward-difference gradients, caching the Brent-Kung group bits across
   evaluations, and the `--public-clusters` fast path -- are all still available
   and all still untaken. Correctness first; these are the obvious follow-up.

## How to verify

```bash
# fast: kernels, segmented scans, dense linear algebra
./mpc-analysis -S kernels

# ingestion plus the descriptive and aggregate nodes; dump the cohort
./mpc-analysis -S describe -r 200 -O /tmp/dump

# the fourteen fits, captured for the oracle to diff (about 30 min at r=400
# under PROTOCOL=1; scale down while iterating)
./mpc-analysis -S models -r 200 > /tmp/models.txt

# independent plaintext oracle: exact on the counting nodes, and it scores the
# fits on the RESULT lines the run above emitted
python3 scripts/testing/validate_mpc_analysis.py /tmp/dump
python3 scripts/testing/validate_mpc_analysis.py /tmp/dump --compare /tmp/models.txt

# read the dumped CSVs back through the per-party ingestion path
./mpc-analysis -S describe -D /tmp/dump
```

The comparison applies two different criteria, because only one model family has
a well-determined answer to compare against:

- **Steps 6a/6b** are plain logistic regressions and the oracle runs the same
  IRLS estimator, so the *coefficients* must agree to about the fixed-point
  floor. Tolerance 1e-3; measured 1.6e-4.
- **The twelve mixed models** are optimisation problems whose likelihood has
  genuinely flat directions — a near-collinear sparse `hispanic` dummy, or a
  variance sitting on the `sigma^2 = 0` boundary that the secure code's
  positivity floor cannot reach. Two optimisers then stop at visibly different
  coefficients that fit equally well, so comparing coefficients would reject a
  correct implementation. What has to hold is that the secure parameters attain
  the same log-likelihood. Tolerance 0.25 in `-logL`; measured worst case 3.4e-2.

  This distinction was not anticipated in the plan and was only visible once the
  fits ran: a naive coefficient diff showed differences up to 1.7e-1 on
  `5b`/non-UMass `hispanic=0`, which looked like a defect until the objective
  values showed both points were equally good optima.

- **Standard errors** are scored at the *secure* optimum, so the check measures
  the finite-difference machinery rather than the gap between the two optima, and
  the time coefficient is held to a tighter bound (10%) than the intercept and
  covariate dummies (40%). See implementation note 9 for why that split is a
  property of the data rather than a convenience.

## Cost optimisation

Measured on blinky with `-S bench`, which times one mixed model (step 5a, the
widest, `dim = 7`) rather than relying on estimates. Both protocols were checked,
because the plan's cost model was about communication while the first
measurements were plaintext-only; the wins turned out to be protocol independent.

### What actually dominated

The plan predicted the segmented scans would be the bottleneck once the boolean
divider was gone. Measurement said otherwise: at the starting point one
`SegTotal` was 0.0039 s against 0.80 s for one objective evaluation — **0.5%**.
The real multiplier was the gradient: central differences cost `2*dim = 14`
objective evaluations, and BFGS wants one gradient per iteration.

### Changes, in the order they were made

| # | Change | Effect |
|---|---|---|
| 1 | **Analytic gradient** of the Laplace objective, derived via the envelope theorem plus the implicit function theorem, sharing the conditional-mode search with the value | gradient **13.4x** faster, and more accurate — no step size |
| 2 | **Removed `BSharedVector::operator/` from `Sigmoid` and `Log`** | `Sigmoid` **7.8x** faster; one objective evaluation 4.0x |
| 3 | **Inner Newton loop 5 -> 3 iterations** | objective a further **1.43x**, no accuracy cost |
| 4 | **Cached per-level scan group bits** | **~4%** locally; much larger on a latency-bound link |
| 5 | **Hessian from gradient differences** instead of objective second differences | **2.5x** cheaper *and* far more accurate (see note 9) |

Change 2 deserves a note: it was the plan's *first* named lever and it had been
missed. `Div`/`Recip` were written and wired into the new code, but the divisions
*inside* `Sigmoid` and `Log` were never replaced — so the 64-iteration
non-restoring divider was still running six times per objective evaluation on
every row. The isolated timings exposed it: `Exp` was 0.0068 s and `Div`
0.0113 s, yet `Sigmoid` — which is just those two plus a sign multiplex — was
0.0838 s. Accuracy after the swap is unchanged to the last printed digit
(`Sigmoid` worst absolute error 3.0e-5 before and after), so the Newton
reciprocal reproduces the divider exactly at a fraction of the cost.

`Sigmoid` and `Log` both know their denominator's interval up front
(`1 + exp(-|eta|)` is in `(1, 2]`; `m + 1` is in `[1.7071, 2.4142]` after Log's
range reduction), so they use `RecipSeeded` — a minimax linear seed for that
interval plus Newton steps, skipping `Div`'s ten-step normalisation ladder
entirely.

### Results

Per objective evaluation and per gradient, at `r = 200` (1024 padded rows):

| | 1PC before | 1PC after | 3PC before | 3PC after |
|---|---|---|---|---|
| `Sigmoid` over n rows | 0.0838 s | 0.0109 s | 0.1966 s | 0.0263 s |
| one objective evaluation | 0.8016 s | 0.1360 s | 1.8595 s | 0.3263 s |
| one gradient | 11.29 s | **0.176 s** | 26.27 s | **0.415 s** |
| gradient speedup | | **64x** | | **63x** |

End to end, all fourteen fits on a 400-patient / 1062-encounter cohort under
`PROTOCOL=1`: **1807.7 s -> 162.5 s, an 11.1x reduction.** The gap between 64x
on the gradient and 11.1x overall is the work that is not the BFGS loop: the
observed-information Hessian (`2*dim` gradient evaluations per mixed model, now
about a third of the run), the two IRLS fits, and the plaintext cross-check.

### `--public-clusters`: measured, and declined

The plan estimated a ~250x win from revealing the cluster boundaries. That
estimate was built on the assumption that the segmented scans dominated, which
measurement disproved. `-S bench` now reports the real ceiling directly: the
per-cluster quantities computed on all n rows (the Newton-step reciprocal, `1/A`
and `log A`) are **45% of an objective evaluation**, and with a mean cluster size
of 2.65 compacting them to one row per cluster could reclaim at most **1.39x**.

That is not worth revealing each patient's encounter count, so the fast path was
**not implemented**. Node `d1a` publishes the visit-count *distribution*; the
per-patient ordering is strictly more than that. The measurement, not a
preference, is the reason — and it is printed by `-S bench` so the trade-off can
be re-examined if the workload ever changes.

### Still on the table

- The Hessian is `2*dim` gradient evaluations. Forward instead of central
  differences would make it `dim + 1` (about 1.2x overall) at the cost of O(h)
  rather than O(h^2) accuracy — not obviously worth it while the standard errors
  are the output under most scrutiny.
- The cached scan's benefit here is bandwidth-shaped, not latency-shaped, because
  everything ran on one machine. On a real cross-org link the equality plus
  `b2a_bit` it removes is roughly ten rounds per network level against one for
  the multiply that remains, and the scans are the bulk of the round count now
  that the transcendental kernels no longer divide privately. That is an
  inference from the construction, not a measurement, and it is the obvious thing
  to check on the task 0008 deployment.

## Validation results

- `Div` / `Recip` / `Sqrt` / `Rsqrt`: ~1 ulp across four orders of magnitude.
- `Exp` / `Log` / `Sigmoid` / softplus: at or near the precision-16 floor.
- Segmented scans on ragged groups: exact against a plaintext `std::map`
  group-by, including the sentinel-padding edge case.
- `Gram` / `CholeskySolve` / `SymmetricInverse`: Gram error 1.6e-4, solve
  residual 1.3e-3, inverse error 2.7e-4.
- Nodes `d1a`, `d1b` and all three `sisa_perct_cnt` tables: **exact** agreement
  with an independent Python recomputation over the dumped cohort -- counts,
  percentages, means and standard deviations to four decimals.
- Steps 6a / 6b against a plaintext IRLS oracle running the same estimator:
  max coefficient difference 1.6e-4.
- All twelve mixed models: `-logL` excess over the oracle's optimum between
  1.0e-5 and 3.4e-2, against a 0.25 tolerance.
- Time-coefficient standard errors: worst relative difference 6.8%, under 4% in
  eleven of twelve fits. Nuisance-term standard errors reach 34% in the
  near-collinear directions, which the program flags at runtime.
- `scripts/testing/validate_mpc_analysis.py --compare` reports
  **VALIDATION: PASS** across all fourteen fits, exit code 0.
- Model 2a on the pooled table, spot-checked term by term against an independent
  scipy BFGS fit of the same Laplace objective: intercept 3.6e-3, slope 1.0e-3,
  variance 2.2e-3.
- CSV round-trip: dumping the synthetic cohort and reading it back through the
  per-party ingestion path reproduces every node bit for bit.
- `-S all` end-to-end: exit 0, both harnesses PASS, fourteen fits, 75 machine-
  readable RESULT lines.
- After optimisation, re-validated unchanged: `-logL` excess worst 3.3e-2,
  standard errors worst 6.3% under a single 10% tolerance, IRLS coefficients
  1.6e-4, **VALIDATION: PASS**.
- The analytic gradient is checked against central differences inside `-S bench`
  on every run: agreement to 0.9-1.1% relative to the largest component, the
  residual being the *numerical* gradient's step error.
- The cached scan is checked against `aggregators::aggregate` row by row in
  `TestSegmented`, to 1e-9, so an index slip in the reimplemented Brent-Kung
  level geometry cannot pass silently.

## Follow-up: header split (merge of `logistic-regression`)

The single-file form was always a consequence of Option C above -- rename and grow
one file -- not a preference. The `logistic-regression` branch had meanwhile
established a header convention for exactly this code (`#pragma once`, namespace
`cdough::regression` with the protocol `using`s inside it, types and constants in
`primitives.h`, a linear include chain), so that branch was merged and the
convention adopted.

`mpc-analysis.cpp` went from 4392 lines to 336: it now holds only the driver --
stage selection, ingestion, and the loop over the nodes.

| Header | Lines | Contents |
|---|---|---|
| `primitives.h` | 706 | types, constants, `Clone`; `ClampAbs`/`ClampRange`/`Abs`, `RecipSeeded`/`Div`/`Recip`, `Sqrt`/`Rsqrt`, `Exp`/`Log`/`Log1p`, `Sigmoid`/`LogOnePlusExp`; share and open helpers |
| `segmented.h` | 309 | `SegScan`/`SegTotal`, the cached-group-bit `SegScanPlanned`/`SegTotalPlanned`, `First`/`LastOfGroup`, `CountDistinct` |
| `linalg.h` | 243 | `Gram`, Cholesky factor / solve / `SymmetricInverse` |
| `optimizer.h` | 667 | the branch's matrix helpers and `NewtonSchulzInverse` (task 0010), plus BFGS, its numerical gradient and the value-plus-gradient objective interface |
| `cohort.h` | 515 | plaintext and secret-shared row layouts, synthetic generator, per-party CSV ingestion |
| `nodes.h` | 321 | `d1a`, `d1b`, `sisa_perct_cnt` x3 |
| `regression.h` | 827 | design matrices, IRLS (steps 6a/6b), flat Laplace mixed model with the analytic gradient |
| `reporting.h` | 284 | coefficient tables and the plaintext IRLS oracle |
| `harness.h` | 546 | accuracy harnesses and the cost breakdown |
| `mixedeffects.h` | 237 | the branch's balanced-cluster mixed-effects model (tasks 0001-0003), used by `secure-logistic-regression.cpp` |

Include order is `primitives` -> {`segmented`, `linalg`, `optimizer`} -> `cohort`
-> `nodes` -> `regression` -> `reporting` -> `harness` -> `mpc-analysis.cpp`.
`mixedeffects.h` hangs off `optimizer.h` on its own and is included only by
`secure-logistic-regression.cpp`, so the two programs share the primitives and
the optimiser without either depending on the other's model code.

The split moved code without changing it, with two mechanical exceptions:

1. The type aliases are now qualified — `using AV =
   COMPILED_MPC_PROTOCOL_NAMESPACE::ASharedVector<DataType>;` rather than the
   bare name. Inside `namespace cdough::regression` the unqualified name is
   ambiguous, because the enclosing `cdough` namespace declares
   `ASharedVector` too and the protocol namespace aliases the same name. This is
   the branch's own convention; it was not optional.
2. The duplicate `Identity` was dropped in favour of the branch's
   `ScaledIdentity` / `Identity` pair, which is strictly more general.

**Verified behaviour-preserving:** all 75 machine-readable `RESULT` lines from
`-S models -r 60` are byte-identical before and after the split, every figure in
the kernel accuracy harness is unchanged to the last digit, `-S all` exits 0 with
both harnesses passing, and the build is warning-free under both `PROTOCOL=1` and
`PROTOCOL=3`.

### Merge resolutions

Nothing from the branch was dropped. Every one of its 126 top-level symbols is
present; the only differences that remain are four local variable names inside
`BfgsInverseUpdate` and `MinimizeBFGS`, whose bodies are equivalent (`Clone(x)`
is exactly the branch's `AV t(x.size(), engine); t = x;` idiom).

- `playground/secure-logistic-regression.cpp` is **kept**, as a second binary.
  One line changed: `#include "./regression.h"` became
  `#include "./mixedeffects.h"`. It now runs against the pipeline's corrected
  kernels, so it is *more* accurate than it was on the branch -- `ConditionalMode`
  agrees with its plaintext oracle to 2.2e-5 and `NegMarginalLogLik` to 1.2e-3,
  where the unfixed `Exp` had been the limiting error.
- The branch's balanced mixed-effects model (`ClusterGroup`, `Dataset`,
  `UnpackParameters`, `ConditionalMode`, `GroupLaplaceLogLik`,
  `NegMarginalLogLik`) is **kept**, moved verbatim into `mixedeffects.h` under the
  same `cdough::regression::mixedeffects` namespace. It is not an older version
  of `regression.h` -- it is a different data layout. One cluster per
  `ClusterGroup` is the natural form for balanced clusters; the pipeline's flat
  padded table with segmented scans exists because real patients have a ragged
  number of encounters. Both are useful, and the header split is what makes
  keeping both cheap.
- `Sum` is **restored** to `primitives.h`, verbatim from the branch.
- `kMaxSeriesTerms` is **restored**, marked as superseded by `kExpSeriesTerms` /
  `kLogSeriesTerms`, which tune the Exp and Log series independently. It no
  longer controls anything, and the comment says so.
- `MinimizeBFGS` now has **two overloads**: the pipeline's
  value-plus-gradient form and the branch's value-only form, which differences
  the objective. They are distinguished by arity, so a lambda selects the right
  one on its own, and the branch's call sites compile unchanged. The
  `MinimizeBFGSNumeric` name introduced during the optimisation work is gone --
  it was the same function under a different name.
- `SecureReciprocal` is kept in `optimizer.h`, where `NewtonSchulzInverse` uses
  it. It goes through the boolean division circuit, so it must not appear on a
  hot path; `Recip` is the replacement there.
- The branch's task 0007 collided with `0007_dockerize-cryptdough.md` already on
  this branch, so the Newton-Schulz task was renumbered to 0010.
- `include/`, `docker/` and `examples/` are unchanged by the merge, so the TLS
  and cross-org deployment work from tasks 0007-0008 is intact.

Both binaries build warning-free under `PROTOCOL=1` and `PROTOCOL=3`, and
`mpc-analysis`'s 75 `RESULT` lines are byte-identical to before the merge.

## Cross-organizational deployment (task 0008 path)

Run across blinky / pinky / inky as three mutually-distrusting organizations:
separate Docker images built independently on each host, mutual TLS with pinned
self-signed certificates, no SSH between parties, and each party mounting only
its own table read-only.

Rank 0 = blinky (owns `any_system`), rank 1 = pinky (`umass_system`), rank 2 =
inky (`nonumass_system`), `base_port` 26000, `-t 1`, `PROTOCOL=3 COMM=NOCOPY
TLS=ON`.

### Two bugs this surfaced, both in the pipeline rather than the deployment

Neither is reachable from a single-host run, because both are about parties
disagreeing about something they each derive locally.

1. **Every party read all three CSVs.** `ReadCohortCsv` was called
   unconditionally for all three tables and `exit(1)`s on a missing file, so each
   organization would have had to hold the other two organizations' data --
   defeating the entire point. Replaced with `LoadOwnedCohort`: the owner opens
   the file, every other party allocates a same-sized placeholder, mirroring the
   `inputCSVTableData` contract. Row counts move into the manifest
   (`--rows-any/--rows-umass/--rows-nonumass`), because a party that does not own
   a table cannot read its length from a file it may not see. `num_subjects` is
   now derived from the shares rather than counted in the local plaintext, for
   the same reason.

2. **A public loop bound was read off private data.** `max_visits` was the
   maximum of `any_system.visit_num` computed on the local plaintext. The owner
   got 15 and the placeholder-holding parties got 1, so node d1a issued 15
   collective operations on one party and 1 on the others. The protocol
   desynchronised: under MPI it aborted with `MPI_ERR_TRUNCATE`, and over the
   nocopy transport it produced opened counts like
   `-85349505764425267205528817037758716928` before failing in `Conn::send_all`.
   Now a public run parameter (`--max-visits`, default 64). Sweeping past the
   true maximum is free -- empty buckets are dropped.

   The generalizable rule is now in `docker/DEPLOYMENT.md`: if removing a party's
   data directory would change how many times a loop runs on that party, the
   bound belongs in the manifest.

### Result

With both fixed, all three parties completed and every published number matched
the independent Python oracle exactly -- `d1a` moments and the full visit
distribution, `d1b` demographics, and all three `sisa_perct_cnt` tables across
all three follow-up windows -- computed over mutual TLS between three hosts that
each held only one of the three tables.

### Certificate pinning is enforced, not decorative

Verified by a negative test rather than assumed. Replacing blinky's copy of
`party2.crt` with a different valid certificate made rank 0 refuse the
connection outright:

```
what():  Peer certificate is not pinned (to rank 2 thread 0); SHA-256
0590be9e...b39f7d. Refusing the connection rather than falling back.
```

That is the right failure: it names the rank, reports the fingerprint actually
presented, and explicitly declines to downgrade. Restoring the correct
certificate reproduced the earlier results identically.

### Also worth recording

- The `-S kernels` stage ran across all three hosts over TLS before either bug
  was fixed, which is what localised the fault to the ingestion path rather than
  the transport.
- An apparent rank-1 hang early on was self-inflicted: overlapping launches left
  by timed-out orchestration commands, plus leftover listeners from a manual
  connectivity test. A clean re-run had all three parties complete. Worth naming
  because it looked exactly like a protocol deadlock.
- `check-manifest.sh` predicted inbound ports 26001 (rank 1) and 26002 / 26005
  (rank 2), which matched the ports measured as reachable beforehand and the
  ports actually opened during the run.

## Change Log
- 2026-09-10: Initial draft created and approved.
- 2026-09-10: Implementation notes and validation results added.
- 2026-09-10: Cost optimisation (11.1x end to end); see that section.
- 2026-09-10: Merged `logistic-regression` and split the program into headers;
  both programs and every branch symbol retained.
- 2026-09-10: Cross-organizational TLS deployment across three hosts; two
  ingestion bugs found and fixed (see that section).
