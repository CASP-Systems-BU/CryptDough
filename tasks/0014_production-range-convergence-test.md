# 0014 — Does the pipeline still converge on the production data ranges?

## Metadata
- Task ID: 0014
- Title: Convergence of the MPC analysis pipeline over the real data's value ranges
- Requested by: Adam Godel
- Owner: Adam Godel
- Date: 2026-09-11
- Status: Done — **no code changes**; findings and follow-ups below
- Estimated effort: Small (test data + runs; no source changes)
- Target completion date: —
- Related documents: [tasks/0009](0009_mpc-analysis-pipeline.md) (the fits, the fixed-point
  kernels, and the 0.25 / 10% / 1e-3 tolerances used here),
  [tasks/0012](0012_sqlite-relational-verification.md) (the oracle this uses),
  [tasks/0013](0013_single-owner-node-runner.md) (used to dump the analysis tables)
- Branch: `orchestration`

## Problem Statement
- **What is the request?** The pipeline was validated against a synthetic cohort whose value
  ranges were chosen for convenience. The production data does not look like that:

  | column | in-binary generator (`etl.h:GenerateBaseTable`) | production |
  |---|---|---|
  | `subject_id` | 1 .. N (N = 200 or 400) | ~1 000 .. ~40 000 |
  | `newage` | 14 .. ~90 | ~1 .. ~100 |
  | `encounter_dt` | 0 .. ~4 000 days from an arbitrary epoch | 45 586 .. 45 950 (day serials) |
  | `visit_num` | 1 .. 16 | ~1 .. ~100 |

  Run the pipeline over the production ranges and establish whether convergence still holds.
- **Why does this matter now?** Everything in the fits runs in fixed point at `precision = 16`
  on `int64_t`, with public clamps (`ClampAbs(eta, +/-10)`, the `sigma^2` band) sized against
  the old ranges. Value ranges an order of magnitude larger are exactly the kind of change
  that invalidates a clamp silently — the arithmetic does not fail, the answer just moves.

## Goals
- Goal 1: exercise the whole pipeline over the production ranges, end to end.
- Goal 2: separate *numeric magnitude* (does bigger arithmetic break the fixed point?) from
  *statistical shape* (does the data these ranges imply make the models harder?).
- Goal 3: say specifically what, if anything, degrades, and by how much.

## Non-Goals
- Non-goal 1: changing any source. This task is a measurement.
- Non-goal 2: re-tuning the clamps, the iteration caps or the tolerances. Findings only.
- Non-goal 3: running at production *scale*. The ranges are the question; ~40 000 patients
  would be a cost study, which is a different task.

## Method

Two instruments, because the question has two halves.

### 1. A production-range cohort, through the CSV ingestion path

A throwaway generator writes the two owners' halves of `cdrcatsse_match_pcc` over the
production ranges and feeds them to `-D`. No C++ changes are needed: the CSV path already
exists for exactly this.

It also writes `flagged_dx_owner_{a,b}.csv` by mirroring `etl.h:DeriveFlagged`, because the
plaintext oracle reads those rather than the base table. The mirror is self-checking: if it
had drifted from the C++, the counting nodes would stop matching. They match exactly —
624 / 686 rows, 1310 rows and 387 / 250 / 227 patients in the three cohorts, identical on
both sides.

### 2. Magnitude-only transforms

Changing a generator parameter re-rolls the whole RNG stream, so two runs differ in every
row and nothing can be attributed. A second script instead rewrites **one already-generated
cohort** so that only the magnitude of a column changes, leaving every statistic the pipeline
fits invariant:

- `--date-offset` adds a constant to `admit_date_pcc` **and** `dob`. Every gap is unchanged,
  so `visit_num`, `index_visit`, `final_visit` and `fu_month` are unchanged; `newage` is a
  difference, so it is unchanged too.
- `--subject-id-span` remaps `subject_id` onto an order-preserving subset of a wider range.
  Grouping and sort order are identical.
- `--age-shift-years` moves `dob` so every recorded age rises by a constant. `newage` enters
  the design linearly and centred at 40, so the age coefficient is invariant and the intercept
  must move by exactly `-shift * beta_age`. The outcome column is untouched.

Anything that moves under one of these is a fixed-point artefact and nothing else.

## Results

All runs on blinky, `PROTOCOL=1`, scored with
`scripts/testing/validate_mpc_analysis.py --compare` at the tolerances tasks/0009 established
(0.25 in `-logL` for the twelve mixed models, 10% relative on standard errors, 1e-3 on the
IRLS coefficients).

### Magnitude is not the problem

| transform | range after | RESULT lines vs control |
|---|---|---|
| `--date-offset 45586` | `encounter_dt` 45 588 .. 47 713 | **all 75 bit-identical** |
| `--subject-id-span 1000:40000` | `subject_id` 1 259 .. 39 639 | **all 75 bit-identical** |
| `--age-shift-years 20` | `newage` 34 .. 101 | see below |
| all three together | — | identical to the age shift alone |

The age shift is the one that must move, because it changes a covariate. It moves exactly as
theory requires. Model 6a, fitted by IRLS:

| term | control | +20 years |
|---|---|---|
| Intercept | -1.34239197 | -1.16203308 |
| visit_num | 0.146118164 | 0.146118164 |
| newage | -0.00901794434 | -0.00901794434 |
| gender=female | -0.161911011 | -0.161911011 |
| gender=male | -0.121276855 | -0.121276855 |
| hispanic=0 | 0.319244385 | 0.319244385 |
| data_source=UMass | 0.202163696 | 0.202163696 |
| data_source=UMass * visit_num | 0.0151977539 | 0.0151977539 |

Every coefficient is bit-identical but the intercept, and the intercept moves by
`-20 * beta_age = +0.18036` — `-1.34239197 + 0.18036 = -1.16203`, matching to every digit
printed. The fit is *exactly* equivariant; the same model, re-parameterised.

**Conclusion: `subject_id` at ~40 000, `encounter_dt` at ~46 000 and `newage` at ~100 are
numerically free.** The sort keys, the date arithmetic and the centred age column all have
ample headroom at `precision = 16` on `int64_t`.

### What does change: the outcome rate, through the visit-number tail

| cohort | rows | SISA rate | max visits | `--compare` |
|---|---|---|---|---|
| in-binary generator, `-r 200` (the published baseline) | 496 | 33.7% | 15 | **PASS** |
| in-binary generator, `-r 400` | 943 | 33.0% | 15 | FAIL — 5a/any 1.46, 5b/any 0.75 |
| production-range generator, old ranges, 200 patients | 560 | 38.0% | 16 | FAIL — 3 criteria |
| production-range generator, old ranges, 400 patients | 1117 | 37.5% | 16 | FAIL — 7 criteria |
| production ranges, 200 patients | 690 | 47.8% | 85 | FAIL — 8 criteria |
| production ranges, 400 patients | 1310 | 45.3% | 88 | FAIL — 13 criteria |

Two effects, and only one of them is about the ranges.

**(a) A pre-existing, size-dependent one.** The repository's *own* generator, at its *own*
ranges, passes at 200 patients and fails at 400 — `5a/any` stops 1.46 log-likelihood units
above the plaintext optimum against a 0.25 tolerance, `5b/any` 0.75. Nothing about the
production data is involved. tasks/0009 validated at `-r 200`; this had not been checked at
400. `MinimizeBFGS` is capped at `kGlmmBfgsIterations = 12` (`regression.h:625`) while the
objective's magnitude grows with n, so the fixed budget buys proportionally less as the
cohort grows. 9 of 12 mixed models hit the cap on the production data; 8 of 12 did at the
published baseline, so the cap binding is not itself new — how far short it leaves the fit is.

**(b) A range-dependent one, via the linear `visit_num` term.** `visit_num` enters the model
as a continuous linear term (lineage fault 4 makes the same point about `fu_month`). Pushing
the tail from 16 to ~100 multiplies its contribution to the linear predictor by six, which
raises the outcome rate from ~34% to ~46%, and on the UMass subset it pushes `eta` past the
public clamp `Exp` and `Sigmoid` depend on:

| model | population | rows | max visit_num | max \|eta\| | visit_num at the clamp | rows clamped |
|---|---|---|---|---|---|---|
| 2a | any_system | 1310 | 88 | 9.53 | 91.9 | 0 |
| 2a | umass_system | 638 | 59 | **11.48** | 52.1 | **7 (1.1%)** |
| 2a | nonumass_system | 672 | 72 | 7.11 | 97.6 | 0 |
| 5a | any_system | 1310 | 88 | 7.52 | 113.5 | 0 |
| 5a | umass_system | 638 | 59 | **10.87** | 54.6 | **5 (0.8%)** |
| 5a | nonumass_system | 672 | 72 | 7.56 | 92.0 | 0 |

`eta` is computed at each model's own fitted slope; the clamp is `ClampAbs(eta, 10)`. On the
UMass subset the slope is steepest (0.21 for 2a), so `eta` crosses 10 at about 52 visits and
1.1% of rows saturate. `ClampAbs` flattens them, `Sigmoid` returns its saturation value, and
`W = p(1-p)` truncates to 0 at `precision = 16` — so those rows contribute nothing to the
Gram matrix and nothing to the gradient. This is precisely the quasi-separation failure mode
tasks/0009's risk register names, reached for the first time by the visit-number range rather
than by an interaction term.

The pooled and non-UMass tables stay inside the clamp, but `any_system` peaks at 9.53 against
a limit of 10 — about 4% of headroom, on this cohort, at this slope.

### Confirming the mechanism: raise the iteration cap, and the failures collapse

A diagnostic build with `kGlmmBfgsIterations` at 40 instead of 12, on the same production
cohort (`data400`, 1310 rows), nothing else changed:

| fit | `-logL` excess, cap 12 | cap 40 | tolerance |
|---|---|---|---|
| 5a any_system | 2.393 | **0.666** | 0.25 |
| 5a umass_system | 0.566 | **0.018** | 0.25 |
| 5a nonumass_system | 1.169 | **0.004** | 0.25 |
| 5b any_system | 2.974 | **0.016** | 0.25 |
| 5b umass_system | 0.004 | 0.003 | 0.25 |
| 5b nonumass_system | 1.943 | **0.022** | 0.25 |
| 2a any_system | 0.033 | 0.003 | 0.25 |

Mixed models reaching the cap: **9 of 12 at cap 12, 1 of 12 at cap 40**. `-logL` failures:
**5 at cap 12, 1 at cap 40**. The optimiser was simply stopping early; the objective, the
gradient and the fixed-point kernels underneath it were fine all along.

The same build on the repository's *own* generator at `-r 400` — effect (a), which has
nothing to do with the production ranges — goes all the way back to green:

| fit | cap 12 | cap 40 |
|---|---|---|
| 5a any_system `-logL` excess | 1.462 | **0.003** |
| 5b any_system `-logL` excess | 0.748 | **0.012** |
| worst relative SE difference | 10.7% | **8.5%** |
| `--compare` verdict | FAIL | **PASS** |

So the `-r 400` regression is entirely the iteration cap, and raising it restores the
published result at the larger cohort size.

What survives the larger budget is the genuinely range-attributable residual, and it is
small: five standard errors at 11-13% against a 10% relative bound, model 6a's coefficients
at 1.33e-3 against 1e-3, and `5a/any` still at 0.666. These are the fixed-point floor moving
under a higher outcome rate and a larger `|eta|` — the saturation effect above — not a
failure of the arithmetic.

### What did NOT degrade

- Every relational and aggregate node is **exact**. `sisa_perct_cnt` and the patient counts
  agree cell for cell with the SQL/Python oracle on the production ranges.
- Ingestion, the oblivious merge and both sequencing passes handle the ranges without
  comment: 1310 rows, padded to 2048, 387 / 250 / 227 patients, matching the oracle.
- Models **2a and 2b** (unadjusted, no covariates) pass on the production ranges at both
  sizes — worst `-logL` excess 7.2e-3 against 0.25.
- Model **6b** passes on the production data (8.3e-5 against 1e-3). **6a** lands at 1.3e-3,
  marginally over.

## Findings, in order of what they should change

1. **The production value ranges are numerically safe.** Bit-identical results under
   date- and subject-id magnitude shifts; exact equivariance under an age shift. No clamp,
   no scale factor and no accumulator needs re-sizing for `subject_id`, `encounter_dt` or
   `newage`.
2. **`ClampAbs(eta, 10)` is no longer comfortably sized.** It is crossed today on the UMass
   subset by ~1% of rows, and the pooled table sits 4% below it. The clamp exists to keep
   `Exp` inside its saturation band, so raising it is not free; the honest options are to
   raise `kMaxExpArg` together with the series terms that support it, to standardise
   `visit_num` the way `newage` is already centred, or to report the saturated-row count
   alongside each fit so a reader can see it happening. The third is cheap and should
   probably happen regardless.
3. **`kGlmmBfgsIterations = 12` is the binding constraint on the mixed models, and it binds
   harder as the cohort grows.** Demonstrated directly: at a cap of 40 the `-logL` failures
   on the production data drop from five to one and eleven of twelve fits converge. This is
   pre-existing — it shows up on the repository's own generator at `-r 400` — but
   production-shaped data reaches it sooner. The published "VALIDATION: PASS" is specific to
   `-r 200`. Raising the cap costs roughly linearly in wall clock (one gradient plus two
   objective evaluations per iteration), so this is a budget decision, not a redesign.
4. **The `--compare` tolerances were calibrated at one cohort size.** A 0.25 absolute bound
   on `-logL` excess does not scale with n; at 1310 rows the same *relative* quality of fit
   reads as a failure. Whether the right answer is a per-row tolerance or a larger iteration
   budget is a judgement call, not a measurement.

## Scope
- In scope: this document and the `tasks/tasks.md` ledger line. Nothing else — no source
  change to the pipeline, to `include/`, or to anything the binary compiles.
- Out of scope, deliberately: every source change implied by the findings above.

> **Note on the instruments.** The two generators described under Method were deliberately
> **not** landed in the tree: they are scaffolding for one measurement, not something to
> maintain. They live alongside the run artefacts in
> `/scratch/adam/CryptDough-orch/runs-real/` on blinky, together with every `models-*.txt`
> and `validate-*.txt` the tables below were read off. Reproducing the findings means going
> there, or writing the generators again from the Method section, which describes them
> completely.

## Risks and Mitigations
- **Risk:** the Python mirror of `DeriveFlagged` drifts from the C++ and the whole comparison
  becomes self-referential.
  - Mitigation: the counting nodes are exact integers computed on both sides independently.
    They agree exactly, which they could not if the mirror had drifted.
- **Risk:** attributing a difference to a range when it is really a different random draw.
  - Mitigation: the magnitude-only transforms, which hold the draw fixed. Every claim about
    magnitude rests on those, not on the generator comparisons.

## Validation Plan
- Automated checks: `scripts/testing/validate_mpc_analysis.py --compare` on every cohort.
- Manual verification: the equivariance check on the age shift, computed from the printed
  coefficients.
- Expected success criteria: identify what moves and what does not, with the mechanism.

## Definition of Done
- The pipeline has been run end to end on the production ranges. **Done.**
- Numeric magnitude has been separated from statistical shape. **Done.**
- The degradations are named, quantified and traced to a mechanism. **Done.**
- No source changes. **Done.**

## Approval
- [x] Findings reviewed
- [x] Approved
- Approver: Adam Godel
- Approval date: 2026-09-11

## Change Log
- 2026-09-11: Initial findings.
