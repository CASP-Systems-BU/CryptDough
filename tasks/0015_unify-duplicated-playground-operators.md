# Task 0015 — Unify the duplicated playground operator libraries

## Metadata
- Task ID: 0015
- Title: Unify the duplicated playground operator libraries (`playground/` vs `playground/library/`)
- Requested by: Adam Godel
- Owner: Adam Godel
- Date: 2026-09-21
- Status: In Progress — `Exp` fixed in `library/` under an approved one-line exception; suite green (104/104); stage 1 underway
- Estimated effort: Medium–Large (the merge is mechanical; the optimizer migration is not)
- Target completion date: TBD
- Related issue or PR: —
- Related documents: `tasks/0009_mpc-analysis-pipeline.md`, `tasks/0010_secure-matrix-inversion-newton-schulz.md`, `tasks/0002_taylor-series-math-approximations.md`
- Branch: `logistic-regression`

## Problem Statement

`playground/` currently contains **two parallel operator libraries** with the same three
filenames:

| Header | Root copy | `library/` copy |
| --- | --- | --- |
| `primitives.h` | 839 lines | 658 lines |
| `optimizer.h` | 671 lines | 881 lines |
| `regression.h` | 851 lines | 807 lines |

They are not stale copies of one another — both were actively developed, on two branches,
after a common ancestor.

### How this happened

- `cc50a11` (2026-09-10) is the common ancestor.
- On `logistic-regression`, commit `3c2c445` *moved* the three headers to `playground/library/`
  (a pure rename, 0 content changes) and continued developing them there (`d9c7da4`, `d561ec3`).
- On `orchestration`, the same three headers stayed at `playground/` root and continued being
  developed there (`db09dc6`, `d3f68ac`, `ea66362`).
- The merge `f5b7bec` brought both branches together. Because git saw *different paths*, there
  was **no conflict** — the duplication landed silently.

### Current usage split

**`library/` operators are used by 7 targets, but not by the production pipeline:**

- `plain-lr.cpp`, `long-lr.cpp`, `secure-logistic-regression.cpp` (standalone drivers)
- all four test binaries: `tests/test_playground_{primitives,optimizer,fixed_effects,mixed_effects}.cpp`
- `tests/profile_playground_primitives.cpp`, `tests/test_util.h`

**The root operators are used by the MPC analysis pipeline, exclusively:**

`mpc-analysis.cpp` → `harness.h`, `output.h`, `reporting.h`, `secure.h`
→ `regression.h`, `sqlite_oracle.h`, `cohort.h`, `linalg.h`, `nodes.h`, `optimizer.h`,
`primitives.h`, `etl.h`, `segmented.h`

So the answer to "to what extent are the `library/` operators used" is: **they carry the entire
test suite and the three standalone LR drivers, and none of the actual MPC analysis pipeline.**
The pipeline that produces the 17 terminal outputs of task 0009 runs on the root copies, which
have **no unit tests at all**.

## The complication: the divergence runs in both directions

The naive reading of the request — "switch to the `library/` versions and delete the old ones" —
would **regress both correctness and performance**. Of the operators present in both copies:

Identical (safe, pure duplicates): `Clone`, `Log1p`, `LogOnePlusExp`, `Sum`, `AsColumnWise`,
`FrobeniusNormSquared`, `Identity`, `ScaledIdentity`, `ScaleMatrix`, `Transpose`, `TransposeData`.

Divergent:

| Operator | Root copy | `library/` copy | Ahead |
| --- | --- | --- | --- |
| `Exp` | has the `kExpRoundBias` floor fix; 5 series terms | **missing the fix**; 3 series terms | **root** |
| `Log` | `RecipSeeded` division; 5 series terms | `a2b`→boolean `/`→`b2a`; 3 terms | **root** |
| `Sigmoid` | `RecipSeeded` division | `a2b`→boolean `/`→`b2a` | **root** |
| `ClampAbs` | `Clone(x)` | explicit copy-construct | equivalent |
| `NewtonSchulzInverse` | hand-rolled `MatMul` | `matrixRightMultiplyWithColumnMatrixVectorized` | **`library/`** |

Two specific regressions that a literal "use the `library/` version" would reintroduce:

1. **An `Exp` correctness bug.** The root copy documents it explicitly: `div_const_a` truncates
   toward zero rather than toward minus infinity, so for `-1 < x/ln2 + 1/2 < 0` the nearest
   integer came out `0` instead of `-1`, `r` escaped `[-ln2/2, ln2/2]`, and **`exp(-1)` returned
   exactly `1/3`**. Root fixes this with a public `kExpRoundBias` before the shift, removed after.
   `library/primitives.h` does not have this fix.
2. **A large performance regression.** Root replaced `BSharedVector::operator/`
   (`circuits.h:39`) with `RecipSeeded` in both `Sigmoid` and `Log`. The root comment records
   that the boolean division was **79% of `Sigmoid`'s cost**, and that since `Sigmoid` runs six
   times per objective evaluation on every row, it was **the largest single cost in the program**.
   `library/` still carries the boolean division.

Conversely, `library/` is genuinely ahead in ways root cannot match:

- **Batched/SIMD optimizer**: `MinimizeBFGSBatched`, `NumericalGradientBatched`,
  `NumericalHessianBatched`, `BfgsInverseUpdateBatched`, `MatVecBatched`, `OuterProduct`,
  `MakeCentralDifferenceSelector`, `MakeLineSearchConstants` — a single packed `AV` across all
  points, versus root's per-element `std::vector<AV>`.
- **Vectorized matrix multiply** inside `NewtonSchulzInverse`.
- **Inference statistics** root has none of: `NormalCdf`, `TwoSidedPValue`, `WaldStatistics`,
  `OddsRatio`, `StandardErrors`, `Covariance`, `SeparationFlag`, `CovarianceFromInformation`.
- **`Multiplex`, `AnyAbsAtLeast`, `SecureSqrt`, `SecureReciprocal`.**
- **The whole test suite.**

And root has ~50 call sites of sharing/opening plumbing that `library/` has **no equivalent for
at all**: `ShareDoubles` (12), `OpenRawToParty` (14), `OpenScalarToParty` (10),
`OpenToDoubles` (12), `OpenToPartyDoubles` (2), `OpenScalar` (3), plus `MakeVector`,
`MakeMatrix`, `RandomRingVector`, `Div`, `Recip`, `Rsqrt`, `SqrtBoth`, `Abs`, `ClampRange`.

## Goals
- No operator defined twice in `playground/`. Where two versions exist, `library/`'s wins and
  root's is deleted.
- `mpc-analysis.cpp` and its tree call `library/` for every operator `library/` defines.
- `library/` is left bit-for-bit unchanged, and its test suite stays green.
- Every numerical movement in the pipeline's 75 `RESULT` lines is accounted for by a named cause.

## Non-Goals
- Rewriting the MPC pipeline's ETL, cohort, or SQL stages.
- Changing pipeline numerical output beyond what removing the duplication requires.
- Migrating `logistic-regression.cpp` (it uses `include/simulated_float.h`, a separate lineage).

## Scope
- In scope: `playground/{primitives,optimizer,regression}.h`, `playground/mixedeffects.h`, and
  the include lines of the pipeline headers.
- **Explicitly out of scope: `playground/library/*`.** It is frozen. Not one line changes.
- Out of scope: `cohort.h`, `segmented.h`, `etl.h`, `nodes.h`, `linalg.h`, `sqlite_oracle.h`,
  `secure.h` internals — these are pipeline-specific and not duplicated. They only need their
  `#include "./primitives.h"` lines repointed.

## Dead code found along the way
- **`playground/mixedeffects.h` (237 lines) is unreachable** — nothing includes it. It is
  superseded by the `mixedeffects` namespace inside `library/regression.h`. **Confirmed for
  deletion** as part of stage 2.

## Impact Assessment
- User impact: none externally; single build tree internally.
- Performance impact: **a pipeline slowdown is expected and accepted.** `Sigmoid` and `Log` move
  onto `library/`'s boolean division, which root measured at 79% of `Sigmoid`'s cost, and
  `Sigmoid` is the program's dominant cost. Partly offset by `library/`'s vectorized
  `NewtonSchulzInverse` and batched optimizer. Net effect to be measured, not predicted.
- Security and privacy impact: none by construction — same protocols, same reveal points. The
  division choice is a cost-and-accuracy question, not a leakage one.
- Backward compatibility impact: `mpc-analysis` output will **not** be byte-identical. The `Exp`
  truncation behaviour and the 5→3 series-term reduction both move the numbers. The acceptance
  test is that every movement traces to one of those two causes.

## Alternatives

### Option A: Merge best-of-both into `library/`, then migrate the pipeline and delete the root copies
Take `library/` as the destination. Port the root wins into it first (the `kExpRoundBias` `Exp`
fix, `RecipSeeded` + its constants, 5-term series, and the whole sharing/opening plumbing block),
keep `library/`'s batched optimizer and vectorized matmul, then repoint the pipeline headers and
delete the three root copies plus dead `mixedeffects.h`.

Pros:
- Ends with exactly one library, which is the stated goal.
- No correctness or performance regression: both root wins are carried over explicitly.
- Pipeline operators inherit the existing test suite.
- The batched optimizer becomes reachable from the pipeline as a follow-up.

Cons:
- Largest single change; touches the pipeline's whole include tree.
- The optimizer migration is not mechanical: root's `ValueGradFn`
  (`std::function<AV(const std::vector<AV>&, std::vector<AV>*)>`) and `library/`'s
  `BatchedObjective` (`std::function<AV(const AV&, size_t)>`) are different data layouts.
  `MinimizeBFGS` (4 sites), `NumericalGradient` (3), `BfgsInverseUpdate` (2), `Dot` (4),
  `MatVec` (2), `MatMul` (3) all need reworking in `regression.h`'s objectives.
- Requires a numerical equivalence run against task 0009's 17 outputs to prove no drift.

### Option B: Merge the primitives only; keep both optimizers for now
Unify `primitives.h` (where the duplication is most dangerous — `Exp`/`Log`/`Sigmoid`) and leave
`optimizer.h`/`regression.h` duplicated until the batched migration is scheduled separately.

Pros:
- Removes the actual correctness hazard (`exp(-1) = 1/3`) and the performance hazard immediately.
- Much smaller, much lower risk; easy to verify.
- Does not block on the `std::vector<AV>` → packed-`AV` rework.

Cons:
- Does not fully satisfy the request — two `optimizer.h` and two `regression.h` remain.
- Leaves a second, smaller divergence to manage.
- `ModelData` and `Dataset`/`BatchedDataset` layouts stay unreconciled.

### Option C: Promote root to be the single library, port `library/`'s additions into it
Keep root as the destination since the pipeline is the production artifact, and port `library/`'s
batching, statistics, and tests into it.

Pros:
- The production pipeline does not move at all — lowest risk to task 0009's outputs.
- Root's performance-tuned primitives stay authoritative by default.

Cons:
- Fights the direction the `logistic-regression` branch already took (`3c2c445` deliberately
  moved these into `library/`).
- Three drivers and all four test binaries would need repointing instead of one pipeline.
- `library/`'s batched optimizer is the larger and more valuable body of code; re-homing it is
  more work than re-homing root's primitives.

### Recommendation (superseded — see Decisions below)

Option A as originally written proposed porting root's `Exp`/`Log`/`Sigmoid` wins *into*
`library/`. That is no longer the plan: see Decisions.

### Decisions (confirmed 2026-09-21)

1. **`library/` is frozen. No active changes to it.** The `Exp` truncation bug and the
   `a2b`→boolean-`/`→`b2a` division in `Log`/`Sigmoid` stay exactly as they are in `library/`.
   The pipeline migrates *onto* them.
2. **Both stages now** — primitives, then optimizer + regression.
3. **Three series terms everywhere.** `kMaxSeriesTerms = 3`, which is what `library/` already
   uses. Root's `kExpSeriesTerms`/`kLogSeriesTerms` = 5 are dropped with root's copies.
4. **Verification on blinky**, against a baseline captured from current `HEAD` before any
   source change.

#### Accepted consequences of decision 1

These are understood and accepted, not oversights:

- The pipeline inherits the `Exp` truncation behaviour, including `exp(-1) → 1/3` at the
  boundary case root's `kExpRoundBias` was added to fix.
- The pipeline inherits the boolean division in `Sigmoid` and `Log`. Root's own measurement put
  that division at **79% of `Sigmoid`'s cost**, with `Sigmoid` the largest single cost in the
  program at six calls per objective evaluation per row. **A substantial pipeline slowdown is
  expected.** It will be measured with `profile_playground_primitives` and reported, not guessed.
- Both effects will move the pipeline's numbers, so the baseline diff is expected to be
  **non-empty**. Its job is to show the movement is confined to what these two changes explain,
  not to show bit-identity.

#### What this means for scope

The request is scoped to operators that exist in **two** versions. Operators that exist in only
**one** version are not duplicates and are therefore out of scope for removal. That splits the
root headers cleanly:

**Removed from root, pipeline switched to `library/` (duplicated):**
`Clone`, `Log1p`, `LogOnePlusExp`, `Sum`, `ClampAbs`, `Exp`, `Log`, `Sigmoid`, `SecureReciprocal`,
`ClampNewtonStep`, `Transpose`, `TransposeData`, `AsColumnWise`, `FrobeniusNormSquared`,
`Identity`, `ScaledIdentity`, `ScaleMatrix`, `NewtonSchulzInverse`, and — as the batched
generation of the same operators — `MinimizeBFGS`, `NumericalGradient`, `MatVec`,
`BfgsInverseUpdate`, plus root's whole GLMM/IRLS implementation (`FlatConditionalMode`,
`FlatObjective`, `FlatNegMarginalLogLik`, `ObservedInformationSE`) in favour of
`library/regression.h`'s `mixedeffects::` and `logistic::` namespaces.

**Kept at root (single version, pipeline-specific — `library/` has no equivalent):**
the party I/O plumbing `ShareDoubles`, `OpenToDoubles`, `OpenScalar`, `OpenRawToParty` (`AV` and
`BV`), `OpenToPartyDoubles`, `OpenScalarToParty`; the construction helpers `MakeVector`,
`MakeMatrix`, `RandomRingVector`; `Abs`, `ClampRange`; and the cohort→design glue `TimeAxis`,
`ModelSpec`, `BuildDesign`, `ColumnGather`, `LinearPredictor`, `CrossProduct`, `FitResult`.

`Div`, `Recip`, `Rsqrt`, `SqrtBoth` are the ambiguous middle: `library/` has `SecureReciprocal`
and `SecureSqrt`, which are different implementations of the same operations. These count as
duplicates, so the pipeline's 3 `Recip`, 3 `Div`, 1 `Rsqrt` and 1 `SqrtBoth` call sites move to
`library/`'s versions, and `RecipSeeded` — which exists only to serve them and root's now-removed
`Sigmoid`/`Log` — is removed with them.

The end state is therefore **not** a single header. It is `library/` untouched and authoritative
for every operator it defines, plus a thin root layer holding only what is unique to the
pipeline, which `#include`s `library/` rather than shadowing it.

## Stage 0 findings — measured on blinky, 2026-09-21

Built `HEAD` (`f5b7bec`) at `/scratch/adam/CryptDough-unify`, `PROTOCOL=1`, reusing the prebuilt
`libOTe-install`/`blaze-install` from `CryptDough-orch/build`. Baseline artefacts are in
`/scratch/adam/CryptDough-unify/runs-verify/`.

### The pipeline baseline is good
`mpc-analysis -S models -D runs-verify/ctrl -ra 549 -rb 568` → **75 `RESULT` lines**, 4m13.9s
wall. Captured as `BASELINE-models.txt`.

### The `library/` test suite is already RED on `HEAD`

This was not known when the decisions above were taken. Three of four binaries **abort**:

| binary | failing check | tolerance | worst error | verdict |
| --- | --- | --- | --- | --- |
| `test_playground_primitives` | `Exp` | 1.0e-3 rel | **3.45e-2** | FAIL, 3 of 10 elements |
| `test_playground_fixed_effects` | `NegLogLikBatched (K = 2)` | 1.5e-2 abs | **8.00e-2** | FAIL |
| `test_playground_mixed_effects` | `ConditionalModeBatched` | 6.0e-4 abs | **9.47e-3** | FAIL |
| `test_playground_optimizer` | — | — | — | all 35 pass |

### The cause is the `library/` `Exp` bug, confirmed empirically

`test_playground_primitives` reports, for `library/primitives.h`'s `Exp`:

```
i=1  input=-2.00000000  expected=0.13533528  actual=0.13401794  error=1.317e-03
i=2  input=-1.00000000  expected=0.36787944  actual=0.33334351  error=3.454e-02
i=3  input=-0.50000000  expected=0.60653066  actual=0.60417175  error=2.359e-03
```

`actual = 0.33334351` at `x = -1` is **1/3**, precisely the symptom root's `kExpRoundBias`
comment documents: the truncation-toward-zero makes `r` escape `[-ln2/2, ln2/2]` and the series
is evaluated at `r = -1`.

A probe running **root's** `Exp` over the same ten inputs (scaffolding, kept at
`/scratch/adam/CryptDough-unify/playground/tests/probe_exp.cpp`, deliberately not landed in the
tree) passes the same 1e-3 tolerance everywhere:

| input | expected | root `Exp` | root rel err | `library/` rel err |
| --- | --- | --- | --- | --- |
| -4.0 | 0.01831564 | 0.01831055 | 2.78e-4 | — |
| -2.0 | 0.13533528 | 0.13533020 | **3.76e-5** | **1.32e-3** FAIL |
| -1.0 | 0.36787944 | 0.36787415 | **1.44e-5** | **3.45e-2** FAIL |
| -0.5 | 0.60653066 | 0.60650635 | **4.01e-5** | **2.36e-3** FAIL |
| 0.0 | 1.0 | 1.0 | 0 | — |
| 1.0 | 2.71828183 | 2.71823120 | 1.86e-5 | — |
| 8.0 | 2980.95799 | 2981.00000 | 1.41e-5 | — |

Root's worst error across the range is **2.78e-4**; `library/`'s is **3.45e-2**, two orders of
magnitude worse, and over tolerance at three of ten points.

The two remaining failures are downstream of this: `NegLogLikBatched` and
`ConditionalModeBatched` both reach `Exp` through `Sigmoid` and `LogOnePlusExp`.

### Why this blocks the migration as specified

`Sigmoid` runs six times per objective evaluation on every row, across all 14 models. Moving the
pipeline onto `library/`'s `Exp` therefore injects a **3.5e-2 relative error at `x = -1`** into
every fit in a clinical analysis pipeline, and does so knowingly. The accepted-consequences note
above treated this as an edge case; it is not — it is the operator's behaviour in the middle of
its working range, and `library/`'s own test suite fails on it today.

**Stage 1 is therefore paused pending a decision.** Nothing has been changed in the source tree.

### Resolution: the one-line exception, applied and verified

Approved exception to the `library/` freeze: add `kExpRoundBias` and the two lines that apply and
remove it. **Three lines of code.** The boolean division in `Log`/`Sigmoid` and
`kMaxSeriesTerms = 3` are untouched, as directed. Diff is 12 insertions, 1 deletion, one file.

The whole suite goes green — one cause, three failures:

| check | tolerance | before | after | improvement |
| --- | --- | --- | --- | --- |
| `Exp` | 1.0e-3 rel | **3.45e-2 FAIL** | 2.99e-4 PASS | 115x |
| `NegLogLikBatched (K = 2)` | 1.5e-2 abs | **8.00e-2 FAIL** | 2.64e-3 PASS | 30x |
| `ConditionalModeBatched` | 6.0e-4 abs | **9.47e-3 FAIL** | 8.66e-5 PASS | 109x |

All four binaries exit 0: **104 checks, 0 failures** (primitives 36, optimizer 34, fixed effects
19, mixed effects 15). Artefacts: `runs-verify/EXPFIX-test_playground_*.txt`.

This gives the migration a green gate to verify against, which it did not have before.

## Implementation Plan

### Stage 0 — baseline (done)
1. Sync `HEAD` (`f5b7bec`) to `/scratch/adam/CryptDough-unify` on blinky; reuse the prebuilt
   `libOTe-install` / `blaze-install` from `CryptDough-orch/build`.
2. `cmake -DPROTOCOL=1`, build `mpc-analysis` + `playground-tests`.
3. Capture `BASELINE-models.txt` (75 `RESULT` lines) on the `ctrl-old` dataset (`-ra 549 -rb 568`)
   and `BASELINE-test_playground_*.txt`, plus a `profile_playground_primitives` cost baseline.

### Stage 1 — primitives
4. Strip from root `primitives.h` every operator `library/primitives.h` already defines, and the
   `RecipSeeded`/`Div`/`Recip`/`SqrtBoth`/`Sqrt`/`Rsqrt` family with their `kRecip*`/`kRsqrt*`/
   `kDivDen*`/`kSqrtArg*`/`k*SeriesTerms` constants.
5. Repoint the remainder at `./library/primitives.h` and rename it to reflect what it now is
   (pipeline-only support), keeping `namespace cdough::regression`.
6. Rewrite the 8 `Recip`/`Div`/`Rsqrt`/`SqrtBoth` call sites onto `SecureReciprocal`/`SecureSqrt`.
7. Repoint `cohort.h`, `linalg.h`, `etl.h`, `segmented.h`, `nodes.h`, `output.h`, `optimizer.h`.
8. Build; run the four test binaries and `profile_playground_primitives`; run the pipeline and
   diff against `BASELINE-models.txt`. Record the slowdown.

### Stage 2 — optimizer + regression
9. Add a `ModelData` → `mixedeffects::BatchedDataset` / `logistic::Dataset` adapter.
   `ModelData` already holds `x`/`y`/`row_mask` as row-major `AV`s, so this is a projection, not
   a rewrite.
10. Move `FitGlmmLaplace` onto `mixedeffects::NegMarginalLogLikBatched` +
    `MinimizeBFGSBatched` + `mixedeffects::Covariance`; move `FitLogisticIrls`'s inference onto
    `logistic::Covariance` / `StandardErrors`.
11. Delete root `optimizer.h`, root `regression.h`'s duplicated half, and the unreachable
    `mixedeffects.h`.
12. Rebuild, rerun all tests and the pipeline, diff, and account for every moved number.

## Stage 2 finding — the mixed-effects half cannot move to `library/`

Stage 2 as planned was to put the pipeline's GLMM fits onto
`library/regression.h`'s `mixedeffects::` namespace. **That is not possible without undoing a
deliberate design decision**, and the evidence is a note the pipeline's own author left in
`primitives.h`:

> NOTE: the balanced ClusterGroup / Dataset representation that used to live here has been
> replaced by the flat, ragged-cluster layout further down (SecureCohort + ModelData +
> FlatNegMarginalLogLik). Real patients have a ragged number of encounters, which the balanced
> form could not express without padding every patient to the largest cluster.

`library/regression.h`'s `mixedeffects::BatchedDataset` **is** that balanced representation:
`num_groups x obs_per_group`, a rectangle, asserting `x_data.size() == groups * obs * fixed`. The
pipeline moved off it on purpose. Migrating back is a regression in both capability and cost.

### Measured cost of going back

Cluster sizes in the control dataset (`runs-verify/ctrl`, both owners):

| statistic | value |
| --- | --- |
| patients | 400 |
| real rows | 1257 |
| mean cluster | 3.14 |
| p95 cluster | 12 |
| **max cluster** | **16** |
| balanced requirement | 400 x 16 = **6400 rows** |

That is **5.1x the real rows** and **3.1x the flat layout's 2048 padded rows** — on the twelve
mixed models, which are the bulk of the pipeline's work. A heavier tail in the production data
makes it worse, since the padding is set by the maximum, not the mean.

### What this means

- The twelve mixed models keep root's flat ragged implementation (`FlatObjective`,
  `FlatConditionalMode`, `FlatNegMarginalLogLik`, `ObservedInformationSE`) and root's
  `MinimizeBFGS` over `std::vector<AV>`. These are **not** duplicates of the `mixedeffects::`
  functions; they are a different algorithm over a different data layout that exists because the
  library's layout was insufficient.
- `MinimizeBFGSBatched` is likewise not a second spelling of `MinimizeBFGS`. Its `BatchedObjective`
  packs several candidate parameter vectors into one `AV` to amortise MPC rounds, which presumes
  the balanced layout. Keeping both is correct.
- Models 6a/6b (fixed effects, IRLS) have no clusters, so `logistic::Dataset`
  (`num_groups == 1`) does fit them. That migration is feasible and is left as a follow-up
  rather than bundled here, because it changes how two of the fourteen models compute their
  inference and deserves its own before/after comparison.

So the deduplication that is actually available in stage 2 is the matrix-helper layer, which was
done, not the optimiser or the GLMM.

## Stage 2c — models 6a/6b migrated onto `logistic::` (requested 2026-09-21)

The two fixed-effects models have no clusters, so the raggedness problem that blocks the other
twelve does not apply: `logistic::Dataset` is `mixedeffects::BatchedDataset` with
`num_groups == 1`, which is the pipeline's flat buffer already. Nothing is regrouped and nothing
is padded to the largest cluster.

### The adapter

`AsLogisticDataset(const ModelData&)` in `regression.h`. `md.x` (row-major `n_pad x p`), `md.y`
and `md.row_mask` map straight across. **The one real conversion is the mask convention**, and it
would have been silently wrong:

- `ModelData::row_mask` is a **raw 0/1** indicator — `BuildDesign` multiplies by it with no
  following `/ scale`.
- `library/regression.h`'s `ApplyMask` computes `(v * m) / scale`, so it needs a **fixed-point
  1.0**.

Passing the raw mask through unchanged would have divided every masked value by `scale`,
i.e. zeroed the design. The adapter multiplies by `scale` explicitly.

### What was migrated, and what deliberately was not

**Migrated — the inference.** `logistic::Covariance` + `logistic::StandardErrors` replace the
local `SymmetricInverse(gram, p)` plus a plaintext `std::sqrt` of the diagonal. Two consequences:

- The covariance is now `(X'WX/n)^-1 / n` by Newton-Schulz instead of a Cholesky inverse of the
  **ridge-augmented** Gram. `kIrlsRidge` is an IRLS damping term; carrying it into the reported
  covariance shrank the standard errors slightly, and it is gone now. Expect 6a/6b standard errors
  to move a little, in the direction of being slightly larger and more correct.
- The square root is taken in MPC by `SecureSqrt`, so the fit **opens `p` standard errors instead
  of the whole `p x p` covariance matrix**. Strictly fewer values leave the computation than
  before — a privacy improvement, not just a deduplication.

**Not migrated — the fit itself.** The IRLS loop and its Cholesky solve stay. `FitLogisticIrls`
carries a documented property the twelve mixed models do not have: *"The ONLY opens in this
estimator are these two... That is a property of IRLS, whose iteration count is fixed and whose
step needs no line search."* `library/optimizer.h`'s `MinimizeBFGSBatched` opens objective values
and directional derivatives to steer its Armijo line search. Swapping the fit would trade an
oblivious estimator for a leaky one, so it was left alone. `SymmetricInverse` stays in `linalg.h`
because `harness.h`'s linear-algebra self-test still exercises it.

**Available but not wired up:** `logistic::SeparationFlag`, `WaldStatistics`, `OddsRatio` and
`TwoSidedPValue` have no counterpart anywhere in the pipeline. `SeparationFlag` in particular is a
real diagnostic for these two models. Adding them would add output lines and so change the
75-`RESULT` comparison, which is why they are noted here rather than landed mid-verification.

## Stage 1 measurement (baseline -> primitives unified)

`mpc-analysis -S models -D runs-verify/ctrl -ra 549 -rb 568`, blinky, `PROTOCOL=1`.

### Cost: 4.33x slower

| | baseline | stage 1 |
| --- | --- | --- |
| real | **4m13.9s** | **18m20.4s** |
| user | 3m53.9s | 18m03.4s |

This is the boolean division in `Sigmoid` and `Log` arriving, and it lands almost exactly where
root's comment predicted: that division was measured at 79% of `Sigmoid`'s cost, `Sigmoid` runs six
times per objective evaluation per row, and it was "the largest single cost in the program". The
accepted trade in decision 1 costs **4.33x wall time** on the pipeline.

### Numbers: CORRECTED BELOW -- the figures in this subsection are wrong

The worst-case figure first recorded here (2.4%) was produced by a comparison script that sorted
relative moves with `nan` values still in the list. Python's sort is not well-defined in the
presence of `nan`, and the true maxima were displaced. The corrected distribution is under
"Final verification" below. The per-row values in the table that follows are accurate; only the
claimed worst case was wrong.

### Numbers: 129 of 150 moved (worst case figure below is WRONG -- see correction above)

All 75 `RESULT` lines still present; 5 lines are byte-identical, 70 moved in at least one column.
Of the 150 numeric cells, 21 are unchanged and 129 moved. Largest relative moves:

| model | scope | term | what | baseline | stage 1 | rel |
| --- | --- | --- | --- | --- | --- | --- |
| 2a | nonumass | random_intercept_variance | est | 0.29710072 | 0.29009442 | **2.36e-2** |
| 2a | any | visit_num | est | 0.17558289 | 0.17684937 | 7.21e-3 |
| 2b | umass | fu_month | est | 0.19888306 | 0.19960022 | 3.61e-3 |
| 2a | umass | random_intercept_variance | est | 0.67475737 | 0.67696431 | 3.27e-3 |
| 2a | any | random_intercept_variance | est | 0.58927069 | 0.59081927 | 2.63e-3 |
| 2a | any | visit_num | se | 0.02407373 | 0.02412843 | 2.27e-3 |
| 2a | umass | Intercept | se | 0.16057059 | 0.16083760 | 1.66e-3 |

The `nan` standard errors on `random_intercept_variance` are present in the baseline too — they
predate this work and are not a regression.

**Attribution.** The only numerical change in stage 1 is `Exp`/`Log` going from 5 series terms to
3, per decision 3. `Sigmoid`, `Log1p`, `LogOnePlusExp`, `Sum`, `Clone` and `ClampAbs` are
value-identical between the two copies (`ClampAbs` differs only in how it takes its working copy,
and both overwrite the precision tag immediately). `kNewtonIterations` was pinned at 3 rather than
inheriting the library's 5, so the conditional-mode solver is unchanged. A worst case of 2.4% on a
variance component is consistent with dropping two Taylor terms and nothing else.

## Final state

```
playground/
  cohort.h  etl.h  harness.h  linalg.h  nodes.h  output.h
  reporting.h  secure.h  segmented.h  sqlite_oracle.h
  mpc-analysis.cpp  plain-lr.cpp  long-lr.cpp
  secure-logistic-regression.cpp  logistic-regression.cpp
  library/
    primitives.h   1127 lines
    optimizer.h    1351 lines
    regression.h   1719 lines
```

Deleted: `playground/primitives.h`, `playground/optimizer.h`, `playground/regression.h`,
`playground/mixedeffects.h` (the last was unreachable).

The pipeline's content was **appended** to the three library headers, after each file's existing
`}  // namespace cdough::regression`, as a new banner-delimited section that reopens the namespace.
No existing line of library code was edited: `git diff` on the three files shows 1852 insertions
and exactly 4 deletions, of which 3 are the closing namespace brace being re-emitted with a
trailing newline and 1 is the `Exp` comment replaced by the approved fix.

**Verified free of duplicate definitions.** 85 function definitions across the three headers; the
only repeated names are three legitimate overloads — `Clone(const AV&)` / `Clone(const
std::vector<AV>&)`, `OpenRawToParty(const AV&)` / `OpenRawToParty(const BV&)`, and the two
`MinimizeBFGS` objective forms, both of which came from the pipeline's own header.

### One collision the merge exposed

`long-lr.cpp` defined its own `double OpenScalar(const AV&)`. Once the pipeline's
`OpenScalar(const AV&, bool scaled = true)` was in `library/primitives.h`, every unqualified call
became ambiguous, because the default argument makes both viable. The local copy's body was the
library one's `scaled == true` path verbatim, so it was removed. `plain-lr` output is bit-identical
before and after the merge, and all 104 library checks still pass.

### Note on layout

`library/regression.h` now includes `../cohort.h`, `../linalg.h` and `../nodes.h`, because the
pipeline's fourteen fits need `SecureCohort`, `ScanPlan`, `MaskCount` and the Cholesky helpers.
That points the library at pipeline-level headers, which is backwards from a layering point of
view, and it means the three standalone LR drivers and the four test binaries now compile the whole
pipeline transitively. Everything builds and passes, so this is a note rather than a defect, but if
the layering matters, the fix is to move `cohort.h`/`linalg.h`/`nodes.h` under `library/` too.

## Stage 2c REVERTED — `logistic::Covariance` fails on these design matrices

The 6a/6b inference migration was implemented, measured, and **backed out**. Models 2a/2b/5a/5b
came through the merge with every cell identical, so the merge itself is sound; 6a/6b broke, and
badly.

### The measurement

Comparing the merged run against stage 1, only 6a/6b moved — 8 of 16 cells in each, all standard
errors:

| model | scope | term | stage 1 | with `logistic::Covariance` | |
| --- | --- | --- | --- | --- | --- |
| 6a | any | `data_source=UMass` | 0.19733330 | **2021.25000000** | 1.1e5 relative |
| 6a | any | `hispanic=0` | 0.20739944 | **0.00083923** | collapsed |
| 6a | any | `gender=female` | 0.15915272 | **0.00083923** | collapsed |
| 6a | any | `gender=male` | 0.15929647 | **0.00083923** | collapsed |
| 6b | any | `gender=female` | 0.16086922 | **0.00083923** | collapsed |
| 6b | any | `gender=male` | 0.16270832 | **0.00083923** | collapsed |
| 6a/6b | any | `newage` | 0.00000000 | 0.00083923 | — |

Four unrelated terms returning the **identical** 0.00083923 is the diagnostic: that is 55 ULPs at
precision 16, i.e. the inverse had lost essentially all of its information and every diagonal entry
converged to the same floor.

### The cause

`logistic::Covariance` forms `(X'WX/n)^-1 / n` with `NewtonSchulzInverse` at the library's
`kMatrixInverseIterations = 14`. Newton-Schulz is an iterative inverse that needs a well-conditioned,
O(1)-scaled operand -- `CovarianceFromInformation`'s own comment says so. **These design matrices are
not well conditioned.** The pipeline reports it itself, in a note it has always emitted:

> NOTE: observed-information 1-norm condition estimate 1.17e+04

and warns when it goes further that the fit is "close to singular; treat every standard error from
this fit as indicative only". Models 6a/6b carry near-collinear dummies (`hispanic`, the `gender`
levels, `data_source=UMass`) by construction. Fourteen Newton-Schulz iterations do not converge
there, and nothing in the library's test suite exercises that regime -- its cases are the
well-conditioned ones the iteration count was calibrated on.

The `n` normalisation is *not* the problem: `inverse(I/n)/n == I^-1` algebraically, so it cancels
whatever `n` is used. Only the conditioning matters, and that is what fails.

### What the code does now

`FitLogisticIrls` keeps the exact Cholesky inverse (`SymmetricInverse`), which handles this
conditioning, so 6a/6b standard errors are back to their stage-1 values. Two things were kept from
the attempt:

- **Only the diagonal is opened.** The standard errors need `p` variances, not the whole `p x p`
  covariance, and `mapping_reference` is a public view, so extracting them is free. This fit now
  opens `p` values where it used to open `p^2` -- a real reduction in disclosure, carried over from
  the migration and independent of it.
- **The plaintext guarded `sqrt` stays**, and the `std::max(0.0, ...)` clamp is load-bearing: a
  near-collinear design can put a small negative on the inverse's diagonal, and these models do go
  near-singular. `SecureSqrt` could not guard that without an extra oblivious comparison, which is
  the second reason not to move this into MPC.
- **`AsLogisticDataset` is retained but unused**, flagged as such in a comment. It is correct, it
  encodes the mask-convention trap, and `logistic::SeparationFlag` / `WaldStatistics` / `OddsRatio`
  would all need it -- none of those touch the failing inverse.

### What would be needed to finish 2c

One of: raising the Newton-Schulz iteration count for ill-conditioned operands (a change inside
`library/`, so out of bounds here); pre-conditioning the information matrix before handing it over;
or giving the library a Cholesky-based inverse to sit alongside Newton-Schulz. All three are real
work with their own verification, and none is a deduplication.

## Final verification

`S3` = the merged tree with 6a/6b reverted to the Cholesky inverse. Run on blinky, `PROTOCOL=1`,
same dataset and seeds. 75 `RESULT` lines, **18m47.3s**.

### The merge is behaviour-preserving

**Stage 1 -> final: 138 cells identical, 12 `nan` on both sides, 0 moved.** Appending the pipeline
onto the library headers, deleting the four root headers, deduplicating the matrix helpers,
removing `long-lr.cpp`'s `OpenScalar`, and switching 6a/6b to open the covariance diagonal instead
of the full matrix together change **nothing** numerically. All 104 library checks pass;
`plain-lr` is bit-identical.

That is the deduplication delivered and verified.

### Baseline -> final: the series-term reduction is worse than first reported

138 comparable cells (12 `nan`-on-both excluded):

| relative move | cells |
| --- | --- |
| exact | 21 |
| < 0.1% | 24 |
| 0.1 - 1% | 41 |
| 1 - 10% | 37 |
| 10 - 100% | 12 |
| > 100% | 3 |
| **sign flip** | **1** |

**The sign flip:** model 5b, `any`, `gender=female` estimate **+0.05374146 -> -0.01293945**.

Largest same-sign moves:

| model | scope | term | baseline | final | rel |
| --- | --- | --- | --- | --- | --- |
| 5a | nonumass | newage | 0.00000000 | 0.00088501 | — |
| 5b | any | `gender=male` | -0.01165771 | -0.11042786 | **8.47** |
| 5b | umass | newage | -0.00715637 | -0.00083923 | 0.88 |
| 5b | nonumass | newage | -0.00178528 | -0.00297546 | 0.67 |
| 5b | any | `hispanic=0` | 0.59550476 | 0.39443970 | 0.34 |
| 5b | any | random_intercept_variance | 0.79416125 | 1.04336384 | 0.31 |
| 5b | any | fu_month | 0.19865417 | 0.24014282 | 0.21 |

### Attribution

The only arithmetic change from baseline is `Exp`/`Log` at 3 series terms instead of 5 (decision 3).
The division change that came with it is, if anything, an accuracy *improvement*: `library/`'s
`a2b` -> boolean `/` -> `b2a` is exact integer division, where root's `RecipSeeded` was a two-step
Newton approximation. So the degradation is the truncated series, not the divide.

Two mechanisms are visible in the table:

1. **ULP-limited cells.** `newage` coefficients live at 1e-3 and the `gender` levels at 1e-2.
   `library/primitives.h`'s own measured-range note gives the rule -- *"relative error ~ (1 to 2) /
   output_ULPs"* -- so a coefficient of 0.00084 is 55 ULPs and carries a percent-level relative
   error floor at precision 16 no matter how many series terms are used. `0.00083923` recurring
   across unrelated terms is that floor.
2. **Convergence-path divergence.** The sign flip and the 8.47x move are not ULP effects:
   0.0537 is ~3500 ULPs. These are model 5b, the near-singular fit that already warns it is
   "close to singular". A noisier objective moves where BFGS stops, and on a weakly determined
   covariate that is enough to change the sign.

Mechanism 1 is inherent to the number format. Mechanism 2 is the cost of the shorter series, and it
is a **statistical** problem rather than an arithmetic one: a sign-flipped covariate coefficient
changes the interpretation of the fit, not just its precision.

### Recommendation

Restore five series terms for `Exp` and `Log`. Root ran five; the pipeline's published results were
produced with five; the measured penalty for three is a sign flip on a reported coefficient. This
means `kMaxSeriesTerms` in `library/primitives.h`, or separate `kExpSeriesTerms` /
`kLogSeriesTerms` as root had -- a constant, not an algorithm, but it is inside `library/` and so
needs the same kind of exception the `Exp` fix got. **Flagged for decision; not changed.**

## Risks and Mitigations
- Risk: silent numerical drift in the pipeline's 17 outputs.
  - Mitigation: byte-diff against a recorded 0009 baseline before deleting anything.
- Risk: reintroducing `exp(-1) = 1/3` by taking `library/`'s `Exp`.
  - Mitigation: explicit test case at `x = -1` in `test_playground_primitives`.
- Risk: losing the `Sigmoid` cost win and slowing the pipeline substantially.
  - Mitigation: `profile_playground_primitives` before/after, on the same host.
- Risk: stage 2's layout change breaks GLMM standard errors subtly.
  - Mitigation: `test_playground_mixed_effects` covers `ObservedInformation` and
    `MixedModelStandardErrors`; extend to the pipeline's `ModelSpec` shapes before migrating.

## Rollback and Recovery
- Rollback plan: each stage is its own commit on `logistic-regression`; revert the commit.
- Recovery steps: the root copies are recoverable from `f5b7bec` / `ea66362` and the `library/`
  copies from `d561ec3` for as long as those commits are reachable.
