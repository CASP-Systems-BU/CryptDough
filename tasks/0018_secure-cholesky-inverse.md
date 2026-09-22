# Task 0018 — Secure Cholesky inverse + oblivious condition estimate

## Metadata
- Task ID: 0018
- Title: Replace NewtonSchulzInverse with a secure Cholesky inverse; make the condition estimate oblivious
- Requested by: Adam Godel
- Owner: Adam Godel
- Date: 2026-09-22
- Status: In Progress — phase A done and measured; phase B underway
- Related documents: [tasks/0017](0017_supersede-pipeline-operators-with-library.md) (where the
  inference half failed), [tasks/0015](0015_unify-duplicated-playground-operators.md),
  [tasks/0010](0010_secure-matrix-inversion-newton-schulz.md) (the Newton-Schulz inverse)
- Branch: `logistic-regression`
- Plan file: `~/.claude/plans/closely-examine-the-current-joyful-reddy.md` (approved)

## Problem Statement

Task 0017 delivered the optimiser half — `MinimizeBFGS` deleted, disclosure during the fit down
from ~2n+21 opened values per iteration to one bit — but its inference half failed:
`NewtonSchulzInverse` returned 6a standard errors 1,700x-111,000x too large.

The obvious fallback, reverting to a plaintext Gauss-Jordan inverse, is a **security regression**:
`2*dim^2` opened values against `dim^2 + p`, with the inversion leaving MPC entirely.

## Acceptance bar

**Beat the plaintext path's disclosure, keep standard errors usable (~10%).**

## Measured basis

Relative error of `diag(A^-1)` — what a standard error reads — Cholesky at n = 7 against an exact
double-precision inverse:

| kappa | no ridge | ridge 1e-3 | ridge bias vs truth |
| --- | --- | --- | --- |
| 1e2 | 7.80e-4 | 1.30e-3 | 8.6% |
| 1e3 | 5.67e-3 | 1.96e-3 | 48% |
| **1e4** | **7.88e-2** | 1.59e-2 | **90%** |
| 2e4 | 1.48e-1 | 1.13e-2 | 95% |

Two corrections to earlier beliefs recorded here so they are not repeated:

1. **Cholesky is NOT condition-insensitive.** Backward stability with growth factor 1 is a statement
   about BACKWARD error; the inverse diagonal is a forward quantity and its error scales with kappa
   whatever the method.
2. **The ridge is not the fix.** It makes the operand tractable by moving the answer 90% away from
   the estimand.

Unridged Cholesky clears the bar: 7.9% at the pipeline's conditioning, and 7 opened values
against the plaintext path's 98.

## Decisions
1. Replace `NewtonSchulzInverse` entirely.
2. **No ridge on the reported covariance.** The IRLS *fit* keeps `AddRidge` (a numerical necessity
   of the Newton step); the *inference* inverts the unridged matrix. This moves 6a/6b standard
   errors off the T1 baseline — they get larger and less biased, because the baseline carried
   `kIrlsRidge` shrinkage.
3. Exact 1-norm condition estimate computed on shares, opening ONE scalar.
4. Re-measure and tighten every affected tolerance.

## Progress

### Phase A — DONE
Restored `Rsqrt`/`SqrtBoth`/`Sqrt` and their constants, reverting the task 0017 phase 6 change that
had made the Cholesky pivot `SecureReciprocal(SecureSqrt(d))` — two boolean division circuits
(~1000 rounds) per pivot against ~10 for the fused seeded Newton.

**Measured:** Cholesky inverse error `||A A^-1 - I||_inf` **7.870e-4 -> 2.729e-4**, exactly
reproducing the task 0009 figure. Solve residual 2.269e-3 -> 1.344e-3.

### Phase B — DONE
`Cell` and `PublicVector` moved to `library/primitives.h`; the Cholesky family moved to
`library/optimizer.h` beside the operator it replaces. `ScalarZero` dropped (it had no callers).
`linalg.h` is down to 100 lines, keeping only `Gram` and `AddRidge` for the IRLS step.

**Solves batched.** `CholeskySolveManyWith` solves all `rhs` right-hand sides at once: the
substitution loops are sequential in `p` either way, so batching leaves the round COUNT unchanged
and makes each multiply `rhs` wide. A full inverse now costs what one solve used to (~p^2 + p
rounds against ~p^3 + p^2; at p = 8, ~72 against ~576).

**Gate: bit-identical.** `TestLinearAlgebra` before and after batching:
Gram 1.582174e-04, solve residual 1.343874e-03, inverse error **2.729076e-04** — unchanged to every
printed digit.

One implementation note: `repeated_subset_reference` is a view and `mapping_reference` asserts
`!has_mapping()`, so a `Cell` slice cannot be broadcast directly — each scalar is `Clone`d into a
named local first, which also keeps the view's referent alive for the statement that uses it.

### Phase C — DONE (6a verified)
Both covariance paths call the new `SecureInverse` wrapper. **The `/n` normalisation was removed
from both**: it existed to satisfy the iterative inverse's "O(1)-scaled" precondition, which
Cholesky does not have, and at n = 2048 it put the operand near 1e-2 where the measured relative
error of `diag(A^-1)` is 8.0e-1 against 7.9e-2 at magnitude 1. Removing it also deletes the
compensating multiply, since `(X'WX)^-1` IS the covariance.

**Model 6a, against the T1 baseline:**

| term | T1 se | Cholesky se | diff |
| --- | --- | --- | --- |
| Intercept | 0.254686 | 0.25444 | 0.10% |
| visit_num | 0.032212 | 0.03220 | 0.04% |
| gender=female | 0.159153 | 0.15907 | 0.05% |
| gender=male | 0.159296 | 0.15927 | 0.02% |
| hispanic=0 | 0.207399 | 0.20731 | 0.04% |
| data_source=UMass | 0.197333 | 0.19730 | 0.02% |

All within **0.1%**, against a 10% acceptance bar — and better than the 7.9% predicted, because
6a's actual conditioning is well below the 1.2e4 that figure assumed (that was a mixed model's).
Estimates identical. Runtime **2.17s against 3.66s** for the Newton-Schulz version: the batched
Cholesky is cheaper than 30 iterations at this size, so this is an accuracy win at negative cost.

One oddity to resolve: `newage` reported se `0.000000` in the baseline and `0.00084` now. The
baseline's zero looks like an underflow rather than a real value; 0.00084 is the 55-ULP floor. Needs
a look before the full run is trusted.

### Model 5a (kappa = 1.11e4, the hard case)

| term | SE change | estimate moved |
| --- | --- | --- |
| Intercept | +0.15% | 0.013 SE |
| visit_num | -0.51% | 0.148 SE |
| **newage** | **-19.67%** | **0.233 SE** |
| gender=female | +0.15% | 0.023 SE |
| gender=male | +0.19% | 0.055 SE |
| hispanic=0 | +0.12% | 0.052 SE |

**Five of six standard errors within 0.51%**, at the conditioning where the iterative inverse
returned 55-ULP noise. That is the plan working.

### Check 1 — `newage`'s standard error: the format, not the inverse

Variances expressed in ULPs at `precision` 16 (ULP = 1.526e-5):

| term | variance | ULPs |
| --- | --- | --- |
| 5a Intercept | 9.10e-02 | 5965 |
| 5a visit_num | 5.85e-04 | 38.3 |
| **5a newage** | **2.37e-05** | **1.55** |
| **6a newage** | **7.06e-07** | **0.05** |

`5a newage`'s variance is held in **1.55 ULPs** — at most one significant figure, so -19.7% is
exactly what the format permits. `6a newage` is at **0.05 ULPs**, below one ULP entirely: the
baseline's `0.000000` was an underflow and the new `0.00084` is the floor. **Neither is a real
number.**

This is inherent to ANY secure inverse at this precision, not a defect of the Cholesky one — the
plaintext baseline escaped it only by inverting in double precision after opening. A standard error
whose variance is O(ULP) cannot be computed in MPC here. It should be reported as unavailable
rather than printed.

(Correction: an earlier note in this task said 1553 ULPs. That was wrong by 1000x.)

### Check 2 — did the phase-6 division change improve the estimates? NOT ESTABLISHED

Task 0017 phase 6 replaced `Div(grad, curv)` with `grad * SecureReciprocal(curv)` inside
`FlatConditionalModeBatched`'s inner Newton. That is a change to the FIT, and it is what moved the
5a estimates (the inverse cannot: 5a was identical to T1 after phase 3).

Measured accuracy of the replacement, from the kernels self-test: exact for most inputs, worst
relative error **8.2e-3** at denominator 1000 — where `1/1000` is 65 ULPs, so the floor again.

But the operator it replaced was **also** ULP-limited: `RecipSeeded`'s documented seed error of
1/17 squares over three Newton steps to ~1.4e-10, far below one ULP. So both are ULP-limited and
**there is no basis to claim the estimates moved toward the truth rather than merely sideways.**
Settling it needs both paths scored against the plaintext oracle
(`scripts/testing/validate_mpc_analysis.py --compare`), which is part of the full-pipeline run.

### A bug found while running check 2

The phase-6 rewrite of the `Divide` self-test in `harness.h` omitted the precision pin that both
production call sites have. A multiply of two operands still carrying `precision` truncates once by
itself, so the explicit `/ scale` divided a second time and the test reported **100% relative error
on a correct operator**. Anyone running `-S kernels` would have seen alarming garbage. Fixed; the
production paths were never affected.

### Phase D — DONE and verified

`OneNorm(SMatrix)` added to `library/optimizer.h`: `Abs` -> column-major gather by
`mapping_reference` (a free public view) -> `chunkedSum(n)` for the column totals -> a tournament
max of pairwise `Multiplex` over `gtez`, `ceil(log2 n)` comparison rounds (3 at n = 7). Entirely on
shares.

`CovarianceFromInformation` gained an optional `AV* condition_out` and fills it with
`||S||_1 * ||S^-1||_1`. Both factors are already in hand, so it costs two `OneNorm` calls and no
extra inverse.

`ObservedInformationSecure` no longer opens the `dim x dim` Hessian. **Disclosure for the inference
path is now `1 + p` opened values**, against `dim^2 + p` before and `2 * dim^2` for the plaintext
Gauss-Jordan fallback — at dim 7 that is 7, against 55 and 98.

**One change of referent to be aware of:** the estimate is now kappa_1 of the matrix ACTUALLY
INVERTED, the Schur complement, rather than of the full `(p+1) x (p+1)` information matrix. That is
the conditioning which governs the accuracy of the standard errors, so it is the more useful
number, but it is a different scale and `kHessianConditionWarn = 1e6` may want recalibrating
against it.

**Verified.** `OneNorm` matches a plaintext max-column-abs-sum exactly at n = 4 and n = 7, and to
1.79e-07 at n = 8. On model 5a the oblivious estimate reads **1.05e+04** against the plaintext
block's 1.11e+04 -- a 5% difference, which is the change of referent plus fixed-point. Coefficients
and standard errors are **identical** to the phase C run, so this phase changed only how the
diagnostic is computed.

**Disclosure counted in the source**, which is the acceptance criterion:

| path | opens | at dim 7 |
| --- | --- | --- |
| mixed models (`ObservedInformationSecure`) | `se_secure` (p) + `kappa` (1) | **7** |
| 6a/6b (`FitLogisticIrls`) | `beta` (p) + `se_secure` (p) | 12 |
| plaintext fallback, for comparison | `2 * dim^2` | 98 |

`OneNorm`, `CholeskyFactor`, `CholeskySolveManyWith` and `SymmetricInverse` contain **zero** opens.
The bar is met with a 14x margin.

### Two implementation traps hit in phase D, both recorded so they are not repeated

1. **`AV` assignment semantics.** `AV a = b` is a SHALLOW copy, but `a = b` is a deep element-wise
   copy that asserts equal sizes. A tournament max has to rebind its accumulator to a shorter
   vector each round, which cannot be expressed that way. The sequential fold used instead is 7
   comparison rounds against 3 at n <= 8 -- not worth the trap.
2. **`make` does not always rebuild when only a header changes.** A crash was chased for one
   12-minute pipeline run and one probe before it turned out the object file was stale and the
   binary still held the old tournament code. `touch` the headers, or change a `.cpp`, before
   trusting a rebuild.

A 2-second probe (`probe_onenorm`) localised what a 12-minute pipeline run could not. Write the
probe first.

### Phase E — prose done; tolerances pending the full run

Zero references to Newton-Schulz remain in `library/regression.h`. Updated: the
Schur-for-conditioning rationale (which still holds, because Cholesky is condition-sensitive too),
`ObservedInformationOverN`'s normalisation note, `ObservedInformationSecure`'s header, the
`FitLogisticIrls` history block, and the user-facing fit note — which had been telling readers the
standard errors came from Newton-Schulz and carried "several percent of residual error".

Tolerances (`kInverseTolerance`, `kCovarianceTolerance`, `kStandardErrorTolerance`,
`kMixedStandardErrorTolerance`, `kSchurTolerance`, and the `kMixedHessianStep` sweep) are all
"measured worst x margin" figures and are re-measured after the full-pipeline run.

## The 128-bit route: scoped, not taken

`__int128_t` is already a first-class share type (instantiated at engine startup, MPI wire support,
byte-oriented PRG, five test binaries). Communication and randomness are NOT blockers. The single
*algorithmic* blocker for this pipeline is boolean division, which `exit(-1)`s at 128 bits
(`circuits.h:41-47`) because the non-restoring circuit needs a `DoubleWidth` intermediate.

**CORRECTION to a first reading of this.** The division circuit *itself* needs only bitwise ops,
shifts, unary negate, `< 0`, `==` and narrow<->wide conversion — no multiply. But **a type cannot be
registered with the runtime for boolean operations only.** `runtime_declare_protocol_functions(T)`
and `TYPE_BITS_*` register arithmetic and boolean functionality as one unit, and instantiating
`Protocol<int256>` instantiates its virtual `multiply_a`, `div_const_a` and `dot_product_a` — which
need `*`, `/` and `%`. So a 256-bit type needs **full signed 256-bit arithmetic**, or a prior
refactor of the `Protocol` virtual surface to permit a boolean-only type. The refactor is probably
the larger and more valuable piece of work.

Two further costs: `std::make_unsigned_t<T>` is used directly in ~15 places
(`b_shared_vector.h:64,106`, `protocol_circuits.h` throughout) and is ill-formed for a class type,
so each must move to the existing `UnsignedTypeSelector` hook; and `communicator.h`'s send/receive
interface is a fixed list of pure-virtual overloads ending at `__int128_t`, needing new virtuals in
three communicators (required for PROTOCOL=3, not for PROTOCOL=1).

The OLE generator can be **struck from scope**: `wide_t` is always `uint16_t` there, it is never
instantiated at 128 bits, and it sits behind `MPC_PROTOCOL_BEAVER_TWO` which is PROTOCOL=2 only.

Precedent worth knowing: `Vector<NTL::GF2E>` already works, so non-POD element types are supported,
and `core/math/util.h:29-84` already hand-rolls 128-bit <-> `NTL::ZZ` limb splitting — a ready
reference implementation to differential-test a hand-rolled `int256` against.

## Unrelated latent bug found while scoping
`sorting.h:335-342`: `PadWidth<__int128_t>` resolves to `int64_t`, **narrower than the input**, so
radix-sorting a 128-bit column silently produces wrong answers. Deserves its own task.

## Phase F — analytic-gradient Hessian (the fix for the three failing standard errors)

The full-pipeline run passed on the fits but the plaintext oracle failed three of twelve mixed
models on standard errors, against a 1.0e-1 relative bar:

| model | scope | before | after | tolerance |
| --- | --- | --- | --- | --- |
| 5a | nonumass | **1.212e-1 FAIL** | 1.601e-2 | 1.0e-1 |
| 5b | any | **1.222e-1 FAIL** | 2.984e-2 | 1.0e-1 |
| 5b | nonumass | **1.099e-1 FAIL** | 2.845e-2 | 1.0e-1 |
| 5a | any | 9.639e-2 | 5.136e-2 | 1.0e-1 |
| 5b | umass | 9.214e-2 | 2.265e-2 | 1.0e-1 |
| 5a | umass | 2.724e-2 | 1.342e-2 | 1.0e-1 |
| 2a | any | 2.219e-2 | 1.309e-2 | 1.0e-1 |
| 2a | nonumass | 1.752e-2 | 1.015e-2 | 1.0e-1 |
| 2a | umass | 2.639e-2 | 1.288e-2 | 1.0e-1 |
| 2b | any | 1.635e-2 | **2.279e-2** | 1.0e-1 |
| 2b | nonumass | 1.071e-2 | 5.189e-3 | 1.0e-1 |
| 2b | umass | 1.822e-2 | 8.531e-3 | 1.0e-1 |

`VALIDATION: *** FAIL ***` -> `VALIDATION: PASS`.

### Diagnosis

The error was in the **Hessian, not the inverse**. The Cholesky inverse measures 0.12-0.51% on 5a at
kappa 1.1e4. The Hessian was built by `NumericalHessianBatched`, from SECOND differences of the
OBJECTIVE — the form this file's own `kHessianStep` calibration note already recorded at 17-46% error
at condition ~2e4 where an analytic-gradient form held 8%. A second difference divides by `h^2` and
so amplifies the objective's fixed-point noise by `1/h^2`; a first difference of the gradient divides
by `h` once.

### Change

`ObservedInformationFromGradient` builds the information from central differences of the **analytic
gradient**: for each axis `k`, perturb the optimum by ±h in coordinate `k`, evaluate
`FlatObjectiveBatched(md, point, 1, &g)`, and take `(g+ - g-)/2h` as column `k`. The result is then
**symmetrised** — the two triangles come from independent difference sets, and
`CovarianceFromInformation` reads `h_bs` from the last column only, which silently assumes symmetry.

**Security is unchanged.** The differences, the Schur complement, the inverse and the square root all
stay on shares; disclosure remains `1 + p`. The predecessor `ObservedInformationSE` opened all `dim`
gradient components at each of the `2*dim` perturbed points only because it went on to invert in
plaintext — which phases B-D removed the need for.

### Cost, which is the real trade

`FlatObjectiveBatched` requires `num_points == 1` when a gradient is requested, because the analytic
gradient is a single-point quantity. So `2*dim` SEQUENTIAL passes replace ONE batched call of
`2*dim*(dim+1)` points: fewer evaluations, more round depth. Measured `-N 5b:any -r 200` at 3m04s for
a single model including startup; the full synthetic `-S models` run amortises startup and completed
well inside the earlier budget.

### Caveats carried forward

- **2b any is the one value that moved the wrong way**, 1.635e-2 -> 2.279e-2. Inside a 1.0e-1 bar and
  not a concern, but it is a real move and should not be rediscovered as a surprise.
- **`kHessianStep = 0.1` is no longer the optimum for the form that uses it.** It was derived as
  `eps^(1/4)` for second differences; a first central difference of the gradient balances nearer
  `eps^(1/3)` ~ 0.05. The comment now records this. Retuning is worth doing and is not urgent.
- **`kHessianConditionWarn = 1e6`** still wants recalibrating: its referent changed to the Schur
  complement, and the 8%-at-2e4 figure it was set against is now the form actually in use.
- **Standard errors whose variance is O(ULP)** (the `newage` terms, 1.55 and 0.05 ULPs) still cannot
  be computed at this precision and should be reported as unavailable rather than printed.
