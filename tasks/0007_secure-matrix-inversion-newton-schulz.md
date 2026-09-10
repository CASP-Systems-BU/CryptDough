# Secure Matrix Inversion via Newton–Schulz Iterations

## Metadata
- Task ID: 0007
- Title: Secure matrix inversion operator using Newton–Schulz iterations
- Requested by: Adam Godel
- Owner: Claude
- Date: 2026-09-09
- Status: Done
- Estimated effort: Medium
- Target completion date: TBD
- Related issue or PR: —
- Related documents (design doc, runbook, specs):
  - Newton–Schulz reference: https://ayushgarg.ca/notes/Newton-Schulz-Iteration
  - Target pipeline: https://cs-people.bu.edu/liagos/pilot/mpc_analysis_lineage.html#/
  - `tasks/0001_mixed-effects-logistic-regression-laplace.md`
- Branch: orchestration

## Problem Statement
- What is the request? Add a matrix inversion operator to
  `playground/secure-logistic-regression.cpp`, implemented with Newton–Schulz iterations under a fixed
  upper bound on the iteration count.
- Why does this matter now? The downstream MPC analysis pipeline needs a matrix inverse in several
  places: the slope-comparison contrast test needs `(L V Lᵀ)⁻¹`, reporting parameter covariance needs the
  observed information matrix inverted, and extending the mixed-effects model to random slopes needs a
  per-group `q×q` curvature inverse. Nothing in the file or the library provides one today.

## Goals
- Goal 1: A `NewtonSchulzInverse(A)` operator over `SecureMatrix<int64_t>` with a compile-time iteration
  bound and no data-dependent control flow.
- Goal 2: The two missing primitives it needs — a secure matrix transpose and a `MatMul` wrapper that
  gets the library's column-wise calling convention right.
- Goal 3: An iteration bound chosen from measurement on the real 3PC LAN deployment, landing in the same
  accuracy band as the file's existing oblivious operators.

## Non-Goals
- Non-goal 1: Changing `MinimizeBFGS` or adding new statistical steps (covariance reporting, Wald tests).
  Those consume this operator in a later task.
- Non-goal 2: Expanding the library API under `include/`. The transpose and scalar helpers stay local to
  the playground file for now.
- Non-goal 3: A separate symmetric-positive-definite fast path.

## Scope
- In scope: `playground/secure-logistic-regression.cpp` (new helpers, the operator, validation and
  calibration blocks in `main()`), plus a small refactor of the secure-reciprocal idiom that is currently
  duplicated three times in that file.
- Out of scope: everything under `include/`, `tests/`, and `bench/`.

## Impact Assessment
- User impact: New operator available to the playground program; existing behaviour unchanged.
- Performance impact: Per inversion — one boolean division circuit (dominates setup) plus
  `2 × iterations` matrix multiplications, each costing one communication round plus one ABY3 truncation
  round. At `n = 3` and the calibrated bound this is small, but it is not free; callers should hoist it
  out of inner loops.
- Security and privacy impact: The operator is fully oblivious — fixed iteration count, no branch on
  shared data, no `open()` calls. Cost and communication pattern depend only on public values (`n`,
  `iterations`). This is a stricter standard than the surrounding `MinimizeBFGS` code, which does open
  values for convergence checks.
- Backward compatibility impact: The `Identity(n, engine)` helper is re-expressed in terms of a new
  `ScaledIdentity(n, c, engine)`; its signature and behaviour are unchanged.

## Context
- Relevant files and modules:
  - `playground/secure-logistic-regression.cpp` — the only file being changed.
  - `include/core/containers/matrix/hybrid/matrix.h:127` — the sole matmul primitive.
  - `include/core/protocols/replicated_3pc.h:207` — the 3PC matmul kernel and its automatic truncation.
  - `include/core/protocols/interface/protocol.h:115` — `handle_precision`, which throws on mismatch.
  - `include/core/containers/mapping_access_vector.h:1268` — `materialize()`, the recipe the transpose
    helper reuses.
  - `include/core/operators/circuits.h:39` — the non-restoring binary division circuit.
- Dependencies: none beyond what the file already includes.
- Constraints and assumptions:
  - Fixed-point `precision = 16`, `DataType = int64_t`.
  - Input `A` must be nonsingular and O(1)-scaled (see Risks).
  - The secure division circuit assumes non-negative operands.

## Alternatives

### Option A: Gaussian elimination / LU with pivoting
Pros:
- Exact in the plaintext sense; `O(n³)` with a small constant.
- Well understood, easy to validate.
Cons:
- Pivot selection is a data-dependent branch. Doing it obliviously requires a secure argmax and an
  oblivious row permutation per column, which is both expensive and easy to get subtly wrong.
- Without pivoting it is numerically fragile in fixed point, and fragility shows up as silent
  wrong answers rather than an error.

### Option B: Cholesky decomposition, then triangular solves
Pros:
- Half the work of LU for symmetric positive-definite inputs, which is the dominant case in this pipeline.
- No pivoting needed for SPD matrices, so it is naturally oblivious.
Cons:
- Needs a secure square root per diagonal entry — another iterative approximation with its own error
  budget and iteration bound, so it does not actually avoid the problem, it adds one.
- Asserts on general (non-symmetric) input, which narrows the operator.

### Option C: Newton–Schulz iteration
Pros:
- Naturally oblivious: a fixed sequence of matrix products, no branching, no comparisons.
- Built almost entirely from the one primitive the library already vectorizes well (matmul), so it maps
  onto few communication rounds.
- Self-correcting — each iteration re-anchors on `A`, so fixed-point truncation error does not accumulate;
  the iteration converges to the noise floor and stays there.
- Quadratic convergence: the error is squared every step.
Cons:
- Needs a scaled initial guess to be in the convergence basin at all.
- Iteration count needed depends on the condition number, so a fixed bound means ill-conditioned inputs
  silently return a poor approximation.

### Recommendation
- Recommended option: **Option C (Newton–Schulz)**, with the initialization `X₀ = Aᵀ / ‖A‖_F²`, which
  guarantees `‖I − A X₀‖₂ < 1` for any nonsingular `A` because `σ_max ≤ ‖A‖_F`.
- Why this option is preferred: it is the only one of the three whose oblivious form is also its natural
  form. A and B both need extra secure machinery (argmax + permutation, or square root) purely to hide
  control flow that Newton–Schulz never has. The fixed-bound requirement is a constraint for A and B and
  a free property for C.
- Open questions needing confirmation: resolved with the requester —
  1. General square matrices, not SPD-only. **Confirmed.**
  2. Operator plus validation only; no wiring into `MinimizeBFGS` in this task. **Confirmed.**
  3. Calibration runs go to blinky/pinky/inky, which are already aliased to `node0`/`node1`/`node2`
     in `/etc/hosts`. **Confirmed.**

## Implementation Plan
1. Add `TransposeData` / `Transpose` / `AsColumnWise` / `MatMul`, getting the column-wise convention right
   (see Risks — this is the highest-risk item and is validated first).
2. Add `ScaledIdentity` and re-express the existing `Identity` in terms of it; add both `ScaleMatrix`
   overloads.
3. Factor out `SecureReciprocal` and refactor the three existing duplicated call sites onto it.
4. Add `FrobeniusNormSquared` and `NewtonSchulzInverse`.
5. Add the validation block to `main()`: a plaintext Gauss-Jordan reference, then `Transpose`/`MatMul`
   tested on a non-symmetric matrix before anything else, then the inverse over a suite covering
   identity, well-conditioned SPD, ill-conditioned SPD, non-symmetric, 2×2, and a realistic `XᵀWX` Gram
   matrix built from the synthetic dataset already in `main()`.
6. Add the iteration-count calibration sweep behind a `constexpr bool` toggle, off by default.
7. Run locally, then on the 3PC LAN cluster; pick the bound from the LAN numbers and bake it in with a
   comment recording the measurement.

## Risks and Mitigations
- Risk: The column-wise calling convention is inverted, so `MatMul` silently computes `A·Bᵀ`.
  Reinterpreting a row-major buffer as column-wise yields `Aᵀ`, not `A` — the obvious spelling is wrong,
  and on symmetric inputs the wrong answer looks correct.
  - Mitigation: validate `MatMul` against a plaintext reference on a **non-symmetric** matrix before
    building anything on top of it.
- Risk: `X₀ = Aᵀ/‖A‖_F²` underflows. If `‖A‖_F²` is large, the entries fall below `2⁻¹⁶`, truncate to
  zero, and the iteration is stuck at `X = 0`.
  - Mitigation: document the O(1)-scaling precondition; verify against the largest-norm test matrix;
    pre-scale by a public constant if the Gram matrix trips it. Note that `X₀`'s *precision* barely
    matters — it only needs to land in the convergence basin, and the iteration cleans up the rest.
- Risk: `handle_precision` throws at runtime, not compile time, if two matmul operands disagree.
  - Mitigation: an explicit `setPrecision(precision)` before every matmul operand, and an early smoke
    test that exercises the path.
- Risk: `SecureMatrix::data()` returns a live reference and `ASharedVector` copies are shallow, so
  `setPrecision` on an apparent copy mutates the source matrix.
  - Mitigation: `Clone(m.data())` in every helper that changes precision.
- Risk: near-singular input converges too slowly for the fixed bound and returns a poor approximation
  with no signal to the caller.
  - Mitigation: inherent to obliviousness; publish the accuracy-vs-condition-number curve from the
    calibration so callers know the operating range.

## Rollback and Recovery
- Rollback plan: the change is additive and confined to one playground file plus this document; revert
  the commit. The only non-additive edit is the `SecureReciprocal` refactor of three existing call sites,
  which is behaviour-preserving and covered by the existing validation output.
- Recovery steps: none needed; no persistent state or deployed service is involved.
- Monitoring and alerts after release: not applicable.

## Validation Plan
- Automated checks: the program's own `main()` prints an abs-error table per operator against a plaintext
  reference, in the style already used for `Exp`/`Log`/`Sigmoid`. New tables cover `Transpose`, `MatMul`,
  and `NewtonSchulzInverse`.
- Manual verification:
  - Local: `../scripts/run_experiment.py -p 3 -r 16 secure-logistic-regression` from `build/`.
  - 3PC LAN: `../scripts/run_experiment.py -s lan -p 3 -r 16 secure-logistic-regression` on
    blinky/pinky/inky.
- Expected success criteria:
  - `Transpose` and `MatMul` match plaintext to within fixed-point noise on non-symmetric input.
  - The inverse's residual `max |A·Â − I|∞` plateaus at or below the band the file's existing operators
    achieve (~1e-4, consistent with `kSmallEpsilon` and `kSeriesTolerance`).
  - The existing BFGS validation still runs and converges unchanged.

## Definition of Done
- DoD criterion 1: `NewtonSchulzInverse` and its supporting primitives are implemented and oblivious.
- DoD criterion 2: `kMatrixInverseIterations` is set from a LAN measurement, with the measurement recorded
  in a comment next to the constant.
- Tests updated or added: validation and calibration blocks added to `main()`.
- Documentation updated: this document, plus the ledger line in `tasks/tasks.md`.

## Approval
- [x] Task document reviewed
- [x] Approved to implement
- Approver: Adam Godel
- Approval date: 2026-09-09

## Results

Calibration sweep on the 3PC LAN cluster (blinky/pinky/inky), residual `max |A*X - I|` by
iteration count. The iteration at which each case reaches its noise floor:

| Matrix | Plateaus at | Noise floor |
|---|---|---|
| identity 3x3 | k = 5 | 0 |
| spd, kappa ~ 4 | k = 8 | ~1e-4 |
| spd, kappa ~ 19 | k = 13 | ~5e-5 |
| non-symmetric 3x3 | k = 6 | ~1e-4 |
| 2x2 | k = 7 | ~5e-5 |
| X^T W X / N (2x2) | k = 5 | ~7e-5 |
| spd, kappa ~ 199 | still descending at k = 20 | out of range by design |

`kMatrixInverseIterations = 14` — the worst in-range case plateaus at 13, plus one iteration of
margin. The resulting floor (5e-5 to 1.2e-4) sits at or below the abs errors the file's other
oblivious operators report (Exp/Log ~1e-4, BfgsInverseUpdate ~5e-4), which was the accuracy bar.
The analytic estimate `k >~ log2(n * kappa^2 * ln(1/epsilon))` predicted 12-15, so the measurement
and the theory agree.

`Transpose` and `MatMul` are exact against the plaintext reference on non-symmetric input, and the
test prints the distance from the `A*B^T` result (4.0) to show the column-wise convention is right
way round.

## Change Log
- 2026-09-09: Initial draft created and approved.
- 2026-09-09: Implemented; iteration bound calibrated on the 3PC LAN cluster; marked Done.
