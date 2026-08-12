# Mixed-Effects Logistic Regression (Laplace, Quasi-Newton) — Plaintext

## How to Use
This is the design document for the task. Implementation must not start until the
document is reviewed and approved (see the Approval section).

## Metadata
- Task ID: 0001
- Title: Plaintext mixed-effects (random-effects) logistic regression via Laplace
  approximation with quasi-Newton maximization
- Requested by: User
- Owner: TBD
- Date: 2026-07-30
- Status: Done
- Estimated effort: ~1 focused implementation pass + validation
- Target completion date: TBD
- Related issue or PR: N/A
- Related documents: SAS PROC GLIMMIX docs (METHOD=LAPLACE, TECHNIQUE=QUANEW)
- Branch: main (current)

## Problem Statement
- What is the request? Implement a mixed-effects (generalized linear mixed model,
  GLMM) logistic regression that mirrors SAS `PROC GLIMMIX` with
  `METHOD=LAPLACE` and quasi-Newton maximization (`TECHNIQUE=QUANEW`). The whole
  implementation must live in the single file
  `playground/logistic-regression.cpp` and run entirely in **plaintext** (no MPC /
  secret sharing for now).
- Why does this matter now? It provides a plaintext reference implementation and a
  correctness baseline before any future secure (MPC) port. It also validates the
  numerical approach (Laplace marginal likelihood + quasi-Newton) independently of
  the cryptographic runtime.

## Goals
- Goal 1: Fit a binary-outcome logistic GLMM with grouped/clustered data using the
  Laplace approximation to the marginal log-likelihood.
- Goal 2: Maximize the (approximate) marginal log-likelihood using a quasi-Newton
  optimizer (BFGS), matching the intent of GLIMMIX `METHOD=LAPLACE TECHNIQUE=QUANEW`.
- Goal 3: Keep everything self-contained in `playground/logistic-regression.cpp`
  using only the C++ standard library (no external linear-algebra dependency).
- Goal 4: Provide a runnable `main()` that demonstrates the fit on synthetic data
  with a known ground truth and prints the estimated fixed effects and variance
  component(s).

## Non-Goals
- Non-goal 1: Any MPC / secret-shared execution. Plaintext only for now.
- Non-goal 2: Full feature parity with GLIMMIX (no adaptive quadrature, no
  R-side/overdispersion structures, no multiple nested/crossed random effects in
  the first pass, no `CLASS`/formula parser).
- Non-goal 3: Production-grade CSV/data ingestion or a general formula interface.

## Scope
- In scope:
  - GLMM with a Bernoulli/logit conditional model.
  - Random effects per group (see Alternatives for intercept-only vs. general).
  - Inner solver for the conditional modes of the random effects (Newton /
    penalized IRLS).
  - Laplace approximation of the per-group integral and the summed marginal
    log-likelihood.
  - Outer quasi-Newton (BFGS) maximization over fixed effects + variance
    component(s).
  - Small dense linear algebra (Cholesky, solve, log-determinant) implemented
    in-file for the low-dimensional matrices involved.
  - Synthetic data generator + `main()` demo and basic sanity checks.
- Out of scope: distributed execution, GPU, sparse solvers, alternative link
  functions, non-Gaussian random effects.

## Impact Assessment
- User impact: Adds a standalone plaintext estimator; no change to existing code.
- Performance impact: New executable only; matrices are small (per-group random
  effect dimension q and fixed-effect dimension p are small), so cost is modest.
- Security and privacy impact: None (plaintext, no secrets, synthetic data).
- Backward compatibility impact: None; single new self-contained source file. The
  file is picked up automatically by `file(GLOB EXE_PLAYGROUND playground/*.cpp)`
  in `CMakeLists.txt`.

## Context
- Relevant files and modules:
  - `playground/logistic-regression.cpp` (currently empty; target file).
  - `CMakeLists.txt` (lines ~241, ~254): playground sources are globbed and each is
    built as its own executable via `configure_target`, which force-includes the
    project PCH (`include/pch.h`) and links the full framework + MPI.
- Dependencies: C++ standard library only for the algorithm. The build still
  force-includes `include/pch.h` (which pulls `mpi.h`, NTL, sodium, cryptoTools)
  and links `LINK_LIBRARIES`. We will **not** call any framework/MPI APIs, so no
  `MPI_Init` is required to run the resulting binary standalone.
- Constraints and assumptions:
  - Everything in one `.cpp` file.
  - No new third-party libraries (Eigen is not part of the core build; only used in
    `sosp-replication`), so linear algebra is hand-rolled.
  - Target C++20 or later, per repo C++ guidelines.

## Statistical / Numerical Design
Model (per observation j in group i):
- Binary outcome $y_{ij} \in \{0,1\}$.
- Fixed effects design row $x_{ij}$ (length $p$), coefficients $\beta$.
- Random effects design row $z_{ij}$ (length $q$), group effect $u_i \sim N(0, G)$.
- Conditional mean: $\operatorname{logit}(p_{ij}) = x_{ij}^\top \beta + z_{ij}^\top u_i$.

Marginal log-likelihood (random effects integrated out):
$$\ell(\beta, \theta) = \sum_i \log \int \exp\big(g_i(u)\big)\, du,$$
with
$$g_i(u) = \sum_j \big[y_{ij}\log p_{ij} + (1-y_{ij})\log(1-p_{ij})\big]
  - \tfrac12 u^\top G^{-1} u - \tfrac12 \log\det(2\pi G).$$

Laplace approximation: let $\hat u_i = \arg\max_u g_i(u)$ (found by inner Newton),
and $H_i = -\nabla^2 g_i(\hat u_i) = Z_i^\top W_i Z_i + G^{-1}$ with
$W_i = \operatorname{diag}(p_{ij}(1-p_{ij}))$. Then
$$\log\int \exp(g_i(u))\,du \approx g_i(\hat u_i) + \tfrac{q}{2}\log(2\pi)
  - \tfrac12 \log\det H_i.$$

Outer objective: maximize $\sum_i$ of the above over $\beta$ and the variance
component(s) $\theta$. To keep $G$ positive definite, $\theta$ is unconstrained via
a log/Cholesky parameterization (e.g. random-intercept: $\theta = \log\sigma^2$).

Optimizers:
- Inner (per group): Newton–Raphson / penalized IRLS to find $\hat u_i$ (a few
  iterations; exact Hessian available in closed form).
- Outer: BFGS quasi-Newton with a backtracking (Armijo/Wolfe) line search. Gradient
  of the outer objective computed by central finite differences (see Alternatives).

Reported output: converged $\hat\beta$, variance component estimate(s), final
marginal log-likelihood, iteration count, and (optionally) standard errors from the
inverse of the outer Hessian approximation.

## Alternatives

### Option A: Random-intercept only (q = 1 per group)
Pros:
- Simplest and by far the most common GLMM; matches the canonical GLIMMIX example.
- Scalar $G = \sigma^2$; inner problem is 1-D per group (fast, robust).
- Easiest to validate against a known truth.
Cons:
- No random slopes; less general.

### Option B: General random effects (arbitrary q, full covariance G)
Pros:
- Supports random slopes and correlated random effects.
- Closer to full GLIMMIX generality.
Cons:
- Requires Cholesky parameterization of $G$, more linear algebra, more edge cases,
  and more validation effort — larger, riskier first pass.

### Option C: Gradient strategy — numerical (finite differences) vs. analytic
Pros (numerical): much simpler, less error-prone, adequate for small parameter
vectors; easy to get BFGS working reliably.
Cons (numerical): more objective evaluations; slightly less precise gradients.
Pros (analytic): fewer evaluations, more precise. Cons: significantly more math and
code (differentiating the Laplace term through $\hat u_i$).

### Recommendation
- Recommended option: **Option A (random intercept)** for the first implementation,
  with the code organized so a later extension to Option B is feasible; and
  **Option C: numerical (central finite-difference) gradient** for the BFGS outer
  loop. This is the smallest, most verifiable implementation that faithfully
  reproduces `METHOD=LAPLACE` + quasi-Newton behavior.
- Why preferred: minimizes risk and code size while covering the dominant use case,
  and keeps the file self-contained and easy to validate.
- Open questions needing confirmation (see below).

## Open Questions (need confirmation before coding)
1. Random-effects structure: is **random intercept only (Option A)** acceptable for
   the first pass, or do you require general random slopes (Option B) now?
2. Data source: OK to **generate synthetic data in-file** with a fixed seed and a
   known ground truth for the demo, or do you want it to read a specific
   dataset/CSV? If a dataset, please provide the format/path.
3. Output/inference: point estimates + final log-likelihood only, or also
   **standard errors / z-tests** for the fixed effects (GLIMMIX-style solution
   table)?
4. Gradient: is a **numerical (finite-difference) gradient** for BFGS acceptable
   (Option C), or do you specifically want an analytic gradient?
5. Optimizer detail: plain **BFGS with backtracking line search** in-file is the
   plan. Any preference for L-BFGS, a trust-region variant, or specific
   convergence tolerances to match GLIMMIX defaults?

## Implementation Plan (pending approval)
1. Define data structures: per-group observations (X rows, Z rows, y), and a model
   config (p, q, parameterization).
2. Implement small dense linear algebra helpers (Cholesky, triangular solve,
   log-determinant, matrix-vector ops) for the low-dimensional matrices.
3. Implement the conditional log-likelihood, gradient, and Hessian in $u$; inner
   Newton solver for $\hat u_i$.
4. Implement the per-group Laplace term and the summed marginal log-likelihood as a
   function of $(\beta, \theta)$ with the chosen parameterization.
5. Implement the BFGS outer optimizer with a backtracking line search and
   central finite-difference gradients; convergence on gradient norm / objective
   change.
6. Implement a synthetic data generator with known truth and a `main()` that fits
   the model and prints results; add lightweight assertions/sanity checks.
7. Build via CMake, run, and verify recovered parameters are close to the truth.

## Risks and Mitigations
- Risk: Inner Newton diverges for extreme starting points / separation.
  - Mitigation: step damping / fallback to a few IRLS iterations; sensible init at
    $u=0$.
- Risk: Variance component drifts to 0 (boundary) causing $\log\det$ issues.
  - Mitigation: log-variance parameterization keeps $\sigma^2 > 0$; guard tiny
    values.
- Risk: BFGS line search failure.
  - Mitigation: Armijo backtracking with curvature check; reset Hessian to identity
    on failure.
- Risk: Framework/MPI linkage side effects when running standalone.
  - Mitigation: avoid all framework/MPI calls; verify the binary runs without
    `mpirun`.

## Rollback and Recovery
- Rollback plan: revert/empty `playground/logistic-regression.cpp`; no other files
  change (aside from this task doc and the ledger).
- Recovery steps: rebuild playground target.
- Monitoring and alerts after release: N/A (offline reference tool).

## Validation Plan
- Automated checks: assertions that recovered $\hat\beta$ and $\hat\sigma^2$ are
  within tolerance of the synthetic ground truth; monotone increase of the marginal
  log-likelihood across accepted BFGS steps.
- Manual verification: compare estimates/log-likelihood against a reference (e.g.,
  R `lme4::glmer` or SAS GLIMMIX Laplace) on the same synthetic data, if available.
- Expected success criteria: converges to near-truth estimates on synthetic data;
  builds cleanly with project warning flags.

## Definition of Done
- DoD criterion 1: `playground/logistic-regression.cpp` compiles under the project
  build and runs standalone, printing fixed-effect estimates, variance
  component(s), and the final marginal log-likelihood.
- DoD criterion 2: On synthetic data with known truth, estimates are within an
  agreed tolerance.
- Tests updated or added: lightweight in-file sanity assertions (no separate test
  target required unless requested).
- Documentation updated: this task doc and `tasks/tasks.md` ledger.

## Approval
- [x] Task document reviewed
- [x] Approved to implement
- Approver: User
- Approval date: 2026-07-30

### Confirmed decisions
1. Random-effects structure: **random intercept only** (Option A).
2. Data source: **synthetic in-file** with fixed seed and known ground truth.
3. Output: **point estimates + final marginal log-likelihood** (no standard errors
   in this pass).
4. Gradient: **numerical central finite differences** for the BFGS outer loop
   (Option C).
5. Optimizer: **BFGS with backtracking line search** in-file.

## Change Log
- 2026-07-30: Initial draft created.
- 2026-07-30: Approved with recommended options (random intercept, synthetic data,
  estimates + log-likelihood, numerical gradient, BFGS). Status → Approved.
- 2026-07-30: Implemented in playground/logistic-regression.cpp; compiles clean
  with -Wall -Wextra -Wpedantic and recovers the synthetic ground truth
  (beta ~= -0.43/1.01/-0.73 vs -0.5/1.0/-0.75, sigma ~= 0.698 vs 0.7) in 13 BFGS
  iterations. Status → Done.
