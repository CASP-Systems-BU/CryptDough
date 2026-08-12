# Taylor-Series Math Approximations with a Compile-Time Toggle

## How to Use
Design document for the task. Implementation must not start until this document
is reviewed and approved (see Approval).

## Metadata
- Task ID: 0002
- Title: Replace direct `std::exp` / `std::log` / `std::log1p` calls in
  `playground/logistic-regression.cpp` with in-file Taylor-series approximations,
  selectable at compile time against the standard-library versions.
- Requested by: User
- Owner: TBD
- Date: 2026-07-31
- Status: Done
- Estimated effort: 1 implementation pass + accuracy validation
- Related documents: tasks/0001_mixed-effects-logistic-regression-laplace.md
- Branch: main

## Problem Statement
- What is the request? The estimator currently calls `std::exp`, `std::log`, and
  `std::log1p` directly (in `Sigmoid`, `LogOnePlusExp`, `GroupLaplaceLogLik`, and
  `UnpackParameters`). We want custom Taylor-series-based implementations of these
  three functions, plus a **compilation flag** that switches between the custom
  approximations and the `std` versions.
- Why now? Groundwork for a future secure (MPC) port, where transcendental
  functions are not available as primitives and must be approximated with
  polynomial series. A plaintext, toggleable reference lets us validate the
  approximation quality against `std` before any secure port.

## Goals
- Goal 1: In-file `Exp`, `Log`, `Log1p` implemented from Taylor series (with the
  range reduction needed to stay accurate over the model's input range).
- Goal 2: A single compile-time switch selecting custom vs. `std` implementations,
  defaulting to `std` so existing behavior is unchanged unless the flag is set.
- Goal 3: All existing `std::exp` / `std::log` / `std::log1p` call sites route
  through the new wrappers. (`std::sqrt` and `std::abs` are out of scope — not in
  the requested set.)
- Goal 4: With the custom path enabled, the demo still recovers the ground truth
  and all sanity assertions still pass.

## Non-Goals
- Non-goal 1: Replacing `std::sqrt`, `std::abs`, `std::max`, or RNG facilities.
- Non-goal 2: Any MPC / secret-shared implementation (still plaintext).
- Non-goal 3: A general-purpose libm replacement beyond these three functions and
  the input ranges this program actually exercises.

## Scope
- In scope: the three functions, the toggle, rerouting call sites, accuracy
  checks against `std` over the exercised ranges.
- Out of scope: other files, other math functions, changing the statistical model.

## Where the functions are used (input ranges)
- `std::exp`:
  - `Sigmoid` via `exp(-|eta|)` — argument ≤ 0, magnitude up to roughly |eta| ~ 20.
  - `UnpackParameters`: `exp(2*s)` with `s = log(sigma)` — argument roughly in
    [-6, 3].
- `std::log1p`: only in `LogOnePlusExp`, and by construction its argument is
  `exp(non-positive)` ∈ (0, 1].
- `std::log`: `GroupLaplaceLogLik` on `sigma2` (>0, ~0.01–10) and `curvature`
  (>0, ~ up to a few tens).

Implication: a **naive** Taylor expansion about 0 is inadequate — `exp` about 0
loses accuracy for large |x|, and the `log(1+t)` series diverges/converges far too
slowly as `t → 1`. Range reduction is required for a correct, accurate result.

## Numerical Design (proposed)
- `Exp(x)`: range-reduce `x = k·ln2 + r`, `k = round(x/ln2)`, so `|r| ≤ ln2/2 ≈
  0.347`; evaluate `exp(r) = Σ_{n≥0} r^n/n!` (fast convergence for small `r`) and
  scale by `2^k`.
- `Log(x)` (x > 0): reduce `x = m·2^e` with `m` centered near 1, then use the
  Taylor/atanh identity `log(m) = 2·Σ_{n odd} w^n/n`, `w = (m-1)/(m+1)` (this is
  the Taylor series of `log`, and `|w|` stays small after centering, giving fast
  convergence); `Log(x) = e·ln2 + log(m)`.
- `Log1p(x)`: computed as `Log(1 + x)` for robustness over the (0, 1] range used
  here (avoids the slow alternating `log(1+t)` series near `t = 1`).
- Series termination: iterate until the next term is below a relative tolerance
  (e.g., 1e-15) or a small fixed max term count — both bounded because the reduced
  arguments are small.

Open design choice: the range reduction needs to split off a power of two. This
can be done with `std::frexp` / `std::ldexp` (pure exponent manipulation, no
transcendental math) or with a fully manual multiply/divide-by-two loop (zero libm
dependency). See Alternatives.

## Alternatives

### Flag mechanism
- Option F1 (recommended): in-file macro `USE_TAYLOR_MATH`, default `0`
  (std path). Enable by compiling with `-DUSE_TAYLOR_MATH=1`. No shared build files
  touched; works with the existing `make logistic-regression` flow via extra flags.
  - Pros: self-contained, no impact on other targets, matches "compilation flag".
  - Cons: user passes the define manually (e.g., via `CXXFLAGS`/`CMAKE_CXX_FLAGS`).
- Option F2: add a CMake `option(USE_TAYLOR_MATH ...)` to `CMakeLists.txt` that
  injects the define.
  - Pros: toggle via `cmake -DUSE_TAYLOR_MATH=ON`.
  - Cons: edits a shared build file that affects the whole project; per the repo's
    "don't restructure without confirmation" rule this needs explicit sign-off.

### Range reduction primitive
- Option R1 (recommended): use `std::frexp`/`std::ldexp` for the power-of-two
  split. These are not transcendental (not in the banned set) and are exact.
- Option R2: fully manual multiply/divide-by-two loops (no libm at all), at a
  small cost in code and speed.

### log1p strategy
- Option L1 (recommended): `Log1p(x) = Log(1 + x)` (robust for our (0,1] range).
- Option L2: dedicated small-`x` Taylor `x − x²/2 + x³/3 − …` with a fallback to
  `Log(1+x)` for larger `x`.

### Recommendation
- **F1** (in-file macro, default std) + **R1** (frexp/ldexp) + **L1**
  (log1p via Log). Smallest, most accurate, no shared-file edits. If you prefer a
  pure-libm-free build, switch R1 → R2; if you want a CMake switch, add F2 as well.

## Open Questions (confirm before coding)
1. Flag: OK with **F1** (in-file `USE_TAYLOR_MATH`, default off, set via
   `-DUSE_TAYLOR_MATH=1`)? Also add the CMake option (F2)?
2. Range reduction: OK to use `std::frexp`/`std::ldexp` (R1), or must the custom
   path avoid all libm and use manual loops (R2)?
3. Default behavior: keep **std** as the default so unflagged builds are unchanged?
4. Accuracy bar: is "recovers ground truth + assertions pass, and max abs error vs
   `std` below ~1e-9 over the exercised ranges" an acceptable success criterion?

## Implementation Plan (pending approval)
1. Add the `USE_TAYLOR_MATH` macro guard and constants (`ln2`, etc.).
2. Implement `Exp`, `Log`, `Log1p` (custom, range-reduced Taylor) and the `std`
   fallbacks behind the switch, as thin wrappers.
3. Replace the four `std::exp` / `std::log` / `std::log1p` call sites with the
   wrappers.
4. (If approved) add an optional accuracy self-check comparing custom vs. `std`
   over sample points, printed or asserted.
5. Build both ways (`std` and `-DUSE_TAYLOR_MATH=1`), run, confirm estimates and
   assertions.

## Risks and Mitigations
- Risk: accuracy loss changing estimates / breaking asserts.
  - Mitigation: range reduction + tolerance-based term counts; validate max error
    vs `std` and rerun the fit.
- Risk: `Log`/`Exp` on out-of-range/degenerate inputs (e.g., non-positive log arg).
  - Mitigation: arguments are provably positive here (sigma2, curvature, 1+x);
    guard defensively if cheap.

## Validation Plan
- Build and run with the default (std) path — unchanged output.
- Build with `-DUSE_TAYLOR_MATH=1` — estimates within tolerance of truth, all
  assertions pass, marginal log-likelihood close to the std run.
- Optional: max abs error of each function vs `std` over sampled points in range.

## Definition of Done
- Custom `Exp`/`Log`/`Log1p` implemented from Taylor series with range reduction.
- Compile-time toggle works both ways; std is the default.
- All original call sites routed through wrappers; program builds and passes
  assertions under both settings.

## Approval
- [x] Task document reviewed
- [x] Approved to implement
- Approver: User
- Approval date: 2026-07-31

### Confirmed decisions
1. Flag: **F1** — in-file macro `USE_TAYLOR_MATH`, default `0` (std path); enable
   with `-DUSE_TAYLOR_MATH=1`. No CMake option added (shared CMakeLists untouched).
2. Range reduction: **R2** — fully manual multiply/divide-by-two loops; the custom
   path calls no libm functions.
3. Default: **std** library path when the flag is unset (unchanged behavior).
4. log1p: **L1** — `Log1p(x) = Log(1 + x)`.

## Change Log
- 2026-07-31: Initial draft created.
- 2026-07-31: Approved with F1 + R2 + std-default + L1. Status → Approved.
- 2026-07-31: Implemented in playground/logistic-regression.cpp. Added toggleable
  `Exp`/`Log`/`Log1p` wrappers (custom range-reduced Taylor series with manual
  power-of-two reduction, no libm on the custom path) and routed all four
  `std::exp`/`std::log`/`std::log1p` call sites through them. Verified three ways
  (default std, standalone `-DUSE_TAYLOR_MATH=1`, and in-tree CMake with the flag):
  identical estimates and marginal log-likelihood (-4143.7827), all assertions
  pass, clean under -Wall -Wextra -Wpedantic. Status → Done.

## Usage
- Default (std library): `make logistic-regression`.
- Custom Taylor path: configure with the define, e.g.
  `cmake -DCMAKE_CXX_FLAGS="-DUSE_TAYLOR_MATH=1" .` then `make logistic-regression`
  (revert with `cmake -DCMAKE_CXX_FLAGS="" .`).
