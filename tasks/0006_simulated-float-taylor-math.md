# SimulatedFloat Taylor-Math Compatibility

## How to Use
Design document for the task. See Approval for sign-off status.

## Metadata
- Task ID: 0006
- Title: Make Taylor math compile and run with `SimulatedFloat`
- Requested by: User
- Owner: Copilot
- Date: 2026-08-15
- Status: Done
- Estimated effort: 1 focused implementation pass + build/runtime validation
- Related documents:
  - tasks/0002_taylor-series-math-approximations.md
  - tasks/0005_simulated-float-value-type.md
- Branch: logistic-regression

## Problem Statement
- What is the request? `playground/logistic-regression.cpp` fails to compile when
  Taylor math and `SimulatedFloat` are both enabled.
- Why does this matter now? The simulated fixed-precision mode is intended to
  exercise arithmetic without native transcendental functions.

## Goals
- Goal 1: Compile the Taylor implementations of `Exp`, `Log`, and `Log1p` with
  `SimulatedFloat`.
- Goal 2: Preserve native `double` and `float` Taylor behavior.
- Goal 3: Run the existing optimizer convergence and estimate assertions in the
  combined Taylor-plus-simulated configuration.

## Non-Goals
- Non-goal 1: Increase the Taylor series order or change its approximation constants.
- Non-goal 2: Redesign `SimulatedFloat` arithmetic or overflow behavior.
- Non-goal 3: Change the statistical model.

## Scope
- In scope: Taylor-specific value declarations and range-reduction conversion in
  `playground/logistic-regression.cpp`, tests/validation, and task documentation.
- Out of scope: Shared MPC math primitives and unrelated executables.

## Impact Assessment
- User impact: The Taylor and simulated-value options can be enabled together.
- Performance impact: None beyond the already selected Taylor/simulated modes.
- Security and privacy impact: None; this remains a plaintext prototype.
- Backward compatibility impact: Native modes retain their existing formulas.

## Context
- Relevant files and modules: `playground/logistic-regression.cpp` and
  `include/simulated_float.h`.
- Dependencies: C++20 standard library only.
- Reproduced failures: Five `constexpr Value` declarations cannot quantize through
  the non-constexpr floating constructor, and Taylor `Exp` cannot cast
  `SimulatedFloat` directly to `long`.

## Alternatives
### Option A: Adapt the Taylor call sites (recommended)
Use runtime `const Value` declarations and convert the range-reduction quotient
through the existing `ToDouble` adapter before converting it to `long`.

Pros:
- Minimal and local to code that requires native conversion.
- Does not broaden `SimulatedFloat`'s implicit conversion surface.
- Preserves the existing Taylor formulas.

Cons:
- Taylor range-reduction integer selection temporarily uses native `double`.

### Option B: Add general integral conversions to `SimulatedFloat`
Add conversion operators that allow direct casts to integral types.

Pros:
- Makes the existing Taylor cast compile unchanged.

Cons:
- Broadens the type API for one call site.
- Makes truncating conversions easier to invoke elsewhere accidentally.

### Recommendation
- Recommended option: Option A.
- Why this option is preferred: It fixes the owning Taylor path with the smallest
  API and behavioral change.
- Open questions needing confirmation: None.

## Implementation Plan
1. Change Taylor `Value` constants that require decimal quantization from
   `constexpr` to `const`.
2. Convert the `Exp` range-reduction quotient through `ToDouble` before selecting
   the integral exponent.
3. Compile with warnings as errors and run Taylor-plus-simulated logistic regression.
4. Recheck native Taylor and focused `SimulatedFloat` arithmetic behavior.

## Risks and Mitigations
- Risk: Coarse Taylor approximations plus fixed-point rounding may prevent optimizer
  convergence even after compilation succeeds.
  - Mitigation: Run the existing end-to-end assertions and adjust only a confirmed
  numerical compatibility issue.
- Risk: The quotient conversion changes range-reduction rounding.
  - Mitigation: It preserves the represented numeric value and the existing
  nearest-integer formula.

## Rollback and Recovery
- Rollback plan: Revert the Taylor-local declaration and conversion changes.
- Recovery steps: Build and run the previous native and simulated non-Taylor modes.
- Monitoring and alerts after release: None.

## Validation Plan
- Automated checks: Warning-clean Taylor-plus-simulated compile and execution;
  focused `SimulatedFloat` tests; native Taylor compile/runtime check.
- Manual verification: Inspect final convergence status and parameter estimates.
- Expected success criteria: Combined mode compiles, converges, and passes existing
  estimate assertions without regressing the other checked modes.

## Definition of Done
- DoD criterion 1: Taylor math compiles with `SimulatedFloat`.
- DoD criterion 2: Combined mode passes runtime assertions.
- DoD criterion 3: Native Taylor and focused arithmetic checks still pass.
- Tests updated or added: Existing executable assertions and focused arithmetic test.
- Documentation updated: This task document and ledger.

## Approval
- [x] Task document reviewed
- [x] Approved to implement
- Approver: User
- Approval date: 2026-08-15

## Change Log
- 2026-08-15: Initial draft created after reproducing six compile errors in the
  Taylor-plus-simulated configuration.
- 2026-08-15: Approved the local Taylor call-site adaptation; implementation started.
- 2026-08-15: Replaced Taylor-only `constexpr Value` declarations with runtime
  constants and adapted exponent selection through `ToDouble`; the warning-clean
  combined mode converges in 26 iterations and passes all estimate assertions.