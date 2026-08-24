# Simulated Fixed-Precision Value Type

## How to Use
Design document for the task. See Approval for sign-off status.

## Metadata
- Task ID: 0005
- Title: Add a `SimulatedFloat` numeric type to the logistic-regression prototype
- Requested by: User
- Owner: Copilot
- Date: 2026-08-15
- Status: Done
- Estimated effort: 1 implementation pass + focused unit/build validation
- Related documents:
  - tasks/0004_factor-out-logistic-regression-value-type.md
- Branch: logistic-regression

## Problem Statement
- What is the request? Add a `SimulatedFloat` type that stores an `int64_t` value
  and a precision, then allow the existing `Value` alias to select it in addition
  to `double` or `float`.
- Why does this matter now? The plaintext prototype needs to model fixed-precision
  arithmetic similar to the representation expected in a future MPC port.

## Goals
- Goal 1: Represent decimal fixed-point values with signed 64-bit storage and an
  explicit precision.
- Goal 2: Support the construction, conversion, comparison, arithmetic, and math
  operations exercised by `playground/logistic-regression.cpp`.
- Goal 3: Define precision propagation and rescaling rules explicitly.
- Goal 4: Preserve the existing `double` and `float` configurations.

## Non-Goals
- Non-goal 1: Replace numeric types throughout the CryptDough library.
- Non-goal 2: Implement secure or secret-shared arithmetic.
- Non-goal 3: Emulate IEEE-754 special values in the integer representation.

## Scope
- In scope: A focused `SimulatedFloat` type, its tests, integration with the local
  logistic-regression `Value` alias, and task documentation.
- Out of scope: Unrelated executables and public MPC APIs.

## Impact Assessment
- User impact: The prototype can exercise deterministic fixed-precision integer
  arithmetic by selecting `SimulatedFloat` as `Value`.
- Performance impact: Arithmetic requires checked rescaling and may be slower than
  native floating-point arithmetic.
- Security and privacy impact: None; the representation remains plaintext.
- Backward compatibility impact: Native `double` remains the default.

## Context
- Relevant files and modules: `playground/logistic-regression.cpp`, a new focused
  header for the type, and a focused test executable or existing test target.
- Dependencies: C++20 standard library only.
- Constraints and assumptions: Values use decimal fixed-point scaling
  `real = data / 10^precision`. The original `{1000, 5} == 0.1` example is
  corrected to `{1000, 4} == 0.1`.

## Alternatives
### Option A: Decimal fractional digits (recommended)
Interpret `precision` as the number of decimal digits after the decimal point:
`real = data / 10^precision`. Thus `{1000, 2}` is `10.00`, `{100, 2}` is `1.00`,
and their product, rescaled to precision 2, is `{1000, 2}`.

Pros:
- Exactly matches the requested multiplication example.
- Uses the conventional decimal fixed-point meaning of precision.
- Rescaling rules are direct and testable.

Cons:
- `{1000, 5}` represents `0.01`, conflicting with the stated `0.1` example.

### Option B: Precision includes the units digit
Interpret the scale as `10^(precision - 1)`, making `{1000, 5}` equal `0.1`.

Pros:
- Exactly matches the requested conversion example.

Cons:
- Does not match the requested multiplication result: `{1000, 2}` times
  `{100, 2}` would not naturally produce `{1000, 2}`.
- Uses an unusual meaning of precision.

### Recommendation
- Recommended option: Option A, with the first example corrected to
  `{data: 1000, precision: 4} == 0.1` or its data corrected to `10000` at
  precision 5.
- Why this option is preferred: It gives conventional fixed-point semantics and
  exactly explains the multiplication example.
- Open questions needing confirmation: None. Mixed-precision results preserve the
  greater operand precision, and overflow is intentionally unchecked.

## Implementation Plan
1. Add `SimulatedFloat` with `int64_t data` and precision storage, conversion from
   numeric values, conversion for output/math interoperability, and checked scale
   helpers.
2. Implement unary, arithmetic, compound-assignment, and comparison operators;
  multiplication and division round back to the greater operand precision.
3. Add focused tests for representation, signs, mixed precision, rounding,
   overflow, and the two confirmed examples.
4. Select `SimulatedFloat` through the logistic-regression `Value` alias and add
   only the math compatibility needed by that executable.
5. Build and run the focused tests plus native and simulated logistic-regression
   configurations.

## Risks and Mitigations
- Risk: The two supplied examples imply different scale rules.
  - Mitigation: Obtain approval for one explicit representation formula first.
- Risk: Scaling and multiplication can overflow 64-bit intermediate values.
  - Mitigation: Overflow is intentionally left unchecked per the approved policy;
    CMake builds use the repository's existing `-fwrapv` compiler flag.
- Risk: Integer quantization may prevent the optimizer from meeting native
  floating-point convergence tolerances.
  - Mitigation: Define tolerances from the selected precision and validate the
  simulated configuration separately.

## Rollback and Recovery
- Rollback plan: Remove the new type and restore the local native `Value` alias.
- Recovery steps: Rebuild and rerun the native playground executable.
- Monitoring and alerts after release: None.

## Validation Plan
- Automated checks: Focused representation/operator tests; compile with warnings
  as errors and execute logistic regression with `double`, `float`, and
  `SimulatedFloat` value types.
- Manual verification: Confirm documented decimal values and raw results for the
  approved examples.
- Expected success criteria: Arithmetic has deterministic documented scaling,
  checked failure behavior, and does not regress native builds.

## Definition of Done
- DoD criterion 1: `SimulatedFloat` stores `int64_t` data and explicit precision.
- DoD criterion 2: Required operators and conversions have documented semantics.
- DoD criterion 3: The logistic-regression prototype builds and runs with the type.
- Tests updated or added: Focused unit tests and executable validation.
- Documentation updated: This task document and ledger.

## Approval
- [x] Task document reviewed
- [x] Approved to implement
- Approver: User
- Approval date: 2026-08-15

## Change Log
- 2026-08-15: Initial draft created; recorded the conflict between the supplied
  conversion and multiplication examples.
- 2026-08-15: Approved decimal fractional-digit scaling, maximum operand result
  precision, nearest rounding, and unchecked overflow; implementation started.
- 2026-08-15: Implemented the type and optimizer adapters; focused arithmetic and
  all three logistic-regression value modes compile cleanly and pass at runtime.