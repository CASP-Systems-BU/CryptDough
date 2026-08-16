# Factor Out Logistic-Regression Value Type

## How to Use
Design document for the task. See Approval for sign-off status.

## Metadata
- Task ID: 0004
- Title: Factor out the stored numeric value type in the plaintext logistic-regression prototype
- Requested by: User
- Owner: Copilot
- Date: 2026-08-14
- Status: Done
- Estimated effort: 1 implementation pass + build validation
- Related documents:
  - tasks/0001_mixed-effects-logistic-regression-laplace.md
  - tasks/0002_taylor-series-math-approximations.md
  - tasks/0003_robust-conditional-mode-and-line-search.md
- Branch: logistic-regression

## Problem Statement
- What is the request? The logistic-regression prototype uses `double` directly for
  stored vectors, dataset values, model parameters, and intermediate scalar values.
- Why does this matter now? The implementation should be easy to build with
  `float` when lower precision or a future MPC-compatible value type is desired.

## Goals
- Goal 1: Introduce one local `Value` type alias that can select `double` or `float`.
- Goal 2: Use the alias consistently for numeric data and model computations.
- Goal 3: Keep numerical differentiation and optimizer convergence behavior valid
  for the selected value precision.
- Goal 4: Ensure Taylor math terminates safely when float underflow creates an
  invalid logarithm input.

## Non-Goals
- Non-goal 1: Changing the statistical model or optimizer behavior.
- Non-goal 2: Refactoring shared library types or unrelated executables.
- Non-goal 3: Adding a build-system option for selecting the precision.

## Scope
- In scope: `playground/logistic-regression.cpp` and this task ledger/document.
- Out of scope: other source files and public library APIs.

## Impact Assessment
- User impact: Precision is selected by the local `Value` alias; both `double` and
  `float` configurations are supported.
- Performance impact: A `float` build may reduce storage and alter numerical precision.
- Security and privacy impact: None; this prototype runs in plaintext.
- Backward compatibility impact: Existing default builds remain `double` builds.

## Alternatives
### Option A: Local `Value` alias (recommended)
Pros:
- Minimal change with no build-system or public-API changes.
- Keeps the precision choice visible at the implementation boundary.

Cons:
- Selecting `float` requires editing the alias.

### Option B: CMake precision option
Pros:
- Precision can be selected from the build command.

Cons:
- Expands scope into build configuration for a single prototype.
- Adds configuration complexity before there is a broader precision policy.

### Recommendation
- Recommended option: Option A.
- Why this option is preferred: It satisfies the request while preserving the existing build and behavior.
- Open questions needing confirmation: None for the scoped prototype change.

## Implementation Plan
1. Add `using Value = double` beside the existing vector aliases.
2. Replace stored/model scalar `double` uses with `Value`, including math wrappers,
   distributions, constants, and numeric containers.
3. Scale finite-difference and convergence tolerances from
  `std::numeric_limits<Value>::epsilon()`.
4. Prevent rounded-zero objective changes from causing false convergence, while
  recognizing steps that are unrepresentable at the selected precision.
5. Guard Taylor `Log` against zero and non-finite inputs so invalid trials return
  a non-finite objective instead of entering an unbounded range-reduction loop.
6. Build and run the playground target with both `double` and `float` aliases.

## Risks and Mitigations
- Risk: A missed `double` can make the float mode inconsistent.
  - Mitigation: Search the edited file for remaining value-level `double` declarations.
- Risk: Float precision may not satisfy the existing assertions.
  - Mitigation: Use a precision-aware finite-difference step and convergence
    tolerance; validate the float variant separately.
- Risk: Taylor `Exp` can underflow a positive value to zero in float mode, and
  passing that value to `Log` can otherwise prevent termination.
  - Mitigation: Validate the logarithm domain before range reduction and let the
    existing objective finiteness guard reject the trial.

## Rollback and Recovery
- Rollback plan: Revert the alias substitutions in the scoped source file.
- Recovery steps: Rebuild the playground target and rerun its executable assertions.
- Monitoring and alerts after release: None.

## Validation Plan
- Automated checks: Build and run the `logistic-regression` playground executable
  with both `double` and `float` aliases.
- Manual verification: Confirm the selected precision uses named, precision-aware
  numerical constants and that convergence is not triggered solely by a rounded
  objective value.
- Expected success criteria: Both configurations compile, run, converge, and pass
  the existing parameter assertions.

## Definition of Done
- DoD criterion 1: Numeric storage and model calculations use `Value` consistently.
- DoD criterion 2: Both `double` and `float` builds pass their assertions.
- DoD criterion 3: The task ledger and document reflect completion.
- Tests updated or added: Existing executable assertions are reused.
- Documentation updated: This task document.

## Approval
- [x] Task document reviewed
- [x] Approved to implement
- Approver: User
- Approval date: 2026-08-14

## Change Log
- 2026-08-14: Initial draft created.
- 2026-08-14: Implemented the local `Value` alias; default build and float compilation/runtime checks completed.
- 2026-08-15: Added precision-aware finite-difference and convergence tolerances;
  extracted named constants and verified the float build passes its assertions.
- 2026-08-15: Guarded Taylor `Log` against zero/non-finite inputs after reproducing
  and fixing the float-plus-Taylor infinite loop; validation passes in 27 iterations.