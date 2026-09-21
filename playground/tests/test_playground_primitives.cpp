// Assertion-based tests for playground/library/primitives.h.
//
// Counterpart to `tests/test_primitives.cpp`, which does the same job for the
// core library. The difference in instrument is forced by the subject: the core
// operators are exact integer circuits and are checked with `==`, while
// everything here is fixed-point at `precision` 16 and is checked against a
// plaintext reference within a stated tolerance. See `test_util.h` for why the
// oracle is `std::` rather than a recorded output.
//
// WHAT REPLACED WHAT. `playground/secure-logistic-regression.cpp` already
// evaluates most of these operators and prints an error table. That table is
// the evidence the tolerances below are calibrated against, and it stays where
// it is -- but it cannot fail, so it cannot catch a regression. This file can.
//
// PRIVACY: a test, not a privacy demonstration. It opens intermediate values
// freely. Nothing here supports any claim in semantic topic 0003.
//
// Run: mpirun -np 3 ./test_playground_primitives
//      (or ../scripts/run_experiment.py -p 3 test_playground_primitives)

#include <algorithm>
#include <cmath>
#include <vector>

#include "cdough.h"

#include "../library/primitives.h"
#include "test_util.h"

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;
using namespace cdough::debug;
using namespace cdough::service;
using namespace cdough::regression;
using namespace playground_test;

namespace {

// ---------------------------------------------------------------------------
// Tolerances
// ---------------------------------------------------------------------------
//
// CALIBRATION. Each bound below was first derived a priori (from the 1.526e-5
// fixed-point resolution, from a figure measured by
// `playground/secure-logistic-regression.cpp` and recorded in semantic task
// 0019, or from an error-composition argument), then TIGHTENED once against the
// measured worst error, which is quoted beside it. The ratio is kept at roughly
// 5x to 15x: close enough that a real regression trips it, loose enough to
// survive a different build configuration.
//
// The measurements come from the 3PC build (PROTOCOL=3, DEFAULT_BITWIDTH=32,
// TRIPLES=DUMMY, DIVISION_CORRECTION=ON) on 2026-09-20, and were bit-identical
// across four consecutive runs -- the truncation error in this configuration is
// deterministic, not sampled. That determinism is a property of THIS build and
// must not be assumed for TRIPLES=REAL or another protocol; the margins below
// exist partly to absorb that.
//
// No bound was ever widened to make a failing check pass. Where a measured worst
// error sits far below its bound, the bound is tightened in a later revision
// WITH the measurement recorded.

// Operators that are pure fixed-point rearrangement -- no series, no division,
// no range reduction. Their only error is the rounding of the inputs at share
// time and single-step truncation, so a handful of ULPs is the whole budget.
constexpr double kExactish = 4.0 * kResolution;  // ~6.1e-5

// The boolean long-division circuit truncates toward zero, so a quotient carries
// up to one ULP from the division plus the operand rounding. Relative, because
// a reciprocal ranges over orders of magnitude.
constexpr double kDivision = 1.0e-4;  // measured worst 9.16e-06 (rel)

// Exp and Log. The measured absolute error in secure-logistic-regression.cpp is
// ~1e-4 over the tested range; the relative form is used for Exp because its
// output spans e^-2 to e^2 here. 1e-3 leaves an order of magnitude of margin
// over the measurement, which is the right posture for a first calibration.
constexpr double kExpTolerance = 1.0e-3;   // relative; measured worst 2.99e-04 -- left as derived
constexpr double kLogTolerance = 5.0e-4;   // absolute; measured worst 6.76e-05 (Log), 4.19e-05 (Log1p)

// Sigmoid and LogOnePlusExp compose Exp with a division (Sigmoid) or with Log1p
// (LogOnePlusExp), so they inherit both errors.
constexpr double kSigmoidTolerance = 6.0e-4;      // absolute; measured worst 1.12e-04
constexpr double kSoftplusTolerance = 1.0e-3;     // absolute; measured worst 1.82e-04

// SecureSqrt is Exp(0.5 * Log(x)): the Log error is multiplied by the output
// magnitude on the way through Exp, so the bound is relative and roughly the sum
// of the two. Task 0019 chose the composed form knowing this and recorded the
// intent to measure it; this is that measurement, re-run on every invocation.
constexpr double kSqrtTolerance = 2.5e-3;  // relative; measured worst 4.74e-04

// NormalCdf. The A&S 26.2.17 approximation contributes ~7.5e-8, far below the
// format, so the budget is the fixed-point evaluation: one reciprocal, one Exp,
// a degree-5 Horner chain.
constexpr double kNormalCdfTolerance = 4.0e-4;  // absolute; measured worst 3.53e-05

// TwoSidedPValue is 2 * NormalCdf(-|z|), so it carries twice the CDF error.
constexpr double kPValueTolerance = 8.0e-4;  // absolute; measured worst 7.07e-05

// ---------------------------------------------------------------------------
// Randomized-check tolerances
// ---------------------------------------------------------------------------
// SEPARATE FROM THE BOUNDS ABOVE, AND FOR A DIFFERENT REASON. The tolerances in
// the preceding block guard the hand-picked edge cases, whose inputs are fixed
// and whose error is therefore deterministic (task 0020 F-02). These guard 2^14
// random draws across each operator's whole documented range, where two things
// change:
//
//   1. The worst case is genuinely larger. Experiment 0002 measured up to 11x the
//      coarse-grid figure for the operators whose transition band is narrow
//      (Sigmoid, LogOnePlusExp, NormalCdf, TwoSidedPValue). Reusing the
//      hand-picked bounds here would produce a test that fails on a good build.
//   2. The draw differs every run, so the bound must absorb sampling variation.
//      Measured spread across independent runs: <= 1.05x (experiment 0002).
//
// Each bound below is **3x the worst value experiment 0002 measured over 2^14
// random samples**, quoted beside it. Three is chosen against the 1.05x measured
// spread, leaving roughly a factor of three of genuine headroom.
//
// NOTE THAT SOME EXCEED THE 1.0e-3 ACCURACY BAR. That is correct and is not a
// weakening. The accuracy bar describes the OPERATOR -- it is what defines the
// documented range, and was frozen before any measurement. These describe the
// TEST, and must clear the operator's real worst case with margin. Conflating
// the two would either mis-document the library or produce a tripwire.
constexpr std::size_t kRandomSamples = 1u << 14;

constexpr double kRandExp = 2.5e-3;               // measured worst 8.22e-04
constexpr double kRandLog = 3.0e-4;               // measured worst 9.85e-05
constexpr double kRandLog1p = 3.0e-4;             // measured worst 9.93e-05
constexpr double kRandReciprocal = 5.0e-5;        // measured worst 1.53e-05
constexpr double kRandSqrt = 2.5e-3;              // measured worst 8.40e-04
constexpr double kRandSigmoid = 6.0e-4;           // measured worst 2.07e-04
constexpr double kRandSoftplus = 1.2e-3;          // measured worst 3.94e-04
constexpr double kRandNormalCdf = 6.0e-4;         // measured worst 1.93e-04
constexpr double kRandPValue = 1.2e-3;            // measured worst 3.93e-04

// ---------------------------------------------------------------------------
// Clone
// ---------------------------------------------------------------------------
// Clone is the library's materialization recipe and is relied on wherever a
// mapped view has to become a real buffer. Two things must hold: the values
// survive, and the copy is genuinely independent of the source.
void TestClone(EngineRef engine) {
    Section("Clone", engine);

    const std::vector<double> inputs = {-3.25, -1.0, -0.125, 0.0, 0.5, 1.75, 4.0, 9.5};
    AV original = ShareVector(inputs, engine);

    AV copy = Clone(original);
    CheckClose("Clone preserves values", OpenVector(copy), RoundTrip(inputs), kExactish, engine,
               ErrorMode::Absolute, inputs);

    // Independence: mutate the clone and confirm the source is untouched. A
    // Clone that returned a view would pass the value check above and fail here,
    // which is the failure mode the operator exists to prevent.
    copy += static_cast<DataType>(scale);
    CheckClose("Clone is independent of source", OpenVector(original), RoundTrip(inputs),
               kExactish, engine, ErrorMode::Absolute, inputs);

    std::vector<double> shifted = RoundTrip(inputs);
    for (double& value : shifted) {
        value += 1.0;
    }
    CheckClose("Clone mutation is local", OpenVector(copy), shifted, kExactish, engine);
}

// ---------------------------------------------------------------------------
// Sum
// ---------------------------------------------------------------------------
void TestSum(EngineRef engine) {
    Section("Sum", engine);

    const std::vector<double> inputs = {1.5, -2.25, 0.75, 4.0, -0.5, 3.125, -6.0, 0.25};
    const std::vector<double> rounded = RoundTrip(inputs);

    double total = 0.0;
    for (const double value : rounded) {
        total += value;
    }

    AV shared = ShareVector(inputs, engine);
    // Fixed-point addition is exact, so this is a near-exact check; a tolerance
    // here would only hide a wiring mistake.
    CheckClose("Sum", OpenVector(Sum(shared)), {total}, kExactish, engine);
}

// ---------------------------------------------------------------------------
// ClampAbs
// ---------------------------------------------------------------------------
void TestClampAbs(EngineRef engine) {
    Section("ClampAbs", engine);

    const double bound = 2.5;
    const DataType bound_scaled = static_cast<DataType>(std::llround(bound * scale));

    // Inputs deliberately straddle both edges and sit exactly on them: the
    // operator is two `gtez` comparisons and the boundary is where an
    // off-by-one in the `-bound - 1` term would show.
    const std::vector<double> inputs = {-6.0, -2.5, -2.4999, -1.0, 0.0, 1.0, 2.4999, 2.5, 7.75};

    std::vector<double> expected;
    expected.reserve(inputs.size());
    for (const double value : inputs) {
        expected.push_back(std::max(-bound, std::min(bound, RoundTrip(value))));
    }

    AV shared = ShareVector(inputs, engine);
    CheckClose("ClampAbs", OpenVector(ClampAbs(shared, bound_scaled)), expected, kExactish, engine,
               ErrorMode::Absolute, inputs);
}

// ---------------------------------------------------------------------------
// Multiplex
// ---------------------------------------------------------------------------
// The whole obliviousness argument of the optimizer rests on this one
// expression, which is exactly why its own comment says it is named rather than
// written inline eight times. It must be exact: it is a selection, not an
// approximation, and any rescaling error here is silent by the operator's own
// admission.
void TestMultiplex(EngineRef engine) {
    Section("Multiplex", engine);

    const std::vector<double> if_false = {1.0, -2.5, 0.25, 8.0, -0.125, 3.5};
    const std::vector<double> if_true = {-7.0, 4.25, -0.5, 0.0, 6.75, -1.25};
    const std::vector<DataType> selector = {0, 1, 0, 1, 1, 0};

    AV a = ShareVector(if_false, engine);
    AV b = ShareVector(if_true, engine);
    AV sel = ShareRaw(selector, engine);

    std::vector<double> expected(if_false.size());
    for (std::size_t i = 0; i < if_false.size(); ++i) {
        expected[i] = RoundTrip(selector[i] == 0 ? if_false[i] : if_true[i]);
    }

    CheckClose("Multiplex elementwise selector", OpenVector(Multiplex(sel, a, b)), expected,
               kExactish, engine);

    // Broadcast form: a single-element selector applied across the vector. The
    // optimizer uses this shape for its per-iteration decisions, so it is not an
    // incidental overload.
    for (const DataType bit : {DataType(0), DataType(1)}) {
        AV broadcast = ShareRaw({bit}, engine);
        std::vector<double> broadcast_expected(if_false.size());
        for (std::size_t i = 0; i < if_false.size(); ++i) {
            broadcast_expected[i] = RoundTrip(bit == 0 ? if_false[i] : if_true[i]);
        }
        CheckClose(bit == 0 ? "Multiplex broadcast sel=0" : "Multiplex broadcast sel=1",
                   OpenVector(Multiplex(broadcast, a, b)), broadcast_expected, kExactish, engine);
    }
}

// ---------------------------------------------------------------------------
// AnyAbsAtLeast
// ---------------------------------------------------------------------------
// Returns a raw 0/1, so it is checked exactly. This is the operator that decides
// the optimizer's one declassified bit per iteration, so a wrong answer here is
// a wrong termination decision, not a small numerical error.
void TestAnyAbsAtLeast(EngineRef engine) {
    Section("AnyAbsAtLeast", engine);

    const double bound = 2.0;
    const DataType bound_scaled = static_cast<DataType>(std::llround(bound * scale));

    struct Case {
        const char* name;
        std::vector<double> values;
        DataType expected;
    };

    const std::vector<Case> cases = {
        {"AnyAbsAtLeast all below", {0.5, -1.25, 1.99, -0.01}, 0},
        {"AnyAbsAtLeast one above (positive)", {0.5, -1.25, 3.5, -0.01}, 1},
        {"AnyAbsAtLeast one above (negative)", {0.5, -4.75, 1.0, -0.01}, 1},
        {"AnyAbsAtLeast exactly at bound", {0.5, 2.0, 1.0, -0.01}, 1},
        {"AnyAbsAtLeast all above", {3.0, -4.75, 2.5, -9.0}, 1},
    };

    for (const Case& test_case : cases) {
        AV shared = ShareVector(test_case.values, engine);
        AV flag = AnyAbsAtLeast(shared, bound_scaled);
        CheckExactRaw(test_case.name, OpenRaw(flag), {test_case.expected}, engine);
    }
}

// ---------------------------------------------------------------------------
// SecureReciprocal
// ---------------------------------------------------------------------------
void TestSecureReciprocal(EngineRef engine) {
    Section("SecureReciprocal", engine);

    // Strictly positive, as the operator's precondition requires: the
    // non-restoring division circuit assumes a non-negative denominator, and
    // handing it a negative one is undefined behaviour in the circuit rather
    // than a wrong number. Out-of-domain behaviour is deliberately NOT asserted.
    const std::vector<double> inputs = {0.05, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 40.0};

    std::vector<double> expected;
    expected.reserve(inputs.size());
    for (const double value : inputs) {
        expected.push_back(1.0 / RoundTrip(value));
    }

    AV shared = ShareVector(inputs, engine);
    CheckClose("SecureReciprocal", OpenVector(SecureReciprocal(shared)), expected, kDivision,
               engine, ErrorMode::Relative, inputs);
}

// ---------------------------------------------------------------------------
// Exp
// ---------------------------------------------------------------------------
void TestExp(EngineRef engine) {
    Section("Exp", engine);

    const std::vector<double> inputs = {-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.5};

    std::vector<double> expected;
    expected.reserve(inputs.size());
    for (const double value : inputs) {
        expected.push_back(std::exp(RoundTrip(value)));
    }

    AV shared = ShareVector(inputs, engine);
    CheckClose("Exp", OpenVector(Exp(shared)), expected, kExpTolerance, engine,
               ErrorMode::Relative, inputs);

    // Saturation. kMaxExpArg is not a hint: past it the oblivious 2^k stage
    // cannot represent the exponent and would wrap, so the operator clamps.
    // SeparationFlag exists downstream precisely because this clamp is silent,
    // and this check is what says the clamp is actually there.
    const std::vector<double> extreme = {-30.0, -15.0, 15.0, 30.0};
    // kMaxExpArg is declared `float` in primitives.h, so widen it once rather
    // than letting every use site deduce a mixed-type std::min.
    const double max_exp_arg = static_cast<double>(kMaxExpArg);
    std::vector<double> clamped_expected;
    clamped_expected.reserve(extreme.size());
    for (const double value : extreme) {
        clamped_expected.push_back(std::exp(std::max(-max_exp_arg, std::min(max_exp_arg, value))));
    }
    AV extreme_shared = ShareVector(extreme, engine);
    CheckClose("Exp saturates at +-kMaxExpArg", OpenVector(Exp(extreme_shared)), clamped_expected,
               kExpTolerance, engine, ErrorMode::Relative, extreme);

    // Supplementary invariant: exp(a) * exp(b) == exp(a + b). Weak on its own --
    // the constant function 1 satisfies it exactly -- but independent of the
    // reference above, so together they constrain more than either does alone.
    const std::vector<double> left = {0.25, -0.5, 1.0, -1.75};
    const std::vector<double> right = {0.75, 1.5, -0.25, 0.5};
    std::vector<double> sums(left.size());
    for (std::size_t i = 0; i < left.size(); ++i) {
        sums[i] = left[i] + right[i];
    }

    const std::vector<double> exp_left = OpenVector(Exp(ShareVector(left, engine)));
    const std::vector<double> exp_right = OpenVector(Exp(ShareVector(right, engine)));
    const std::vector<double> exp_sum = OpenVector(Exp(ShareVector(sums, engine)));

    std::vector<double> products(left.size());
    for (std::size_t i = 0; i < left.size(); ++i) {
        products[i] = exp_left[i] * exp_right[i];
    }
    // Two independent Exp evaluations multiplied in the clear, so the bound is
    // twice the single-call relative tolerance.
    CheckClose("Exp(a)*Exp(b) == Exp(a+b)", products, exp_sum, 2.0 * kExpTolerance, engine,
               ErrorMode::Relative, sums);
}

// ---------------------------------------------------------------------------
// Log and Log1p
// ---------------------------------------------------------------------------
void TestLog(EngineRef engine) {
    Section("Log / Log1p", engine);

    // Inside the documented usable interval, roughly [3e-5, 4e4]. Outside it the
    // range reduction saturates and the answer is wrong rather than imprecise;
    // the library does not define that case, so it is not asserted.
    const std::vector<double> inputs = {0.01, 0.1, 0.25, 0.5, 0.7071, 1.0,
                                        1.4142, 2.0, 4.0, 10.0, 20.0, 100.0};

    std::vector<double> expected;
    expected.reserve(inputs.size());
    for (const double value : inputs) {
        expected.push_back(std::log(RoundTrip(value)));
    }

    AV shared = ShareVector(inputs, engine);
    CheckClose("Log", OpenVector(Log(shared)), expected, kLogTolerance, engine,
               ErrorMode::Absolute, inputs);

    // Log1p's domain is x > -1; the values below keep 1 + x inside Log's range.
    const std::vector<double> log1p_inputs = {-0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0, 5.0};
    std::vector<double> log1p_expected;
    log1p_expected.reserve(log1p_inputs.size());
    for (const double value : log1p_inputs) {
        log1p_expected.push_back(std::log1p(RoundTrip(value)));
    }

    AV log1p_shared = ShareVector(log1p_inputs, engine);
    CheckClose("Log1p", OpenVector(Log1p(log1p_shared)), log1p_expected, kLogTolerance, engine,
               ErrorMode::Absolute, log1p_inputs);
}

// ---------------------------------------------------------------------------
// SecureSqrt
// ---------------------------------------------------------------------------
void TestSecureSqrt(EngineRef engine) {
    Section("SecureSqrt", engine);

    // x > 0, and inside Log's usable interval since SecureSqrt routes through it.
    // The intended consumer is a variance (a covariance diagonal), so the range
    // tested is the range a variance plausibly occupies.
    const std::vector<double> inputs = {0.01, 0.05, 0.25, 1.0, 2.0, 9.0, 25.0, 100.0};

    std::vector<double> expected;
    expected.reserve(inputs.size());
    for (const double value : inputs) {
        expected.push_back(std::sqrt(RoundTrip(value)));
    }

    AV shared = ShareVector(inputs, engine);
    CheckClose("SecureSqrt", OpenVector(SecureSqrt(shared)), expected, kSqrtTolerance, engine,
               ErrorMode::Relative, inputs);
}

// ---------------------------------------------------------------------------
// Sigmoid and LogOnePlusExp
// ---------------------------------------------------------------------------
void TestSigmoid(EngineRef engine) {
    Section("Sigmoid / LogOnePlusExp", engine);

    // Straddles zero, which is where the oblivious sign split happens. An error
    // in the multiplex would show as a discontinuity there rather than as a
    // uniformly larger error, so the sampling is dense around 0.
    const std::vector<double> inputs = {-6.0, -5.0, -2.0, -1.0, -0.25, -0.01,
                                        0.0,  0.01, 0.25, 1.0,  2.0,   5.0, 6.0};

    std::vector<double> sigmoid_expected;
    std::vector<double> softplus_expected;
    sigmoid_expected.reserve(inputs.size());
    softplus_expected.reserve(inputs.size());
    for (const double value : inputs) {
        sigmoid_expected.push_back(RefSigmoid(RoundTrip(value)));
        softplus_expected.push_back(RefLogOnePlusExp(RoundTrip(value)));
    }

    AV shared = ShareVector(inputs, engine);
    const std::vector<double> sigmoid_actual = OpenVector(Sigmoid(shared));
    CheckClose("Sigmoid", sigmoid_actual, sigmoid_expected, kSigmoidTolerance, engine,
               ErrorMode::Absolute, inputs);

    AV softplus_shared = ShareVector(inputs, engine);
    CheckClose("LogOnePlusExp", OpenVector(LogOnePlusExp(softplus_shared)), softplus_expected,
               kSoftplusTolerance, engine, ErrorMode::Absolute, inputs);

    // Supplementary invariant: sigmoid(x) + sigmoid(-x) == 1. This is the
    // property the oblivious sign split is supposed to preserve exactly, and it
    // is sensitive to the multiplex in a way the pointwise check is not.
    std::vector<double> negated(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        negated[i] = -inputs[i];
    }
    const std::vector<double> sigmoid_negated = OpenVector(Sigmoid(ShareVector(negated, engine)));

    std::vector<double> sums(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        sums[i] = sigmoid_actual[i] + sigmoid_negated[i];
    }
    CheckClose("Sigmoid(x) + Sigmoid(-x) == 1", sums, std::vector<double>(inputs.size(), 1.0),
               2.0 * kSigmoidTolerance, engine, ErrorMode::Absolute, inputs);
}

// ---------------------------------------------------------------------------
// NormalCdf and TwoSidedPValue
// ---------------------------------------------------------------------------
void TestNormalCdf(EngineRef engine) {
    Section("NormalCdf / TwoSidedPValue", engine);

    const std::vector<double> inputs = {-3.0, -2.5, -1.96, -1.0, -0.5, 0.0,
                                        0.5,  1.0,  1.96,  2.5,  3.0};

    std::vector<double> expected;
    expected.reserve(inputs.size());
    for (const double value : inputs) {
        expected.push_back(RefNormalCdf(RoundTrip(value)));
    }

    AV shared = ShareVector(inputs, engine);
    const std::vector<double> actual = OpenVector(NormalCdf(shared));
    CheckClose("NormalCdf", actual, expected, kNormalCdfTolerance, engine, ErrorMode::Absolute,
               inputs);

    // Reflection Phi(x) + Phi(-x) == 1, the identity the operator's own sign
    // multiplex implements. Checking it separately tests the multiplex rather
    // than the polynomial.
    std::vector<double> negated(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        negated[i] = -inputs[i];
    }
    const std::vector<double> reflected = OpenVector(NormalCdf(ShareVector(negated, engine)));
    std::vector<double> sums(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        sums[i] = actual[i] + reflected[i];
    }
    CheckClose("NormalCdf(x) + NormalCdf(-x) == 1", sums, std::vector<double>(inputs.size(), 1.0),
               2.0 * kNormalCdfTolerance, engine, ErrorMode::Absolute, inputs);

    // TwoSidedPValue. The operator computes 2 * Phi(-|z|) rather than
    // 2 * (1 - Phi(|z|)) specifically to avoid the cancellation near 1, so the
    // reference is written the same way.
    //
    // Range note from the operator's own comment: at `precision` 16 the
    // representable resolution is 1.5e-5, so a p-value below roughly 1e-4 should
    // be read as "small" rather than at face value. The inputs stay inside |z| <= 3
    // (p >= 2.7e-3) so the check is meaningful rather than a test of the format's
    // floor.
    std::vector<double> p_expected;
    p_expected.reserve(inputs.size());
    for (const double value : inputs) {
        p_expected.push_back(RefTwoSidedPValue(RoundTrip(value)));
    }
    AV z_shared = ShareVector(inputs, engine);
    CheckClose("TwoSidedPValue", OpenVector(TwoSidedPValue(z_shared)), p_expected,
               kPValueTolerance, engine, ErrorMode::Absolute, inputs);

    Note("p-values below ~1e-4 are below the fixed-point resolution and are not asserted", engine);
}


// ---------------------------------------------------------------------------
// Randomized sweeps over each operator's documented range
// ---------------------------------------------------------------------------
// The hand-picked checks above and these are complementary, and neither replaces
// the other:
//
//   * the fixed vectors land exactly on boundaries -- x == bound, x == 0, the
//     sign-change point -- which random sampling essentially never hits, and
//     which is where comparison and multiplex defects live;
//   * these cover the operator's whole documented range densely, which is where
//     experiment 0002 found errors up to 11x what the coarse fixed grids showed.
//
// Ranges are the measured operating ranges concluded by experiment 0002. They are
// not guesses and not the header's older derived figures -- notably `Log` is
// exercised over [1.53e-5, 1e5], wider at both ends than the "[3e-5, 4e4]" the
// header used to derive.
//
// Only party 0 holds the drawn inputs, so only party 0 checks -- see
// CheckCloseOnReporter in test_util.h.
void TestRandomSweeps(EngineRef engine) {
    Section("randomized sweeps (2^14 samples per operator)", engine);

    const bool kLog = true;
    const bool kLinear = false;

    struct Case {
        const char* name;
        double low;
        double high;
        bool log_spaced;
        double tolerance;
        ErrorMode mode;
        AV (*secure)(const AV&);
        double (*plain)(double);
    };

    // Wrappers: the library operators have defaulted or reference-qualified
    // signatures that do not convert to a plain function pointer directly.
    struct Op {
        static AV Exp_(const AV& x) { return Exp(x); }
        static AV Log_(const AV& x) { return Log(x); }
        static AV Log1p_(const AV& x) { return Log1p(x); }
        static AV Recip_(const AV& x) { return SecureReciprocal(x); }
        static AV Sqrt_(const AV& x) { return SecureSqrt(x); }
        static AV Sigmoid_(const AV& x) { return Sigmoid(x); }
        static AV Softplus_(const AV& x) { return LogOnePlusExp(x); }
        static AV Cdf_(const AV& x) { return NormalCdf(x); }
        static AV PValue_(const AV& x) { return TwoSidedPValue(x); }

        static double pExp(double x) { return std::exp(x); }
        static double pLog(double x) { return std::log(x); }
        static double pLog1p(double x) { return std::log1p(x); }
        static double pRecip(double x) { return 1.0 / x; }
        static double pSqrt(double x) { return std::sqrt(x); }
        static double pSigmoid(double x) { return RefSigmoid(x); }
        static double pSoftplus(double x) { return RefLogOnePlusExp(x); }
        static double pCdf(double x) { return RefNormalCdf(x); }
        static double pPValue(double x) { return RefTwoSidedPValue(x); }
    };

    const std::vector<Case> cases = {
        // Exp: upper bound is the kMaxExpArg clamp; below it there is no bound,
        // the output simply underflows the format.
        {"Exp [rand]", -10.0, 10.0, kLinear, kRandExp, ErrorMode::Relative, Op::Exp_, Op::pExp},
        // Log: [1.53e-5, 1e5]. Lower edge is the input's own representability
        // (1 ULP); upper edge measured between 1e5 and 3.16e5.
        {"Log [rand]", 1.53e-5, 1.0e5, kLog, kRandLog, ErrorMode::Absolute, Op::Log_, Op::pLog},
        {"SecureReciprocal [rand]", 1.0e-4, 1.0e4, kLog, kRandReciprocal, ErrorMode::Relative,
         Op::Recip_, Op::pRecip},
        {"SecureSqrt [rand]", 1.53e-5, 1.0e5, kLog, kRandSqrt, ErrorMode::Relative, Op::Sqrt_,
         Op::pSqrt},
        {"Sigmoid [rand]", -20.0, 20.0, kLinear, kRandSigmoid, ErrorMode::Absolute, Op::Sigmoid_,
         Op::pSigmoid},
        {"LogOnePlusExp [rand]", -20.0, 20.0, kLinear, kRandSoftplus, ErrorMode::Absolute,
         Op::Softplus_, Op::pSoftplus},
        {"NormalCdf [rand]", -5.0, 5.0, kLinear, kRandNormalCdf, ErrorMode::Absolute, Op::Cdf_,
         Op::pCdf},
        {"TwoSidedPValue [rand]", 0.0, 5.0, kLinear, kRandPValue, ErrorMode::Absolute,
         Op::PValue_, Op::pPValue},
    };

    for (const Case& test_case : cases) {
        const std::vector<double> inputs =
            RandomInputs(kRandomSamples, test_case.low, test_case.high, test_case.log_spaced,
                         engine);
        const std::vector<double> rounded = RoundTrip(inputs);

        AV shared = ShareVector(inputs, engine);
        const std::vector<double> actual = OpenVector(test_case.secure(shared));

        std::vector<double> expected(rounded.size());
        for (std::size_t i = 0; i < rounded.size(); ++i) {
            expected[i] = test_case.plain(rounded[i]);
        }

        CheckCloseOnReporter(test_case.name, actual, expected, test_case.tolerance, engine,
                             test_case.mode, rounded);
    }

    // Log1p separately: it is sampled log-spaced in (1 + x) so the draw
    // concentrates where 1 + x approaches Log's lower limit, which a log draw on
    // x itself cannot do because x is negative there.
    {
        std::vector<double> shifted =
            RandomInputs(kRandomSamples, 0.01, 1.0e4 + 1.0, kLog, engine);
        for (double& value : shifted) {
            value -= 1.0;
        }
        const std::vector<double> rounded = RoundTrip(shifted);
        AV shared = ShareVector(shifted, engine);
        const std::vector<double> actual = OpenVector(Log1p(shared));
        std::vector<double> expected(rounded.size());
        for (std::size_t i = 0; i < rounded.size(); ++i) {
            expected[i] = std::log1p(rounded[i]);
        }
        CheckCloseOnReporter("Log1p [rand]", actual, expected, kRandLog1p, engine,
                             ErrorMode::Absolute, rounded);
    }

    // Sum, at the sample size this file uses. Its safe range is the only one that
    // depends on the vector length: N * magnitude must stay under ~1.4e14, and
    // above that it wraps to a plausible-looking number of the WRONG SIGN with no
    // flag of any kind (experiment 0002 E7). At 2^14 elements the per-element
    // ceiling is 8.6e9; drawing from [-1e3, 1e3] leaves six orders of margin.
    {
        const std::vector<double> values =
            RandomInputs(kRandomSamples, -1.0e3, 1.0e3, kLinear, engine);
        const std::vector<double> rounded = RoundTrip(values);
        double total = 0.0;
        for (const double value : rounded) {
            total += value;
        }
        AV shared = ShareVector(values, engine);
        // Addition in fixed point is exact below the overflow ceiling, so this is
        // held to a few ULPs rather than to a tolerance.
        CheckCloseOnReporter("Sum [rand, 2^14 elements]", OpenVector(Sum(shared)), {total},
                             kExactish, engine);
    }
}

}  // namespace

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);

    if (engine.getPartyID() == 0) {
        std::cout << "\n########## playground/library/primitives.h ##########\n"
                  << "precision = " << precision << "   scale = " << static_cast<long long>(scale)
                  << "   resolution = " << std::scientific << std::setprecision(3) << kResolution
                  << std::endl;
    }

    TestClone(engine);
    TestSum(engine);
    TestClampAbs(engine);
    TestMultiplex(engine);
    TestAnyAbsAtLeast(engine);
    TestSecureReciprocal(engine);
    TestExp(engine);
    TestLog(engine);
    TestSecureSqrt(engine);
    TestSigmoid(engine);
    TestNormalCdf(engine);
    TestRandomSweeps(engine);

    if (engine.getPartyID() == 0) {
        std::cout << "\nprimitives: all checks passed" << std::endl;
    }

    return 0;
}
