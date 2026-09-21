// Accuracy profiler for playground/library/primitives.h.
//
// WHAT THIS IS, AND WHAT IT IS NOT. This is an *exploratory* instrument, not a
// test. It asserts nothing and always exits 0. Its job is to show how each
// primitive's error moves as a function of its input, so that the operating
// ranges and the sweep design for experiment 0002 are chosen from a picture
// rather than from algebra.
//
// It proves nothing on its own. The frozen experiment produces the evidence;
// this chooses the instrument. That separation is the one recorded in semantic
// personality note 0009, Conflict C-01.
//
// PRECEDENT. `optimizer.h` already works this way: `kRunMatrixInverseCalibration`
// is an exploratory sweep that chose `kMatrixInverseIterations`, and the
// committed comment table documents the result separately. This file is the same
// idea for the primitives' input ranges.
//
// STAGE S1 (task 0021): the eight approximate unary primitives, each on one
// coarse grid small enough to read. `Sum` and the exact operators have a
// different shape and are handled separately.
//
// Run: mpirun -np 3 ./profile_playground_primitives

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <cstdint>
#include <string>
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

// A secure operator under profile, and the plaintext function it approximates.
using SecureOp = std::function<AV(const AV&)>;
using PlainOp = std::function<double(double)>;

// Below one ULP the expected value cannot be represented at all, so its relative
// error measures the number format rather than the operator. An expected value of
// exactly zero is NOT underflow -- zero is exactly representable, and Log(1.0)
// hitting it is the operator being exactly right.
constexpr double kUnderflowUlps = 1.0;

// An absolute error at or below this many ULPs means the operator is sitting on
// the format's own floor: whatever relative error the point shows is quantisation,
// not arithmetic. Reported as a label rather than used to exclude the point,
// because "the operator is as accurate as the format allows here" is information,
// not a reason to look away.
//
// Chosen as 8 ULP because S1 measured every primitive except Exp at 1-7 ULP of
// absolute error across its whole range (task 0021 F-06).
constexpr double kFormatNoiseUlps = 8.0;

// Per-row tables print only below this size; above it, summary only.
constexpr std::size_t kMaxRowsToPrint = 32;

// The accuracy bar, fixed in task 0021 D-4 BEFORE any measurement and frozen in
// experiment 0002. The pilot did not move it and analysis must not.
constexpr double kAccuracyBar = 1.0e-3;

// Sample count for the random pass, per the user's choice on 2026-09-21.
constexpr std::size_t kRandomSamples = 1u << 14;

// A point is called saturated when the operator has stopped tracking its input
// at all. That is a STRUCTURAL condition -- the input is outside a documented
// clamp -- not a magnitude of error, so the caller states it explicitly per
// operator rather than the profiler guessing from the error.
//
// The S0 pilot had this wrong: it flagged anything with relative error above
// 1e-2, which labelled Exp(-9) "SATURATED?" when that point is merely
// quantisation-dominated (8.1 ULP of output). Conflating the two would have made
// every operator's lower range edge unreadable.

// Evaluates `secure` at every point of `inputs` in ONE call -- these operators
// are vectorized, so a grid costs about the same as a single point -- and prints
// input, expected, actual, absolute error, TRUE relative error, and the expected
// magnitude measured in ULPs.
//
// WHY TRUE RELATIVE ERROR, AND WHY ULPs. `test_util.h`'s ErrorMode::Relative
// divides by max(1, |expected|). That floor is right for a *test* -- it stops the
// measure exploding near zero -- but it is wrong for *characterisation*, because
// it hides underflow: an expected value of 6.1e-06 sits below the 1.526e-05
// resolution and cannot be represented at all, yet the floored measure reports a
// 2.4e-05 "relative" error and looks excellent. The S0 pilot hit exactly this at
// Exp(-12). So this profiler reports the undivided ratio, plus |expected| in
// ULPs, which makes the three distinct failure modes separable:
//
//   truncation  -- rel error grows while the value is comfortably representable
//   underflow   -- |expected| falls below ~1 ULP; the output is quantisation noise
//   saturation  -- the operator clamps and the value stops tracking the input
// `clamp_low`/`clamp_high` bound the operator's documented input domain; points
// outside are labelled SATURATED and excluded from the worst-error summary.
// Pass -inf/+inf when the operator has no clamp.
void Profile(const std::string& name, const std::vector<double>& inputs, const SecureOp& secure,
             const PlainOp& plain, EngineRef engine,
             double clamp_low = -std::numeric_limits<double>::infinity(),
             double clamp_high = std::numeric_limits<double>::infinity()) {
    const std::vector<double> rounded = RoundTrip(inputs);

    AV shared = ShareVector(inputs, engine);
    const auto start = std::chrono::steady_clock::now();
    AV result = secure(shared);
    const std::vector<double> actual = OpenVector(result);
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    if (!IsReporter(engine)) {
        return;
    }

    // At 2^14 samples a per-row table is 16384 lines, which is not inspectable
    // and defeats the purpose. Rows print only for grids small enough to read;
    // larger runs report the summary plus the input range actually drawn, which
    // is the check that the shaping function did not collapse the sample into
    // one corner of the range.
    const bool print_rows = inputs.size() <= kMaxRowsToPrint;

    std::cout << "\n=== " << name << "  (n = " << inputs.size() << ", "
              << std::fixed << std::setprecision(3) << elapsed << " s) ===" << std::endl;
    if (print_rows) {
        std::cout << std::right << std::setw(12) << "input" << std::setw(16) << "expected"
                  << std::setw(16) << "actual" << std::setw(12) << "abs err"
                  << std::setw(12) << "rel err" << std::setw(12) << "floored"
                  << std::setw(12) << "exp ULPs" << "  note" << std::endl;
    }

    double worst_absolute = 0.0;
    double worst_relative_representable = 0.0;
    double worst_floored = 0.0;
    double worst_error_ulps = 0.0;
    double drawn_low = std::numeric_limits<double>::infinity();
    double drawn_high = -std::numeric_limits<double>::infinity();
    std::size_t excluded = 0;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const double expected = plain(rounded[i]);
        const double absolute = std::abs(actual[i] - expected);
        // Undivided. Guarded only at exactly zero, where no ratio exists.
        const double relative = (expected == 0.0) ? absolute : absolute / std::abs(expected);
        const double ulps = std::abs(expected) / kResolution;

        // Three distinct conditions, tested in order of precedence. Saturation is
        // structural (input outside the documented clamp); underflow is a
        // property of the expected value, not of the operator; what remains is
        // the operator's own arithmetic, and only that feeds the summary.
        const bool saturated = (rounded[i] < clamp_low) || (rounded[i] > clamp_high);
        // Exactly zero is representable; only a nonzero value too small to encode
        // is underflow.
        const bool underflow = (expected != 0.0) && (ulps < kUnderflowUlps);
        const bool measures_operator = !saturated && !underflow;

        const double error_ulps = absolute / kResolution;
        const char* note = saturated      ? "SATURATED"
                           : underflow    ? "UNDERFLOW"
                           : (error_ulps <= kFormatNoiseUlps) ? "format-limited"
                                                              : "";

        if (measures_operator) {
            worst_absolute = std::max(worst_absolute, absolute);
            worst_error_ulps = std::max(worst_error_ulps, error_ulps);
            worst_relative_representable = std::max(worst_relative_representable, relative);
        } else {
            ++excluded;
        }

        // The measure the TEST uses: absolute below 1, relative above it. Printed
        // alongside the true relative error so the two are comparable at a glance.
        const double floored = absolute / std::max(1.0, std::abs(expected));
        if (measures_operator) {
            worst_floored = std::max(worst_floored, floored);
        }

        drawn_low = std::min(drawn_low, rounded[i]);
        drawn_high = std::max(drawn_high, rounded[i]);
        if (!print_rows) {
            continue;
        }
        std::cout << std::right << std::setw(12) << std::fixed << std::setprecision(4) << rounded[i]
                  << std::setw(16) << std::setprecision(6) << expected
                  << std::setw(16) << actual[i]
                  << std::setw(12) << std::scientific << std::setprecision(2) << absolute
                  << std::setw(12) << relative
                  << std::setw(12) << floored
                  << std::setw(12) << std::fixed << std::setprecision(1) << ulps
                  << "  " << note << std::endl;
        (void)0;
    }

    // The headline is absolute error IN ULPs. Task 0021 F-06: that is the
    // operator-intrinsic quantity; true relative error is a property of the
    // output's magnitude, not of the operator, so it is shown per row for reading
    // but is not the summary.
    std::cout << "  input drawn: [" << std::scientific << std::setprecision(3) << drawn_low
              << ", " << drawn_high << "]   excluded (saturated/underflow): " << excluded
              << " of " << inputs.size()
              << "\n  worst over operator-measuring points (excludes saturated and underflow):"
              << "\n    abs = " << std::fixed << std::setprecision(1) << worst_error_ulps
              << " ULP (" << std::scientific << std::setprecision(2) << worst_absolute << ")"
              << "   floored = " << worst_floored << "   <- the measure the test applies"
              << "\n    worst true rel = " << worst_relative_representable
              << "  (a property of output magnitude, not of the operator)" << std::endl;
}

// Logarithmically spaced grid, inclusive of both endpoints. The right shape for
// operators whose input spans orders of magnitude (Log, SecureReciprocal,
// SecureSqrt): a linear grid over [1e-5, 1e5] would put every sample at the top
// of the range and never probe the small end at all, which is where these
// operators actually fail.
std::vector<double> LogGrid(double low, double high, std::size_t count) {
    std::vector<double> grid(count);
    const double log_low = std::log(low);
    const double step = (std::log(high) - log_low) / static_cast<double>(count - 1);
    for (std::size_t i = 0; i < count; ++i) {
        grid[i] = std::exp(log_low + step * static_cast<double>(i));
    }
    return grid;
}

// Linearly spaced grid, inclusive of both endpoints.
std::vector<double> LinearGrid(double low, double high, std::size_t count) {
    std::vector<double> grid(count);
    const double step = (high - low) / static_cast<double>(count - 1);
    for (std::size_t i = 0; i < count; ++i) {
        grid[i] = low + step * static_cast<double>(i);
    }
    return grid;
}



// Random input generation lives in test_util.h, shared with the test binary so
// the profiler and the test draw their samples identically.

// ---------------------------------------------------------------------------
// Sum: the one primitive whose safe range depends on the vector length
// ---------------------------------------------------------------------------
// Every other primitive is elementwise, so its range is a property of one value.
// `Sum` accumulates, so the quantity that can overflow is N * magnitude. At
// `DataType` = int64_t and `precision` = 16 the largest representable value is
// 2^63 / 2^16 = 2^47 ~ 1.4e14, so the predicted ceiling is
//
//     N * magnitude < ~1.4e14
//
// which at N = 16384 -- the sample size this task's randomized tests will use --
// puts the per-element limit at about 8.6e9. Overflow in a two's-complement ring
// wraps rather than saturating, so the failure is a sign flip or a wildly wrong
// magnitude, not a clamp. That is worth seeing rather than deriving.
//
// All elements are set to the same value: the worst case for accumulation, and
// deterministic, so the table can be read directly.
void ProfileSum(EngineRef engine) {
    if (IsReporter(engine)) {
        std::cout << "\n=== Sum (accumulation limit) ===\n"
                  << "  predicted ceiling: N * magnitude < 2^47 ~ 1.4e14\n"
                  << std::right << std::setw(8) << "N" << std::setw(12) << "magnitude"
                  << std::setw(16) << "expected" << std::setw(16) << "actual"
                  << std::setw(12) << "rel err" << "  note" << std::endl;
    }

    for (const std::size_t count : {std::size_t(16), std::size_t(256), std::size_t(4096),
                                    std::size_t(16384)}) {
        for (const double magnitude : {1.0, 1.0e3, 1.0e6, 1.0e9, 1.0e12}) {
            const std::vector<double> values(count, magnitude);
            AV shared = ShareVector(values, engine);
            const std::vector<double> opened = OpenVector(Sum(shared));

            if (!IsReporter(engine)) {
                continue;
            }
            const double expected = static_cast<double>(count) * RoundTrip(magnitude);
            const double actual = opened[0];
            const double relative = std::abs(actual - expected) / std::abs(expected);
            const double predicted = static_cast<double>(count) * magnitude;
            const bool predicted_overflow = predicted > 1.4e14;

            std::cout << std::right << std::setw(8) << count
                      << std::setw(12) << std::scientific << std::setprecision(1) << magnitude
                      << std::setw(16) << std::setprecision(6) << expected
                      << std::setw(16) << actual
                      << std::setw(12) << std::setprecision(2) << relative << "  "
                      << (relative > 1e-6 ? "WRONG" : "ok")
                      << (predicted_overflow ? "  (overflow predicted)" : "") << std::endl;
        }
    }
}

// ---------------------------------------------------------------------------
// The exact operators: where does fixed-point representation give out?
// ---------------------------------------------------------------------------
// ClampAbs, Multiplex and AnyAbsAtLeast are selections and comparisons, not
// approximations. They have no accuracy range -- they are either exactly right or
// broken -- so the only question is the magnitude at which the representation
// itself fails. Checked directly rather than assumed from the bit width.
void ProfileExactOperators(EngineRef engine) {
    if (IsReporter(engine)) {
        std::cout << "\n=== exact operators (magnitude limit) ===\n"
                  << "  these are selections, not approximations: exact or broken\n"
                  << std::right << std::setw(14) << "magnitude" << std::setw(14) << "ClampAbs"
                  << std::setw(14) << "Multiplex" << std::setw(16) << "AnyAbsAtLeast" << std::endl;
    }

    for (const double magnitude : {1.0, 1.0e3, 1.0e6, 1.0e9, 1.0e12, 1.0e13, 1.0e14}) {
        // ClampAbs with a bound well above the magnitude: the value must pass
        // through untouched.
        const std::vector<double> values = {magnitude, -magnitude, magnitude / 2.0};
        AV shared = ShareVector(values, engine);
        const DataType bound = static_cast<DataType>(
            std::llround(std::min(magnitude * 4.0, 1.0e14) * static_cast<double>(scale)));
        const std::vector<double> clamped = OpenVector(ClampAbs(shared, bound));

        // Multiplex must select exactly.
        AV a = ShareVector(values, engine);
        AV b = ShareVector({-magnitude, magnitude, 0.0}, engine);
        AV selector = ShareRaw({0, 1, 0}, engine);
        const std::vector<double> selected = OpenVector(Multiplex(selector, a, b));

        // AnyAbsAtLeast against a bound just under the magnitude: must be 1.
        AV probe = ShareVector(values, engine);
        const DataType probe_bound =
            static_cast<DataType>(std::llround(magnitude * 0.5 * static_cast<double>(scale)));
        const std::vector<DataType> flag = OpenRaw(AnyAbsAtLeast(probe, probe_bound));

        if (!IsReporter(engine)) {
            continue;
        }
        const double tolerance = std::max(1.0e-9, magnitude * 1.0e-9);
        const bool clamp_ok = std::abs(clamped[0] - RoundTrip(magnitude)) < tolerance;
        const bool mux_ok = std::abs(selected[1] - RoundTrip(magnitude)) < tolerance;
        const bool any_ok = (flag[0] == 1);

        std::cout << std::right << std::setw(14) << std::scientific << std::setprecision(1)
                  << magnitude << std::setw(14) << (clamp_ok ? "exact" : "BROKEN")
                  << std::setw(14) << (mux_ok ? "exact" : "BROKEN")
                  << std::setw(16) << (any_ok ? "exact" : "BROKEN") << std::endl;
    }
}


// ---------------------------------------------------------------------------
// The random pass: experiment 0002, Approach A
// ---------------------------------------------------------------------------
// Ranges are FROZEN in experiment 0002's design table and are reproduced here
// verbatim. Changing one is a deviation and must be recorded as such in that
// record -- not edited quietly here.
void ProfileRandom(EngineRef engine) {
    if (IsReporter(engine)) {
        std::cout << "\n\n##### RANDOM PASS (experiment 0002, Approach A) #####\n"
                  << "  " << kRandomSamples << " samples per operator, drawn with "
                  << "engine.populateLocalRandom on party 0\n"
                  << "  accuracy bar: floored <= " << std::scientific << std::setprecision(1)
                  << kAccuracyBar << "  (frozen in task 0021 D-4, before any measurement)"
                  << std::endl;
    }

    const bool kLog = true;
    const bool kLinear = false;

    Profile("Exp [rand]", RandomInputs(kRandomSamples, -10.0, 10.0, kLinear, engine),
            [](const AV& x) { return Exp(x); }, [](double x) { return std::exp(x); }, engine,
            -static_cast<double>(kMaxExpArg), static_cast<double>(kMaxExpArg));

    Profile("Log [rand]", RandomInputs(kRandomSamples, 1.5e-5, 1.0e5, kLog, engine),
            [](const AV& x) { return Log(x); }, [](double x) { return std::log(x); }, engine);

    // Log-spaced in (1 + x), so the sample concentrates where 1 + x approaches
    // Log's lower limit rather than spreading uniformly over a huge upper range.
    {
        std::vector<double> shifted = RandomInputs(kRandomSamples, 0.01, 1.0e4 + 1.0, kLog, engine);
        for (double& value : shifted) {
            value -= 1.0;
        }
        Profile("Log1p [rand]", shifted, [](const AV& x) { return Log1p(x); },
                [](double x) { return std::log1p(x); }, engine);
    }

    Profile("SecureReciprocal [rand]", RandomInputs(kRandomSamples, 1.0e-4, 1.0e4, kLog, engine),
            [](const AV& x) { return SecureReciprocal(x); }, [](double x) { return 1.0 / x; },
            engine);

    Profile("SecureSqrt [rand]", RandomInputs(kRandomSamples, 1.0e-4, 1.0e4, kLog, engine),
            [](const AV& x) { return SecureSqrt(x); }, [](double x) { return std::sqrt(x); },
            engine);

    Profile("Sigmoid [rand]", RandomInputs(kRandomSamples, -20.0, 20.0, kLinear, engine),
            [](const AV& x) { return Sigmoid(x); }, [](double x) { return RefSigmoid(x); }, engine);

    Profile("LogOnePlusExp [rand]", RandomInputs(kRandomSamples, -20.0, 20.0, kLinear, engine),
            [](const AV& x) { return LogOnePlusExp(x); },
            [](double x) { return RefLogOnePlusExp(x); }, engine);

    Profile("NormalCdf [rand]", RandomInputs(kRandomSamples, -5.0, 5.0, kLinear, engine),
            [](const AV& x) { return NormalCdf(x); }, [](double x) { return RefNormalCdf(x); },
            engine);

    Profile("TwoSidedPValue [rand]", RandomInputs(kRandomSamples, 0.0, 5.0, kLinear, engine),
            [](const AV& z) { return TwoSidedPValue(z); },
            [](double z) { return RefTwoSidedPValue(z); }, engine);
}

// ---------------------------------------------------------------------------
// Edge location: experiment 0002, Approach B
// ---------------------------------------------------------------------------
// Where does each multiplicative operator actually cross the bar? The frozen plan
// said "extend by factors of 10, then bisect twice", which brackets an edge to a
// factor of 2.5. A single log grid at factor 3.16 per step reaches comparable
// resolution in one run instead of several, so that is what is done -- recorded
// as a deviation in experiment 0002 rather than passed off as the plan.
void ProfileEdges(EngineRef engine) {
    if (IsReporter(engine)) {
        std::cout << "\n\n##### EDGE LOCATION (experiment 0002, Approach B) #####\n"
                  << "  log grid at ~3.16x per step, extended well past each documented range"
                  << std::endl;
    }

    Profile("Log [edges]", LogGrid(1.0e-7, 1.0e7, 29), [](const AV& x) { return Log(x); },
            [](double x) { return std::log(x); }, engine);

    Profile("SecureReciprocal [edges]", LogGrid(1.0e-7, 1.0e7, 29),
            [](const AV& x) { return SecureReciprocal(x); }, [](double x) { return 1.0 / x; },
            engine);

    Profile("SecureSqrt [edges]", LogGrid(1.0e-7, 1.0e7, 29),
            [](const AV& x) { return SecureSqrt(x); }, [](double x) { return std::sqrt(x); },
            engine);
}

}  // namespace

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);

    if (engine.getPartyID() == 0) {
        std::cout << "\n########## primitive accuracy profile (task 0021, stage S0) ##########\n"
                  << "precision = " << precision
                  << "   resolution = " << std::scientific << std::setprecision(3) << kResolution
                  << "   kMaxExpArg = " << static_cast<double>(kMaxExpArg)
                  << "\nExploratory only -- asserts nothing, proves nothing." << std::endl;
    }

    // ---- Exp ----
    // Spans past the +-kMaxExpArg clamp deliberately, so the clamp shows up in
    // the profile rather than having to be taken on trust. 17 points at a step
    // of 1.5 is small enough to read in one screen.
    Profile("Exp", LinearGrid(-12.0, 12.0, 17), [](const AV& x) { return Exp(x); },
            [](double x) { return std::exp(x); }, engine,
            -static_cast<double>(kMaxExpArg), static_cast<double>(kMaxExpArg));

    // ---- Log ----
    // Log-spaced across ten orders of magnitude, straddling the range-reduction
    // limits the header derives as roughly [3e-5, 4e4]. Those figures have never
    // been measured; this is the measurement.
    //
    // No clamp is declared: Log has no saturation guard, so whatever happens at
    // the edges is the operator's real behaviour rather than a documented cutoff.
    // Its OUTPUT is O(10) and never underflows, so neither label should fire and
    // any error shown is genuinely the operator's.
    Profile("Log", LogGrid(1.0e-5, 1.0e5, 21), [](const AV& x) { return Log(x); },
            [](double x) { return std::log(x); }, engine);

    // ---- Log1p ----
    // Domain is x > -1. The negative end is where 1 + x approaches Log's lower
    // limit, so it is sampled toward -1 rather than uniformly.
    {
        std::vector<double> grid;
        for (const double value : {-0.9, -0.75, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0, 2.0,
                                   5.0, 10.0, 100.0, 1000.0, 10000.0}) {
            grid.push_back(value);
        }
        Profile("Log1p", grid, [](const AV& x) { return Log1p(x); },
                [](double x) { return std::log1p(x); }, engine);
    }

    // ---- SecureReciprocal ----
    // Requires x > 0 (the non-restoring division circuit assumes a non-negative
    // denominator). Both edges are interesting: small x drives the output toward
    // overflow, large x drives it under the resolution floor.
    Profile("SecureReciprocal", LogGrid(1.0e-4, 1.0e4, 17),
            [](const AV& x) { return SecureReciprocal(x); },
            [](double x) { return 1.0 / x; }, engine);

    // ---- SecureSqrt ----
    // Exp(0.5 * Log(x)), so it inherits Log's range and composes the two errors.
    // Its intended consumer is a variance, so the range a variance occupies is
    // the range that matters.
    Profile("SecureSqrt", LogGrid(1.0e-4, 1.0e4, 17), [](const AV& x) { return SecureSqrt(x); },
            [](double x) { return std::sqrt(x); }, engine);

    // ---- Sigmoid ----
    // No clamp declared deliberately. Sigmoid routes through Exp, so |eta| > 10
    // does saturate internally -- but its output there is already far below one
    // ULP, so the UNDERFLOW label is the honest explanation and declaring a clamp
    // would mask which of the two limits actually binds.
    Profile("Sigmoid", LinearGrid(-20.0, 20.0, 21), [](const AV& x) { return Sigmoid(x); },
            [](double x) { return RefSigmoid(x); }, engine);

    // ---- LogOnePlusExp ----
    Profile("LogOnePlusExp", LinearGrid(-20.0, 20.0, 21),
            [](const AV& x) { return LogOnePlusExp(x); },
            [](double x) { return RefLogOnePlusExp(x); }, engine);

    // ---- NormalCdf ----
    // The tails are the whole question: Phi(-5) is 2.9e-07, about 0.02 ULP, so it
    // cannot be represented at all. Where that boundary sits determines how small
    // a p-value the Wald layer may report.
    Profile("NormalCdf", LinearGrid(-5.0, 5.0, 21), [](const AV& x) { return NormalCdf(x); },
            [](double x) { return RefNormalCdf(x); }, engine);

    // ---- TwoSidedPValue ----
    // Symmetric, so only the non-negative half is profiled.
    Profile("TwoSidedPValue", LinearGrid(0.0, 5.0, 11),
            [](const AV& z) { return TwoSidedPValue(z); },
            [](double z) { return RefTwoSidedPValue(z); }, engine);

    ProfileSum(engine);
    ProfileExactOperators(engine);
    ProfileEdges(engine);
    ProfileRandom(engine);

    if (engine.getPartyID() == 0) {
        std::cout << "\nprofile complete (exploratory; no assertions)" << std::endl;
    }

    return 0;
}
