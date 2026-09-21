// Shared assertion harness for the playground library tests.
//
// WHY THIS EXISTS, AND WHY IT IS NOT tests/util.h
// -----------------------------------------------
// The repository's `tests/util.h` compares exactly (`ASSERT_SAME` is `==` on
// opened vectors). That is the right instrument for the core library, whose
// operators are exact integer circuits. It is the wrong one here: everything in
// `playground/library/` is fixed-point arithmetic at `precision` 16, so the
// correct answer and the computed answer differ by construction, and the only
// meaningful question is by how much.
//
// So this header provides a tolerance check instead. Three properties matter,
// and each is a deliberate choice rather than a convenience:
//
//   1. EVERY CHECK PRINTS ITS TOLERANCE AND ITS WORST OBSERVED ERROR, pass or
//      fail. A bare `assert` tells you an operator is inside the bar but not
//      whether it is inside by a factor of 300 or by 2%. An operator drifting
//      toward its tolerance is the thing worth catching early, and it is
//      invisible unless the margin is printed on every run.
//
//   2. THE ORACLE IS A PLAINTEXT REFERENCE, NOT A RECORDED OUTPUT. Golden
//      vectors certify current behaviour, correct or not; they would have been
//      perfectly content with the flattened-matrix defect of semantic task 0011.
//      Algebraic invariants (used here only as a supplement) pass for whole
//      families of wrong functions -- `Exp(x) = 1` satisfies the addition law
//      exactly. Comparing against `std::` is the only oracle of the three that
//      can say "this operator is wrong", as opposed to "this operator changed".
//
//   3. FAILURE IS FATAL AND LOUD. `assert`, so the process exits non-zero and
//      `scripts/testing/run_tests.sh`-style drivers notice.
//
// PRIVACY: THESE ARE TESTS, NOT A PRIVACY DEMONSTRATION.
// A test must open its results to check them, and these files open freely --
// intermediate values, inputs, everything. Nothing here is evidence about the
// obliviousness of any operator, and no claim in semantic topic 0003 may be
// supported by pointing at this harness. The obliviousness argument lives in the
// operators themselves; see the contract comment in `optimizer.h`.
//
// Deliberately does NOT enable LOGISTIC_REGRESSION_LAYER_PRINT.
//
// DIAGNOSTIC OUTPUT: compile with -DPLAYGROUND_TEST_PRINT (CMake:
// -DPLAYGROUND_TEST_PRINT=ON) to additionally dump input / expected / actual /
// error for the first `kPrintElements` entries of every checked vector.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "cdough.h"

#include "../library/primitives.h"

namespace playground_test {

// `EngineRef` is minted by the init_mpc_system macro into the compiled
// protocol's namespace, not into cdough::, so it has to be named explicitly.
using COMPILED_MPC_PROTOCOL_NAMESPACE::EngineRef;

using cdough::regression::AV;
using cdough::regression::DataType;
using cdough::regression::PMatrix;
using cdough::regression::SMatrix;
using cdough::regression::precision;
using cdough::regression::scale;

// The fixed-point resolution: 2^-16 ~ 1.526e-5. No tolerance below this is
// meaningful, because it is the gap between representable neighbours. Every
// tolerance in these tests is stated as a multiple of it, as a relative error,
// or as a figure measured elsewhere and cited.
constexpr double kResolution = 1.0 / static_cast<double>(scale);

// How many elements the PLAYGROUND_TEST_PRINT dump shows per check.
constexpr std::size_t kPrintElements = 8;

// ---------------------------------------------------------------------------
// Sharing and opening
// ---------------------------------------------------------------------------

// Shares `values` as fixed-point at `precision`, dealt by party 0.
//
// Note that this rounds: the shared value is llround(v * scale) / scale, not v.
// Expected values must therefore be computed from the ROUNDED inputs wherever a
// test claims near-exactness, or the input rounding shows up as operator error.
// `RoundTrip` below exists for exactly that.
inline AV ShareVector(const std::vector<double>& values, EngineRef engine) {
    cdough::Vector<DataType> plain(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        plain[i] = static_cast<DataType>(std::llround(values[i] * scale));
    }
    return engine.secret_share_a(plain, 0, precision);
}

// Shares raw integers at precision 0. Used for the 0/1 selectors that `Multiplex`
// and the comparison primitives consume: those expect a RAW 0/1, not fixed-point
// 1.0, and handing them a scaled one is silently wrong rather than an error.
inline AV ShareRaw(const std::vector<DataType>& values, EngineRef engine) {
    cdough::Vector<DataType> plain(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        plain[i] = values[i];
    }
    AV shared = engine.secret_share_a(plain, 0, 0);
    shared.setPrecision(0);
    return shared;
}

// The value a double actually takes once shared at `precision`.
inline double RoundTrip(double value) {
    return static_cast<double>(std::llround(value * scale)) / static_cast<double>(scale);
}

inline std::vector<double> RoundTrip(const std::vector<double>& values) {
    std::vector<double> result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        result[i] = RoundTrip(values[i]);
    }
    return result;
}

inline std::vector<double> OpenVector(const AV& value) {
    auto opened = value.open();
    std::vector<double> result(value.size(), 0.0);
    for (std::size_t i = 0; i < value.size(); ++i) {
        result[i] = static_cast<double>(opened[i]) / static_cast<double>(scale);
    }
    return result;
}

// Opens a vector carrying raw integers (a 0/1 flag, an unscaled count) rather
// than fixed point.
inline std::vector<DataType> OpenRaw(const AV& value) {
    auto opened = value.open();
    std::vector<DataType> result(value.size(), 0);
    for (std::size_t i = 0; i < value.size(); ++i) {
        result[i] = static_cast<DataType>(opened[i]);
    }
    return result;
}

inline std::vector<double> OpenMatrix(const SMatrix& m) { return OpenVector(m.data()); }

// Shares a row-major (rows x cols) buffer as a SecureMatrix at `precision`.
inline SMatrix ShareMatrix(const std::vector<double>& row_major, std::size_t rows,
                           std::size_t cols, EngineRef engine) {
    assert(row_major.size() == rows * cols);
    cdough::Vector<DataType> plain(row_major.size());
    for (std::size_t i = 0; i < row_major.size(); ++i) {
        plain[i] = static_cast<DataType>(std::llround(row_major[i] * scale));
    }
    PMatrix plain_matrix(plain, rows, cols, false);
    SMatrix shared = engine.secret_share_matrix(plain_matrix, 0);
    shared.setPrecision(precision);
    return shared;
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

inline bool IsReporter(EngineRef engine) { return engine.getPartyID() == 0; }

inline void Section(const std::string& title, EngineRef engine) {
    if (!IsReporter(engine)) {
        return;
    }
    std::cout << "\n=== " << title << " ===" << std::endl;
}

inline void Note(const std::string& text, EngineRef engine) {
    if (!IsReporter(engine)) {
        return;
    }
    std::cout << "    note: " << text << std::endl;
}

// ---------------------------------------------------------------------------
// The tolerance check
// ---------------------------------------------------------------------------

// How an element's error is measured.
enum class ErrorMode {
    // |actual - expected|. For quantities whose magnitude is bounded and known,
    // such as a probability or a log.
    Absolute,
    // |actual - expected| / max(1, |expected|). For quantities that range over
    // orders of magnitude, such as Exp or a reciprocal, where a single absolute
    // bound is either vacuous at the top of the range or unmeetable at the
    // bottom. The max(1, .) floor keeps it from exploding near zero.
    Relative,
};

inline const char* ErrorModeName(ErrorMode mode) {
    return mode == ErrorMode::Absolute ? "abs" : "rel";
}

inline double ElementError(double actual, double expected, ErrorMode mode) {
    const double difference = std::abs(actual - expected);
    if (mode == ErrorMode::Absolute) {
        return difference;
    }
    return difference / std::max(1.0, std::abs(expected));
}

// Compares an opened result against a plaintext reference.
//
// Prints one summary line -- always, pass or fail -- carrying the tolerance that
// was applied and the worst error observed against it, so that a passing run
// still shows how much margin the operator had. On failure it prints the
// offending elements before asserting.
//
// `inputs` is optional and used only for diagnostics; pass an empty vector when
// there is no single natural input per element (a matrix product, say).
//
// The assert fires on EVERY party, not just the reporter. All parties see the
// same opened values and compute the same reference, so a failure that is real
// aborts the whole job rather than one rank.
inline void CheckClose(const std::string& name, const std::vector<double>& actual,
                       const std::vector<double>& expected, double tolerance, EngineRef engine,
                       ErrorMode mode = ErrorMode::Absolute,
                       const std::vector<double>& inputs = {}) {
    assert(actual.size() == expected.size());
    assert(tolerance > 0.0);

    double worst_error = 0.0;
    std::size_t worst_index = 0;
    std::size_t num_failures = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double error = ElementError(actual[i], expected[i], mode);
        if (error > worst_error) {
            worst_error = error;
            worst_index = i;
        }
        if (error > tolerance) {
            ++num_failures;
        }
    }

    if (IsReporter(engine)) {
        std::cout << "  " << std::left << std::setw(34) << name << std::right << " n="
                  << std::setw(6) << actual.size() << "  tol=" << std::scientific
                  << std::setprecision(2) << std::setw(9) << tolerance << " ("
                  << ErrorModeName(mode) << ")"
                  << "  worst=" << std::setw(9) << worst_error << " @" << std::setw(5)
                  << worst_index << "   " << (num_failures == 0 ? "PASS" : "FAIL") << std::endl;

#ifdef PLAYGROUND_TEST_PRINT
        const std::size_t shown = std::min(kPrintElements, actual.size());
        std::cout << "      " << std::left << std::setw(6) << "i" << std::setw(16) << "input"
                  << std::setw(16) << "expected" << std::setw(16) << "actual" << std::setw(14)
                  << "error" << std::right << std::endl;
        for (std::size_t i = 0; i < shown; ++i) {
            std::cout << "      " << std::left << std::setw(6) << i << std::setw(16);
            if (i < inputs.size()) {
                std::cout << std::fixed << std::setprecision(6) << inputs[i];
            } else {
                std::cout << "-";
            }
            std::cout << std::setw(16) << std::fixed << std::setprecision(6) << expected[i]
                      << std::setw(16) << actual[i] << std::setw(14) << std::scientific
                      << std::setprecision(3) << ElementError(actual[i], expected[i], mode)
                      << std::right << std::endl;
        }
#endif

        if (num_failures != 0) {
            std::cerr << "    " << num_failures << " of " << actual.size()
                      << " elements exceeded the tolerance. Offenders:" << std::endl;
            std::size_t reported = 0;
            for (std::size_t i = 0; i < actual.size() && reported < kPrintElements; ++i) {
                const double error = ElementError(actual[i], expected[i], mode);
                if (error <= tolerance) {
                    continue;
                }
                std::cerr << "      i=" << i;
                if (i < inputs.size()) {
                    std::cerr << "  input=" << std::fixed << std::setprecision(8) << inputs[i];
                }
                std::cerr << "  expected=" << std::fixed << std::setprecision(8) << expected[i]
                          << "  actual=" << actual[i] << "  error=" << std::scientific
                          << std::setprecision(3) << error << std::endl;
                ++reported;
            }
        }
    }

    assert(num_failures == 0 && "playground test tolerance exceeded");
}

// Exact comparison for the operators that ARE exact: index permutations, 0/1
// flags, and pattern constructors. Tolerance would only hide a defect in these.
inline void CheckExactRaw(const std::string& name, const std::vector<DataType>& actual,
                          const std::vector<DataType>& expected, EngineRef engine) {
    assert(actual.size() == expected.size());

    std::size_t num_failures = 0;
    std::size_t first_bad = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            if (num_failures == 0) {
                first_bad = i;
            }
            ++num_failures;
        }
    }

    if (IsReporter(engine)) {
        std::cout << "  " << std::left << std::setw(34) << name << std::right << " n="
                  << std::setw(6) << actual.size() << "  tol=" << std::setw(9) << "exact"
                  << "       "
                  << "  worst=" << std::setw(9) << (num_failures == 0 ? "0" : "mismatch")
                  << "      " << "   " << (num_failures == 0 ? "PASS" : "FAIL") << std::endl;

        if (num_failures != 0) {
            std::cerr << "    " << num_failures << " mismatched; first at i=" << first_bad
                      << ": expected=" << static_cast<long long>(expected[first_bad])
                      << " actual=" << static_cast<long long>(actual[first_bad]) << std::endl;
        }
    }

    assert(num_failures == 0 && "playground test exact comparison failed");
}


// ---------------------------------------------------------------------------
// Random input generation
// ---------------------------------------------------------------------------
// Follows `tests/test_primitives.cpp`: party 0 draws with
// `engine.populateLocalRandom` and is the dealer for the shares, so it alone
// holds the plaintext and it alone can check. Other parties' buffers are ignored
// by `secret_share_a(..., 0, ...)`.
//
// The raw draw spans the whole integer range, which is meaningless as an operator
// input, so it is shaped into the target interval -- the same thing
// `test_private_division` does with `(ub % (scale - 1)) + 1`, generalised to an
// arbitrary linear or logarithmic range. Log spacing is the right shape whenever
// the operator's input spans orders of magnitude; a linear draw over [1e-5, 1e5]
// would put essentially every sample at the top of the range.
//
// REPRODUCIBILITY, STATED PLAINLY. `populateLocalRandom` is not seedable from the
// test, so a draw that produces a failure CANNOT BE RE-RUN (semantic task 0021,
// decision D-1). The mitigation: party 0 holds the inputs in the clear and
// `CheckClose` prints the offending values. To make such a case permanent, copy
// the printed inputs into one of the fixed hand-picked vectors above it.
inline std::vector<double> RandomInputs(std::size_t count, double low, double high,
                                        bool log_spaced, EngineRef engine) {
    cdough::Vector<DataType> raw(count);
    if (engine.getPartyID() == 0) {
        engine.populateLocalRandom(raw);
    }

    std::vector<double> shaped(count, low);
    if (engine.getPartyID() != 0) {
        return shaped;  // ignored downstream: party 0 is the dealer
    }

    const double log_low = log_spaced ? std::log(low) : 0.0;
    const double log_span = log_spaced ? (std::log(high) - log_low) : 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        // Top 53 bits of the unsigned reinterpretation give a double in [0, 1).
        const auto bits = static_cast<std::uint64_t>(raw[i]);
        const double unit = static_cast<double>(bits >> 11) / 9007199254740992.0;  // 2^53
        shaped[i] = log_spaced ? std::exp(log_low + unit * log_span)
                               : low + unit * (high - low);
    }
    return shaped;
}

// Tolerance check for a randomized sample, where only party 0 holds the inputs
// and can compute the reference.
//
// This is the one place the suite departs from asserting on every party, and the
// reason is structural rather than a preference: with `populateLocalRandom` the
// plaintext exists only on the dealer. It matches `tests/test_primitives.cpp`,
// which guards its assertions the same way for the same reason.
inline void CheckCloseOnReporter(const std::string& name, const std::vector<double>& actual,
                                 const std::vector<double>& expected, double tolerance,
                                 EngineRef engine, ErrorMode mode = ErrorMode::Absolute,
                                 const std::vector<double>& inputs = {}) {
    if (!IsReporter(engine)) {
        return;
    }
    CheckClose(name, actual, expected, tolerance, engine, mode, inputs);
}

// ---------------------------------------------------------------------------
// Plaintext references
// ---------------------------------------------------------------------------
// These mirror what the secure operators compute. They are written out longhand
// rather than pulled from a library so that the reference and the operator do
// not share an implementation -- a shared helper would let the same mistake
// appear on both sides of the comparison and cancel.

// The numerically stable logistic function, matching the branch structure the
// secure Sigmoid multiplexes over.
inline double RefSigmoid(double eta) {
    if (eta >= 0.0) {
        const double z = std::exp(-eta);
        return 1.0 / (1.0 + z);
    }
    const double z = std::exp(eta);
    return z / (1.0 + z);
}

inline double RefLogOnePlusExp(double eta) {
    if (eta >= 0.0) {
        return eta + std::log1p(std::exp(-eta));
    }
    return std::log1p(std::exp(eta));
}

// The exact standard normal CDF, via erfc. NOT the Abramowitz & Stegun 26.2.17
// approximation the secure operator uses: the point of the comparison is to
// bound the TOTAL error against the true Phi, and A&S 26.2.17 contributes about
// 7.5e-8 of that, which is two orders of magnitude below the 1.5e-5 the number
// format can represent. Reimplementing the approximation here would test only
// the fixed-point evaluation of it and would silently excuse an error in the
// coefficients.
inline double RefNormalCdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

inline double RefTwoSidedPValue(double z) { return 2.0 * RefNormalCdf(-std::abs(z)); }

// Row-major (rows_a x inner) * (inner x cols_b) -> row-major (rows_a x cols_b).
inline std::vector<double> RefMatMul(const std::vector<double>& a, const std::vector<double>& b,
                                     std::size_t rows_a, std::size_t inner, std::size_t cols_b) {
    assert(a.size() == rows_a * inner);
    assert(b.size() == inner * cols_b);
    std::vector<double> result(rows_a * cols_b, 0.0);
    for (std::size_t i = 0; i < rows_a; ++i) {
        for (std::size_t j = 0; j < cols_b; ++j) {
            double total = 0.0;
            for (std::size_t k = 0; k < inner; ++k) {
                total += a[i * inner + k] * b[k * cols_b + j];
            }
            result[i * cols_b + j] = total;
        }
    }
    return result;
}

inline std::vector<double> RefTranspose(const std::vector<double>& m, std::size_t rows,
                                        std::size_t cols) {
    assert(m.size() == rows * cols);
    std::vector<double> result(rows * cols, 0.0);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            result[j * rows + i] = m[i * cols + j];
        }
    }
    return result;
}

inline std::vector<double> RefIdentity(std::size_t n, double c = 1.0) {
    std::vector<double> result(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        result[i * n + i] = c;
    }
    return result;
}

// Gauss-Jordan inverse with partial pivoting. Returns false for a singular
// matrix; the caller is expected to treat that as a defect in the test's own
// fixture, not as a result.
inline bool RefInvert(std::vector<double>& m, std::size_t n) {
    assert(m.size() == n * n);
    std::vector<double> inverse = RefIdentity(n);

    for (std::size_t column = 0; column < n; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < n; ++row) {
            if (std::abs(m[row * n + column]) > std::abs(m[pivot * n + column])) {
                pivot = row;
            }
        }
        if (std::abs(m[pivot * n + column]) < 1e-12) {
            return false;
        }
        if (pivot != column) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(m[column * n + j], m[pivot * n + j]);
                std::swap(inverse[column * n + j], inverse[pivot * n + j]);
            }
        }

        const double diagonal = m[column * n + column];
        for (std::size_t j = 0; j < n; ++j) {
            m[column * n + j] /= diagonal;
            inverse[column * n + j] /= diagonal;
        }

        for (std::size_t row = 0; row < n; ++row) {
            if (row == column) {
                continue;
            }
            const double factor = m[row * n + column];
            if (factor == 0.0) {
                continue;
            }
            for (std::size_t j = 0; j < n; ++j) {
                m[row * n + j] -= factor * m[column * n + j];
                inverse[row * n + j] -= factor * inverse[column * n + j];
            }
        }
    }

    m = inverse;
    return true;
}

}  // namespace playground_test
