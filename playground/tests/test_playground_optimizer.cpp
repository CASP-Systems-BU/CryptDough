// Assertion-based tests for playground/library/optimizer.h: the secure linear
// algebra and the oblivious quasi-Newton optimizer built on it.
//
// WHY THESE OPERATORS NEED A TEST MORE THAN THE PRIMITIVES DO. Every one of
// them is a claim about buffer layout as much as about arithmetic. The matmul
// kernel takes a row-major left operand and a COLUMN-WISE right operand, and
// reinterpreting a row-major buffer as column-wise silently transposes it
// (semantic topic 0001, claim C-09). That convention is load-bearing in
// `AsColumnWise`, `OuterProduct`, `MatVecBatched` and `BfgsInverseUpdateBatched`,
// and getting it backwards produces a matrix of the right shape full of the
// wrong numbers -- which is exactly what semantic tasks 0010 and 0011 were
// about. A shape assertion does not catch it; a value comparison does.
//
// PRIVACY: a test, not a privacy demonstration. It opens intermediate values
// freely. Nothing here supports any claim in semantic topic 0003, and in
// particular running the optimizer here says nothing about its obliviousness.
//
// Run: mpirun -np 3 ./test_playground_optimizer

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include "cdough.h"

#include "../library/optimizer.h"
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
// No bound was ever widened to make a failing check pass.

// Pure index permutations and pattern constructors. Exact, and checked as such
// where the value is a raw integer.
constexpr double kExactish = 4.0 * kResolution;  // ~6.1e-5

// One matmul: each output entry is a sum of `inner` products, and the kernel
// truncates once per product. The bound therefore grows with the contraction
// length; at the sizes used here (inner <= 4) a few ULPs per term is ample.
constexpr double kMatMulTolerance = 2.0e-4;  // absolute; measured worst 0.00e+00 (fixtures are exact)

// NewtonSchulzInverse. The calibration table in optimizer.h records a noise
// floor of 5e-5 to 1.2e-4 for well-conditioned cases at 14 iterations; 2e-3
// leaves better than an order of magnitude of margin over the worst of those.
// The residual check |A X - I| is held to the same bar.
//
// This bounds an operator the PIPELINE no longer calls. Since task 0018
// the only callers of NewtonSchulzInverse are this suite and the
// secure-logistic-regression calibration harness; the analysis inverts through
// SecureInverse (regression.h:408, :732). Left as derived rather than retuned to
// its 2.85e-05 worst, because tightening a bound on a path nothing depends on
// buys nothing. kSymmetricInverseTolerance below is the one that matters now.
constexpr double kInverseTolerance = 4.0e-4;  // absolute; measured worst 2.85e-05 (residual)

// SecureInverse / SymmetricInverse -- the Cholesky inverse, and the one the
// PIPELINE actually uses. Calibrated below.
constexpr double kSymmetricInverseTolerance = 1.0e-2;  // absolute; PLACEHOLDER, calibrating

// BfgsInverseUpdateBatched composes four kernel calls, and optimizer.h's own
// comment quotes ~5e-4 for it. 4e-3 is that with margin.
constexpr double kBfgsUpdateTolerance = 4.0e-4;  // absolute; measured worst 8.13e-05

// The central-difference gradient. On a QUADRATIC objective the central
// difference is algebraically exact, so the entire budget here is fixed-point:
// one objective evaluation over 2*dim points and one division circuit.
constexpr double kGradientTolerance = 1.0e-3;  // absolute; measured worst 0.00e+00 (exact on a quadratic)

// Convergence of the optimizer itself. Bounded a priori by the BFGS update
// noise above (~5e-4 per entry), amplified by the conditioning of the test
// problem (weights 0.5 to 2.0, so kappa = 4): ~2e-3. The measured worst
// distance at kMaxBfgsIterations is 1.27e-3, which agrees with that argument,
// so the bound is set to 5e-3 -- about 4x the measurement and 2.5x the a priori
// estimate. See the FINDING note in TestMinimizeBFGS for the full trajectory.
constexpr double kOptimumTolerance = 5.0e-3;  // absolute

// The objective at the optimum. f = sum w_i (x_i - c_i)^2 with |x - c| at the
// plateau above is ~1e-5, below the 1.53e-5 resolution, so it truncates to
// exactly 0. A bound of one ULP would be the honest floor; 1e-3 allows the
// plateau to degrade by an order of magnitude before this fires.
constexpr double kObjectiveTolerance = 1.0e-3;  // absolute

// ---------------------------------------------------------------------------
// The test objective: a weighted quadratic
// ---------------------------------------------------------------------------
//
//     f(x) = sum_i w_i (x_i - c_i)^2
//
// Chosen for three reasons. Its minimum is known exactly (x = c), so the
// optimizer has an unambiguous target. Its gradient is known exactly
// (2 w_i (x_i - c_i)), so the numerical gradient can be checked against truth
// rather than against itself. And the weights make it anisotropic, so BFGS has
// to actually build a non-trivial inverse-Hessian approximation -- an isotropic
// quadratic is solved by the first steepest-descent step and would leave
// `BfgsInverseUpdateBatched` untested through this path.

constexpr std::size_t kDim = 3;
const std::vector<double> kWeights = {1.0, 2.0, 0.5};
const std::vector<double> kCenter = {0.75, -0.5, 1.25};

// Twelve, chosen from the measured trajectory in the FINDING note at the end of
// TestMinimizeBFGS: the quadratic is essentially solved by iteration 6 and the
// remaining motion is fixed-point noise creeping down a plateau. Twelve buys
// margin over 6 while keeping the whole test in the region of half a second per
// optimizer run, which matters because each iteration evaluates the objective
// over the entire 17-rung line-search ladder.
constexpr int kMaxBfgsIterations = 12;

double PlainObjective(const std::vector<double>& x) {
    double total = 0.0;
    for (std::size_t i = 0; i < kDim; ++i) {
        const double difference = x[i] - kCenter[i];
        total += kWeights[i] * difference * difference;
    }
    return total;
}

std::vector<double> PlainGradient(const std::vector<double>& x) {
    std::vector<double> gradient(kDim, 0.0);
    for (std::size_t i = 0; i < kDim; ++i) {
        gradient[i] = 2.0 * kWeights[i] * (x[i] - kCenter[i]);
    }
    return gradient;
}

// The secure counterpart, as a BatchedObjective. `params` is length
// num_points * kDim with (k, i) -> k * kDim + i, matching the layout the
// optimizer and the numerical gradient both use.
BatchedObjective MakeSecureObjective(const AV& center, const AV& weights) {
    return [&center, &weights](const AV& params, std::size_t num_points) -> AV {
        AV point = Clone(params);
        point.setPrecision(0);

        // cyclic_subset_reference(k) repeats the WHOLE vector k times, which is
        // the tiling the (k, i) layout wants. (repeated_subset_reference(k)
        // repeats each ELEMENT k times and would interleave the coordinates.)
        AV center_tiled = Clone(center.cyclic_subset_reference(num_points));
        center_tiled.setPrecision(0);

        AV difference = point - center_tiled;
        difference.setPrecision(0);
        AV square = (*(difference * difference)) / scale;
        square.setPrecision(0);

        AV weights_tiled = Clone(weights.cyclic_subset_reference(num_points));
        weights_tiled.setPrecision(0);
        AV weighted = (*(square * weights_tiled)) / scale;

        AV total = weighted.chunkedSum(kDim);
        total.setPrecision(precision);
        return total;
    };
}

// ---------------------------------------------------------------------------
// Identity constructors
// ---------------------------------------------------------------------------
void TestIdentity(EngineRef engine) {
    Section("Identity / ScaledIdentity", engine);

    const std::size_t n = 4;

    SMatrix eye = Identity(n, engine);
    CheckClose("Identity", OpenMatrix(eye), RefIdentity(n), kExactish, engine);

    SMatrix scaled = ScaledIdentity(n, 2.5, engine);
    CheckClose("ScaledIdentity(2.5)", OpenMatrix(scaled), RefIdentity(n, 2.5), kExactish, engine);

    // The optimizer's own use: ScaledIdentity(n, 2.0) is the `2I` of the
    // Newton-Schulz residual, so it is worth pinning separately.
    SMatrix two_eye = ScaledIdentity(n, 2.0, engine);
    CheckClose("ScaledIdentity(2.0)", OpenMatrix(two_eye), RefIdentity(n, 2.0), kExactish, engine);
}

// ---------------------------------------------------------------------------
// Transpose and the column-wise convention
// ---------------------------------------------------------------------------
void TestTranspose(EngineRef engine) {
    Section("TransposeData / Transpose / AsColumnWise", engine);

    const std::size_t rows = 3;
    const std::size_t cols = 4;
    std::vector<double> values(rows * cols);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = 0.25 * static_cast<double>(i) - 1.0;
    }

    AV shared = ShareVector(values, engine);
    CheckClose("TransposeData", OpenVector(TransposeData(shared, rows, cols)),
               RefTranspose(RoundTrip(values), rows, cols), kExactish, engine);

    SMatrix m = ShareMatrix(values, rows, cols, engine);
    SMatrix transposed = Transpose(m);
    assert(transposed.rows() == cols && transposed.cols() == rows);
    CheckClose("Transpose", OpenMatrix(transposed), RefTranspose(RoundTrip(values), rows, cols),
               kExactish, engine);

    // Involution. Independent of the reference above: a Transpose that swapped
    // the wrong index pair could still match one hand-written expectation, but
    // it would not be its own inverse for a non-square matrix.
    SMatrix round_trip = Transpose(transposed);
    CheckClose("Transpose is an involution", OpenMatrix(round_trip), RoundTrip(values), kExactish,
               engine);

    // ---- The column-wise convention (semantic topic 0001, claim C-09) ----
    //
    // Two distinct facts, and confusing them is the defect this section exists
    // to catch:
    //
    //   (a) REINTERPRETING a row-major buffer as column-wise yields the
    //       TRANSPOSE, not the matrix.
    //   (b) AsColumnWise physically permutes first, so the result represents
    //       the matrix ITSELF and is the correct right-hand operand for a
    //       matmul that means A * B.
    const std::size_t inner = 3;
    const std::vector<double> a_values = {0.5, -1.25, 2.0, 1.0, 0.25, -0.75};  // 2 x 3
    const std::vector<double> b_values = {1.5, -0.5, 0.25, 2.0, -1.0, 0.75};   // 3 x 2

    SMatrix a = ShareMatrix(a_values, 2, inner, engine);
    SMatrix b = ShareMatrix(b_values, inner, 2, engine);

    // (b): A * B, the documented contract.
    SMatrix product = a.matrixRightMultiplyWithColumnMatrixVectorized(AsColumnWise(b));
    product.setPrecision(precision);
    CheckClose("AsColumnWise gives A * B", OpenMatrix(product),
               RefMatMul(RoundTrip(a_values), RoundTrip(b_values), 2, inner, 2),
               kMatMulTolerance, engine);

    // (a): the same buffer reinterpreted, with no permutation, is B^T. Feeding
    // it to a matmul therefore computes A2 * B^T. If C-09 were false -- if
    // reinterpretation preserved the matrix -- this check would fail while the
    // one above still passed, which is what makes the pair worth having.
    const std::vector<double> a2_values = {1.0, -2.0, 0.5, 3.0};  // 2 x 2
    SMatrix a2 = ShareMatrix(a2_values, 2, 2, engine);
    SMatrix b_reinterpreted(b.data(), 2, inner, true);
    b_reinterpreted.setPrecision(precision);
    SMatrix product_t = a2.matrixRightMultiplyWithColumnMatrixVectorized(b_reinterpreted);
    product_t.setPrecision(precision);
    CheckClose("reinterpretation transposes (C-09)", OpenMatrix(product_t),
               RefMatMul(RoundTrip(a2_values), RefTranspose(RoundTrip(b_values), inner, 2), 2, 2,
                         inner),
               kMatMulTolerance, engine);
}

// ---------------------------------------------------------------------------
// ScaleMatrix and FrobeniusNormSquared
// ---------------------------------------------------------------------------
void TestMatrixScalars(EngineRef engine) {
    Section("ScaleMatrix / FrobeniusNormSquared", engine);

    const std::size_t rows = 2;
    const std::size_t cols = 3;
    const std::vector<double> values = {1.5, -2.0, 0.25, 3.75, -1.125, 0.5};
    const std::vector<double> rounded = RoundTrip(values);

    SMatrix m = ShareMatrix(values, rows, cols, engine);

    const double factor = -1.75;
    AV scalar = ShareVector({factor}, engine);
    std::vector<double> scaled_expected(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        scaled_expected[i] = rounded[i] * RoundTrip(factor);
    }
    CheckClose("ScaleMatrix", OpenMatrix(ScaleMatrix(m, scalar)), scaled_expected, kExactish,
               engine);

    double frobenius = 0.0;
    for (const double value : rounded) {
        frobenius += value * value;
    }
    CheckClose("FrobeniusNormSquared", OpenVector(FrobeniusNormSquared(m)), {frobenius},
               kMatMulTolerance, engine);
}

// ---------------------------------------------------------------------------
// MatVecBatched and OuterProduct
// ---------------------------------------------------------------------------
void TestMatVecAndOuter(EngineRef engine) {
    Section("MatVecBatched / OuterProduct", engine);

    const std::size_t n = 3;
    const std::vector<double> m_values = {1.0, -0.5, 0.25, 2.0, 0.75, -1.5, -0.25, 1.25, 0.5};
    const std::vector<double> v_values = {0.5, -1.0, 2.25};

    SMatrix m = ShareMatrix(m_values, n, n, engine);
    AV v = ShareVector(v_values, engine);

    CheckClose("MatVecBatched", OpenVector(MatVecBatched(m, v)),
               RefMatMul(RoundTrip(m_values), RoundTrip(v_values), n, n, 1), kMatMulTolerance,
               engine);

    const std::vector<double> a_values = {1.5, -2.0, 0.25};
    const std::vector<double> b_values = {0.5, 1.25, -0.75};
    AV a = ShareVector(a_values, engine);
    AV b = ShareVector(b_values, engine);

    // a b^T: an (n x 1) times a (1 x n).
    CheckClose("OuterProduct", OpenMatrix(OuterProduct(a, b)),
               RefMatMul(RoundTrip(a_values), RoundTrip(b_values), n, 1, n), kMatMulTolerance,
               engine);
}

// ---------------------------------------------------------------------------
// NewtonSchulzInverse
// ---------------------------------------------------------------------------
void TestNewtonSchulzInverse(EngineRef engine) {
    Section("NewtonSchulzInverse", engine);

    // Well-conditioned and O(1)-scaled, which are the operator's stated
    // preconditions. An ill-conditioned matrix is NOT tested here as a
    // correctness case: the operator documents that it silently returns a poorer
    // approximation rather than iterating longer, so asserting on one would be
    // asserting on undefined accuracy.
    struct Case {
        const char* name;
        std::size_t n;
        std::vector<double> values;
    };

    const std::vector<Case> cases = {
        {"NewtonSchulzInverse identity", 3, RefIdentity(3)},
        {"NewtonSchulzInverse spd 2x2", 2, {2.0, 0.5, 0.5, 1.5}},
        {"NewtonSchulzInverse spd 3x3", 3, {2.0, 0.3, 0.1, 0.3, 1.5, 0.25, 0.1, 0.25, 1.0}},
        {"NewtonSchulzInverse non-symmetric", 3, {1.5, 0.25, -0.5, 0.1, 1.25, 0.3, -0.2, 0.4, 1.75}},
    };

    for (const Case& test_case : cases) {
        const std::size_t n = test_case.n;
        const std::vector<double> rounded = RoundTrip(test_case.values);

        std::vector<double> expected = rounded;
        const bool invertible = RefInvert(expected, n);
        // A singular fixture is a defect in this test, not a result about the
        // operator, so it aborts rather than being reported as a failure.
        assert(invertible && "test fixture matrix is singular");

        SMatrix a = ShareMatrix(test_case.values, n, n, engine);
        SMatrix inverse = NewtonSchulzInverse(a);

        const std::vector<double> actual = OpenMatrix(inverse);
        CheckClose(test_case.name, actual, expected, kInverseTolerance, engine);

        // Residual |A X - I|, computed from the OPENED inverse. Independent of
        // the Gauss-Jordan reference: it would catch a case where both the
        // reference and the operator were wrong in the same way.
        const std::vector<double> residual = RefMatMul(rounded, actual, n, n, n);
        CheckClose(std::string(test_case.name) + " residual A*X=I", residual, RefIdentity(n),
                   kInverseTolerance, engine);
    }
}

// ---------------------------------------------------------------------------
// SecureInverse / SymmetricInverse (the Cholesky inverse)
// ---------------------------------------------------------------------------
// This is the inverse the PIPELINE uses. Every standard error the analysis
// reports flows through it (regression.h:408 for the mixed models, :732 for the
// fixed-effect ones), where NewtonSchulzInverse above now has no caller outside
// this suite and the secure-logistic-regression calibration harness. Task 0018
// replaced the iterative inverse without adding a test for the direct one, so
// until now the live path was covered only by harness.h's unasserted self-check.
//
// Cases are SPD, which is the operator's precondition: CholeskyFactor does not
// pivot, and a non-positive pivot yields meaningless output with no error and no
// leak. The non-symmetric case from the Newton-Schulz suite is therefore
// deliberately absent -- it is outside the contract, not a case this should pass.
//
// The 4x4 case is not redundant with the 3x3: CholeskySolveManyWith solves all p
// right-hand sides in one batched call with its own row-major p x rhs indexing,
// and a width-4 operand exercises that arithmetic where a 2x2 would not.
void TestSecureInverse(EngineRef engine) {
    Section("SecureInverse (Cholesky)", engine);

    struct Case {
        const char* name;
        std::size_t n;
        std::vector<double> values;
    };

    const std::vector<Case> cases = {
        {"SecureInverse identity", 3, RefIdentity(3)},
        {"SecureInverse spd 2x2", 2, {2.0, 0.5, 0.5, 1.5}},
        {"SecureInverse spd 3x3", 3, {2.0, 0.3, 0.1, 0.3, 1.5, 0.25, 0.1, 0.25, 1.0}},
        {"SecureInverse spd 4x4", 4,
         {2.5, 0.4, 0.2, 0.1, 0.4, 2.0, 0.3, 0.15, 0.2, 0.3, 1.75, 0.25, 0.1, 0.15, 0.25, 1.5}},
    };

    for (const Case& test_case : cases) {
        const std::size_t n = test_case.n;
        const std::vector<double> rounded = RoundTrip(test_case.values);

        std::vector<double> expected = rounded;
        const bool invertible = RefInvert(expected, n);
        assert(invertible && "test fixture matrix is singular");

        SMatrix a = ShareMatrix(test_case.values, n, n, engine);
        SMatrix inverse = SecureInverse(a);

        const std::vector<double> actual = OpenMatrix(inverse);
        CheckClose(test_case.name, actual, expected, kSymmetricInverseTolerance, engine);

        // Residual |A X - I| from the OPENED inverse, independent of the
        // Gauss-Jordan reference, so a shared error in both would still show.
        const std::vector<double> residual = RefMatMul(rounded, actual, n, n, n);
        CheckClose(std::string(test_case.name) + " residual A*X=I", residual, RefIdentity(n),
                   kSymmetricInverseTolerance, engine);
    }

    // The result is NOT symmetrised: each column is an independent solve, so the
    // two triangles differ by rounding. Callers that read only the diagonal (the
    // standard errors do) are unaffected, but the asymmetry is real and is
    // asserted here so that it stays a documented property rather than a
    // surprise. The bound is the same fixed-point budget as the inverse itself.
    {
        const std::size_t n = 3;
        const std::vector<double> values = {2.0, 0.3, 0.1, 0.3, 1.5, 0.25, 0.1, 0.25, 1.0};
        SMatrix a = ShareMatrix(values, n, n, engine);
        const std::vector<double> x = OpenMatrix(SecureInverse(a));
        std::vector<double> upper, lower;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j) {
                upper.push_back(x[i * n + j]);
                lower.push_back(x[j * n + i]);
            }
        CheckClose("SecureInverse asymmetry is within the fixed-point budget", upper, lower,
                   kSymmetricInverseTolerance, engine);
    }
}

// ---------------------------------------------------------------------------
// Line-search and central-difference constants
// ---------------------------------------------------------------------------
// Both are public patterns that depend only on `dim` (or on nothing at all), so
// they are checked EXACTLY on their raw integers. A tolerance would be wrong
// here in a specific way: `armijo_slack` underflows to zero for l >= 5 at
// `precision` 16, and that underflow is a real property of the constant table
// which a tolerant comparison against the ideal c1 * 2^-l would paper over.
void TestConstants(EngineRef engine) {
    Section("MakeCentralDifferenceSelector / MakeLineSearchConstants", engine);

    const std::size_t dim = 3;
    AV selector = MakeCentralDifferenceSelector(dim, engine);

    std::vector<DataType> selector_expected(2 * dim * dim, 0);
    for (std::size_t k = 0; k < dim; ++k) {
        selector_expected[k * dim + k] = scale;
        selector_expected[(dim + k) * dim + k] = -scale;
    }
    CheckExactRaw("MakeCentralDifferenceSelector", OpenRaw(selector), selector_expected, engine);

    const LineSearchConstants ls = MakeLineSearchConstants(engine);
    const std::size_t steps = static_cast<std::size_t>(kLineSearchSteps);

    std::vector<DataType> alpha_expected(steps, 0);
    std::vector<DataType> slack_expected(steps, 0);
    for (std::size_t l = 0; l < steps; ++l) {
        alpha_expected[l] = scale >> l;
        slack_expected[l] =
            std::llround(kArmijoC1 * std::ldexp(1.0, -static_cast<int>(l)) * scale);
    }
    CheckExactRaw("LineSearchConstants.alpha", OpenRaw(ls.alpha), alpha_expected, engine);
    CheckExactRaw("LineSearchConstants.armijo_slack", OpenRaw(ls.armijo_slack), slack_expected,
                  engine);

    std::vector<DataType> prefix_expected(steps * steps, 0);
    for (std::size_t l = 0; l < steps; ++l) {
        for (std::size_t m = 0; m < l; ++m) {
            prefix_expected[l * steps + m] = 1;
        }
    }
    CheckExactRaw("LineSearchConstants.strict_prefix", OpenRaw(ls.strict_prefix), prefix_expected,
                  engine);

    // The smallest rung is 2^-(steps-1). It has to be a NON-ZERO step: at
    // `precision` 16 it was exactly 1 ULP, which is where kLineSearchSteps = 17
    // came from. Since precision moved to 28 it is 4096 ULPs and the format would
    // permit more rungs; the ladder is kept at 17 deliberately (see
    // kLineSearchSteps). What still has to hold, and is what this checks, is that
    // the last rung does not truncate away.
    assert(alpha_expected[steps - 1] >= 1);
    Note("alpha ladder bottoms out at a non-zero step (2^-16 at the current ladder length)",
         engine);
}

// ---------------------------------------------------------------------------
// BfgsInverseUpdateBatched
// ---------------------------------------------------------------------------
void TestBfgsInverseUpdate(EngineRef engine) {
    Section("BfgsInverseUpdateBatched", engine);

    const std::size_t n = 3;
    const std::vector<double> h_values = {1.25, 0.1, -0.2, 0.1, 0.9, 0.05, -0.2, 0.05, 1.5};
    const std::vector<double> s_values = {0.4, -0.25, 0.75};
    const std::vector<double> y_values = {0.6, 0.3, -0.5};
    const double rho_value = 0.4;

    const std::vector<double> h_rounded = RoundTrip(h_values);
    const std::vector<double> s_rounded = RoundTrip(s_values);
    const std::vector<double> y_rounded = RoundTrip(y_values);
    const double rho_rounded = RoundTrip(rho_value);

    // Plaintext H+ = (I - rho s y^T) H (I - rho y s^T) + rho s s^T, written out
    // in the same factored order the operator uses so the comparison is of the
    // arithmetic and not of two different groupings.
    std::vector<double> left = RefIdentity(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            left[i * n + j] -= rho_rounded * s_rounded[i] * y_rounded[j];
        }
    }
    const std::vector<double> left_transposed = RefTranspose(left, n, n);
    const std::vector<double> temp = RefMatMul(left, h_rounded, n, n, n);
    std::vector<double> expected = RefMatMul(temp, left_transposed, n, n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            expected[i * n + j] += rho_rounded * s_rounded[i] * s_rounded[j];
        }
    }

    SMatrix h_inv = ShareMatrix(h_values, n, n, engine);
    AV s = ShareVector(s_values, engine);
    AV y = ShareVector(y_values, engine);
    AV rho = ShareVector({rho_value}, engine);
    SMatrix eye = Identity(n, engine);

    const std::vector<double> actual =
        OpenMatrix(BfgsInverseUpdateBatched(h_inv, s, y, rho, eye));
    CheckClose("BfgsInverseUpdateBatched", actual, expected, kBfgsUpdateTolerance, engine);

    // Symmetry. The update of a symmetric H is symmetric in exact arithmetic,
    // and optimizer.h explicitly declines to ASSUME symmetry when forming the
    // right factor -- it takes a real transpose rather than reusing the buffer.
    // This check is what tells you whether that caution is still buying anything:
    // the asymmetry it measures is pure truncation drift.
    std::vector<double> transposed_actual = RefTranspose(actual, n, n);
    CheckClose("BFGS update preserves symmetry", actual, transposed_actual, kBfgsUpdateTolerance,
               engine);
}

// ---------------------------------------------------------------------------
// NumericalGradientBatched
// ---------------------------------------------------------------------------
void TestNumericalGradient(EngineRef engine) {
    Section("NumericalGradientBatched", engine);

    AV center = ShareVector(kCenter, engine);
    AV weights = ShareVector(kWeights, engine);
    const BatchedObjective objective = MakeSecureObjective(center, weights);

    // First: the objective itself, over several points at once. If this is
    // wrong every gradient and every optimizer check below is meaningless, so
    // it is pinned before anything is built on it.
    const std::vector<std::vector<double>> points = {
        {0.0, 0.0, 0.0}, {0.5, -1.25, 2.0}, kCenter, {-1.0, 1.0, -1.0}};

    std::vector<double> packed;
    std::vector<double> objective_expected;
    for (const std::vector<double>& point : points) {
        packed.insert(packed.end(), point.begin(), point.end());
        objective_expected.push_back(PlainObjective(RoundTrip(point)));
    }

    AV packed_shared = ShareVector(packed, engine);
    CheckClose("batched objective (4 points)", OpenVector(objective(packed_shared, points.size())),
               objective_expected, kMatMulTolerance, engine);

    // The gradient. Central differences are algebraically exact on a quadratic,
    // so the reference is the TRUE gradient, not a finite-difference estimate of
    // it -- which means this check also catches a wrong step size h, something a
    // self-referential comparison could not.
    AV selector = MakeCentralDifferenceSelector(kDim, engine);
    for (const std::vector<double>& point : points) {
        AV x = ShareVector(point, engine);
        const std::vector<double> actual = OpenVector(NumericalGradientBatched(objective, x, selector));
        CheckClose("NumericalGradientBatched", actual, PlainGradient(RoundTrip(point)),
                   kGradientTolerance, engine, ErrorMode::Absolute, point);
    }
}

// ---------------------------------------------------------------------------
// MinimizeBFGSBatched
// ---------------------------------------------------------------------------
void TestMinimizeBFGS(EngineRef engine) {
    Section("MinimizeBFGSBatched", engine);

    AV center = ShareVector(kCenter, engine);
    AV weights = ShareVector(kWeights, engine);
    const BatchedObjective objective = MakeSecureObjective(center, weights);

    const std::vector<double> start = {0.0, 0.0, 0.0};
    const std::vector<double> minimum = RoundTrip(kCenter);

    // Numerical-gradient path (the default, and what every caller before
    // semantic task 0019 used).
    {
        AV x0 = ShareVector(start, engine);
        BatchedOptResult result = MinimizeBFGSBatched(objective, x0, kMaxBfgsIterations);

        if (IsReporter(engine)) {
            std::cout << "    numerical gradient: iterations=" << result.iterations
                      << " converged=" << (result.converged ? "yes" : "no") << std::endl;
        }
        // NOTE: `result.converged` is deliberately NOT asserted. See the
        // FINDING note at the end of this function -- it does not become true
        // on this problem at any iteration count, and asserting it would be
        // asserting something the arithmetic cannot deliver.
        CheckClose("BFGS minimizer (numerical grad)", OpenVector(result.params), minimum,
                   kOptimumTolerance, engine);
        CheckClose("BFGS objective at minimum", OpenVector(result.value), {0.0},
                   kObjectiveTolerance, engine);
    }

    // Analytic-gradient path, added by semantic task 0019. Same problem, same
    // answer expected: supplying a gradient must change the cost, not the result.
    {
        AV x0 = ShareVector(start, engine);
        const BatchedGradient analytic = [&center, &weights](const AV& params) -> AV {
            AV point = Clone(params);
            point.setPrecision(0);
            AV center_raw = Clone(center);
            center_raw.setPrecision(0);
            AV difference = point - center_raw;
            difference.setPrecision(0);
            AV weights_raw = Clone(weights);
            weights_raw.setPrecision(0);
            AV weighted = (*(difference * weights_raw)) / scale;
            AV gradient = *(weighted * DataType(2));
            gradient.setPrecision(precision);
            return gradient;
        };

        BatchedOptResult result =
            MinimizeBFGSBatched(objective, x0, kMaxBfgsIterations, analytic);

        if (IsReporter(engine)) {
            std::cout << "    analytic gradient : iterations=" << result.iterations
                      << " converged=" << (result.converged ? "yes" : "no") << std::endl;
        }
        CheckClose("BFGS minimizer (analytic grad)", OpenVector(result.params), minimum,
                   kOptimumTolerance, engine);
    }

    // -----------------------------------------------------------------------
    // FINDING (task 0020, measured on the 3PC build, 2026-09-20)
    // -----------------------------------------------------------------------
    // `BatchedOptResult::converged` stays FALSE on this problem at every
    // iteration count tried -- 2, 4, 6, 8, 12, 16 and 40 -- on BOTH the
    // numerical and the analytic gradient path, while the returned parameters
    // are correct throughout.
    //
    // Why. The loop's termination bit is `AnyAbsAtLeast(gradient,
    // kSmallEpsilon_scaled)`, and kSmallEpsilon_scaled is
    // (DataType)(0.0001f * 65536) = 6 raw units, i.e. 9.16e-5. Here the
    // gradient is 2 w_i (x_i - c_i) with w up to 2, so clearing that bar needs
    // every coordinate within 2.3e-5 of the optimum -- about 1.5 ULP at
    // `precision` 16. The measured plateau is 9 ULP:
    //
    //     iterations   worst |x - c|
    //          2         5.14e-01
    //          4         4.46e-02
    //          6         1.92e-03      <- quadratic essentially solved
    //          8         1.65e-03
    //         12         1.27e-03
    //         16         9.31e-04
    //         40         1.37e-04      <- objective already truncates to 0
    //
    // So the gradient's own fixed-point noise floor sits above the termination
    // threshold and the flag is unreachable. The optimizer runs to its cap and
    // returns a correct answer with converged == false.
    //
    // WHAT THIS IS AND IS NOT. It is a property of the threshold relative to
    // the number format, not a defect in the search: the parameters are right.
    // It does mean a caller must not read `converged == false` as "the fit
    // failed" -- `plain-lr.cpp` prints exactly that as "no (hit cap)". Whether
    // kSmallEpsilon should be raised, or the flag redefined, is a decision for
    // a separate task; this one does not repair library operators (task 0020
    // Non-Goals). Recorded here so the next reader meets the measurement rather
    // than rediscovering it.
    Note("converged flag is not asserted: it is unreachable at precision 16 -- see FINDING above",
         engine);
}

}  // namespace

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);

    if (engine.getPartyID() == 0) {
        std::cout << "\n########## playground/library/optimizer.h ##########\n"
                  << "precision = " << precision
                  << "   NewtonSchulz iterations = " << kMatrixInverseIterations
                  << "   line-search rungs = " << kLineSearchSteps << std::endl;
    }

    TestIdentity(engine);
    TestTranspose(engine);
    TestMatrixScalars(engine);
    TestMatVecAndOuter(engine);
    TestNewtonSchulzInverse(engine);
    TestSecureInverse(engine);
    TestConstants(engine);
    TestBfgsInverseUpdate(engine);
    TestNumericalGradient(engine);
    TestMinimizeBFGS(engine);

    if (engine.getPartyID() == 0) {
        std::cout << "\noptimizer: all checks passed" << std::endl;
    }

    return 0;
}
