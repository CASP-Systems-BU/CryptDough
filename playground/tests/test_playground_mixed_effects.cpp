// Assertion-based tests for cdough::regression::mixedeffects in
// playground/library/regression.h -- the vectorized Laplace-approximation
// objective for a random-intercept logistic model.
//
// WHAT MAKES THIS LAYER EASY TO GET WRONG. Every operator here is an
// aggregation over a padded, batched buffer with the layout
//
//     observation-level   (k, g, j) -> k * (G * N) + g * N + j
//     group-level         (k, g)    -> k * G + g
//
// and the aggregations are performed by `chunkedSum(N)` and by two DIFFERENT
// broadcast mappings that are easy to confuse: `cyclic_subset_reference(k)`
// repeats the whole vector k times, `repeated_subset_reference(k)` repeats each
// element k times. Swapping them produces a buffer of exactly the right length
// with the coordinates interleaved wrongly -- no assertion fires, and the
// likelihood is merely wrong. The only instrument that catches it is a
// plaintext mirror computed independently, which is what this file is.
//
// The padding mask gets the same treatment: a mask that is applied in the wrong
// place still yields a number, and that number is the likelihood of a dataset
// that includes its own padding rows.
//
// PRIVACY: a test, not a privacy demonstration. Nothing here supports any claim
// in semantic topic 0003.
//
// Run: mpirun -np 3 ./test_playground_mixed_effects

#include <algorithm>
#include <cmath>
#include <vector>

#include "cdough.h"

#include "../library/regression.h"
#include "test_util.h"

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;
using namespace cdough::debug;
using namespace cdough::service;
using namespace cdough::regression;
using namespace playground_test;

namespace {

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------
// Small on purpose. These operators' cost is dominated by Sigmoid, Log and
// LogOnePlusExp over K*G*N elements; correctness of the layout does not need
// scale, and semantic topic 0002 is where scaling behaviour belongs.
//
// The last observation of group 1 is PADDING (mask 0) with deliberately absurd
// y and x values. If the mask is ever dropped or applied in the wrong place,
// those values poison the group's aggregate and the comparison fails loudly --
// which is the point of choosing them rather than leaving padding rows zero,
// since a zero row contributes almost nothing either way and would hide the bug.

constexpr std::size_t kGroups = 3;
constexpr std::size_t kObsPerGroup = 4;
constexpr std::size_t kNumFixed = 2;
constexpr std::size_t kTotalRows = kGroups * kObsPerGroup;
constexpr std::size_t kNumPoints = 2;

// (G*N) x p row-major, row = g * N + j. Column 0 is the intercept.
const std::vector<double> kDesign = {
    // group 0
    1.0, 0.50, 1.0, -0.75, 1.0, 1.25, 1.0, 0.00,
    // group 1 (last row is padding)
    1.0, -1.50, 1.0, 0.25, 1.0, 0.80, 9.0, 9.00,
    // group 2
    1.0, 0.10, 1.0, -0.40, 1.0, 1.75, 1.0, -1.10,
};

const std::vector<double> kOutcome = {
    1.0, 0.0, 1.0, 0.0,  //
    0.0, 1.0, 1.0, 1.0,  // last is padding
    1.0, 1.0, 0.0, 0.0,
};

const std::vector<double> kMask = {
    1.0, 1.0, 1.0, 1.0,  //
    1.0, 1.0, 1.0, 0.0,  // padding
    1.0, 1.0, 1.0, 1.0,
};

// Packed parameter points, length K * (p + 1); the trailing entry of each point
// is s, with sigma^2 = exp(2s).
const std::vector<double> kParams = {
    0.25, -0.60, -0.20,  // point 0: beta = (0.25, -0.60), s = -0.20
    -0.40, 0.80, 0.10,   // point 1: beta = (-0.40, 0.80), s =  0.10
};

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

constexpr double kExactish = 4.0 * kResolution;  // ~6.1e-5

// One matmul contracting over p = 2.
constexpr double kLinearPredictorTolerance = 2.0e-4;  // absolute; measured worst 2.38e-05

// sigma^2 = Exp(2s): one Exp, relative.
constexpr double kSigmaSquaredTolerance = 1.0e-3;  // relative; measured worst 1.73e-04 -- left as derived

// The conditional mode. Five damped Newton steps, each costing one Sigmoid over
// K*G*N elements (measured error ~1.1e-4 each) and one division circuit. The
// iteration is contracting, so the errors do not compound across steps; the
// bound is a few Sigmoid errors plus the division truncation, widened for the
// fact that the mode is the ARGUMENT of everything downstream.
constexpr double kConditionalModeTolerance = 6.0e-4;  // absolute; measured worst 8.66e-05

// The per-group Laplace log-likelihood. Sums N = 4 masked observation terms,
// each a LogOnePlusExp (~1.8e-4) plus a product, then adds two Logs (~7e-5 each)
// and a penalty built from the conditional mode above. Composition bound:
// 4 * 2e-4 + 2 * 7e-5 + the mode's contribution ~ 2e-3.
constexpr double kGroupLogLikTolerance = 4.0e-3;  // absolute; measured worst 7.05e-04

// The marginal objective sums G = 3 group contributions, so three times the
// above with margin.
constexpr double kMarginalTolerance = 1.0e-2;  // absolute; measured worst 1.78e-03

// ---------------------------------------------------------------------------
// Plaintext mirror
// ---------------------------------------------------------------------------
// Written out independently of the secure implementation: same formulas, no
// shared helper. A shared helper would let one mistake appear on both sides of
// the comparison and cancel, which is the failure mode this whole file exists
// to avoid.

struct PlainParameters {
    std::vector<double> beta;  // K x p row-major
    std::vector<double> sigma2;  // length K
};

PlainParameters UnpackPlain(const std::vector<double>& params) {
    const std::size_t dim = kNumFixed + 1;
    PlainParameters unpacked;
    unpacked.beta.resize(kNumPoints * kNumFixed);
    unpacked.sigma2.resize(kNumPoints);
    for (std::size_t k = 0; k < kNumPoints; ++k) {
        for (std::size_t i = 0; i < kNumFixed; ++i) {
            unpacked.beta[k * kNumFixed + i] = params[k * dim + i];
        }
        unpacked.sigma2[k] = std::exp(2.0 * params[k * dim + kNumFixed]);
    }
    return unpacked;
}

// Beta (K x p) * X^T (p x G*N) -> K x (G*N), row-major, in the (k, g, j) layout.
std::vector<double> PlainLinearPredictors(const std::vector<double>& design,
                                          const std::vector<double>& beta) {
    std::vector<double> eta(kNumPoints * kTotalRows, 0.0);
    for (std::size_t k = 0; k < kNumPoints; ++k) {
        for (std::size_t row = 0; row < kTotalRows; ++row) {
            double total = 0.0;
            for (std::size_t i = 0; i < kNumFixed; ++i) {
                total += beta[k * kNumFixed + i] * design[row * kNumFixed + i];
            }
            eta[k * kTotalRows + row] = total;
        }
    }
    return eta;
}

// The same damped Newton iteration as ConditionalModeBatched: kNewtonIterations
// steps, each clamped to +-kMaxNewtonStep.
std::vector<double> PlainConditionalMode(const std::vector<double>& x_beta,
                                         const std::vector<double>& sigma2,
                                         const std::vector<double>& outcome,
                                         const std::vector<double>& mask) {
    std::vector<double> u(kNumPoints * kGroups, 0.0);
    const double clamp = static_cast<double>(kMaxNewtonStep);

    for (int iteration = 0; iteration < kNewtonIterations; ++iteration) {
        for (std::size_t k = 0; k < kNumPoints; ++k) {
            const double inverse_variance = 1.0 / sigma2[k];
            for (std::size_t g = 0; g < kGroups; ++g) {
                const double mode = u[k * kGroups + g];
                double gradient_sum = 0.0;
                double curvature_sum = 0.0;
                for (std::size_t j = 0; j < kObsPerGroup; ++j) {
                    const std::size_t row = g * kObsPerGroup + j;
                    const double eta = x_beta[k * kTotalRows + row] + mode;
                    const double probability = RefSigmoid(eta);
                    gradient_sum += mask[row] * (outcome[row] - probability);
                    curvature_sum += mask[row] * probability * (1.0 - probability);
                }
                const double gradient = -mode * inverse_variance + gradient_sum;
                const double curvature = inverse_variance + curvature_sum;
                const double step = std::max(-clamp, std::min(clamp, gradient / curvature));
                u[k * kGroups + g] = mode + step;
            }
        }
    }
    return u;
}

std::vector<double> PlainGroupLaplaceLogLik(const std::vector<double>& x_beta,
                                            const std::vector<double>& sigma2,
                                            const std::vector<double>& mode,
                                            const std::vector<double>& outcome,
                                            const std::vector<double>& mask) {
    std::vector<double> result(kNumPoints * kGroups, 0.0);

    for (std::size_t k = 0; k < kNumPoints; ++k) {
        const double inverse_variance = 1.0 / sigma2[k];
        for (std::size_t g = 0; g < kGroups; ++g) {
            const double u = mode[k * kGroups + g];
            double conditional_log_lik = 0.0;
            double curvature_sum = 0.0;
            for (std::size_t j = 0; j < kObsPerGroup; ++j) {
                const std::size_t row = g * kObsPerGroup + j;
                const double eta = x_beta[k * kTotalRows + row] + u;
                const double probability = RefSigmoid(eta);
                conditional_log_lik +=
                    mask[row] * (outcome[row] * eta - RefLogOnePlusExp(eta));
                curvature_sum += mask[row] * probability * (1.0 - probability);
            }
            const double curvature = inverse_variance + curvature_sum;
            const double penalty_half = 0.5 * u * u * inverse_variance;
            const double log_terms_half = 0.5 * (std::log(sigma2[k]) + std::log(curvature));
            result[k * kGroups + g] = conditional_log_lik - penalty_half - log_terms_half;
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Shared fixture construction
// ---------------------------------------------------------------------------

struct Fixture {
    mixedeffects::BatchedDataset data;
    std::vector<double> design;   // rounded
    std::vector<double> outcome;  // rounded
    std::vector<double> mask;     // rounded
    PlainParameters unpacked;
    std::vector<double> x_beta;
    std::vector<double> mode;
};

Fixture MakeFixture(EngineRef engine) {
    Fixture fixture{
        mixedeffects::BatchedDataset(ShareVector(kDesign, engine), ShareVector(kOutcome, engine),
                                     ShareVector(kMask, engine), kGroups, kObsPerGroup,
                                     kNumFixed),
        RoundTrip(kDesign),
        RoundTrip(kOutcome),
        RoundTrip(kMask),
        {},
        {},
        {}};

    fixture.unpacked = UnpackPlain(RoundTrip(kParams));
    fixture.x_beta = PlainLinearPredictors(fixture.design, fixture.unpacked.beta);
    fixture.mode = PlainConditionalMode(fixture.x_beta, fixture.unpacked.sigma2, fixture.outcome,
                                        fixture.mask);
    return fixture;
}

// ---------------------------------------------------------------------------
// BatchedDataset
// ---------------------------------------------------------------------------
void TestDataset(EngineRef engine) {
    Section("BatchedDataset", engine);

    Fixture fixture = MakeFixture(engine);

    assert(fixture.data.num_groups == kGroups);
    assert(fixture.data.obs_per_group == kObsPerGroup);
    assert(fixture.data.num_fixed == kNumFixed);
    assert(fixture.data.total_rows() == kTotalRows);

    // The constructor's own assertions already enforce the buffer sizes; what is
    // worth checking is that it did not reorder or lose anything on the way in.
    CheckClose("BatchedDataset.x_data round trip", OpenVector(fixture.data.x_data),
               fixture.design, kExactish, engine);
    CheckClose("BatchedDataset.y round trip", OpenVector(fixture.data.y), fixture.outcome,
               kExactish, engine);
    CheckClose("BatchedDataset.mask round trip", OpenVector(fixture.data.mask), fixture.mask,
               kExactish, engine);
}

// ---------------------------------------------------------------------------
// ApplyMask
// ---------------------------------------------------------------------------
void TestApplyMask(EngineRef engine) {
    Section("ApplyMask", engine);

    Fixture fixture = MakeFixture(engine);

    const std::vector<double> values = {1.5,  -2.0, 0.25, 3.0, -1.75, 0.5,
                                        2.25, 8.0,  -0.5, 1.0, 0.75,  -3.25};
    const std::vector<double> rounded = RoundTrip(values);

    std::vector<double> expected(kTotalRows);
    for (std::size_t i = 0; i < kTotalRows; ++i) {
        expected[i] = rounded[i] * fixture.mask[i];
    }

    AV shared = ShareVector(values, engine);
    CheckClose("ApplyMask", OpenVector(mixedeffects::ApplyMask(shared, fixture.data.mask)),
               expected, kExactish, engine, ErrorMode::Absolute, values);

    // The padded entry must be exactly zero, not merely small. A mask applied
    // with the wrong scaling would leave a residue here that the tolerance above
    // might swallow.
    const std::vector<double> masked =
        OpenVector(mixedeffects::ApplyMask(ShareVector(values, engine), fixture.data.mask));
    assert(masked[7] == 0.0 && "padded row survived ApplyMask");
    Note("padded row (index 7) is exactly zero after masking", engine);
}

// ---------------------------------------------------------------------------
// UnpackParametersBatched
// ---------------------------------------------------------------------------
void TestUnpackParameters(EngineRef engine) {
    Section("UnpackParametersBatched", engine);

    Fixture fixture = MakeFixture(engine);
    AV params = ShareVector(kParams, engine);

    mixedeffects::UnpackedParameters unpacked =
        mixedeffects::UnpackParametersBatched(params, kNumPoints, kNumFixed);

    // beta is a pure index mapping out of the packed buffer, so it is exact.
    CheckClose("UnpackParametersBatched.beta", OpenVector(unpacked.beta_data),
               fixture.unpacked.beta, kExactish, engine);

    // sigma^2 = Exp(2s) carries one Exp.
    CheckClose("UnpackParametersBatched.sigma2", OpenVector(unpacked.sigma2),
               fixture.unpacked.sigma2, kSigmaSquaredTolerance, engine, ErrorMode::Relative);
}

// ---------------------------------------------------------------------------
// LinearPredictors
// ---------------------------------------------------------------------------
void TestLinearPredictors(EngineRef engine) {
    Section("LinearPredictors", engine);

    Fixture fixture = MakeFixture(engine);
    AV beta = ShareVector(fixture.unpacked.beta, engine);

    AV eta = mixedeffects::LinearPredictors(fixture.data, beta, kNumPoints);
    CheckClose("LinearPredictors", OpenVector(eta), fixture.x_beta, kLinearPredictorTolerance,
               engine);

    // Layout check, independent of the values: point 1's block must start at
    // offset kTotalRows. If the K dimension were laid out interleaved rather
    // than blocked, the value check above could still pass for a symmetric
    // fixture, so the two parameter points are given deliberately different
    // coefficients and this slice is compared on its own.
    const std::vector<double> opened = OpenVector(eta);
    std::vector<double> second_point(opened.begin() + kTotalRows, opened.end());
    std::vector<double> second_expected(fixture.x_beta.begin() + kTotalRows,
                                        fixture.x_beta.end());
    CheckClose("LinearPredictors point-1 block", second_point, second_expected,
               kLinearPredictorTolerance, engine);
}

// ---------------------------------------------------------------------------
// ConditionalModeBatched
// ---------------------------------------------------------------------------
void TestConditionalMode(EngineRef engine) {
    Section("ConditionalModeBatched", engine);

    Fixture fixture = MakeFixture(engine);

    std::vector<double> inverse_variance(kNumPoints);
    for (std::size_t k = 0; k < kNumPoints; ++k) {
        inverse_variance[k] = 1.0 / fixture.unpacked.sigma2[k];
    }

    AV x_beta = ShareVector(fixture.x_beta, engine);
    AV inv_sigma2 = ShareVector(inverse_variance, engine);

    AV mode = mixedeffects::ConditionalModeBatched(fixture.data, x_beta, inv_sigma2, kNumPoints);
    CheckClose("ConditionalModeBatched", OpenVector(mode), fixture.mode,
               kConditionalModeTolerance, engine);
}

// ---------------------------------------------------------------------------
// GroupLaplaceLogLikBatched and NegMarginalLogLikBatched
// ---------------------------------------------------------------------------
void TestLogLikelihood(EngineRef engine) {
    Section("GroupLaplaceLogLikBatched / NegMarginalLogLikBatched", engine);

    Fixture fixture = MakeFixture(engine);

    const std::vector<double> group_expected = PlainGroupLaplaceLogLik(
        fixture.x_beta, fixture.unpacked.sigma2, fixture.mode, fixture.outcome, fixture.mask);

    AV beta = ShareVector(fixture.unpacked.beta, engine);
    AV sigma2 = ShareVector(fixture.unpacked.sigma2, engine);

    AV group_actual =
        mixedeffects::GroupLaplaceLogLikBatched(fixture.data, beta, sigma2, kNumPoints);
    CheckClose("GroupLaplaceLogLikBatched", OpenVector(group_actual), group_expected,
               kGroupLogLikTolerance, engine);

    // The marginal objective: sum the G contributions of each point, negate.
    std::vector<double> marginal_expected(kNumPoints, 0.0);
    for (std::size_t k = 0; k < kNumPoints; ++k) {
        double total = 0.0;
        for (std::size_t g = 0; g < kGroups; ++g) {
            total += group_expected[k * kGroups + g];
        }
        marginal_expected[k] = -total;
    }

    AV params = ShareVector(kParams, engine);
    AV marginal_actual =
        mixedeffects::NegMarginalLogLikBatched(fixture.data, params, kNumPoints);
    CheckClose("NegMarginalLogLikBatched", OpenVector(marginal_actual), marginal_expected,
               kMarginalTolerance, engine);

    // The objective must be usable as a BatchedObjective, which is the only
    // thing the optimizer requires of it. Evaluating a single point through the
    // same entry point checks the K = 1 path, which every real fit's line search
    // does NOT exercise (it always passes K = kLineSearchSteps).
    const std::vector<double> single_point(kParams.begin(),
                                           kParams.begin() + (kNumFixed + 1));
    AV single_shared = ShareVector(single_point, engine);
    AV single_actual = mixedeffects::NegMarginalLogLikBatched(fixture.data, single_shared, 1);
    CheckClose("NegMarginalLogLikBatched (K = 1)", OpenVector(single_actual),
               {marginal_expected[0]}, kMarginalTolerance, engine);
}

}  // namespace

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);

    if (engine.getPartyID() == 0) {
        std::cout << "\n########## regression.h :: mixedeffects ##########\n"
                  << "groups = " << kGroups << "   obs/group = " << kObsPerGroup
                  << "   fixed = " << kNumFixed << "   points = " << kNumPoints
                  << "   Newton iterations = " << kNewtonIterations << std::endl;
    }

    TestDataset(engine);
    TestApplyMask(engine);
    TestUnpackParameters(engine);
    TestLinearPredictors(engine);
    TestConditionalMode(engine);
    TestLogLikelihood(engine);

    if (engine.getPartyID() == 0) {
        std::cout << "\nmixedeffects: all checks passed" << std::endl;
    }

    return 0;
}
