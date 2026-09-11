// Secure mixed-effects logistic regression (Laplace approximation + BFGS
// quasi-Newton) at the scale of the plaintext reference in
// `playground/logistic-regression.cpp`.
//
// This is the MPC equivalent of that file's `main`: it generates the same
// synthetic dataset, secret-shares it, fits the model with the secure operators
// in primitives.h / optimizer.h / regression.h, and scores the estimates
// against ground truth with the same tolerance.
//
// Run with:
//   ../scripts/run_experiment.py -p 3 -s same long-lr
//
// IMPLEMENTATION -- this program now uses the VECTORIZED objective and optimizer
// (semantic task 0014). The whole dataset lives in one buffer and every
// operation is applied to it at once, so circuit depth no longer scales with
// group count or parameter count. The serial implementation it replaced is
// preserved as `cdough::regression::reference` for differential testing and is
// scheduled for deletion in semantic task 0015.
//
// COST -- the numbers below are for the SERIAL implementation and are kept only
// as the "before" side of the comparison. They are NOT valid for this program
// any more.
//
//   serial, dim 13:  ~11-14 s per group per BFGS iteration; 500 observations in
//                    20 groups cost ~6x the same 500 in 5 groups, because groups
//                    were walked sequentially.
//
// Measured after vectorization, on the secure-logistic-regression differential
// harness (8 groups x 16 obs, dim 3, 5 BFGS iterations):
//
//   serial  147.1 s      batched  12.9 s      ~11.4x
//
// That is the *smallest* configuration measured; the serial side scales with
// group count and parameter count while the batched side largely does not, so
// the gap should widen at this program's scale. No measurement at 300 groups
// exists yet, so treat the reference-scale runtime as unknown rather than
// projected -- see semantic personality note 0004 P-01.
//
// VALIDATION STATE (see semantic tasks 0013 and 0014):
//   Exercised   -- compiles clean. The batched objective, gradient, and BFGS are
//                  differentially validated against the serial implementation on
//                  identical secret values in secure-logistic-regression,
//                  including ragged group sizes and a full 5-iteration fit that
//                  tracks the serial trajectory step for step.
//   NOT exercised -- this program at the reference's 300 x 25 configuration. No
//                  run at that scale has completed, so its runtime, convergence,
//                  and accumulated fixed-point error are unmeasured. The hard
//                  tolerance assertions only bind there (see
//                  kEnforceToleranceAsserts), so they have never fired.
//   Watch item   -- an earlier serial run at 5 x 100 estimated sigma at 0.5273
//                  against a truth of 0.7000, the only plainly systematic error
//                  in that fit. Most likely the known downward bias of
//                  variance-component estimation with few clusters, but
//                  unconfirmed: run the plaintext reference at the same seed and
//                  group count to see whether it agrees.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cdough.h"

#include "./regression.h"

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;
using namespace cdough::debug;
using namespace cdough::service;
using namespace cdough::regression;

namespace {

// ---------------------------------------------------------------------------
// Workload configuration
// ---------------------------------------------------------------------------

// Scale. After vectorization `kNumGroups` no longer drives circuit depth, but it
// does drive vector width and memory -- see the COST note above.
constexpr std::size_t kNumGroups = 5;
constexpr std::size_t kObsPerGroup = 100;
constexpr unsigned int kSeed = 20260730u;

// BFGS iteration cap, matching the plaintext reference's default.
// MinimizeBFGSBatched normally exits on its convergence or "no descent found"
// branch well before this (7-30 iterations in every run observed so far), so the
// cap is a safety net rather than the expected count.
constexpr int kMaxBfgsIterations = 300;

// Ground truth: an intercept plus eleven covariates, and a moderate
// random-intercept standard deviation.
const std::vector<double> kTrueBeta = {-0.5, 1.0, -0.75, 0.3, -0.5, 1.0,
                                       -0.75, 0.3, -0.5, 1.0, -0.75, 0.3};
constexpr double kTrueSigma = 0.7;

// The reference asserts every estimate lands within this distance of truth.
constexpr double kEstimateTolerance = 0.25;

// The reference's tolerance assertions are premised on "with this much data".
// They are only meaningful at the full data volume, so they are enforced as
// hard assertions there and reported without aborting at any smaller
// configuration -- a reduced run that misses 0.25 has told you about its group
// count, not about the secure arithmetic. Derived rather than a separate knob so
// that lowering kNumGroups cannot leave it inconsistent.
constexpr std::size_t kFullScaleGroups = 300;
constexpr std::size_t kFullScaleObsPerGroup = 25;
constexpr bool kEnforceToleranceAsserts =
    (kNumGroups >= kFullScaleGroups && kObsPerGroup >= kFullScaleObsPerGroup);

// ---------------------------------------------------------------------------
// Plaintext data generation
// ---------------------------------------------------------------------------

// Row-wise synthetic dataset, matching the plaintext reference's `Dataset`.
// Secret sharing flattens it into the single group-major buffer that
// mixedeffects::BatchedDataset expects.
struct PlainDataset {
    std::vector<std::vector<std::vector<double>>> x;  // [group][obs][fixed]
    std::vector<std::vector<double>> y;               // [group][obs]
    std::vector<double> u;                            // true random effects
    std::size_t num_fixed = 0;
};

// Numerically stable logistic function, as in the plaintext reference.
double Sigmoid(double eta) {
    if (eta >= 0.0) {
        const double z = std::exp(-eta);
        return 1.0 / (1.0 + z);
    }
    const double z = std::exp(eta);
    return z / (1.0 + z);
}

// Mirrors `GenerateSyntheticData` in the plaintext reference, including the
// order in which the three distributions are drawn: one random effect per
// group, then per observation `num_fixed - 1` covariates followed by one
// uniform for the outcome. The draw order is what makes both programs see the
// identical dataset for a given seed, so it must not be rearranged.
PlainDataset GenerateSyntheticData(std::size_t num_groups, std::size_t obs_per_group,
                                   const std::vector<double>& true_beta, double true_sigma,
                                   unsigned int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> covariate_dist(0.0, 1.0);
    std::normal_distribution<double> random_effect_dist(0.0, true_sigma);
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

    const std::size_t num_fixed = true_beta.size();
    PlainDataset data;
    data.num_fixed = num_fixed;
    data.x.assign(num_groups, {});
    data.y.assign(num_groups, {});
    data.u.assign(num_groups, 0.0);

    for (std::size_t i = 0; i < num_groups; ++i) {
        data.x[i].reserve(obs_per_group);
        data.y[i].reserve(obs_per_group);
        const double u = random_effect_dist(rng);
        data.u[i] = u;
        for (std::size_t j = 0; j < obs_per_group; ++j) {
            std::vector<double> row(num_fixed, 0.0);
            row[0] = 1.0;  // Intercept column.
            for (std::size_t k = 1; k < num_fixed; ++k) {
                row[k] = covariate_dist(rng);
            }
            double eta = 0.0;
            for (std::size_t k = 0; k < num_fixed; ++k) {
                eta += row[k] * true_beta[k];
            }
            eta += u;
            const double p = Sigmoid(eta);
            data.x[i].push_back(std::move(row));
            data.y[i].push_back(uniform_dist(rng) < p ? 1.0 : 0.0);
        }
    }
    return data;
}

// ---------------------------------------------------------------------------
// Reporting helpers
// ---------------------------------------------------------------------------

// Prints a labeled fixed-effect estimate alongside its true value, in the
// plaintext reference's format.
void PrintEstimate(const std::string& label, double estimate, double truth) {
    std::cout << "  " << std::left << std::setw(12) << label << std::right << std::setw(12)
              << std::fixed << std::setprecision(4) << estimate << std::setw(12) << truth << '\n';
}

// Secret-shares one scalar as a 1-element AV at `precision`.
AV ShareScalar(double value, EngineRef engine) {
    cdough::Vector<DataType> plain(1, precision);
    plain[0] = static_cast<DataType>(std::llround(value * scale));
    return engine.secret_share_a(plain, 0, precision);
}

// Opens a 1-element AV back to a double. Every party must call this: open() is
// a communication round.
double OpenScalar(const AV& value) {
    auto opened = value.open();
    return static_cast<double>(opened[0]) / scale;
}

}  // namespace

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    const auto pID = engine.getPartyID();

    const std::size_t num_fixed = kTrueBeta.size();
    const std::size_t num_obs_total = kNumGroups * kObsPerGroup;

    // -----------------------------------------------------------------------
    // 1. Generate the dataset. Every party derives it from the same fixed seed,
    //    so all parties agree; party 0 is the dealer for the shares below.
    // -----------------------------------------------------------------------
    const PlainDataset data =
        GenerateSyntheticData(kNumGroups, kObsPerGroup, kTrueBeta, kTrueSigma, kSeed);

    // Diagnostics. `max |eta|` matters because the secure Exp saturates its
    // argument at kMaxExpArg: beyond that the logistic is already 0.99995, so
    // clamping barely moves the likelihood, but it should be visible rather
    // than silent.
    if (pID == 0) {
        std::size_t num_ones = 0;
        double max_abs_eta = 0.0;
        for (std::size_t g = 0; g < kNumGroups; ++g) {
            for (std::size_t j = 0; j < kObsPerGroup; ++j) {
                num_ones += static_cast<std::size_t>(data.y[g][j]);
                double eta = data.u[g];
                for (std::size_t k = 0; k < num_fixed; ++k) {
                    eta += data.x[g][j][k] * kTrueBeta[k];
                }
                max_abs_eta = std::max(max_abs_eta, std::abs(eta));
            }
        }
        std::cout << "Secure mixed-effects logistic regression (Laplace + BFGS quasi-Newton)\n";
        std::cout << "Groups: " << kNumGroups << ", observations/group: " << kObsPerGroup
                  << ", total: " << num_obs_total << "\n";
        std::cout << "Fixed effects: " << num_fixed << ", parameters: " << num_fixed + 1
                  << " (beta + s),  seed: " << kSeed << "\n";
        std::cout << "  class balance: " << num_ones << "/" << num_obs_total << " ones\n";
        std::cout << "  max |eta| = " << std::fixed << std::setprecision(4) << max_abs_eta
                  << " (secure Exp clamp is " << kMaxExpArg << ")"
                  << (max_abs_eta > kMaxExpArg ? "  <- SATURATING" : "") << "\n"
                  << std::endl;
    }

    // -----------------------------------------------------------------------
    // 2. Secret-share it. mixedeffects::ClusterGroup stores the design matrix
    //    column-wise, one AV of length kObsPerGroup per fixed effect, so each
    //    group's rows are transposed on the way in.
    // -----------------------------------------------------------------------
    // One flat buffer for the whole design matrix, rows ordered group-major
    // (row = g * kObsPerGroup + j), plus y and the padding mask. Groups here are
    // uniform, so the mask is all ones -- it is carried anyway so that the one
    // code path serves ragged data too.
    cdough::Vector<DataType> x_flat(num_obs_total * num_fixed, precision);
    cdough::Vector<DataType> y_flat(num_obs_total, precision);
    cdough::Vector<DataType> mask_flat(num_obs_total, precision);
    for (std::size_t g = 0; g < kNumGroups; ++g) {
        for (std::size_t j = 0; j < kObsPerGroup; ++j) {
            const std::size_t row = g * kObsPerGroup + j;
            y_flat[row] = static_cast<DataType>(std::llround(data.y[g][j] * scale));
            mask_flat[row] = static_cast<DataType>(scale);
            for (std::size_t k = 0; k < num_fixed; ++k) {
                x_flat[row * num_fixed + k] =
                    static_cast<DataType>(std::llround(data.x[g][j][k] * scale));
            }
        }
    }
    mixedeffects::BatchedDataset secure_dataset(engine.secret_share_a(x_flat, 0, precision),
                                                engine.secret_share_a(y_flat, 0, precision),
                                                engine.secret_share_a(mask_flat, 0, precision),
                                                kNumGroups, kObsPerGroup, num_fixed);

    if (pID == 0) {
        std::cout << "Secret-shared " << kNumGroups << " groups (" << num_fixed
                  << " design columns each)." << std::endl;
    }

    // -----------------------------------------------------------------------
    // 3. Fit. Objective: the negative Laplace marginal log-likelihood as a
    //    function of the packed parameter vector [beta_0..beta_{p-1}, s] with
    //    sigma = exp(s). Start the fixed effects at 0 and sigma at 1 (s = 0).
    // -----------------------------------------------------------------------
    const BatchedObjective objective = [&secure_dataset](const AV& params,
                                                         std::size_t num_points) -> AV {
        return mixedeffects::NegMarginalLogLikBatched(secure_dataset, params, num_points);
    };

    // The whole parameter vector is one AV of length num_fixed + 1.
    cdough::Vector<DataType> initial_flat(num_fixed + 1, precision);
    AV initial_params = engine.secret_share_a(initial_flat, 0, precision);

    if (pID == 0) {
        std::cout << "Running MinimizeBFGS (cap " << kMaxBfgsIterations << " iterations)..."
                  << std::endl;
    }

    const BatchedOptResult fit =
        MinimizeBFGSBatched(objective, initial_params, kMaxBfgsIterations);

    // -----------------------------------------------------------------------
    // 4. Recover the estimates. sigma = exp(s), so sigma_hat is Exp(s_hat);
    //    there is no secure square root, and none is needed. The arithmetic
    //    stays secure and only the reporting boundary opens values.
    // -----------------------------------------------------------------------
    AV s_hat = fit.params.slice(num_fixed, num_fixed + 1);
    s_hat.setPrecision(precision);
    AV sigma_hat_shared = Exp(s_hat);
    sigma_hat_shared.setPrecision(precision);

    // One open for the whole parameter vector, where the serial version needed
    // one per coefficient.
    auto opened_params = fit.params.open();
    std::vector<double> beta_hat;
    beta_hat.reserve(num_fixed);
    for (std::size_t k = 0; k < num_fixed; ++k) {
        beta_hat.push_back(static_cast<double>(opened_params[k]) / scale);
    }
    const double sigma_hat = OpenScalar(sigma_hat_shared);
    const double final_objective = OpenScalar(fit.value);
    const double final_log_lik = -final_objective;

    // -----------------------------------------------------------------------
    // 5. Report and check.
    // -----------------------------------------------------------------------
    bool all_within_tolerance = true;
    for (std::size_t k = 0; k < num_fixed; ++k) {
        if (std::abs(beta_hat[k] - kTrueBeta[k]) >= kEstimateTolerance) {
            all_within_tolerance = false;
        }
    }
    if (std::abs(sigma_hat - kTrueSigma) >= kEstimateTolerance) {
        all_within_tolerance = false;
    }

    if (pID == 0) {
        std::cout << "\nConverged: " << (fit.converged ? "yes" : "no") << " in " << fit.iterations
                  << " iterations\n";
        std::cout << "Final marginal log-likelihood: " << std::fixed << std::setprecision(4)
                  << final_log_lik << "\n\n";

        std::cout << "  " << std::left << std::setw(12) << "Parameter" << std::right
                  << std::setw(12) << "Estimate" << std::setw(12) << "Truth" << '\n';
        PrintEstimate("Intercept", beta_hat[0], kTrueBeta[0]);
        for (std::size_t k = 1; k < num_fixed; ++k) {
            PrintEstimate("beta_" + std::to_string(k), beta_hat[k], kTrueBeta[k]);
        }
        PrintEstimate("sigma", sigma_hat, kTrueSigma);

        // Per-parameter deviations, so a miss says which parameter and by how
        // much instead of only tripping an assert.
        double worst_deviation = std::abs(sigma_hat - kTrueSigma);
        std::size_t worst_index = num_fixed;
        for (std::size_t k = 0; k < num_fixed; ++k) {
            const double deviation = std::abs(beta_hat[k] - kTrueBeta[k]);
            if (deviation > worst_deviation) {
                worst_deviation = deviation;
                worst_index = k;
            }
        }
        std::cout << "\nWorst deviation: "
                  << (worst_index == num_fixed ? "sigma" : "beta_" + std::to_string(worst_index))
                  << " off by " << std::fixed << std::setprecision(4) << worst_deviation
                  << " (tolerance " << kEstimateTolerance << ")\n";
        std::cout << "All estimates within tolerance: " << (all_within_tolerance ? "yes" : "no")
                  << (kEnforceToleranceAsserts ? "" : "   (reported only: reduced scale)")
                  << std::endl;
    }

    // BFGS must converge at any scale -- this is a property of the optimizer,
    // not of the data volume.
    assert(fit.converged && "BFGS optimization failed to converge");

    // Same sanity checks as the plaintext reference: with this much data the
    // Laplace estimates should land close to the ground truth. Enforced only at
    // the full data volume; see kEnforceToleranceAsserts.
    if constexpr (kEnforceToleranceAsserts) {
        for (std::size_t k = 0; k < num_fixed; ++k) {
            assert(std::abs(beta_hat[k] - kTrueBeta[k]) < kEstimateTolerance);
        }
        assert(std::abs(sigma_hat - kTrueSigma) < kEstimateTolerance);
    }

    return 0;
}
