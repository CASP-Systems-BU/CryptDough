// Secure fixed-effects logistic regression with Wald inference.
//
// This is the MPC equivalent of step 6b of `notes/MPC_analysis_Mar26.sas`:
//
//   proc glimmix data=a.any_system;
//      class data_source subject_id Gender hispanic;
//      model suicide_acutecare_icd_narrow(event='1') =
//             newage Gender hispanic data_source fu_month data_source*fu_month
//            / dist=binary link=logit s;
//      lsmeans data_source / diff;
//      estimate 'Diff in slopes' data_source*fu_month  1 -1;
//   run;
//
// That step is the only GLIMMIX in the file with no `random intercept /
// subject=subject_id`, so as written it fits a plain fixed-effects binary logit --
// which `mixedeffects` cannot produce, because its variance goes through Exp(2s)
// and SecureReciprocal and sigma -> 0 is not representable. Hence this program and
// the `logistic` namespace it exercises. See semantic task 0019.
//
// WHAT THIS REPORTS, AND WHY IT IS NOT LSMEANS. `lsmeans data_source / diff` does
// not give a difference in slopes: with the interaction in the model it is the
// difference in marginal means at the average fu_month, an intercept-type
// contrast. The difference in slopes *is* the data_source*fu_month coefficient,
// and with data_source at two levels the `estimate ... 1 -1` statement recovers
// exactly that coefficient. So this program reports coefficients, standard errors,
// Wald statistics, two-sided p-values, and odds ratios with limits, and calls out
// the interaction coefficient as the slope difference. No general LSMEANS
// machinery is needed or built.
//
// Run with:
//   ../scripts/run_experiment.py -p 3 -s same plain-lr
//
// PRIVACY. The fit is oblivious apart from what MinimizeBFGSBatched already
// declassifies (one bit per iteration, the termination flag; see the contract in
// optimizer.h) plus ONE additional bit: the separation flag, opened after
// convergence. Everything else -- the covariance, the standard errors, the Wald
// statistics, the p-values -- is computed on shares and opened only at the end, as
// the results the analysis is asking for. In particular the p-value goes through
// the secure NormalCdf rather than by opening the test statistic.
//
// SCALING IS A PRECONDITION, NOT A PREFERENCE. The covariates below are generated
// on a standardized scale, and real preprocessing must do the same before sharing.
// Two independent reasons: fixed point at `precision` 16 has limited headroom, and
// NewtonSchulzInverse requires an O(1)-scaled operand, which is why the
// information matrix is formed as X'WX/n. A design mixing an age in years with 0/1
// indicators will not invert well.
//
// VALIDATION STATE:
//   Exercised   -- fits against a plaintext IRLS reference computed in this same
//                  program on the identical dataset, and reports the coefficient
//                  and standard-error errors against it.
//   NOT exercised -- the real PointClickCare extract, at its scale or with its
//                  actual class levels and sparsity.

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

#include "./library/regression.h"

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;
using namespace cdough::debug;
using namespace cdough::service;
using namespace cdough::regression;

namespace {

// ---------------------------------------------------------------------------
// Workload configuration
// ---------------------------------------------------------------------------

constexpr std::size_t kNumObs = 2000;
constexpr unsigned int kSeed = 20260917u;
constexpr int kMaxBfgsIterations = 60;

// Ridge penalty. Zero by default: PROC GLIMMIX applies none, so a non-zero value
// would make these estimates disagree with the SAS fit they reproduce. See
// logistic::L2Penalty for what to respect if this is ever switched on.
constexpr double kRidgeLambda = 0.0;

// Design, in the order of the SAS model statement. The intercept is column 0.
const std::vector<std::string> kCoefficientNames = {
    "Intercept", "newage", "Gender", "hispanic", "data_source", "fu_month", "ds*fu_month"};

// Ground truth. The interaction is the quantity step 6 exists to estimate, so it
// is given a clearly non-zero value.
const std::vector<double> kTrueBeta = {-1.20, 0.35, -0.40, 0.25, 0.60, -0.30, 0.45};

// Index of the data_source*fu_month coefficient -- the "difference in slopes".
constexpr std::size_t kInteractionIndex = 6;

constexpr double kEstimateTolerance = 0.25;

// ---------------------------------------------------------------------------
// Plaintext data generation and reference fit
// ---------------------------------------------------------------------------

struct PlainDataset {
    std::vector<std::vector<double>> x;  // [obs][fixed], row 0 is the intercept column
    std::vector<double> y;
    std::size_t num_fixed = 0;
};

double Sigmoid(double eta) {
    if (eta >= 0.0) {
        const double z = std::exp(-eta);
        return 1.0 / (1.0 + z);
    }
    const double z = std::exp(eta);
    return z / (1.0 + z);
}

// Builds the step 6b design on a standardized scale. `fu_month` takes the three
// follow-up windows the SAS script's own SQL filters on (1, 3, 6), centered and
// scaled; the SAS treats it as a continuous linear term, which is an assumption
// about linearity on the logit scale across those three points, not a fact.
PlainDataset GenerateSyntheticData(std::size_t num_obs, const std::vector<double>& true_beta,
                                   unsigned int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> standard_normal(0.0, 1.0);
    std::bernoulli_distribution gender_dist(0.55);
    std::bernoulli_distribution hispanic_dist(0.20);
    std::bernoulli_distribution source_dist(0.50);
    std::uniform_int_distribution<int> window_index(0, 2);
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

    const double windows[] = {1.0, 3.0, 6.0};
    const double window_mean = 10.0 / 3.0;
    const double window_sd = 2.0567;  // sd of {1, 3, 6}

    const std::size_t num_fixed = true_beta.size();
    PlainDataset data;
    data.num_fixed = num_fixed;
    data.x.reserve(num_obs);
    data.y.reserve(num_obs);

    for (std::size_t i = 0; i < num_obs; ++i) {
        const double source = source_dist(rng) ? 1.0 : 0.0;
        const double follow_up = (windows[window_index(rng)] - window_mean) / window_sd;

        std::vector<double> row(num_fixed, 0.0);
        row[0] = 1.0;                                    // Intercept
        row[1] = standard_normal(rng);                   // newage, standardized
        row[2] = gender_dist(rng) ? 1.0 : 0.0;           // Gender
        row[3] = hispanic_dist(rng) ? 1.0 : 0.0;         // hispanic
        row[4] = source;                                 // data_source
        row[5] = follow_up;                              // fu_month, standardized
        row[6] = source * follow_up;                     // data_source*fu_month

        double eta = 0.0;
        for (std::size_t j = 0; j < num_fixed; ++j) {
            eta += row[j] * true_beta[j];
        }
        data.x.push_back(std::move(row));
        data.y.push_back(uniform_dist(rng) < Sigmoid(eta) ? 1.0 : 0.0);
    }
    return data;
}

// In-place Gauss-Jordan inverse of a small dense matrix. Returns false if the
// matrix is singular to working precision.
bool InvertInPlace(std::vector<std::vector<double>>& m) {
    const std::size_t n = m.size();
    std::vector<std::vector<double>> inverse(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        inverse[i][i] = 1.0;
    }

    for (std::size_t column = 0; column < n; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < n; ++row) {
            if (std::abs(m[row][column]) > std::abs(m[pivot][column])) {
                pivot = row;
            }
        }
        if (std::abs(m[pivot][column]) < 1e-12) {
            return false;
        }
        std::swap(m[column], m[pivot]);
        std::swap(inverse[column], inverse[pivot]);

        const double diagonal = m[column][column];
        for (std::size_t j = 0; j < n; ++j) {
            m[column][j] /= diagonal;
            inverse[column][j] /= diagonal;
        }
        for (std::size_t row = 0; row < n; ++row) {
            if (row == column) {
                continue;
            }
            const double factor = m[row][column];
            if (factor == 0.0) {
                continue;
            }
            for (std::size_t j = 0; j < n; ++j) {
                m[row][j] -= factor * m[column][j];
                inverse[row][j] -= factor * inverse[column][j];
            }
        }
    }
    m = inverse;
    return true;
}

struct ReferenceFit {
    std::vector<double> beta;
    std::vector<double> standard_errors;
    bool converged = false;
    int iterations = 0;
};

// Plaintext maximum-likelihood fit by iteratively reweighted least squares, plus
// the observed-information standard errors. This is the ground truth the secure
// fit is scored against: it is the same estimator SAS computes for a model with no
// random effects, so any disagreement is secure-arithmetic error rather than a
// difference of method.
ReferenceFit FitReference(const PlainDataset& data, int max_iterations = 50) {
    const std::size_t n = data.y.size();
    const std::size_t p = data.num_fixed;

    ReferenceFit fit;
    fit.beta.assign(p, 0.0);
    fit.standard_errors.assign(p, 0.0);

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        fit.iterations = iteration + 1;
        std::vector<double> gradient(p, 0.0);
        std::vector<std::vector<double>> information(p, std::vector<double>(p, 0.0));

        for (std::size_t i = 0; i < n; ++i) {
            double eta = 0.0;
            for (std::size_t j = 0; j < p; ++j) {
                eta += data.x[i][j] * fit.beta[j];
            }
            const double probability = Sigmoid(eta);
            const double weight = probability * (1.0 - probability);
            const double residual = data.y[i] - probability;
            for (std::size_t j = 0; j < p; ++j) {
                gradient[j] += data.x[i][j] * residual;
                for (std::size_t k = 0; k < p; ++k) {
                    information[j][k] += data.x[i][j] * weight * data.x[i][k];
                }
            }
        }

        std::vector<std::vector<double>> covariance = information;
        if (!InvertInPlace(covariance)) {
            return fit;
        }

        double max_step = 0.0;
        for (std::size_t j = 0; j < p; ++j) {
            double step = 0.0;
            for (std::size_t k = 0; k < p; ++k) {
                step += covariance[j][k] * gradient[k];
            }
            fit.beta[j] += step;
            max_step = std::max(max_step, std::abs(step));
        }

        if (max_step < 1e-10) {
            for (std::size_t j = 0; j < p; ++j) {
                fit.standard_errors[j] = std::sqrt(covariance[j][j]);
            }
            fit.converged = true;
            return fit;
        }
    }
    return fit;
}

// ---------------------------------------------------------------------------
// Sharing and reporting helpers
// ---------------------------------------------------------------------------

AV ShareVector(const std::vector<double>& values, EngineRef engine) {
    cdough::Vector<DataType> plain(values.size(), precision);
    for (std::size_t i = 0; i < values.size(); ++i) {
        plain[i] = static_cast<DataType>(std::llround(values[i] * scale));
    }
    return engine.secret_share_a(plain, 0, precision);
}

std::vector<double> OpenVector(const AV& value) {
    auto opened = value.open();
    std::vector<double> result(value.size(), 0.0);
    for (std::size_t i = 0; i < value.size(); ++i) {
        result[i] = static_cast<double>(opened[i]) / scale;
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    const auto pID = engine.getPartyID();

    const std::size_t num_fixed = kTrueBeta.size();
    assert(kCoefficientNames.size() == num_fixed);

    // -----------------------------------------------------------------------
    // 1. Generate the dataset and fit the plaintext reference. Every party
    //    derives both from the same fixed seed, so all parties agree; party 0 is
    //    the dealer for the shares below.
    // -----------------------------------------------------------------------
    const PlainDataset data = GenerateSyntheticData(kNumObs, kTrueBeta, kSeed);
    const ReferenceFit reference = FitReference(data);

    if (pID == 0) {
        std::size_t num_events = 0;
        double max_abs_eta = 0.0;
        for (std::size_t i = 0; i < kNumObs; ++i) {
            num_events += static_cast<std::size_t>(data.y[i]);
            double eta = 0.0;
            for (std::size_t j = 0; j < num_fixed; ++j) {
                eta += data.x[i][j] * kTrueBeta[j];
            }
            max_abs_eta = std::max(max_abs_eta, std::abs(eta));
        }
        std::cout << "\n=== Secure fixed-effects logistic regression (SAS step 6b) ===\n"
                  << "  observations      : " << kNumObs << "\n"
                  << "  coefficients      : " << num_fixed << "\n"
                  << "  events (y = 1)    : " << num_events << "  ("
                  << std::fixed << std::setprecision(1)
                  << (100.0 * static_cast<double>(num_events) / static_cast<double>(kNumObs))
                  << "%)\n"
                  << "  max |eta| (truth) : " << std::setprecision(3) << max_abs_eta
                  << "   (secure Exp saturates at " << kMaxExpArg << ")\n"
                  << "  ridge lambda      : " << kRidgeLambda << "\n"
                  << "  plaintext IRLS    : "
                  << (reference.converged ? "converged" : "DID NOT CONVERGE") << " in "
                  << reference.iterations << " iterations" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 2. Secret-share the design, the outcome, and the mask.
    // -----------------------------------------------------------------------
    std::vector<double> flat_x;
    flat_x.reserve(kNumObs * num_fixed);
    for (std::size_t i = 0; i < kNumObs; ++i) {
        flat_x.insert(flat_x.end(), data.x[i].begin(), data.x[i].end());
    }
    // No padding in this program, so the mask is all ones. It is still carried and
    // applied, so the padded path is exercised rather than bypassed.
    const std::vector<double> flat_mask(kNumObs, 1.0);

    logistic::Dataset secure_data(ShareVector(flat_x, engine), ShareVector(data.y, engine),
                                  ShareVector(flat_mask, engine), kNumObs, num_fixed);

    const DataType lambda_scaled = static_cast<DataType>(std::llround(kRidgeLambda * scale));

    // -----------------------------------------------------------------------
    // 3. Fit. The analytic gradient replaces the central-difference one, so the
    //    per-iteration cost no longer grows with the coefficient count.
    // -----------------------------------------------------------------------
    const BatchedObjective objective = [&](const AV& params, std::size_t num_points) {
        return logistic::PenalizedNegLogLikBatched(secure_data, params, num_points, lambda_scaled);
    };
    const BatchedGradient gradient = [&](const AV& params) {
        return logistic::LogisticGradient(secure_data, params, lambda_scaled);
    };

    AV start(num_fixed, engine);  // beta starts at 0
    start.setPrecision(precision);

    BatchedOptResult result =
        MinimizeBFGSBatched(objective, start, kMaxBfgsIterations, gradient);

    // -----------------------------------------------------------------------
    // 4. Inference. All of this is computed on shares; nothing is opened until
    //    the reporting block below.
    // -----------------------------------------------------------------------
    SMatrix covariance = logistic::Covariance(secure_data, result.params);
    AV standard_errors = logistic::StandardErrors(covariance);
    AV wald = logistic::WaldStatistics(result.params, standard_errors);
    AV p_values = TwoSidedPValue(wald);
    logistic::OddsRatioEstimate odds = logistic::OddsRatio(result.params, standard_errors);

    // The one extra declassification beyond the optimizer's own: a single bit
    // saying whether the fit ran into the Exp clamp.
    AV separation = logistic::SeparationFlag(secure_data, result.params);

    // Every party must reach each open(): it is a communication round.
    const std::vector<double> beta_hat = OpenVector(result.params);
    const std::vector<double> se_hat = OpenVector(standard_errors);
    const std::vector<double> z_hat = OpenVector(wald);
    const std::vector<double> p_hat = OpenVector(p_values);
    const std::vector<double> or_hat = OpenVector(odds.ratio);
    const std::vector<double> or_low = OpenVector(odds.lower);
    const std::vector<double> or_high = OpenVector(odds.upper);
    auto opened_separation = separation.open();
    const bool hit_clamp = static_cast<DataType>(opened_separation[0]) != 0;

    // -----------------------------------------------------------------------
    // 5. Report.
    // -----------------------------------------------------------------------
    if (pID == 0) {
        std::cout << "\n[fit] iterations = " << result.iterations
                  << "   converged = " << (result.converged ? "yes" : "no (hit cap)")
                  << std::endl;

        std::cout << "\n--- Coefficients ---\n"
                  << std::left << std::setw(14) << "Term" << std::right << std::setw(10) << "Truth"
                  << std::setw(12) << "Secure" << std::setw(12) << "Plaintext" << std::setw(11)
                  << "AbsErr" << std::setw(11) << "SE" << std::setw(11) << "SE(ref)"
                  << std::setw(10) << "z" << std::setw(11) << "p" << std::endl;

        double max_beta_error = 0.0;
        double max_se_error = 0.0;
        for (std::size_t j = 0; j < num_fixed; ++j) {
            const double beta_error = std::abs(beta_hat[j] - reference.beta[j]);
            const double se_error = std::abs(se_hat[j] - reference.standard_errors[j]);
            max_beta_error = std::max(max_beta_error, beta_error);
            max_se_error = std::max(max_se_error, se_error);

            std::cout << std::left << std::setw(14) << kCoefficientNames[j] << std::right
                      << std::fixed << std::setprecision(4) << std::setw(10) << kTrueBeta[j]
                      << std::setw(12) << beta_hat[j] << std::setw(12) << reference.beta[j]
                      << std::setw(11) << std::scientific << std::setprecision(2) << beta_error
                      << std::fixed << std::setprecision(4) << std::setw(11) << se_hat[j]
                      << std::setw(11) << reference.standard_errors[j] << std::setw(10)
                      << std::setprecision(3) << z_hat[j] << std::setw(11) << std::setprecision(4)
                      << p_hat[j] << std::endl;
        }

        std::cout << "\n--- Odds ratios (95% Wald limits) ---\n"
                  << std::left << std::setw(14) << "Term" << std::right << std::setw(12) << "OR"
                  << std::setw(12) << "Lower" << std::setw(12) << "Upper" << std::setw(14)
                  << "OR(ref)" << std::endl;
        for (std::size_t j = 0; j < num_fixed; ++j) {
            std::cout << std::left << std::setw(14) << kCoefficientNames[j] << std::right
                      << std::fixed << std::setprecision(4) << std::setw(12) << or_hat[j]
                      << std::setw(12) << or_low[j] << std::setw(12) << or_high[j] << std::setw(14)
                      << std::exp(reference.beta[j]) << std::endl;
        }

        // The quantity the SAS `estimate 'Diff in slopes'` statement asks for.
        std::cout << "\n--- Difference in slopes (data_source * fu_month) ---\n"
                  << "  estimate   = " << std::fixed << std::setprecision(4)
                  << beta_hat[kInteractionIndex] << "   (plaintext "
                  << reference.beta[kInteractionIndex] << ", truth "
                  << kTrueBeta[kInteractionIndex] << ")\n"
                  << "  std error  = " << se_hat[kInteractionIndex] << "   (plaintext "
                  << reference.standard_errors[kInteractionIndex] << ")\n"
                  << "  z          = " << std::setprecision(3) << z_hat[kInteractionIndex] << "\n"
                  << "  p (2-sided)= " << std::setprecision(4) << p_hat[kInteractionIndex]
                  << std::endl;

        std::cout << "\n--- Diagnostics ---\n"
                  << "  separation flag           : " << (hit_clamp ? "SET" : "clear")
                  << (hit_clamp ? "  <-- some |eta| reached the Exp clamp; treat the estimates"
                                  " as a diagnostic, not a result"
                                : "")
                  << "\n"
                  << "  max |beta - plaintext|    : " << std::scientific << std::setprecision(3)
                  << max_beta_error << "\n"
                  << "  max |se - plaintext|      : " << max_se_error << std::endl;

        double max_truth_error = 0.0;
        for (std::size_t j = 0; j < num_fixed; ++j) {
            max_truth_error = std::max(max_truth_error, std::abs(beta_hat[j] - kTrueBeta[j]));
        }
        std::cout << "  max |beta - truth|        : " << max_truth_error << "  (tolerance "
                  << std::fixed << std::setprecision(2) << kEstimateTolerance
                  << "; sampling error at this n, not secure-arithmetic error)" << std::endl;
    }

    return 0;
}
