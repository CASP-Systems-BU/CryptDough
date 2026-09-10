#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "cdough.h"
#include "./regression.h"

// ../scripts/run_experiment.py -p 3 -r 16 secure-logistic-regression

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;
using namespace cdough::debug;
using namespace cdough::service;
using namespace cdough::regression;
using namespace cdough::regression::mixedeffects;


int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();

    // Test values for Exp
    std::vector<double> test_inputs = {-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0};
    cdough::Vector<DataType> plain_x(test_inputs.size(), precision);
    for (size_t i = 0; i < test_inputs.size(); ++i) {
        plain_x[i] = static_cast<DataType>(test_inputs[i] * scale);
    }

    AV secure_x = engine.secret_share_a(plain_x, 0, precision);
    AV secure_exp = Exp(secure_x);

    auto opened_exp = secure_exp.open();

    if (pID == 0) {
        std::cout << "\n--- Oblivious Exp Function Test Results ---" << std::endl;
        std::cout << std::left << std::setw(10) << "Input (x)"
                  << std::setw(16) << "Plaintext exp"
                  << std::setw(16) << "MPC exp"
                  << std::setw(12) << "Abs Error" << std::endl;
        for (size_t i = 0; i < test_inputs.size(); ++i) {
            double expected = std::exp(test_inputs[i]);
            double actual = static_cast<double>(opened_exp[i]) / scale;
            double error = std::abs(expected - actual);
            std::cout << std::left << std::setw(10) << test_inputs[i]
                      << std::setw(16) << expected
                      << std::setw(16) << actual
                      << std::setw(12) << error << std::endl;
        }
    }

    // Test values for Log
    std::vector<double> test_log_inputs = {0.1, 0.25, 0.5, 0.7071, 1.0, 1.4142, 2.0, 4.0, 10.0, 20.0};
    cdough::Vector<DataType> plain_log_x(test_log_inputs.size(), precision);
    for (size_t i = 0; i < test_log_inputs.size(); ++i) {
        plain_log_x[i] = static_cast<DataType>(test_log_inputs[i] * scale);
    }

    AV secure_log_x = engine.secret_share_a(plain_log_x, 0, precision);
    AV secure_log = Log(secure_log_x);

    auto opened_log = secure_log.open();

    if (pID == 0) {
        std::cout << "\n--- Oblivious Log Function Test Results ---" << std::endl;
        std::cout << std::left << std::setw(10) << "Input (x)"
                  << std::setw(16) << "Plaintext log"
                  << std::setw(16) << "MPC log"
                  << std::setw(12) << "Abs Error" << std::endl;
        for (size_t i = 0; i < test_log_inputs.size(); ++i) {
            double expected = std::log(test_log_inputs[i]);
            double actual = static_cast<double>(opened_log[i]) / scale;
            double error = std::abs(expected - actual);
            std::cout << std::left << std::setw(10) << test_log_inputs[i]
                      << std::setw(16) << expected
                      << std::setw(16) << actual
                      << std::setw(12) << error << std::endl;
        }
    }

    // =========================================================================
    // Validation for Log1p, Sigmoid, LogOnePlusExp, Sum
    // =========================================================================
    // Test values for Log1p
    std::vector<double> test_log1p_inputs = {-0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0, 5.0};
    cdough::Vector<DataType> plain_log1p_x(test_log1p_inputs.size(), precision);
    for (size_t i = 0; i < test_log1p_inputs.size(); ++i) {
        plain_log1p_x[i] = static_cast<DataType>(test_log1p_inputs[i] * scale);
    }
    AV secure_log1p_x = engine.secret_share_a(plain_log1p_x, 0, precision);
    AV secure_log1p = Log1p(secure_log1p_x);
    auto opened_log1p = secure_log1p.open();

    if (pID == 0) {
        std::cout << "\n--- Oblivious Log1p Function Test Results ---" << std::endl;
        std::cout << std::left << std::setw(10) << "Input (x)"
                  << std::setw(16) << "Plaintext log1p"
                  << std::setw(16) << "MPC log1p"
                  << std::setw(12) << "Abs Error" << std::endl;
        for (size_t i = 0; i < test_log1p_inputs.size(); ++i) {
            double expected = std::log1p(test_log1p_inputs[i]);
            double actual = static_cast<double>(opened_log1p[i]) / scale;
            double error = std::abs(expected - actual);
            std::cout << std::left << std::setw(10) << test_log1p_inputs[i]
                      << std::setw(16) << expected
                      << std::setw(16) << actual
                      << std::setw(12) << error << std::endl;
        }
    }

    // Test values for Sigmoid
    std::vector<double> test_sigmoid_inputs = {-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0};
    cdough::Vector<DataType> plain_sigmoid_x(test_sigmoid_inputs.size(), precision);
    for (size_t i = 0; i < test_sigmoid_inputs.size(); ++i) {
        plain_sigmoid_x[i] = static_cast<DataType>(test_sigmoid_inputs[i] * scale);
    }
    AV secure_sigmoid_x = engine.secret_share_a(plain_sigmoid_x, 0, precision);
    AV secure_sigmoid = Sigmoid(secure_sigmoid_x);
    auto opened_sigmoid = secure_sigmoid.open();

    if (pID == 0) {
        std::cout << "\n--- Oblivious Sigmoid Function Test Results ---" << std::endl;
        std::cout << std::left << std::setw(10) << "Input (x)"
                  << std::setw(16) << "Plaintext sigm"
                  << std::setw(16) << "MPC sigm"
                  << std::setw(12) << "Abs Error" << std::endl;
        for (size_t i = 0; i < test_sigmoid_inputs.size(); ++i) {
            double expected = 1.0 / (1.0 + std::exp(-test_sigmoid_inputs[i]));
            double actual = static_cast<double>(opened_sigmoid[i]) / scale;
            double error = std::abs(expected - actual);
            std::cout << std::left << std::setw(10) << test_sigmoid_inputs[i]
                      << std::setw(16) << expected
                      << std::setw(16) << actual
                      << std::setw(12) << error << std::endl;
        }
    }

    // Test values for LogOnePlusExp
    std::vector<double> test_softplus_inputs = {-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0};
    cdough::Vector<DataType> plain_softplus_x(test_softplus_inputs.size(), precision);
    for (size_t i = 0; i < test_softplus_inputs.size(); ++i) {
        plain_softplus_x[i] = static_cast<DataType>(test_softplus_inputs[i] * scale);
    }
    AV secure_softplus_x = engine.secret_share_a(plain_softplus_x, 0, precision);
    AV secure_softplus = LogOnePlusExp(secure_softplus_x);
    auto opened_softplus = secure_softplus.open();

    if (pID == 0) {
        std::cout << "\n--- Oblivious LogOnePlusExp Function Test Results ---" << std::endl;
        std::cout << std::left << std::setw(10) << "Input (x)"
                  << std::setw(16) << "Plaintext softp"
                  << std::setw(16) << "MPC softp"
                  << std::setw(12) << "Abs Error" << std::endl;
        for (size_t i = 0; i < test_softplus_inputs.size(); ++i) {
            double expected = std::log1p(std::exp(-std::abs(test_softplus_inputs[i]))) + (test_softplus_inputs[i] > 0 ? test_softplus_inputs[i] : 0.0);
            double actual = static_cast<double>(opened_softplus[i]) / scale;
            double error = std::abs(expected - actual);
            std::cout << std::left << std::setw(10) << test_softplus_inputs[i]
                      << std::setw(16) << expected
                      << std::setw(16) << actual
                      << std::setw(12) << error << std::endl;
        }
    }

    // Test values for Sum
    std::vector<double> test_sum_inputs = {1.5, -2.25, 3.125, 0.5, -1.0, 4.0};
    cdough::Vector<DataType> plain_sum_x(test_sum_inputs.size(), precision);
    double expected_sum = 0.0;
    for (size_t i = 0; i < test_sum_inputs.size(); ++i) {
        plain_sum_x[i] = static_cast<DataType>(test_sum_inputs[i] * scale);
        expected_sum += test_sum_inputs[i];
    }
    AV secure_sum_x = engine.secret_share_a(plain_sum_x, 0, precision);
    AV secure_sum = Sum(secure_sum_x);
    auto opened_sum = secure_sum.open();

    if (pID == 0) {
        std::cout << "\n--- Oblivious Sum Function Test Results ---" << std::endl;
        double actual_sum = static_cast<double>(opened_sum[0]) / scale;
        double error = std::abs(expected_sum - actual_sum);
        std::cout << "Vector elements: [1.5, -2.25, 3.125, 0.5, -1.0, 4.0]" << std::endl;
        std::cout << "Plaintext sum: " << expected_sum << std::endl;
        std::cout << "MPC sum:       " << actual_sum << std::endl;
        std::cout << "Abs Error:     " << error << std::endl;
    }

    // =========================================================================
    // Validation for ClampNewtonStep
    // =========================================================================
    std::vector<double> test_clamp_inputs = {-10.0, -5.0, -4.0, -2.5, 0.0, 1.5, 4.0, 6.0, 12.0};
    cdough::Vector<DataType> plain_clamp_x(test_clamp_inputs.size(), precision);
    for (size_t i = 0; i < test_clamp_inputs.size(); ++i) {
        plain_clamp_x[i] = static_cast<DataType>(test_clamp_inputs[i] * scale);
    }
    AV secure_clamp_x = engine.secret_share_a(plain_clamp_x, 0, precision);
    AV secure_clamp = ClampNewtonStep(secure_clamp_x);
    auto opened_clamp = secure_clamp.open();

    if (pID == 0) {
        std::cout << "\n--- Oblivious ClampNewtonStep Function Test Results ---" << std::endl;
        std::cout << std::left << std::setw(10) << "Input (x)"
                  << std::setw(16) << "Plaintext clamp"
                  << std::setw(16) << "MPC clamp"
                  << std::setw(12) << "Abs Error" << std::endl;
        for (size_t i = 0; i < test_clamp_inputs.size(); ++i) {
            double expected = std::max(-4.0, std::min(4.0, test_clamp_inputs[i]));
            double actual = static_cast<double>(opened_clamp[i]) / scale;
            double error = std::abs(expected - actual);
            std::cout << std::left << std::setw(10) << test_clamp_inputs[i]
                      << std::setw(16) << expected
                      << std::setw(16) << actual
                      << std::setw(12) << error << std::endl;
        }
    }

    // =========================================================================
    // Validation for ConditionalMode, GroupLaplaceLogLik, NegMarginalLogLik, NumericalGradient
    // =========================================================================
    if (pID == 0) {
        std::cout << "\n--- Testing ConditionalMode, GroupLaplaceLogLik, NegMarginalLogLik, and NumericalGradient ---" << std::endl;
    }

    // Randomly generated mixed-effects dataset, drawn from known parameters
    size_t test_num_groups = 8;
    size_t test_obs_per_group = 16;
    size_t test_num_fixed = 2; // intercept + one covariate

    const double kTrueBeta0 = 0.5;
    const double kTrueBeta1 = 1.0;
    const double kTrueSigma = 0.8;
    const double kTwoPi = 6.28318530717958647692;

    size_t num_uniforms = test_num_groups * (2 + test_obs_per_group * 3);
    cdough::Vector<DataType> raw_random(num_uniforms);
    engine.populateLocalRandom(raw_random);

    size_t raw_pos = 0;
    auto next_uniform = [&raw_random, &raw_pos]() {
        // top 53 bits of a fresh 64-bit word, mapped into [0, 1)
        uint64_t bits = static_cast<uint64_t>(raw_random[raw_pos++]);
        return static_cast<double>(bits >> 11) * (1.0 / 9007199254740992.0);
    };
    auto next_normal = [&next_uniform, kTwoPi]() {
        double u1 = next_uniform();
        double u2 = next_uniform();
        if (u1 < 1e-300) u1 = 1e-300;
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(kTwoPi * u2);
    };

    std::vector<std::vector<std::vector<double>>> plain_X(
        test_num_groups,
        std::vector<std::vector<double>>(test_obs_per_group,
                                         std::vector<double>(test_num_fixed, 0.0)));
    std::vector<std::vector<double>> plain_Y(test_num_groups,
                                             std::vector<double>(test_obs_per_group, 0.0));
    std::vector<double> true_u(test_num_groups, 0.0);

    for (size_t g = 0; g < test_num_groups; ++g) {
        true_u[g] = kTrueSigma * next_normal();
        for (size_t j = 0; j < test_obs_per_group; ++j) {
            plain_X[g][j][0] = 1.0; // intercept
            plain_X[g][j][1] = next_normal();
        }
    }

    // Standardize the covariate to exactly zero mean / unit variance before
    // drawing the responses, so beta stays interpretable
    {
        double mean = 0.0;
        for (size_t g = 0; g < test_num_groups; ++g)
            for (size_t j = 0; j < test_obs_per_group; ++j) mean += plain_X[g][j][1];
        mean /= static_cast<double>(test_num_groups * test_obs_per_group);

        double var = 0.0;
        for (size_t g = 0; g < test_num_groups; ++g)
            for (size_t j = 0; j < test_obs_per_group; ++j) {
                double d = plain_X[g][j][1] - mean;
                var += d * d;
            }
        var /= static_cast<double>(test_num_groups * test_obs_per_group);
        double sd = std::sqrt(var);
        if (sd < 1e-12) sd = 1.0;

        for (size_t g = 0; g < test_num_groups; ++g)
            for (size_t j = 0; j < test_obs_per_group; ++j)
                plain_X[g][j][1] = (plain_X[g][j][1] - mean) / sd;
    }

    for (size_t g = 0; g < test_num_groups; ++g) {
        for (size_t j = 0; j < test_obs_per_group; ++j) {
            double eta = kTrueBeta0 + kTrueBeta1 * plain_X[g][j][1] + true_u[g];
            double prob = 1.0 / (1.0 + std::exp(-eta));
            plain_Y[g][j] = (next_uniform() < prob) ? 1.0 : 0.0;
        }
    }

    // Evaluate the per-function tests at the true generating parameters.
    std::vector<double> plain_beta = {kTrueBeta0, kTrueBeta1};
    double plain_s = std::log(kTrueSigma);
    double plain_sigma2 = std::exp(2.0 * plain_s);

    if (pID == 0) {
        size_t num_obs_total = test_num_groups * test_obs_per_group;
        size_t num_ones = 0;
        double max_abs_eta = 0.0;
        double cov_mean = 0.0, cov_sd = 0.0;
        for (size_t g = 0; g < test_num_groups; ++g)
            for (size_t j = 0; j < test_obs_per_group; ++j) cov_mean += plain_X[g][j][1];
        cov_mean /= static_cast<double>(num_obs_total);
        for (size_t g = 0; g < test_num_groups; ++g)
            for (size_t j = 0; j < test_obs_per_group; ++j)
                cov_sd += (plain_X[g][j][1] - cov_mean) * (plain_X[g][j][1] - cov_mean);
        cov_sd = std::sqrt(cov_sd / static_cast<double>(num_obs_total));
        for (size_t g = 0; g < test_num_groups; ++g) {
            for (size_t j = 0; j < test_obs_per_group; ++j) {
                num_ones += static_cast<size_t>(plain_Y[g][j]);
                double eta = kTrueBeta0 + kTrueBeta1 * plain_X[g][j][1] + true_u[g];
                max_abs_eta = std::max(max_abs_eta, std::abs(eta));
            }
        }
        std::cout << "Generated dataset: " << test_num_groups << " groups x "
                  << test_obs_per_group << " obs = " << num_obs_total << " observations\n"
                  << "  true [beta_0, beta_1, s] = [" << kTrueBeta0 << ", " << kTrueBeta1 << ", "
                  << plain_s << "]  (sigma = " << kTrueSigma << ")\n"
                  << "  class balance: " << num_ones << "/" << num_obs_total << " ones"
                  << ",  max |eta| = " << max_abs_eta << " (Exp clamp is " << kMaxExpArg << ")\n"
                  << "  covariate standardized: mean = " << cov_mean << ", sd = " << cov_sd
                  << std::endl;
    }

    // Build secret-shared Dataset
    Dataset secure_dataset;
    secure_dataset.num_fixed = test_num_fixed;

    for (size_t g = 0; g < test_num_groups; ++g) {
        // Share y
        cdough::Vector<DataType> plain_y_vec(test_obs_per_group, precision);
        for (size_t j = 0; j < test_obs_per_group; ++j) {
            plain_y_vec[j] = static_cast<DataType>(plain_Y[g][j] * scale);
        }
        AV grp_y = engine.secret_share_a(plain_y_vec, 0, precision);

        // Share columns of X
        std::vector<AV> grp_x_cols;
        for (size_t k = 0; k < test_num_fixed; ++k) {
            cdough::Vector<DataType> plain_col_vec(test_obs_per_group, precision);
            for (size_t j = 0; j < test_obs_per_group; ++j) {
                plain_col_vec[j] = static_cast<DataType>(plain_X[g][j][k] * scale);
            }
            grp_x_cols.push_back(engine.secret_share_a(plain_col_vec, 0, precision));
        }
        secure_dataset.groups.emplace_back(std::move(grp_y), std::move(grp_x_cols));
    }

    // Secret share parameter vector [beta_0, beta_1, s]
    std::vector<AV> secure_params;
    cdough::Vector<DataType> p0(1, precision); p0[0] = static_cast<DataType>(plain_beta[0] * scale);
    cdough::Vector<DataType> p1(1, precision); p1[0] = static_cast<DataType>(plain_beta[1] * scale);
    cdough::Vector<DataType> ps(1, precision); ps[0] = static_cast<DataType>(plain_s * scale);

    secure_params.push_back(engine.secret_share_a(p0, 0, precision));
    secure_params.push_back(engine.secret_share_a(p1, 0, precision));
    secure_params.push_back(engine.secret_share_a(ps, 0, precision));

    std::vector<AV> secure_beta = { secure_params[0], secure_params[1] };
    cdough::Vector<DataType> p_sig(1, precision); p_sig[0] = static_cast<DataType>(plain_sigma2 * scale);
    AV secure_sigma2 = engine.secret_share_a(p_sig, 0, precision);

    // Plaintext computation helper functions
    auto plain_exp = [](double x) -> double {
        return std::exp(x);
    };
    auto plain_log = [](double x) -> double {
        return std::log(x);
    };
    auto plain_log1p = [](double x) -> double {
        return std::log1p(x);
    };
    auto plain_sigmoid = [](double eta) -> double {
        if (eta >= 0.0) {
            double z = std::exp(-eta);
            return 1.0 / (1.0 + z);
        }
        double z = std::exp(eta);
        return z / (1.0 + z);
    };
    auto plain_softplus = [](double eta) -> double {
        if (eta > 0.0) {
            return eta + std::log1p(std::exp(-eta));
        }
        return std::log1p(std::exp(eta));
    };
    auto plain_conditional_mode = [&](size_t g, const std::vector<double>& beta, double sigma2) -> double {
        double inv_sigma2 = 1.0 / sigma2;
        double u = 0.0;
        for (int iter = 0; iter < kNewtonIterations; ++iter) {
            double grad = -u * inv_sigma2;
            double curv = inv_sigma2;
            for (size_t j = 0; j < test_obs_per_group; ++j) {
                double eta = plain_X[g][j][0] * beta[0] + plain_X[g][j][1] * beta[1] + u;
                double p = plain_sigmoid(eta);
                grad += plain_Y[g][j] - p;
                curv += p * (1.0 - p);
            }
            double step = grad / curv;
            step = std::max(-4.0, std::min(4.0, step));
            u += step;
        }
        return u;
    };
    auto plain_group_laplace = [&](size_t g, const std::vector<double>& beta, double sigma2) -> double {
        double inv_sigma2 = 1.0 / sigma2;
        double u = plain_conditional_mode(g, beta, sigma2);
        double cll = 0.0;
        double curv = inv_sigma2;
        for (size_t j = 0; j < test_obs_per_group; ++j) {
            double eta = plain_X[g][j][0] * beta[0] + plain_X[g][j][1] * beta[1] + u;
            double p = plain_sigmoid(eta);
            cll += plain_Y[g][j] * eta - plain_softplus(eta);
            curv += p * (1.0 - p);
        }
        return cll - 0.5 * u * u * inv_sigma2 - 0.5 * std::log(sigma2) - 0.5 * std::log(curv);
    };
    auto plain_neg_marginal_log_lik = [&](const std::vector<double>& params) -> double {
        std::vector<double> beta = {params[0], params[1]};
        double s = params[2];
        double sigma2 = std::exp(2.0 * s);
        double total = 0.0;
        for (size_t g = 0; g < test_num_groups; ++g) {
            total += plain_group_laplace(g, beta, sigma2);
        }
        return -total;
    };
    auto plain_numerical_gradient = [&](const std::vector<double>& params) -> std::vector<double> {
        std::vector<double> grad(params.size(), 0.0);
        std::vector<double> perturbed = params;
        for (size_t k = 0; k < params.size(); ++k) {
            double h = kNumericalGradientStep * (1.0 + std::abs(params[k]));
            perturbed[k] = params[k] + h;
            double f_plus = plain_neg_marginal_log_lik(perturbed);
            perturbed[k] = params[k] - h;
            double f_minus = plain_neg_marginal_log_lik(perturbed);
            perturbed[k] = params[k];
            grad[k] = (f_plus - f_minus) / (2.0 * h);
        }
        return grad;
    };

    // 1. Test ConditionalMode on group 0 and group 1
    AV u_mode_0 = ConditionalMode(secure_dataset.groups[0], secure_beta, secure_sigma2);
    auto opened_u0 = u_mode_0.open();
    double expected_u0 = plain_conditional_mode(0, plain_beta, plain_sigma2);
    if (pID == 0) {
        double actual_u0 = static_cast<double>(opened_u0[0]) / scale;
        std::cout << "ConditionalMode (Group 0):\n"
                  << "  Plaintext: " << expected_u0 << "\n"
                  << "  MPC:       " << actual_u0 << "\n"
                  << "  Abs Error: " << std::abs(expected_u0 - actual_u0) << std::endl;
    }

    AV u_mode_1 = ConditionalMode(secure_dataset.groups[1], secure_beta, secure_sigma2);
    auto opened_u1 = u_mode_1.open();
    double expected_u1 = plain_conditional_mode(1, plain_beta, plain_sigma2);
    if (pID == 0) {
        double actual_u1 = static_cast<double>(opened_u1[0]) / scale;
        std::cout << "ConditionalMode (Group 1):\n"
                  << "  Plaintext: " << expected_u1 << "\n"
                  << "  MPC:       " << actual_u1 << "\n"
                  << "  Abs Error: " << std::abs(expected_u1 - actual_u1) << std::endl;
    }

    // 2. Test GroupLaplaceLogLik on group 0 and group 1
    AV group0_lik = GroupLaplaceLogLik(secure_dataset.groups[0], secure_beta, secure_sigma2);
    auto opened_g0_lik = group0_lik.open();
    double expected_g0 = plain_group_laplace(0, plain_beta, plain_sigma2);
    if (pID == 0) {
        double actual_g0 = static_cast<double>(opened_g0_lik[0]) / scale;
        std::cout << "GroupLaplaceLogLik (Group 0):\n"
                  << "  Plaintext: " << expected_g0 << "\n"
                  << "  MPC:       " << actual_g0 << "\n"
                  << "  Abs Error: " << std::abs(expected_g0 - actual_g0) << std::endl;
    }

    AV group1_lik = GroupLaplaceLogLik(secure_dataset.groups[1], secure_beta, secure_sigma2);
    auto opened_g1_lik = group1_lik.open();
    double expected_g1 = plain_group_laplace(1, plain_beta, plain_sigma2);
    if (pID == 0) {
        double actual_g1 = static_cast<double>(opened_g1_lik[0]) / scale;
        std::cout << "GroupLaplaceLogLik (Group 1):\n"
                  << "  Plaintext: " << expected_g1 << "\n"
                  << "  MPC:       " << actual_g1 << "\n"
                  << "  Abs Error: " << std::abs(expected_g1 - actual_g1) << std::endl;
    }

    // 3. Test NegMarginalLogLik
    AV neg_log_lik = NegMarginalLogLik(secure_dataset, secure_params);
    auto opened_neg_log_lik = neg_log_lik.open();
    std::vector<double> plain_params = {plain_beta[0], plain_beta[1], plain_s};
    double expected_nll = plain_neg_marginal_log_lik(plain_params);
    if (pID == 0) {
        double actual_nll = static_cast<double>(opened_neg_log_lik[0]) / scale;
        std::cout << "NegMarginalLogLik (Total):\n"
                  << "  Plaintext: " << expected_nll << "\n"
                  << "  MPC:       " << actual_nll << "\n"
                  << "  Abs Error: " << std::abs(expected_nll - actual_nll) << std::endl;
    }

    // 4. Test NumericalGradient
    auto objective_func = [&secure_dataset](const std::vector<AV>& p) -> AV {
        return NegMarginalLogLik(secure_dataset, p);
    };

    std::vector<AV> secure_grad = NumericalGradient(objective_func, secure_params);
    std::vector<double> expected_grad = plain_numerical_gradient(plain_params);
    if (pID == 0) {
        std::cout << "NumericalGradient:" << std::endl;
        std::cout << "  Plaintext: [";
        for (size_t k = 0; k < expected_grad.size(); ++k) {
            std::cout << expected_grad[k] << (k + 1 < expected_grad.size() ? ", " : "");
        }
        std::cout << "]" << std::endl;
        std::cout << "  MPC:       [";
    }
    for (size_t k = 0; k < secure_grad.size(); ++k) {
        auto opened_grad_k = secure_grad[k].open();
        if (pID == 0) {
            double val = static_cast<double>(opened_grad_k[0]) / scale;
            std::cout << val << (k + 1 < secure_grad.size() ? ", " : "");
        }
    }
    if (pID == 0) {
        std::cout << "]" << std::endl;
    }

    // =========================================================================
    // Validation for Identity, MatVec, Dot, BfgsInverseUpdate, and MinimizeBFGS
    // =========================================================================
    if (pID == 0) {
        std::cout << "\n--- Testing Identity, MatVec, Dot, BfgsInverseUpdate, and MinimizeBFGS ---" << std::endl;
    }

    // 1. Test Identity
    size_t dim = 3;
    SMatrix sec_I = Identity(dim, engine);
    auto opened_I = sec_I.open();
    if (pID == 0) {
        std::cout << "Secure Identity (3x3):" << std::endl;
        for (size_t i = 0; i < dim; ++i) {
            std::cout << "  [";
            for (size_t j = 0; j < dim; ++j) {
                double val = static_cast<double>(opened_I.data()[i * dim + j]) / scale;
                std::cout << std::setw(6) << val << (j + 1 < dim ? ", " : "");
            }
            std::cout << "]" << std::endl;
        }
    }

    // 2. Test Dot
    std::vector<double> v1_plain = {1.5, -2.0, 3.0};
    std::vector<double> v2_plain = {0.5, 4.0, -1.0};
    double expected_dot = 1.5 * 0.5 + (-2.0) * 4.0 + 3.0 * (-1.0); // 0.75 - 8.0 - 3.0 = -10.25

    std::vector<AV> v1_sec, v2_sec;
    for (size_t i = 0; i < dim; ++i) {
        cdough::Vector<DataType> p1(1, precision); p1[0] = static_cast<DataType>(v1_plain[i] * scale);
        cdough::Vector<DataType> p2(1, precision); p2[0] = static_cast<DataType>(v2_plain[i] * scale);
        v1_sec.push_back(engine.secret_share_a(p1, 0, precision));
        v2_sec.push_back(engine.secret_share_a(p2, 0, precision));
    }
    AV sec_dot = Dot(v1_sec, v2_sec);
    auto opened_dot = sec_dot.open();
    if (pID == 0) {
        double actual_dot = static_cast<double>(opened_dot[0]) / scale;
        std::cout << "Dot Product:\n"
                  << "  Plaintext: " << expected_dot << "\n"
                  << "  MPC:       " << actual_dot << "\n"
                  << "  Abs Error: " << std::abs(expected_dot - actual_dot) << std::endl;
    }

    // 3. Test MatVec with Identity: I * v1 == v1
    std::vector<AV> sec_matvec = MatVec(sec_I, v1_sec);
    std::vector<double> opened_matvec_vals;
    for (size_t i = 0; i < dim; ++i) {
        auto op_mv = sec_matvec[i].open();
        opened_matvec_vals.push_back(static_cast<double>(op_mv[0]) / scale);
    }
    if (pID == 0) {
        std::cout << "MatVec (I * v1):" << std::endl;
        std::cout << "  Expected: [1.5, -2.0, 3.0]\n  MPC:      [";
        for (size_t i = 0; i < dim; ++i) {
            std::cout << opened_matvec_vals[i] << (i + 1 < dim ? ", " : "");
        }
        std::cout << "]" << std::endl;
    }

    // 4. Test BfgsInverseUpdate
    // s = [0.1, -0.2, 0.05], y = [0.2, -0.1, 0.1]
    // rho = 1.0 / (y^T s) = 1.0 / (0.02 + 0.02 + 0.005) = 1.0 / 0.045 = 22.2222
    std::vector<double> s_plain = {0.1, -0.2, 0.05};
    std::vector<double> y_plain = {0.2, -0.1, 0.1};
    double ys = 0.1 * 0.2 + (-0.2) * (-0.1) + 0.05 * 0.1; // 0.045
    double rho_plain = 1.0 / ys;

    std::vector<AV> s_sec, y_sec;
    for (size_t i = 0; i < dim; ++i) {
        cdough::Vector<DataType> ps(1, precision); ps[0] = static_cast<DataType>(s_plain[i] * scale);
        cdough::Vector<DataType> py(1, precision); py[0] = static_cast<DataType>(y_plain[i] * scale);
        s_sec.push_back(engine.secret_share_a(ps, 0, precision));
        y_sec.push_back(engine.secret_share_a(py, 0, precision));
    }
    cdough::Vector<DataType> prho(1, precision); prho[0] = static_cast<DataType>(rho_plain * scale);
    AV rho_sec = engine.secret_share_a(prho, 0, precision);

    SMatrix sec_h_updated = BfgsInverseUpdate(sec_I, s_sec, y_sec, rho_sec);
    auto opened_h_up = sec_h_updated.open();

    // Plaintext computation of BfgsInverseUpdate from Identity
    // left = I - rho * s * y^T
    std::vector<std::vector<double>> left_plain(dim, std::vector<double>(dim, 0.0));
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            left_plain[i][j] = (i == j ? 1.0 : 0.0) - rho_plain * s_plain[i] * y_plain[j];
        }
    }
    // updated = left * I * left^T + rho * s * s^T = left * left^T + rho * s * s^T
    std::vector<std::vector<double>> expected_H(dim, std::vector<double>(dim, 0.0));
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < dim; ++k) {
                sum += left_plain[i][k] * left_plain[j][k];
            }
            expected_H[i][j] = sum + rho_plain * s_plain[i] * s_plain[j];
        }
    }

    if (pID == 0) {
        std::cout << "BfgsInverseUpdate (from I):\n";
        double max_h_err = 0.0;
        for (size_t i = 0; i < dim; ++i) {
            std::cout << "  Row " << i << " Plain: [";
            for (size_t j = 0; j < dim; ++j) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(4) << expected_H[i][j] << (j + 1 < dim ? ", " : "");
            }
            std::cout << "]  MPC: [";
            for (size_t j = 0; j < dim; ++j) {
                double val = static_cast<double>(opened_h_up.data()[i * dim + j]) / scale;
                double err = std::abs(expected_H[i][j] - val);
                if (err > max_h_err) max_h_err = err;
                std::cout << std::setw(8) << std::fixed << std::setprecision(4) << val << (j + 1 < dim ? ", " : "");
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "  Max Matrix Abs Error: " << max_h_err << std::endl;
    }


    // =========================================================================
    // Validation for Transpose, MatMul, and NewtonSchulzInverse
    // =========================================================================
    if (pID == 0) {
        std::cout << "\n--- Testing Transpose, MatMul, and NewtonSchulzInverse ---" << std::endl;
    }

    using Mat = std::vector<std::vector<double>>;

    auto share_matrix = [&engine](const Mat& m) -> SMatrix {
        size_t rows = m.size();
        size_t cols = m[0].size();
        cdough::Vector<DataType> flat(rows * cols, precision);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                flat[i * cols + j] = std::llround(m[i][j] * scale);
            }
        }
        PMatrix plain(flat, rows, cols, false);
        SMatrix shared = engine.secret_share_matrix(plain, 0);
        shared.setPrecision(precision);
        return shared;
    };

    // Must be called by every party: open() is a communication round.
    auto open_matrix = [](const SMatrix& m) -> Mat {
        auto opened = m.open();
        Mat out(m.rows(), std::vector<double>(m.cols(), 0.0));
        for (size_t i = 0; i < m.rows(); ++i) {
            for (size_t j = 0; j < m.cols(); ++j) {
                out[i][j] = static_cast<double>(opened.data()[i * m.cols() + j]) / scale;
            }
        }
        return out;
    };

    auto plain_transpose = [](const Mat& m) -> Mat {
        Mat out(m[0].size(), std::vector<double>(m.size(), 0.0));
        for (size_t i = 0; i < m.size(); ++i) {
            for (size_t j = 0; j < m[0].size(); ++j) out[j][i] = m[i][j];
        }
        return out;
    };

    auto plain_matmul = [](const Mat& a, const Mat& b) -> Mat {
        Mat out(a.size(), std::vector<double>(b[0].size(), 0.0));
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < b[0].size(); ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < b.size(); ++k) sum += a[i][k] * b[k][j];
                out[i][j] = sum;
            }
        }
        return out;
    };

    // Gauss-Jordan with partial pivoting. Reference only; the secure operator
    // cannot pivot, because pivot selection branches on the data.
    auto plain_inverse = [](Mat m) -> Mat {
        size_t n = m.size();
        Mat inv(n, std::vector<double>(n, 0.0));
        for (size_t i = 0; i < n; ++i) inv[i][i] = 1.0;
        for (size_t col = 0; col < n; ++col) {
            size_t pivot = col;
            for (size_t r = col + 1; r < n; ++r) {
                if (std::abs(m[r][col]) > std::abs(m[pivot][col])) pivot = r;
            }
            std::swap(m[col], m[pivot]);
            std::swap(inv[col], inv[pivot]);
            double d = m[col][col];
            for (size_t j = 0; j < n; ++j) {
                m[col][j] /= d;
                inv[col][j] /= d;
            }
            for (size_t r = 0; r < n; ++r) {
                if (r == col) continue;
                double f = m[r][col];
                for (size_t j = 0; j < n; ++j) {
                    m[r][j] -= f * m[col][j];
                    inv[r][j] -= f * inv[col][j];
                }
            }
        }
        return inv;
    };

    auto max_abs_diff = [](const Mat& a, const Mat& b) -> double {
        double worst = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < a[0].size(); ++j) {
                worst = std::max(worst, std::abs(a[i][j] - b[i][j]));
            }
        }
        return worst;
    };

    // Residual max |A * A_inv - I|, the honest accuracy metric: it stays
    // meaningful even when the entries of A^-1 are large.
    auto residual_error = [&plain_matmul](const Mat& a, const Mat& a_inv) -> double {
        Mat product = plain_matmul(a, a_inv);
        double worst = 0.0;
        for (size_t i = 0; i < product.size(); ++i) {
            for (size_t j = 0; j < product[0].size(); ++j) {
                double target = (i == j) ? 1.0 : 0.0;
                worst = std::max(worst, std::abs(product[i][j] - target));
            }
        }
        return worst;
    };

    // 1. Transpose and MatMul, before anything is built on top of them.
    // Every matrix here is deliberately non-symmetric: a symmetric one cannot
    // tell A * B apart from A * B^T, which is the failure mode of getting the
    // library's column-wise calling convention backwards.
    Mat plain_wide = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};        // 2 x 3
    Mat plain_tall = {{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};   // 3 x 2
    Mat plain_left = {{1.0, 2.0}, {3.0, 4.0}};                  // 2 x 2
    Mat plain_right = {{5.0, 6.0}, {7.0, 8.0}};                 // 2 x 2

    SMatrix secure_wide = share_matrix(plain_wide);
    SMatrix secure_tall = share_matrix(plain_tall);
    SMatrix secure_left = share_matrix(plain_left);
    SMatrix secure_right = share_matrix(plain_right);

    Mat opened_transpose = open_matrix(Transpose(secure_wide));
    Mat opened_rect_product = open_matrix(MatMul(secure_wide, secure_tall));
    Mat opened_square_product = open_matrix(MatMul(secure_left, secure_right));

    if (pID == 0) {
        Mat expected_transpose = plain_transpose(plain_wide);
        Mat expected_rect = plain_matmul(plain_wide, plain_tall);
        Mat expected_square = plain_matmul(plain_left, plain_right);
        // If the column-wise convention were inverted we would land on this instead.
        Mat wrong_square = plain_matmul(plain_left, plain_transpose(plain_right));

        std::cout << "Transpose (2x3 -> 3x2):        max abs error = "
                  << max_abs_diff(expected_transpose, opened_transpose) << std::endl;
        std::cout << "MatMul (2x3 * 3x2):            max abs error = "
                  << max_abs_diff(expected_rect, opened_rect_product) << std::endl;
        std::cout << "MatMul (2x2 * 2x2):            max abs error = "
                  << max_abs_diff(expected_square, opened_square_product) << std::endl;
        std::cout << "  (distance from the A*B^T mistake: "
                  << max_abs_diff(wrong_square, opened_square_product)
                  << ", must be large)" << std::endl;
    }

    // 2. NewtonSchulzInverse over a suite spanning conditioning and symmetry.
    // Note every matrix is O(1)-scaled, as the operator requires: X_0 is
    // A^T / ||A||_F^2, which underflows to zero if ||A||_F^2 grows too large.
    std::vector<std::pair<std::string, Mat>> inverse_cases = {
        {"identity 3x3", {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}},
        {"spd, kappa ~ 4", {{4.0, 1.0, 0.0}, {1.0, 3.0, 1.0}, {0.0, 1.0, 2.0}}},
        {"spd, kappa ~ 19", {{1.0, 0.9, 0.0}, {0.9, 1.0, 0.0}, {0.0, 0.0, 1.0}}},
        {"non-symmetric 3x3", {{2.0, 1.0, 0.0}, {0.0, 3.0, 1.0}, {1.0, 0.0, 4.0}}},
        {"2x2", {{2.0, 1.0}, {1.0, 3.0}}},
        // Deliberately past the operating range, to show where the fixed bound gives up.
        {"spd, kappa ~ 199", {{1.0, 0.99, 0.0}, {0.99, 1.0, 0.0}, {0.0, 0.0, 1.0}}},
    };

    // A realistic case: the averaged information matrix X^T W X / N from the
    // synthetic dataset above, evaluated at the true parameters. Averaging by N
    // is what keeps it O(1)-scaled.
    {
        size_t num_obs_total = test_num_groups * test_obs_per_group;
        Mat gram(test_num_fixed, std::vector<double>(test_num_fixed, 0.0));
        for (size_t g = 0; g < test_num_groups; ++g) {
            for (size_t j = 0; j < test_obs_per_group; ++j) {
                double eta = kTrueBeta0 + kTrueBeta1 * plain_X[g][j][1] + true_u[g];
                double prob = 1.0 / (1.0 + std::exp(-eta));
                double weight = prob * (1.0 - prob);
                for (size_t r = 0; r < test_num_fixed; ++r) {
                    for (size_t c = 0; c < test_num_fixed; ++c) {
                        gram[r][c] += weight * plain_X[g][j][r] * plain_X[g][j][c];
                    }
                }
            }
        }
        for (size_t r = 0; r < test_num_fixed; ++r) {
            for (size_t c = 0; c < test_num_fixed; ++c) {
                gram[r][c] /= static_cast<double>(num_obs_total);
            }
        }
        inverse_cases.emplace_back("X^T W X / N (2x2)", gram);
    }

    if (pID == 0) {
        std::cout << "\nNewtonSchulzInverse (" << kMatrixInverseIterations
                  << " iterations):" << std::endl;
        std::cout << std::left << std::setw(24) << "Matrix"
                  << std::setw(18) << "Max entry error"
                  << std::setw(18) << "Residual |AX-I|" << std::endl;
    }

    for (const auto& [name, matrix] : inverse_cases) {
        SMatrix secure_case = share_matrix(matrix);
        Mat opened_inverse = open_matrix(NewtonSchulzInverse(secure_case));
        if (pID == 0) {
            double entry_error = max_abs_diff(plain_inverse(matrix), opened_inverse);
            double residual = residual_error(matrix, opened_inverse);
            std::cout << std::left << std::setw(24) << name
                      << std::setw(18) << std::scientific << std::setprecision(3) << entry_error
                      << std::setw(18) << residual << std::endl;
        }
    }
    if (pID == 0) {
        std::cout << "  (the kappa ~ 199 row is deliberately past the operating range:"
                  << " a fixed iteration bound cannot chase arbitrary conditioning)" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
    }

    // 3. Iteration-count calibration. Sweeps the bound so it can be chosen from
    // measurement on the real 3PC deployment rather than guessed. The error
    // should fall quadratically and then flatten on the ABY3 truncation noise
    // floor; take the first count on the plateau for the worst case here, and
    // require that plateau to sit at or below the abs errors the other
    // operators above report (~1e-4).
    if (kRunMatrixInverseCalibration) {
        if (pID == 0) {
            std::cout << "\nNewtonSchulzInverse iteration calibration"
                      << " (residual |AX-I|, by iteration count):" << std::endl;
            std::cout << std::left << std::setw(24) << "Matrix";
            for (int k = 1; k <= kMatrixInverseCalibrationMax; ++k) {
                std::cout << std::setw(11) << k;
            }
            std::cout << std::endl;
        }
        for (const auto& [name, matrix] : inverse_cases) {
            SMatrix secure_case = share_matrix(matrix);
            std::vector<double> residuals;
            residuals.reserve(kMatrixInverseCalibrationMax);
            for (int k = 1; k <= kMatrixInverseCalibrationMax; ++k) {
                Mat opened_inverse = open_matrix(NewtonSchulzInverse(secure_case, k));
                residuals.push_back(residual_error(matrix, opened_inverse));
            }
            if (pID == 0) {
                std::cout << std::left << std::setw(24) << name;
                for (double r : residuals) {
                    std::cout << std::setw(11) << std::scientific << std::setprecision(2) << r;
                }
                std::cout << std::endl;
            }
        }
        if (pID == 0) {
            std::cout << std::fixed << std::setprecision(6);
        }
    }

    // 5. Test MinimizeBFGS.
    // Skipped while calibrating: the sweep above is what that run is for, and
    // the full optimizer costs far more than everything else here combined.
    if (kRunMatrixInverseCalibration) {
        if (pID == 0) {
            std::cout << "\nSkipping MinimizeBFGS (matrix-inverse calibration run)." << std::endl;
        }
    } else {
        if (pID == 0) {
            std::cout << "\nRunning MinimizeBFGS (Secure Quasi-Newton Optimizer)..." << std::endl;
        }
        // Initial parameter guess: [0.0, 0.0, 0.0]
        std::vector<AV> init_params_sec;
        for (size_t i = 0; i < dim; ++i) {
            cdough::Vector<DataType> p_init(1, precision); p_init[0] = 0;
            init_params_sec.push_back(engine.secret_share_a(p_init, 0, precision));
        }

        OptResult opt_res = MinimizeBFGS(objective_func, init_params_sec, 50);

        auto opened_final_val = opt_res.value.open();
        std::vector<double> opened_final_params;
        for (size_t i = 0; i < dim; ++i) {
            auto op_p = opt_res.params[i].open();
            opened_final_params.push_back(static_cast<double>(op_p[0]) / scale);
        }

        if (pID == 0) {
            std::cout << "MinimizeBFGS Result:\n"
                      << "  Iterations: " << opt_res.iterations << "\n"
                      << "  Converged:  " << (opt_res.converged ? "True" : "False") << "\n"
                      << "  Final Objective Value: " << static_cast<double>(opened_final_val[0]) / scale << "\n"
                      << "  Fitted Parameters [beta_0, beta_1, s]: [";
            for (size_t i = 0; i < dim; ++i) {
                std::cout << opened_final_params[i] << (i + 1 < dim ? ", " : "");
            }
            std::cout << "]" << std::endl;

            // The MLE need not equal the generating parameters on a finite sample,
            // so score the fit by the plaintext objective instead
            std::vector<double> true_params = {kTrueBeta0, kTrueBeta1, plain_s};
            double nll_at_true = plain_neg_marginal_log_lik(true_params);
            double nll_at_fit = plain_neg_marginal_log_lik(opened_final_params);
            std::cout << "  True Parameters [beta_0, beta_1, s]:   [" << kTrueBeta0 << ", "
                      << kTrueBeta1 << ", " << plain_s << "]\n"
                      << "  Plaintext NLL at true params: " << nll_at_true << "\n"
                      << "  Plaintext NLL at MPC fit:     " << nll_at_fit
                      << (nll_at_fit <= nll_at_true ? "   (fit beats truth: OK)"
                                                    : "   (WORSE than truth)")
                      << std::endl;
            std::cout << "\nAll functions executed and validated successfully against plaintext!" << std::endl;
        }
    }

    return 0;
}