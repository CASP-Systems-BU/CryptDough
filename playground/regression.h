#pragma once

#include "./optimizer.h"
#include "./primitives.h"

namespace cdough::regression {

// Secure clamping to [-kMaxNewtonStep, kMaxNewtonStep]
AV ClampNewtonStep(const AV& step) { return ClampAbs(step, kMaxNewtonStep_scaled); }

namespace mixedeffects {

    // Data structures for secure mixed-effects logistic regression
    struct ClusterGroup {
        AV y;                    // Binary outcomes (0/1), length N (group size)
        std::vector<AV> x_cols;  // num_fixed columns, each of size N

        ClusterGroup(AV y_in, std::vector<AV> x_cols_in)
            : y(std::move(y_in)), x_cols(std::move(x_cols_in)) {}

        size_t size() const { return y.size(); }
    };

    struct Dataset {
        std::vector<ClusterGroup> groups;
        size_t num_fixed = 0;
    };

    // Helpers
    // Splits a packed parameter vector [beta..., s] into `beta` and sigma^2, where
    // the variance is parameterized as sigma = exp(s) to keep it strictly positive.
    void UnpackParameters(const std::vector<AV>& params, size_t num_fixed, std::vector<AV>& beta,
                          AV& sigma2) {
        beta.clear();
        for (size_t k = 0; k < num_fixed; ++k) {
            beta.push_back(params[k]);
            beta.back().setPrecision(precision);
        }
        AV s = params[num_fixed];
        s.setPrecision(0);
        AV two_s = *(s * DataType(2));
        two_s.setPrecision(precision);
        sigma2 = Exp(two_s);
        sigma2.setPrecision(precision);
    }

    // Finds the conditional mode u_hat = argmax_u g_i(u) for one group, given the
    // fixed effects `beta` (std::vector<AV> of size p, each size 1) and variance `sigma2` (AV of
    // size 1). The objective g_i is strictly concave in u, so a damped Newton iteration converges
    // reliably.
    AV ConditionalMode(const ClusterGroup& group, const std::vector<AV>& beta, const AV& sigma2) {
        EngineRef engine = sigma2.engine;
        size_t num_obs = group.size();
        size_t num_fixed = beta.size();

        // inv_sigma2 = 1.0 / sigma2 (in fixed-point: scale^2 / sigma2)
        AV inv_sigma2 = SecureReciprocal(sigma2);
        inv_sigma2.setPrecision(0);

        // Compute fixed effects linear predictor X_beta = sum_{k=0}^{p-1} X_col[k] * beta[k]
        // X_beta has length num_obs
        AV x_beta(num_obs, engine);
        x_beta.setPrecision(0);
        for (size_t k = 0; k < num_fixed; ++k) {
            // Broadcast beta[k] (size 1) to size num_obs using repeated_subset_reference
            AV beta_rep = beta[k].repeated_subset_reference(num_obs);
            beta_rep.setPrecision(0);
            AV col_k = group.x_cols[k];
            col_k.setPrecision(0);
            AV term = (*(col_k * beta_rep)) / scale;
            x_beta += term;
        }

        AV u(1, engine);  // Initial mode u = 0 (size 1)
        u.setPrecision(0);

        for (int iter = 0; iter < kNewtonIterations; ++iter) {
            // Broadcast u to size num_obs
            AV u_rep = u.repeated_subset_reference(num_obs);
            u_rep.setPrecision(0);

            AV eta = x_beta + u_rep;
            eta.setPrecision(precision);
            AV p = Sigmoid(eta);  // size num_obs
            p.setPrecision(0);

            // gradient = -u * inv_sigma2 + sum_j (y_j - p_j)
            AV u_inv = (*(u * inv_sigma2)) / scale;
            AV grad_prior = -u_inv;  // size 1

            AV one_minus_p = -p;
            one_minus_p += scale;
            AV var_p = (*(p * one_minus_p)) / scale;  // p * (1 - p), size num_obs

            AV y_copy = group.y;
            y_copy.setPrecision(0);
            AV diff_y_p = y_copy - p;  // size num_obs
            diff_y_p.setPrecision(precision);
            AV sum_grad = diff_y_p.chunkedSum(diff_y_p.size());  // size 1
            sum_grad.setPrecision(0);
            AV gradient = grad_prior + sum_grad;  // size 1

            // curvature = inv_sigma2 + sum_j p_j * (1 - p_j)
            var_p.setPrecision(precision);
            AV sum_curv = var_p.chunkedSum(var_p.size());  // size 1
            sum_curv.setPrecision(0);
            AV curvature = inv_sigma2 + sum_curv;  // size 1

            // step = gradient / curvature
            auto grad_scaled_b = (*(gradient * scale)).a2b();
            auto curv_b = curvature.a2b();
            auto step_b = (*grad_scaled_b) / (*curv_b);
            AV step = *(step_b->b2a());
            step.setPrecision(precision);

            // Clamp step to [-kMaxNewtonStep, kMaxNewtonStep]
            AV step_clamped = ClampNewtonStep(step);
            step_clamped.setPrecision(0);
            u += step_clamped;
        }

        u.setPrecision(precision);
        return u;
    }

    // Laplace-approximated marginal log-likelihood contribution of a single group.
    // log L_i ~= cll(u_hat_i) - 0.5 * u_hat_i^2 / sigma^2 - 0.5 * log(sigma^2) - 0.5 * log(A_i)
    AV GroupLaplaceLogLik(const ClusterGroup& group, const std::vector<AV>& beta,
                          const AV& sigma2) {
        EngineRef engine = sigma2.engine;
        size_t num_obs = group.size();
        size_t num_fixed = beta.size();

        // inv_sigma2 = scale^2 / sigma2
        AV inv_sigma2 = SecureReciprocal(sigma2);
        inv_sigma2.setPrecision(0);

        AV u_hat = ConditionalMode(group, beta, sigma2);  // size 1
        u_hat.setPrecision(0);

        // Compute linear predictor X_beta + u_hat
        AV x_beta(num_obs, engine);
        x_beta.setPrecision(0);
        for (size_t k = 0; k < num_fixed; ++k) {
            AV beta_rep = beta[k].repeated_subset_reference(num_obs);
            beta_rep.setPrecision(0);
            AV col_k = group.x_cols[k];
            col_k.setPrecision(0);
            AV term = (*(col_k * beta_rep)) / scale;
            x_beta += term;
        }
        AV u_hat_rep = u_hat.repeated_subset_reference(num_obs);
        u_hat_rep.setPrecision(0);
        AV eta = x_beta + u_hat_rep;
        eta.setPrecision(precision);

        AV p = Sigmoid(eta);
        p.setPrecision(0);
        AV log_one_plus_exp = LogOnePlusExp(eta);
        log_one_plus_exp.setPrecision(0);

        // conditional_log_lik = sum_j (y_j * eta_j - LogOnePlusExp(eta_j))
        AV y_copy = group.y;
        y_copy.setPrecision(0);
        eta.setPrecision(0);
        AV y_eta = (*(y_copy * eta)) / scale;
        AV obs_cll = y_eta - log_one_plus_exp;
        obs_cll.setPrecision(precision);
        AV conditional_log_lik = obs_cll.chunkedSum(obs_cll.size());  // size 1
        conditional_log_lik.setPrecision(0);

        // curvature A_i = inv_sigma2 + sum_j p_j * (1 - p_j)
        AV one_minus_p = -p;
        one_minus_p += scale;
        AV var_p = (*(p * one_minus_p)) / scale;
        var_p.setPrecision(precision);
        AV sum_var_p = var_p.chunkedSum(var_p.size());
        sum_var_p.setPrecision(0);
        AV curvature = inv_sigma2 + sum_var_p;  // size 1

        // Gaussian prior quadratic penalty: 0.5 * u_hat^2 * inv_sigma2
        AV u_sq = (*(u_hat * u_hat)) / scale;
        AV penalty = (*(u_sq * inv_sigma2)) / scale;
        AV penalty_half = (*(penalty * kHalf_scaled)) / scale;

        // Log terms: 0.5 * log(sigma2) + 0.5 * log(curvature)
        AV sigma2_copy = sigma2;
        sigma2_copy.setPrecision(precision);
        AV log_sigma2 = Log(sigma2_copy);
        log_sigma2.setPrecision(0);

        curvature.setPrecision(precision);
        AV log_curv = Log(curvature);
        log_curv.setPrecision(0);

        AV log_terms = log_sigma2 + log_curv;
        AV log_terms_half = (*(log_terms * kHalf_scaled)) / scale;

        AV result = conditional_log_lik - penalty_half - log_terms_half;
        result.setPrecision(precision);
        return result;
    }

    // Negative marginal log-likelihood over the whole dataset (the BFGS objective).
    AV NegMarginalLogLik(const Dataset& data, const std::vector<AV>& params) {
        std::vector<AV> beta;
        AV sigma2(1, params[0].engine);
        sigma2.setPrecision(precision);
        UnpackParameters(params, data.num_fixed, beta, sigma2);

        AV total(1, params[0].engine);
        total.setPrecision(0);
        for (const auto& group : data.groups) {
            AV group_lik = GroupLaplaceLogLik(group, beta, sigma2);
            group_lik.setPrecision(0);
            total += group_lik;
        }
        AV neg_total = -total;
        neg_total.setPrecision(precision);
        return neg_total;
    }

}  // namespace mixedeffects

}  // namespace cdough::regression