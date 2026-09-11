#pragma once

#include "./optimizer.h"
#include "./primitives.h"

namespace cdough::regression {

// Secure clamping to [-kMaxNewtonStep, kMaxNewtonStep]
AV ClampNewtonStep(const AV& step) { return ClampAbs(step, kMaxNewtonStep_scaled); }

// =============================================================================
// mixedeffects: vectorized (batched) implementation
// =============================================================================
//
// Circuit depth does not scale
// with the number of groups or with the parameter count. The whole dataset lives
// in one buffer and every operation is applied to it at once; per-group
// quantities are recovered with chunkedSum and broadcast back with index
// mappings, which cost no communication.
//
// Layout convention, used everywhere below. Let
//   G = num_groups, N = obs_per_group (padded), p = num_fixed,
//   K = number of parameter points evaluated simultaneously.
// Observation-level buffers have length K * G * N, indexed
//   (k, g, j) -> k * (G * N) + g * N + j
// so a group's observations are CONSECUTIVE. That is what makes chunkedSum(N)
// the per-group aggregator. Group-level buffers have length K * G, indexed
//   (k, g) -> k * G + g
// and parameter-point-level buffers have length K.
// =============================================================================
namespace mixedeffects {

// One padded, batched dataset.
//
// Groups are padded to a common width `obs_per_group`; `mask` is 1 on real rows
// and 0 on padding, and is multiplied in before every per-group aggregation so
// padded rows contribute exactly zero.
//
// PRIVACY: only `num_groups` and the padded width `obs_per_group` are public.
// Because the mask is secret-shared, the *actual* per-group sizes stay private —
// unlike the serial implementation, where each group's size is a loop bound. To
// keep that property, choose `obs_per_group` from a data-independent bound; if
// you pad to the observed maximum group size, you leak that maximum.
struct BatchedDataset {
    AV x_data;  // K-independent design matrix, (G*N) x p row-major: row = g*N + j
    AV y;       // length G*N
    AV mask;    // length G*N, fixed-point 1.0 on real rows, 0 on padding
    size_t num_groups = 0;
    size_t obs_per_group = 0;  // padded width N
    size_t num_fixed = 0;      // p

    BatchedDataset(AV x_in, AV y_in, AV mask_in, size_t groups, size_t obs, size_t fixed)
        : x_data(std::move(x_in)),
          y(std::move(y_in)),
          mask(std::move(mask_in)),
          num_groups(groups),
          obs_per_group(obs),
          num_fixed(fixed) {
        assert(x_data.size() == groups * obs * fixed);
        assert(y.size() == groups * obs);
        assert(mask.size() == groups * obs);
    }

    size_t total_rows() const { return num_groups * obs_per_group; }
};

// Zeroes padded entries so they cannot reach a per-group sum. One vectorized
// multiply; the mask is secret-shared because the library has no elementwise
// multiply against a public vector.
AV ApplyMask(const AV& values, const AV& mask) {
    AV v = values;
    AV m = mask;
    v.setPrecision(0);
    m.setPrecision(0);
    AV out = (*(v * m)) / scale;
    out.setPrecision(precision);
    return out;
}

// The design coefficients and variances of a batch of parameter points.
struct UnpackedParameters {
    AV beta_data;  // K x p row-major, materialized (the matmul kernel rejects views)
    AV sigma2;     // length K
};

// Splits a batch of packed parameter points into the design coefficients and the
// variances. `params` is length K * dim with (k, i) -> k * dim + i, where
// dim = p + 1 and the trailing entry of each point is s, with sigma = exp(s).
//
// Returns by value rather than filling out-parameters: an out-parameter would
// have to be constructed at some size first, and AV copy-assignment asserts
// equal sizes (semantic topic 0001, claim C-06), so a pre-sized placeholder
// aborts the moment the real size differs. It would also mean constructing
// one-element AVs, which is the pattern this rewrite exists to remove.
UnpackedParameters UnpackParametersBatched(const AV& params, size_t num_points,
                                           size_t num_fixed) {
    const size_t dim = num_fixed + 1;
    assert(params.size() == num_points * dim);
    AV beta_data(num_points * num_fixed, params.engine);
    AV sigma2(num_points, params.engine);

    std::vector<size_t> beta_map(num_points * num_fixed);
    for (size_t k = 0; k < num_points; ++k) {
        for (size_t i = 0; i < num_fixed; ++i) {
            beta_map[k * num_fixed + i] = k * dim + i;
        }
    }
    // Clone materializes, dropping the mapping.
    beta_data = Clone(params.mapping_reference(beta_map));
    beta_data.setPrecision(precision);

    std::vector<size_t> s_map(num_points);
    for (size_t k = 0; k < num_points; ++k) s_map[k] = k * dim + num_fixed;
    AV s = params.mapping_reference(s_map);
    s.setPrecision(0);
    AV two_s = *(s * DataType(2));
    two_s.setPrecision(precision);
    sigma2 = Exp(two_s);
    sigma2.setPrecision(precision);

    return UnpackedParameters{std::move(beta_data), std::move(sigma2)};
}

// All fixed-effect linear predictors for all parameter points, in ONE matmul.
//
// The trick is that a row-major (G*N) x p buffer is bit-identical to a
// column-wise p x (G*N) matrix, so X^T is available with no data movement (see
// semantic topic 0001, claim C-09). The kernel wants a column-wise right-hand
// side, so this is exactly the orientation it needs:
//
//   Beta (K x p, row-major) * X^T (p x (G*N), column-wise) -> K x (G*N) row-major
//
// and the output lands already in the (k, g, j) layout chunkedSum consumes.
AV LinearPredictors(const BatchedDataset& data, const AV& beta_data, size_t num_points) {
    SMatrix beta_m(beta_data, num_points, data.num_fixed, false);
    beta_m.setPrecision(precision);

    SMatrix xt(data.x_data, data.num_fixed, data.total_rows(), true);
    xt.setPrecision(precision);

    SMatrix eta = beta_m.matrixRightMultiplyWithColumnMatrixVectorized(xt);
    eta.setPrecision(precision);
    return eta.data();
}

// Conditional modes u_hat for every (parameter point, group), by the same damped
// Newton iteration as the pre-vectorization implementation (see semantic task 0014).
//
// Depth is kNewtonIterations, independent of G and K: each iteration issues one
// Sigmoid over K*G*N elements and one boolean division circuit over K*G, where
// the serial version issued K*G of each.
AV ConditionalModeBatched(const BatchedDataset& data, const AV& x_beta, const AV& inv_sigma2,
                          size_t num_points) {
    const size_t G = data.num_groups;
    const size_t N = data.obs_per_group;
    const size_t rows = data.total_rows();
    EngineRef engine = data.y.engine;

    assert(x_beta.size() == num_points * rows);
    assert(inv_sigma2.size() == num_points);

    // Tile the observation-level inputs across the K parameter points, and
    // broadcast the per-point variance down to per (k, g). Both are mappings.
    AV y_rep = data.y.cyclic_subset_reference(num_points);
    AV mask_rep = data.mask.cyclic_subset_reference(num_points);
    AV inv_sigma2_kg = inv_sigma2.repeated_subset_reference(G);
    inv_sigma2_kg.setPrecision(0);

    AV u(num_points * G, engine);  // modes start at 0
    u.setPrecision(0);

    for (int iteration = 0; iteration < kNewtonIterations; ++iteration) {
        // One mode per group, repeated across that group's observations.
        AV u_rows = u.repeated_subset_reference(N);
        u_rows.setPrecision(0);

        AV eta = x_beta + u_rows;
        eta.setPrecision(precision);
        AV p = Sigmoid(eta);
        p.setPrecision(0);

        // gradient = -u * inv_sigma2 + sum_j (y_j - p_j)
        AV y_copy = y_rep;
        y_copy.setPrecision(0);
        AV diff_y_p = y_copy - p;
        diff_y_p.setPrecision(precision);
        AV sum_grad = ApplyMask(diff_y_p, mask_rep).chunkedSum(N);
        sum_grad.setPrecision(0);

        AV u_inv = (*(u * inv_sigma2_kg)) / scale;
        AV gradient = -u_inv;
        gradient += sum_grad;

        // curvature = inv_sigma2 + sum_j p_j * (1 - p_j)
        AV one_minus_p = -p;
        one_minus_p += scale;
        AV var_p = (*(p * one_minus_p)) / scale;
        var_p.setPrecision(precision);
        AV sum_curv = ApplyMask(var_p, mask_rep).chunkedSum(N);
        sum_curv.setPrecision(0);
        AV curvature = inv_sigma2_kg + sum_curv;

        // All K*G Newton steps in a single division circuit.
        auto grad_scaled_b = (*(gradient * scale)).a2b();
        auto curv_b = curvature.a2b();
        auto step_b = (*grad_scaled_b) / (*curv_b);
        AV step = *(step_b->b2a());
        step.setPrecision(precision);

        AV step_clamped = ClampNewtonStep(step);
        step_clamped.setPrecision(0);
        u += step_clamped;
    }

    u.setPrecision(precision);
    return u;
}

// Laplace log-likelihood contribution of every (parameter point, group).
// Returns length K*G.
AV GroupLaplaceLogLikBatched(const BatchedDataset& data, const AV& beta_data, const AV& sigma2,
                             size_t num_points) {
    const size_t G = data.num_groups;
    const size_t N = data.obs_per_group;

    AV inv_sigma2 = SecureReciprocal(sigma2);
    inv_sigma2.setPrecision(0);

    AV x_beta = LinearPredictors(data, beta_data, num_points);
    x_beta.setPrecision(0);

    AV u_hat = ConditionalModeBatched(data, x_beta, inv_sigma2, num_points);
    u_hat.setPrecision(0);

    AV y_rep = data.y.cyclic_subset_reference(num_points);
    AV mask_rep = data.mask.cyclic_subset_reference(num_points);

    AV u_hat_rows = u_hat.repeated_subset_reference(N);
    u_hat_rows.setPrecision(0);
    AV eta = x_beta + u_hat_rows;
    eta.setPrecision(precision);

    AV p = Sigmoid(eta);
    p.setPrecision(0);
    AV log_one_plus_exp = LogOnePlusExp(eta);
    log_one_plus_exp.setPrecision(0);

    // conditional_log_lik = sum_j (y_j * eta_j - LogOnePlusExp(eta_j))
    AV y_copy = y_rep;
    y_copy.setPrecision(0);
    eta.setPrecision(0);
    AV y_eta = (*(y_copy * eta)) / scale;
    AV obs_cll = y_eta - log_one_plus_exp;
    obs_cll.setPrecision(precision);
    AV conditional_log_lik = ApplyMask(obs_cll, mask_rep).chunkedSum(N);
    conditional_log_lik.setPrecision(0);

    // curvature A = inv_sigma2 + sum_j p_j * (1 - p_j)
    AV one_minus_p = -p;
    one_minus_p += scale;
    AV var_p = (*(p * one_minus_p)) / scale;
    var_p.setPrecision(precision);
    AV sum_var_p = ApplyMask(var_p, mask_rep).chunkedSum(N);
    sum_var_p.setPrecision(0);
    AV inv_sigma2_kg = inv_sigma2.repeated_subset_reference(G);
    inv_sigma2_kg.setPrecision(0);
    AV curvature = inv_sigma2_kg + sum_var_p;

    // Gaussian prior penalty 0.5 * u_hat^2 / sigma2
    AV u_sq = (*(u_hat * u_hat)) / scale;
    AV penalty = (*(u_sq * inv_sigma2_kg)) / scale;
    AV penalty_half = (*(penalty * kHalf_scaled)) / scale;

    // 0.5 * (log(sigma2) + log(curvature))
    AV sigma2_kg = sigma2.repeated_subset_reference(G);
    sigma2_kg.setPrecision(precision);
    AV log_sigma2 = Log(sigma2_kg);
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

// Negative Laplace marginal log-likelihood for every parameter point.
// `params` is length K * (num_fixed + 1); the result is length K.
AV NegMarginalLogLikBatched(const BatchedDataset& data, const AV& params, size_t num_points) {
    UnpackedParameters unpacked = UnpackParametersBatched(params, num_points, data.num_fixed);

    AV per_group =
        GroupLaplaceLogLikBatched(data, unpacked.beta_data, unpacked.sigma2, num_points);
    per_group.setPrecision(precision);

    // Sum the G contributions belonging to each parameter point.
    AV total = per_group.chunkedSum(data.num_groups);
    AV neg_total = -total;
    neg_total.setPrecision(precision);
    return neg_total;
}

}  // namespace mixedeffects

}  // namespace cdough::regression