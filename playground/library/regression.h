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

// =============================================================================
// logistic: fixed-effects binary logistic regression
// =============================================================================
//
// The model `mixedeffects` above fits with the random intercept removed:
//
//     logit P(y_i = 1) = x_i' beta
//
// This is not reachable by degenerating the mixed-effects objective. That
// objective routes the variance through Exp(2s) and SecureReciprocal, so
// sigma -> 0 is not representable and the Laplace correction does not vanish
// smoothly. It needs its own objective, which is what this namespace provides.
//
// Everything expensive is reused: the design matrix, the mask, and the batched
// linear predictor all come from `mixedeffects` with a single group, so
// LinearPredictors applies verbatim and the (k, g, j) layout collapses to (k, i).
//
// Introduced by task 0019 to support step 6 of notes/MPC_analysis_Mar26.sas.
// =============================================================================
namespace logistic {

// One padded dataset of `num_obs` observations on `num_fixed` covariates.
//
// Wraps a single-group BatchedDataset so that LinearPredictors and ApplyMask are
// reused unchanged, and additionally caches X^T.
//
// WHY CACHE X^T. The matmul kernel takes a row-major left operand and a
// column-wise right operand, and a column-wise (n x p) matrix reads element
// (i, j) from data[j * n + i] -- which is the row-major buffer of X^T, not of X
// (semantic topic 0001, claim C-09). So the transposed buffer serves both
// orientations this namespace needs: as a row-major (p x n) matrix it is X^T,
// the left operand of the gradient and the information matrix; as a column-wise
// (n x p) matrix it is X itself, their right operand. Building it is a pure index
// permutation, so it costs no communication, and doing it once in the constructor
// keeps it out of the per-iteration path.
//
// PRIVACY: `num_obs` (the padded width) and `num_fixed` are public. The real row
// count stays private behind `mask`, provided the padding bound is chosen
// independently of the data -- the same caveat as mixedeffects::BatchedDataset.
// Note that `num_fixed` being public means the number of levels of each
// categorical predictor leaks through the column count.
struct Dataset {
    mixedeffects::BatchedDataset design;  // single group: num_groups == 1
    AV xt_data;                           // p x n row-major, i.e. X^T

    Dataset(AV x_in, AV y_in, AV mask_in, size_t num_obs, size_t num_fixed)
        : design(std::move(x_in), std::move(y_in), std::move(mask_in), 1, num_obs, num_fixed),
          xt_data(TransposeData(design.x_data, num_obs, num_fixed)) {
        xt_data.setPrecision(precision);
    }

    size_t num_obs() const { return design.obs_per_group; }
    size_t num_fixed() const { return design.num_fixed; }
};

// Negative log-likelihood of every parameter point.
//
//     -log L(beta) = sum_i [ log(1 + exp(eta_i)) - y_i * eta_i ],   eta = X beta
//
// `beta` is length K * p with (k, j) -> k * p + j; the result is length K.
//
// This is GroupLaplaceLogLikBatched with the conditional-mode Newton loop, the
// Gaussian prior penalty, and the 0.5 * (log sigma^2 + log A) correction removed,
// leaving one matmul and one LogOnePlusExp per evaluation.
AV NegLogLikBatched(const Dataset& data, const AV& beta, size_t num_points) {
    const size_t rows = data.num_obs();
    assert(beta.size() == num_points * data.num_fixed());

    AV eta = mixedeffects::LinearPredictors(data.design, beta, num_points);
    eta.setPrecision(precision);

    AV log_one_plus_exp = LogOnePlusExp(eta);
    log_one_plus_exp.setPrecision(0);

    AV y_rep = data.design.y.cyclic_subset_reference(num_points);
    AV y_copy = y_rep;
    y_copy.setPrecision(0);
    eta.setPrecision(0);
    AV y_eta = (*(y_copy * eta)) / scale;

    AV obs_neg_log_lik = log_one_plus_exp - y_eta;
    obs_neg_log_lik.setPrecision(precision);

    AV mask_rep = data.design.mask.cyclic_subset_reference(num_points);
    AV total = mixedeffects::ApplyMask(obs_neg_log_lik, mask_rep).chunkedSum(rows);
    total.setPrecision(precision);
    return total;
}

// Ridge penalty lambda * sum_{j >= 1} beta_j^2 for every parameter point, with
// the intercept (column 0) excluded. Returns length K.
//
// `lambda_scaled` DEFAULTS TO ZERO throughout this namespace, and that default is
// deliberate: PROC GLIMMIX applies no penalty, so a non-zero default would make
// these estimates disagree with the SAS fit they are meant to reproduce. The
// lever exists because separation has no oblivious remedy -- see SeparationFlag.
//
// Two properties a caller must respect when switching it on. The penalty is
// applied on each column's own scale, so columns should be standardised first or
// a covariate measured in years is shrunk far less than a 0/1 indicator. And a
// penalised fit does not carry the usual asymptotic distribution, so the Wald
// quantities below lose their nominal coverage.
AV L2Penalty(const AV& beta, size_t num_points, size_t num_fixed, DataType lambda_scaled) {
    AV penalty(num_points, beta.engine);
    penalty.setPrecision(precision);
    if (lambda_scaled == 0 || num_fixed <= 1) {
        return penalty;
    }

    const size_t num_penalized = num_fixed - 1;
    std::vector<size_t> slope_map(num_points * num_penalized);
    for (size_t k = 0; k < num_points; ++k) {
        for (size_t j = 1; j < num_fixed; ++j) {
            slope_map[k * num_penalized + (j - 1)] = k * num_fixed + j;
        }
    }

    // Clone materializes, dropping the mapping.
    AV slopes = Clone(beta.mapping_reference(slope_map));
    slopes.setPrecision(0);
    AV squares = (*(slopes * slopes)) / scale;
    AV sum_squares = squares.chunkedSum(num_penalized);
    AV scaled = (*(sum_squares * lambda_scaled)) / scale;
    scaled.setPrecision(precision);
    return scaled;
}

// The objective handed to MinimizeBFGSBatched: negative log-likelihood, plus the
// ridge penalty when one is requested.
AV PenalizedNegLogLikBatched(const Dataset& data, const AV& beta, size_t num_points,
                             DataType lambda_scaled = 0) {
    AV neg_log_lik = NegLogLikBatched(data, beta, num_points);
    if (lambda_scaled == 0) {
        return neg_log_lik;
    }

    AV penalty = L2Penalty(beta, num_points, data.num_fixed(), lambda_scaled);
    neg_log_lik.setPrecision(precision);
    penalty.setPrecision(precision);
    AV total = neg_log_lik + penalty;
    total.setPrecision(precision);
    return total;
}

// Analytic gradient at a single parameter point:
//
//     d(-log L)/d(beta) = X' (sigmoid(X beta) - y),   plus 2 * lambda * beta
//
// One Sigmoid over n elements and one matmul, so unlike the central-difference
// gradient its cost does not grow with the parameter count.
AV LogisticGradient(const Dataset& data, const AV& beta, DataType lambda_scaled = 0) {
    const size_t rows = data.num_obs();
    const size_t num_fixed = data.num_fixed();
    assert(beta.size() == num_fixed);

    AV eta = mixedeffects::LinearPredictors(data.design, beta, 1);
    eta.setPrecision(precision);
    AV probability = Sigmoid(eta);
    probability.setPrecision(precision);

    AV y_copy = Clone(data.design.y);
    y_copy.setPrecision(precision);
    AV residual = probability - y_copy;
    residual.setPrecision(precision);
    AV masked_residual = mixedeffects::ApplyMask(residual, data.design.mask);

    // X' r: a row-major (p x n) left operand (the cached transpose) against the
    // residual as a column-wise (n x 1) right operand. An n x 1 matrix has
    // identical row-major and column-wise layouts, so the residual needs no
    // transpose of its own.
    SMatrix xt(data.xt_data, num_fixed, rows, false);
    xt.setPrecision(precision);
    SMatrix residual_col(masked_residual, rows, 1, true);
    residual_col.setPrecision(precision);

    SMatrix product = xt.matrixRightMultiplyWithColumnMatrixVectorized(residual_col);
    product.setPrecision(precision);
    AV gradient = product.data();
    gradient.setPrecision(precision);

    if (lambda_scaled == 0) {
        return gradient;
    }

    // d(lambda * sum_{j>=1} beta_j^2)/d(beta) = 2 * lambda * beta, with the
    // intercept zeroed. The pattern is public but has to be secret-shared: the
    // library has no elementwise multiply against a public vector.
    cdough::Vector<DataType> pattern(num_fixed, scale);
    pattern[0] = 0;
    AV selector = beta.engine.secret_share_a(pattern, 0, precision);
    selector.setPrecision(0);

    AV beta_raw = Clone(beta);
    beta_raw.setPrecision(0);
    AV slopes = (*(beta_raw * selector)) / scale;
    AV penalty_gradient = (*(slopes * (2 * lambda_scaled))) / scale;
    penalty_gradient.setPrecision(precision);

    AV total = gradient + penalty_gradient;
    total.setPrecision(precision);
    return total;
}

// =============================================================================
// Wald inference
// =============================================================================

// Observed information X' W X / n at `beta`, with W = diag(p_i (1 - p_i)) masked
// so padded rows contribute nothing.
//
// Returned already divided by n. NewtonSchulzInverse requires an O(1)-scaled
// operand -- at n observations the raw Gram matrix is O(n), and its initializer
// A'/||A||_F^2 would underflow to zero -- and the `X^T W X / N` case in that
// operator's calibration table is exactly this matrix.
SMatrix ObservedInformationOverN(const Dataset& data, const AV& beta) {
    const size_t rows = data.num_obs();
    const size_t num_fixed = data.num_fixed();

    AV eta = mixedeffects::LinearPredictors(data.design, beta, 1);
    eta.setPrecision(precision);
    AV probability = Sigmoid(eta);
    probability.setPrecision(0);

    AV one_minus_probability = -probability;
    one_minus_probability += scale;
    AV weight = (*(probability * one_minus_probability)) / scale;
    weight.setPrecision(precision);
    weight = mixedeffects::ApplyMask(weight, data.design.mask);

    // W X: scale row i of X by weight_i. Repeating each weight p times lines it
    // up with the row-major (n x p) buffer, so this is one vectorized multiply.
    AV weight_rows = weight.repeated_subset_reference(num_fixed);
    weight_rows.setPrecision(0);
    AV x_raw = Clone(data.design.x_data);
    x_raw.setPrecision(0);
    AV weighted_x = (*(x_raw * weight_rows)) / scale;

    // A = (W X)' X. W is diagonal, so (W X)' X == X' W X.
    // Left: (W X)' as a row-major (p x n) matrix -- one local permutation.
    // Right: X as a column-wise (n x p) matrix, which is the cached X^T buffer
    //        reinterpreted, so it is free.
    SMatrix weighted_x_t(TransposeData(weighted_x, rows, num_fixed), num_fixed, rows, false);
    weighted_x_t.setPrecision(precision);
    SMatrix x_col(data.xt_data, rows, num_fixed, true);
    x_col.setPrecision(precision);

    SMatrix information = weighted_x_t.matrixRightMultiplyWithColumnMatrixVectorized(x_col);
    information.setPrecision(precision);

    const DataType inverse_n_scaled = std::llround(scale / static_cast<double>(rows));
    AV information_raw = Clone(information.data());
    information_raw.setPrecision(0);
    AV normalized = (*(information_raw * inverse_n_scaled)) / scale;

    SMatrix result(normalized, num_fixed, num_fixed, false);
    result.setPrecision(precision);
    return result;
}

// Parameter covariance (X' W X)^-1 at `beta`.
//
// Inverting the normalized matrix and undoing the scaling afterwards:
// (A/n)^-1 = n A^-1, so A^-1 = (A/n)^-1 / n.
SMatrix Covariance(const Dataset& data, const AV& beta) {
    const size_t rows = data.num_obs();
    const size_t num_fixed = data.num_fixed();

    SMatrix normalized = ObservedInformationOverN(data, beta);
    SMatrix inverse = NewtonSchulzInverse(normalized);

    const DataType inverse_n_scaled = std::llround(scale / static_cast<double>(rows));
    AV inverse_raw = Clone(inverse.data());
    inverse_raw.setPrecision(0);
    AV covariance_data = (*(inverse_raw * inverse_n_scaled)) / scale;

    SMatrix covariance(covariance_data, num_fixed, num_fixed, false);
    covariance.setPrecision(precision);
    return covariance;
}

// Per-coefficient standard errors: the square roots of the covariance diagonal.
// Extracting the diagonal is an index mapping, so it costs no communication.
AV StandardErrors(const SMatrix& covariance) {
    const size_t num_fixed = covariance.rows();
    assert(covariance.cols() == num_fixed);

    std::vector<size_t> diagonal_map(num_fixed);
    for (size_t j = 0; j < num_fixed; ++j) {
        diagonal_map[j] = j * num_fixed + j;
    }

    AV variances = Clone(covariance.data().mapping_reference(diagonal_map));
    variances.setPrecision(precision);
    return SecureSqrt(variances);
}

// Wald statistics z_j = beta_j / se_j.
//
// The division is computed on |beta| and the sign reattached afterwards, because
// the non-restoring division circuit assumes a non-negative numerator. Handing it
// a negative coefficient is undefined behaviour inside the circuit, not merely a
// wrong result -- the same hazard SecureReciprocal documents.
AV WaldStatistics(const AV& beta, const AV& standard_errors) {
    assert(beta.size() == standard_errors.size());

    AV beta_raw = Clone(beta);
    beta_raw.setPrecision(0);
    AV sign = *(beta_raw.gtez());
    sign.setPrecision(0);
    AV two_sign = *(sign * DataType(2));
    two_sign -= DataType(1);
    two_sign.setPrecision(0);
    AV abs_beta = *(two_sign * beta_raw);
    abs_beta.setPrecision(0);

    AV standard_errors_raw = Clone(standard_errors);
    standard_errors_raw.setPrecision(0);

    auto numerator_b = (*(abs_beta * scale)).a2b();
    auto denominator_b = standard_errors_raw.a2b();
    auto quotient_b = (*numerator_b) / (*denominator_b);
    AV abs_z = *(quotient_b->b2a());
    abs_z.setPrecision(0);

    AV z = *(two_sign * abs_z);
    z.setPrecision(precision);
    return z;
}

// Odds ratio exp(beta) with its confidence limits exp(beta +/- z * se).
// `z_critical` is public (1.96 for a 95% interval), so it needs no protection.
struct OddsRatioEstimate {
    AV ratio;
    AV lower;
    AV upper;
};

OddsRatioEstimate OddsRatio(const AV& beta, const AV& standard_errors, double z_critical = 1.96) {
    assert(beta.size() == standard_errors.size());

    const DataType z_scaled = std::llround(z_critical * scale);
    AV standard_errors_raw = Clone(standard_errors);
    standard_errors_raw.setPrecision(0);
    AV margin = (*(standard_errors_raw * z_scaled)) / scale;
    margin.setPrecision(precision);

    AV center = Clone(beta);
    center.setPrecision(precision);
    AV lower_log = center - margin;
    lower_log.setPrecision(precision);
    AV upper_log = center + margin;
    upper_log.setPrecision(precision);

    return OddsRatioEstimate{Exp(center), Exp(lower_log), Exp(upper_log)};
}

// 1 iff some |eta_i| has reached `bound_scaled` at the fitted `beta`, as a
// 1-element raw 0/1 intended to be opened.
//
// WHY THIS EXISTS. Without a random intercept, a rare outcome and a sparse
// indicator column drive beta unbounded. Past kMaxExpArg the secure Exp saturates,
// the likelihood is flat, the gradient vanishes, and BFGS reports convergence on a
// meaningless estimate -- a failure that is silent and looks exactly like success.
//
// In plaintext the analyst sees the warning or the absurd estimate, cross-tabulates
// the offending variable against the outcome, and drops it; that is what the
// comment in notes/MPC_analysis_Mar26.sas records for `hispanic`. That remedy is
// "inspect the data, then decide", which is what MPC forbids, so the safeguard
// protecting the plaintext analysis has no secure analogue.
//
// This is the cheapest substitute: ONE bit, not the value of max |eta|. Opening it
// is a declassification and belongs in the obliviousness contract, but it discloses
// strictly less than a magnitude would, and without it the fit is uninterpretable.
AV SeparationFlag(const Dataset& data, const AV& beta,
                  DataType bound_scaled = kMaxExpArg_scaled) {
    AV eta = mixedeffects::LinearPredictors(data.design, beta, 1);
    eta.setPrecision(precision);
    return AnyAbsAtLeast(eta, bound_scaled);
}

}  // namespace logistic

}  // namespace cdough::regression