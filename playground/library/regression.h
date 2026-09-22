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


// =============================================================================
// Mixed-model inference (semantic task 0022)
// =============================================================================

// Fixed-effect covariance from the observed information of the Laplace marginal
// likelihood, by Schur complement.
//
// `information` is the (p+1) x (p+1) Hessian of the NEGATIVE marginal log
// likelihood at the optimum, ordered [beta_0 .. beta_{p-1}, s] where
// sigma^2 = exp(2s). `num_obs` is the padded observation count used to normalise.
//
// WHY A SCHUR COMPLEMENT AND NOT A FULL INVERSE. Two reasons, and the second is
// the one that matters here.
//
//   1. Only the beta block is reported, so inverting (p+1) x (p+1) does more work
//      than the answer needs.
//   2. CONDITIONING. Coefficients and the variance parameter live on different
//      scales, so the full matrix is markedly worse conditioned than its beta
//      block. NewtonSchulzInverse degrades sharply with condition number -- its
//      own calibration table in optimizer.h is still descending at 20 iterations
//      for kappa ~ 199 -- so handing it the better-conditioned matrix is not an
//      optimisation, it is what makes the result usable.
//
// The identity: for
//     H = [ H_bb   h_bs ]
//         [ h_bs'  h_ss ]
// the beta block of H^-1 is (H_bb - h_bs h_bs' / h_ss)^-1. Note this is NOT
// H_bb^-1 -- it accounts for the covariance between the coefficients and the
// variance parameter, and using H_bb^-1 instead understates the standard errors.
// That distinction is the classic error in mixed-model inference and is why the
// plaintext oracle in the test inverts the full matrix independently.
//
// The only division is one reciprocal of the SCALAR h_ss, which is far cheaper
// and better behaved than a matrix inverse of the same information.
//
// PRECONDITION: h_ss > 0, which holds at a genuine minimum of the negative log
// likelihood. It is not checked -- a non-positive curvature there means the
// optimiser did not reach a minimum, which is a defect upstream rather than a
// case to handle. Compare SecureSqrt's treatment of a non-positive variance.
//
// Obliviousness: straight-line. No branch on shared data, no open().
SMatrix CovarianceFromInformation(const SMatrix& information, size_t num_fixed,
                                  size_t num_obs) {
    const size_t dim = information.rows();
    assert(dim == num_fixed + 1);
    assert(information.cols() == dim);

    // Normalise by n before inverting: NewtonSchulzInverse needs an O(1)-scaled
    // operand, and at n observations the raw information is O(n). Undone below.
    const DataType inverse_n_scaled = std::llround(scale / static_cast<double>(num_obs));
    AV information_raw = Clone(information.data());
    information_raw.setPrecision(0);
    AV normalized = (*(information_raw * inverse_n_scaled)) / scale;
    normalized.setPrecision(precision);

    // Split into blocks. All three are index mappings, so they cost nothing.
    std::vector<size_t> bb_map(num_fixed * num_fixed);
    for (size_t i = 0; i < num_fixed; ++i) {
        for (size_t j = 0; j < num_fixed; ++j) {
            bb_map[i * num_fixed + j] = i * dim + j;
        }
    }
    std::vector<size_t> bs_map(num_fixed);
    for (size_t i = 0; i < num_fixed; ++i) {
        bs_map[i] = i * dim + num_fixed;
    }
    const std::vector<size_t> ss_map = {num_fixed * dim + num_fixed};

    AV h_bb = Clone(normalized.mapping_reference(bb_map));
    AV h_bs = Clone(normalized.mapping_reference(bs_map));
    AV h_ss = Clone(normalized.mapping_reference(ss_map));
    h_bb.setPrecision(precision);
    h_bs.setPrecision(precision);
    h_ss.setPrecision(precision);

    // Schur complement: H_bb - h_bs h_bs' / h_ss.
    AV inverse_h_ss = SecureReciprocal(h_ss);
    SMatrix outer = OuterProduct(h_bs, h_bs);
    SMatrix correction = ScaleMatrix(outer, inverse_h_ss);
    SMatrix schur = SMatrix(h_bb, num_fixed, num_fixed, false) - correction;
    schur.setPrecision(precision);

    SMatrix inverse = NewtonSchulzInverse(schur);

    // Undo the normalisation: (A/n)^-1 = n A^-1, so A^-1 = (A/n)^-1 / n.
    AV inverse_raw = Clone(inverse.data());
    inverse_raw.setPrecision(0);
    AV covariance_data = (*(inverse_raw * inverse_n_scaled)) / scale;

    SMatrix covariance(covariance_data, num_fixed, num_fixed, false);
    covariance.setPrecision(precision);
    return covariance;
}

// Observed information for the Laplace marginal likelihood, numerically.
//
// STAGE S1 of semantic task 0022: the numerical route, built first as a sanity
// check on the whole chain (oracle -> Hessian -> Schur -> inverse -> standard
// errors) before any new mathematics is introduced. Its accuracy is bounded by
// the objective's own error amplified by 1/h^2, so `step_scaled` is a calibrated
// constant, not a reused gradient step.
SMatrix ObservedInformation(const BatchedDataset& data, const AV& params,
                            DataType step_scaled) {
    const BatchedObjective objective = [&data](const AV& points, size_t num_points) {
        return NegMarginalLogLikBatched(data, points, num_points);
    };
    return NumericalHessianBatched(objective, params, step_scaled);
}

// Fixed-effect covariance at `params`, the composition of the two above.
SMatrix Covariance(const BatchedDataset& data, const AV& params, DataType step_scaled) {
    SMatrix information = ObservedInformation(data, params, step_scaled);
    return CovarianceFromInformation(information, data.num_fixed, data.total_rows());
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


// =============================================================================
// The fourteen regression fits: design-matrix construction, IRLS for the
// fixed-effects models (steps 6a/6b), and the flat ragged-cluster
// Laplace-approximated mixed model with an analytic gradient (steps 2 and 5).
// =============================================================================

#include <cassert>
#include <string>
#include <vector>

#include "../cohort.h"
#include "../linalg.h"
#include "../nodes.h"

namespace cdough::regression {

// Newton iterations in FlatConditionalMode's inner solve.
//
// The pipeline has always run three. library/primitives.h declares its own
// kNewtonIterations = 5 for the library's conditional-mode solver, and now that
// this tree includes that header the unqualified name would resolve to 5 and
// silently change both the pipeline's numbers and 67% more work in the hottest
// loop it has. The pipeline's own figure therefore gets its own name rather
// than inheriting a constant tuned for a different solver.
// Raised from 3 to 5 with the BFGS budget below (task 0016). The Laplace
// objective AND its analytic gradient are both evaluated at the conditional mode
// this loop finds, so an under-solved mode biases the function BFGS is
// minimising. Raising the outer budget without raising this one would let 5a/5b
// converge to a stationary point of the wrong objective and report success --
// worse than the honest non-convergence it replaced. Five is what
// library/regression.h's own mixedeffects::ConditionalModeBatched uses, via
// kNewtonIterations, for the identical solve.
constexpr int kFlatConditionalModeNewtonIterations = 5;

// =============================================================================
// Model specifications and design matrices
// =============================================================================

enum class TimeAxis { VisitNum, FuMonth };

struct ModelSpec {
    std::string step;                          // "2a" ... "6b"
    TimeAxis time = TimeAxis::VisitNum;
    bool covars = false;                       // newage + gender + hispanic
    bool interaction = false;                  // data_source main effect + data_source*time
    bool random_intercept = true;              // false only for steps 6a/6b
    bool drop_hispanic = false;                // model 5b on the UMass subset
    SystemScope scope = SystemScope::Any;
};

const char* TimeName(TimeAxis t) { return t == TimeAxis::VisitNum ? "visit_num" : "fu_month"; }

// Age is centred at a public constant purely for conditioning: at precision 16 an
// uncentred age column makes the entries of X^T W X span many orders of
// magnitude. Centring shifts only the intercept, and the shift is undone exactly
// when the coefficients are reported, so nothing about the fit changes.
const double kAgeCentre = 40.0;

// One model's design matrix, held row-major so that eta = X*beta is a single
// chunked dot product and X^T W X needs no transpose.
struct ModelData {
    size_t n_pad = 0;
    size_t p = 0;
    AV x;         // n_pad * p, row-major, scaled
    AV y;         // n_pad, 0/1
    AV row_mask;  // n_pad, 1 on rows this model actually uses
    std::vector<BV> keys;
    const ScanPlan* scan_plan = nullptr;  // owned by the cohort
    AV last_of_subject;
    std::vector<std::string> terms;
    long rows_used = 0;
    long rows_dropped_null_fu = 0;

    ModelData(EngineRef engine, size_t np, size_t width)
        : n_pad(np), p(width), x(np * width, engine), y(np, engine), row_mask(np, engine),
          last_of_subject(np, engine) {}
};

// `reveal_to` names the party entitled to this model's output; -1 publishes to
// everybody, which is the default and the two-owner behaviour. The only values
// this function opens are the two row counts it reports, so they follow the same
// rule as the coefficients.
ModelData BuildDesign(const SecureCohort& c, const ModelSpec& spec, int reveal_to = -1,
                      int party_id = 0) {
    EngineRef engine = c.valid.engine;

    // Column list, in the order SAS lists them.
    std::vector<std::string> terms{"Intercept", TimeName(spec.time)};
    if (spec.covars) {
        terms.push_back("newage");
        for (size_t i = 0; i + 1 < kGenderLevels.size(); ++i)
            terms.push_back(std::string("gender=") + kGenderNames[i]);
        if (!spec.drop_hispanic)
            for (size_t i = 0; i + 1 < kHispanicLevels.size(); ++i)
                terms.push_back("hispanic=" + std::to_string(kHispanicLevels[i]));
    }
    if (spec.interaction) {
        terms.push_back("data_source=UMass");
        terms.push_back(std::string("data_source=UMass * ") + TimeName(spec.time));
    }

    const size_t np = c.n_pad;
    const size_t p = terms.size();
    ModelData md(engine, np, p);
    md.terms = terms;
    md.keys = c.keys;
    md.scan_plan = &c.scan_plan;
    md.last_of_subject = c.last_of_subject;

    // Rows this model uses. A b-axis model silently drops every row whose
    // fu_month is null -- that is the source script's behaviour, and the count is
    // reported rather than left invisible.
    AV mask = Clone(c.valid);
    mask.setPrecision(0);
    if (spec.time == TimeAxis::FuMonth) {
        mask = *(mask * c.fu_present);
        mask.setPrecision(0);
    }
    md.row_mask = mask;

    AV time_col = spec.time == TimeAxis::VisitNum ? Clone(c.visit_num) : Clone(c.fu_month);
    time_col.setPrecision(0);

    AV age_centred = Clone(c.newage);
    age_centred.setPrecision(0);
    age_centred -= static_cast<DataType>(std::llround(kAgeCentre * scale));

    // Assemble the row-major design matrix column by column, writing into a
    // strided view of the flat buffer.
    std::vector<AV> cols;
    {
        AV ones(np, engine);
        ones.setPrecision(0);
        ones += DataType(scale);
        cols.push_back(ones);
    }
    cols.push_back(time_col);
    if (spec.covars) {
        cols.push_back(age_centred);
        // SAS CLASS makes the LAST sorted level the reference, so the final
        // level of each list is omitted.
        for (size_t i = 0; i + 1 < kGenderLevels.size(); ++i) {
            AV d = *(c.gender_is[i] * DataType(scale));
            d.setPrecision(0);
            cols.push_back(d);
        }
        if (!spec.drop_hispanic) {
            for (size_t i = 0; i + 1 < kHispanicLevels.size(); ++i) {
                AV d = *(c.hispanic_is[i] * DataType(scale));
                d.setPrecision(0);
                cols.push_back(d);
            }
        }
    }
    if (spec.interaction) {
        AV ds = *(c.umass * DataType(scale));
        ds.setPrecision(0);
        cols.push_back(ds);
        AV inter = *(*(ds * time_col) / scale);
        inter.setPrecision(0);
        cols.push_back(inter);
    }
    assert(cols.size() == p);

    md.x.setPrecision(0);
    for (size_t k = 0; k < p; ++k) {
        // Column k of a row-major n x p matrix is the stride-p view starting at k.
        AV dest = md.x.simple_subset_reference(k, p, (np - 1) * p + k);
        AV src = *(cols[k] * md.row_mask);  // zero the rows this model does not use
        dest = src;
    }
    md.x.setPrecision(0);

    AV y = *(c.sisa * md.row_mask);
    y.setPrecision(0);
    md.y = *(y * DataType(scale));
    md.y.setPrecision(0);

    md.rows_used = static_cast<long>(
        std::llround(OpenScalarToParty(MaskCount(md.row_mask), reveal_to, party_id, false)));
    AV dropped = c.valid - md.row_mask;
    md.rows_dropped_null_fu = static_cast<long>(
        std::llround(OpenScalarToParty(MaskCount(dropped), reveal_to, party_id, false)));
    return md;
}

// Gather the columns of a row-major n x p matrix end to end, which is the layout
// a chunked dot_product needs. mapping_reference is a public view: free.
AV ColumnGather(const AV& x_rm, size_t n, size_t p) {
    std::vector<cdough::VectorSizeType> map(n * p);
    for (size_t k = 0; k < p; ++k)
        for (size_t i = 0; i < n; ++i)
            map[k * n + i] = static_cast<cdough::VectorSizeType>(i * p + k);
    AV out = x_rm.mapping_reference(map);
    out.setPrecision(0);
    return out;
}

// eta = X * beta for X row-major (n x p) and beta a length-p vector.
AV LinearPredictor(const AV& x_rm, const AV& beta, size_t n, size_t p) {
    AV x_ = x_rm;
    x_.setPrecision(0);
    AV b_ = beta;
    b_.setPrecision(0);
    AV tiled = b_.cyclic_subset_reference(n);  // [b_0..b_{p-1}] repeated n times
    tiled.setPrecision(0);
    AV eta = *(*x_.dot_product(tiled, p) / scale);
    eta.setPrecision(0);
    return eta;
}

// X^T r, for r a length-n residual vector.
AV CrossProduct(const AV& x_rm, const AV& r, size_t n, size_t p) {
    AV cols = ColumnGather(x_rm, n, p);
    AV r_ = r;
    r_.setPrecision(0);
    AV tiled = r_.cyclic_subset_reference(p);  // r repeated p times: column-major order
    tiled.setPrecision(0);
    AV out = *(*cols.dot_product(tiled, n) / scale);
    out.setPrecision(0);
    return out;
}

// =============================================================================
// Fitting
// =============================================================================

struct FitResult {
    std::string step;
    SystemScope scope = SystemScope::Any;
    std::vector<std::string> terms;
    std::vector<double> estimate;
    std::vector<double> se;
    bool has_random = false;
    double sigma2 = 0.0;
    int iterations = 0;
    long rows_used = 0;
    long rows_dropped_null_fu = 0;
    bool converged = false;
    double hessian_condition = 0.0;  // mixed models only; 0 when not computed
    std::vector<std::string> notes;
};

const double kIrlsRidge = 1e-3;
constexpr int kIrlsIterations = 8;

// Steps 6a and 6b: an ordinary fixed-effects logistic regression, fitted by
// IRLS / Fisher scoring.
//
// No random effect, faithfully. The SAS declares subject_id in CLASS and then
// never uses it -- there is no RANDOM statement -- so PROC GLIMMIX treats every
// encounter as independent. That is reproduced here rather than corrected, and
// flagged in the output.
//
// IRLS rather than BFGS because the Hessian is analytic: convergence is
// quadratic, and the converged X^T W X IS the inverse asymptotic covariance, so
// the standard errors come out exact instead of approximated.
// ModelData -> logistic::Dataset, for the two fixed-effects models.
//
// CURRENTLY UNUSED, and kept deliberately. The 6a/6b inference migration this was
// written for was reverted after measurement (see FitLogisticIrls below for the
// numbers): logistic::Covariance's Newton-Schulz inverse does not converge on
// these design matrices. The adapter itself is correct and is the part that was
// difficult to get right -- see the mask note below -- so it stays for the retry,
// and for logistic::SeparationFlag / WaldStatistics / OddsRatio, none of which
// depend on that inverse and none of which the pipeline currently has.
//
// ModelData is already the single-group case of the flat ragged layout: md.x is
// the (n_pad x p) row-major design, md.y the scaled outcome, md.row_mask the rows
// this model uses. logistic::Dataset is mixedeffects::BatchedDataset with
// num_groups == 1, which is that same flat buffer -- so nothing is regrouped and
// nothing is padded to the largest cluster. That padding is only a problem for the
// twelve mixed models, and 6a/6b have no clusters at all: they are fixed-effects
// fits, one group by construction.
//
// The one real conversion is the mask convention. ModelData::row_mask is a raw
// 0/1 indicator -- BuildDesign multiplies by it with no following `/ scale` --
// while library/regression.h's ApplyMask computes (v * m) / scale and so needs a
// fixed-point 1.0.
logistic::Dataset AsLogisticDataset(const ModelData& md) {
    AV x = Clone(md.x);
    x.setPrecision(precision);

    AV y = Clone(md.y);
    y.setPrecision(precision);

    AV mask_raw = Clone(md.row_mask);
    mask_raw.setPrecision(0);
    AV mask = *(mask_raw * DataType(scale));
    mask.setPrecision(precision);

    return logistic::Dataset(std::move(x), std::move(y), std::move(mask), md.n_pad, md.p);
}

FitResult FitLogisticIrls(const ModelData& md, const ModelSpec& spec, int reveal_to = -1,
                          int party_id = 0) {
    EngineRef engine = md.y.engine;
    const size_t n = md.n_pad, p = md.p;

    AV beta(p, engine);
    beta.setPrecision(0);

    AV gram(p * p, engine);
    gram.setPrecision(0);

    for (int it = 0; it < kIrlsIterations; ++it) {
        AV eta = LinearPredictor(md.x, beta, n, p);
        eta = ClampAbs(eta, kMaxExpArg_scaled);  // keep Exp inside its saturation band
        eta.setPrecision(0);

        AV pr = Sigmoid(eta);
        pr.setPrecision(0);
        AV one_minus = -pr;
        one_minus += DataType(scale);
        AV w = *(*(pr * one_minus) / scale);
        w = *(w * md.row_mask);
        w.setPrecision(0);

        AV resid = md.y - pr;
        resid = *(resid * md.row_mask);
        resid.setPrecision(0);

        AV rhs = CrossProduct(md.x, resid, n, p);
        gram = Gram(md.x, w, n, p);
        AddRidge(gram, p, kIrlsRidge);

        AV delta = CholeskySolve(gram, rhs, p);
        delta = ClampNewtonStep(delta);  // the same damping the inner Newton uses
        delta.setPrecision(0);
        beta += delta;
    }

    // Standard errors from the Cholesky inverse of the ridge-augmented Gram.
    //
    // logistic::Covariance was tried here and REJECTED on measurement. It forms
    // (X' W X / n)^-1 / n with NewtonSchulzInverse, and Newton-Schulz at the
    // library's kMatrixInverseIterations = 14 does not converge on these design
    // matrices: model 6a's observed information carries a 1-norm condition
    // estimate around 1.2e4 -- the fit itself reports it, and warns that it is
    // close to singular. The measured result was garbage. Several terms collapsed
    // to an identical 0.00083923 (55 ULPs at precision 16, i.e. the inverse had
    // lost all of its information) and `data_source=UMass` came back as 2021.25
    // against a true 0.197. The Cholesky factorisation below is exact and handles
    // that conditioning, so it stays. Newton-Schulz is calibrated for O(1),
    // well-conditioned operands, which the library's own test cases are and these
    // are not.
    //
    // Only the DIAGONAL is opened: the standard errors need p variances, not the
    // whole p x p covariance, and mapping_reference is a public view so extracting
    // them costs nothing. That is p opened values where this used to open p^2.
    AV cov = SymmetricInverse(gram, p);

    std::vector<cdough::VectorSizeType> diagonal(p);
    for (size_t k = 0; k < p; ++k) diagonal[k] = static_cast<cdough::VectorSizeType>(k * p + k);
    AV variances = cov.mapping_reference(diagonal);
    variances.setPrecision(0);

    FitResult r;
    r.step = spec.step;
    r.scope = spec.scope;
    r.terms = md.terms;
    r.has_random = false;
    r.iterations = kIrlsIterations;
    r.rows_used = md.rows_used;
    r.rows_dropped_null_fu = md.rows_dropped_null_fu;
    r.converged = true;

    // The ONLY opens in this estimator are these two, so with `reveal_to` set the
    // fit discloses nothing to any other party -- not an intermediate, not a
    // branch. That is a property of IRLS, whose iteration count is fixed and
    // whose step needs no line search; the mixed models below cannot say it.
    const std::vector<double> beta_open = OpenToPartyDoubles(beta, reveal_to, party_id);
    const std::vector<double> var_open = OpenToPartyDoubles(variances, reveal_to, party_id);
    if (beta_open.empty()) return r;  // not this party's output
    for (size_t k = 0; k < p; ++k) {
        r.estimate.push_back(beta_open[k]);
        // The clamp is load-bearing: on a near-collinear design the inverse can
        // put a small negative on the diagonal, and these models do go
        // near-singular. A secure sqrt could not guard this without an extra
        // oblivious comparison, which is the other reason the plaintext sqrt of
        // an opened variance stays.
        r.se.push_back(std::sqrt(std::max(0.0, var_open[k])));
    }
    return r;
}

// =============================================================================
// Steps 2a / 2b / 5a / 5b: random-intercept logistic mixed model, Laplace
//
// The original code held the data as balanced ClusterGroups and looped over them
// one at a time. Real patients have a ragged number of encounters, so everything
// here works on the FLAT table instead: every cluster iterates in lockstep, and
// the per-cluster sums come from the segmented scans above. Cost is O(n log n)
// in the number of encounters, with no per-patient padding and no dependence on
// the largest cluster.
// =============================================================================

// Broadcast a 1-element vector across n rows. A free public view.
AV Broadcast(const AV& scalar, size_t n) {
    AV out = scalar.repeated_subset_reference(n);
    out.setPrecision(0);
    return out;
}

// sigma^2 is kept inside a public band: Log needs a strictly positive argument,
// and a runaway variance would push eta outside Exp's saturation range.
const DataType kSigma2Min_scaled = std::llround(1e-3 * scale);
const DataType kSigma2Max_scaled = std::llround(1e3 * scale);

// The conditional mode u_hat for EVERY cluster at once.
//
// u is carried broadcast on every row. It stays constant inside a cluster by
// construction -- each row of a cluster sees the same segmented gradient and
// curvature -- so no extra broadcast is needed to maintain it.
AV FlatConditionalMode(const ModelData& md, const AV& x_beta, const AV& inv_sigma2_row) {
    EngineRef engine = md.y.engine;
    const size_t n = md.n_pad;

    AV u(n, engine);
    u.setPrecision(0);

    for (int it = 0; it < kFlatConditionalModeNewtonIterations; ++it) {
        AV eta = x_beta + u;
        eta = ClampAbs(eta, kMaxExpArg_scaled);
        eta.setPrecision(0);

        AV pr = Sigmoid(eta);
        pr.setPrecision(0);
        AV omp = -pr;
        omp += DataType(scale);

        AV varp = *(*(pr * omp) / scale);
        varp = *(varp * md.row_mask);
        varp.setPrecision(0);

        AV resid = md.y - pr;
        resid = *(resid * md.row_mask);
        resid.setPrecision(0);

        // One call, both columns, with the group bits taken from the cohort's
        // precomputed plan rather than recomputed per level per call.
        std::vector<AV> in{resid, varp};
        std::vector<AV> out;
        out.emplace_back(n, engine);
        out.emplace_back(n, engine);
        if (g_use_cached_scans) {
            SegTotalPlanned(*md.scan_plan, in, out);
        } else {
            std::vector<BV> keys = md.keys;
            SegTotal(keys, in, out);
        }

        AV grad = out[0] - *(*(u * inv_sigma2_row) / scale);
        grad.setPrecision(0);
        AV curv = out[1] + inv_sigma2_row;
        curv.setPrecision(0);

        AV step = Div(grad, curv);
        step = ClampNewtonStep(step);
        step.setPrecision(0);
        u += step;
    }
    return u;
}

// Negative Laplace-approximated marginal log-likelihood: the BFGS objective.
//
// The per-cluster terms -0.5 log(sigma^2) - 0.5 log(A_i) and the prior penalty
// must be counted ONCE PER CLUSTER, not once per row. The forward scan leaves
// each cluster's totals on its last row, and multiplying by last_of_subject --
// which is zero everywhere else, and zero on the sentinel pad block -- picks
// exactly those rows out. A cluster with no usable rows contributes
// -0.5 log(sigma^2) - 0.5 log(1/sigma^2) = 0, so it drops out on its own.
// Counts calls to the marginal likelihood. The number of objective evaluations
// is the multiplier on everything else in a mixed-model fit, so it is the first
// thing to look at when the cost is wrong.
long g_objective_evaluations = 0;

// Negative Laplace-approximated marginal log-likelihood and, optionally, its
// ANALYTIC gradient.
//
// The gradient is worth deriving rather than differencing. Central differences
// cost 2*dim objective evaluations, which measurement showed to be the single
// largest multiplier in the whole program; the analytic form shares the
// conditional-mode search with the value and costs about one extra segmented
// scan. It is also strictly more accurate, because it never divides by a step.
//
// Derivation. Write g_i(u) for the exponent inside the integral, so that
//     l_i = g_i(u_i) - 0.5 log(sigma^2) - 0.5 log(A_i),
//     A_i = 1/sigma^2 + sum_j v_ij,        v_ij = p_ij (1 - p_ij),
// with u_i the conditional mode. Because u_i is a stationary point of g_i, the
// envelope theorem kills the du_i/dbeta terms in d g_i(u_i)/dbeta and it
// collapses to the partial derivative at fixed u:
//     d/dbeta_k [ g_i(u_i) ] = sum_j (y_ij - p_ij) x_ijk.
// The log(A_i) term gets no such cancellation. Differentiating A_i through both
// x and u_i, using
//     du_i/dbeta_k  = -(sum_j v_ij x_ijk) / A_i          (implicit function theorem)
//     dv_ij/deta_ij = v_ij (1 - 2 p_ij)
// gives dA_i/dbeta_k = S3_k - S1_k S2_i / A_i, and therefore
//     dl_i/dbeta_k = S4_k - (S3_k - S1_k S2_i / A_i) / (2 A_i)
// where, all sums being over the rows of cluster i,
//     S1_k = sum v_ij x_ijk,                 S2_i = sum v_ij (1 - 2 p_ij),
//     S3_k = sum v_ij (1 - 2 p_ij) x_ijk,    S4_k = sum (y_ij - p_ij) x_ijk.
// For the variance, parameterised as sigma^2 = exp(2s) to keep it positive:
//     du_i/ds  = 2 u_i / (sigma^2 A_i),
//     dA_i/ds  = -2/sigma^2 + S2_i du_i/ds,
//     dl_i/ds  = u_i^2/sigma^2 - 1 - (dA_i/ds) / (2 A_i).
//
// Every S is a per-cluster sum of a per-row quantity, so all 3p+2 of them come
// out of ONE forward segmented scan alongside the two the value already needs.
//
// Pass nullptr for `gradient_out` to get the value alone, which is what the
// line search wants.
AV FlatObjective(const ModelData& md, const std::vector<AV>& params,
                 std::vector<AV>* gradient_out) {
    ++g_objective_evaluations;

    EngineRef engine = md.y.engine;
    const size_t n = md.n_pad, p = md.p;
    const bool want_gradient = (gradient_out != nullptr);

    AV beta(p, engine);
    beta.setPrecision(0);
    for (size_t k = 0; k < p; ++k) {
        AV cell = Cell(beta, k);
        AV src = Clone(params[k]);
        src.setPrecision(0);
        cell = src;
    }

    AV s = Clone(params[p]);
    s.setPrecision(0);
    AV two_s = *(s * DataType(2));
    two_s.setPrecision(precision);
    AV sigma2 = Exp(two_s);
    sigma2 = ClampRange(sigma2, kSigma2Min_scaled, kSigma2Max_scaled);
    sigma2.setPrecision(0);

    AV inv_sigma2 = Recip(sigma2);
    inv_sigma2.setPrecision(0);
    AV log_sigma2 = Log(sigma2);
    log_sigma2.setPrecision(0);

    AV inv_sigma2_row = Broadcast(inv_sigma2, n);
    AV log_sigma2_row = Broadcast(log_sigma2, n);

    AV x_beta = LinearPredictor(md.x, beta, n, p);
    AV u = FlatConditionalMode(md, x_beta, inv_sigma2_row);

    AV eta = x_beta + u;
    eta = ClampAbs(eta, kMaxExpArg_scaled);
    eta.setPrecision(0);

    AV pr = Sigmoid(eta);
    pr.setPrecision(0);
    AV sp = LogOnePlusExp(eta);
    sp.setPrecision(0);

    AV y_eta = *(*(md.y * eta) / scale);
    AV cll_row = y_eta - sp;
    cll_row = *(cll_row * md.row_mask);
    cll_row.setPrecision(0);

    AV omp = -pr;
    omp += DataType(scale);
    AV varp_row = *(*(pr * omp) / scale);
    varp_row = *(varp_row * md.row_mask);
    varp_row.setPrecision(0);

    // Columns to scan. The first two are what the value needs; the rest are the
    // gradient's per-cluster sums, folded into the same pass.
    std::vector<AV> columns{cll_row, varp_row};
    AV resid(n, engine), skew(n, engine);
    if (want_gradient) {
        // skew = v * (1 - 2p), the derivative of v with respect to eta.
        AV one_minus_two_p = *(pr * DataType(2));
        one_minus_two_p = -one_minus_two_p;
        one_minus_two_p += DataType(scale);
        skew = *(*(varp_row * one_minus_two_p) / scale);
        skew.setPrecision(0);

        resid = md.y - pr;
        resid = *(resid * md.row_mask);
        resid.setPrecision(0);

        columns.push_back(skew);
        for (size_t k = 0; k < p; ++k) {
            AV xk = md.x.simple_subset_reference(k, p, (n - 1) * p + k);
            xk.setPrecision(0);
            columns.push_back(*(*(varp_row * xk) / scale));
            columns.back().setPrecision(0);
        }
        for (size_t k = 0; k < p; ++k) {
            AV xk = md.x.simple_subset_reference(k, p, (n - 1) * p + k);
            xk.setPrecision(0);
            columns.push_back(*(*(skew * xk) / scale));
            columns.back().setPrecision(0);
        }
        for (size_t k = 0; k < p; ++k) {
            AV xk = md.x.simple_subset_reference(k, p, (n - 1) * p + k);
            xk.setPrecision(0);
            columns.push_back(*(*(resid * xk) / scale));
            columns.back().setPrecision(0);
        }
    }

    std::vector<AV> pre;
    pre.reserve(columns.size());
    for (size_t k = 0; k < columns.size(); ++k) pre.emplace_back(n, engine);
    if (g_use_cached_scans) {
        SegScanPlanned(*md.scan_plan, columns, pre, SegDirection::Forward);
    } else {
        std::vector<BV> keys = md.keys;
        SegScan(keys, columns, pre, SegDirection::Forward);
    }

    AV a_row = pre[1] + inv_sigma2_row;
    a_row.setPrecision(0);
    AV log_a = Log(a_row);
    log_a.setPrecision(0);

    AV u_sq = *(*(u * u) / scale);
    AV pen = *(*(u_sq * inv_sigma2_row) / scale);
    AV pen_half = *(pen / DataType(2));

    AV log_terms = log_sigma2_row + log_a;
    AV log_half = *(log_terms / DataType(2));

    AV per_row = pre[0] - pen_half - log_half;
    per_row.setPrecision(0);

    AV contrib = *(per_row * md.last_of_subject);
    AV total = contrib.chunkedSum(n);
    total.setPrecision(0);

    if (want_gradient) {
        // One reciprocal serves the whole gradient. It is computed on every row
        // even though only the last row of each cluster is used, because a
        // vectorised division over n rows is cheaper than compacting first.
        AV inv_a = Recip(a_row);
        inv_a.setPrecision(0);
        AV s2 = pre[2];  // S2_i, the skew total

        gradient_out->clear();
        gradient_out->reserve(p + 1);

        for (size_t k = 0; k < p; ++k) {
            AV s1 = pre[3 + k];
            AV s3 = pre[3 + p + k];
            AV s4 = pre[3 + 2 * p + k];

            // dA/dbeta_k = S3_k - S1_k * S2 / A
            AV s1_s2 = *(*(s1 * s2) / scale);
            AV shift = *(*(s1_s2 * inv_a) / scale);
            AV d_a = s3 - shift;
            d_a.setPrecision(0);

            // dl/dbeta_k = S4_k - (dA/dbeta_k) / (2 A)
            AV correction = *(*(d_a * inv_a) / scale);
            AV half_correction = *(correction / DataType(2));
            AV row = s4 - half_correction;
            row.setPrecision(0);

            AV masked = *(row * md.last_of_subject);
            AV summed = masked.chunkedSum(n);
            AV negated = -summed;  // gradient of the NEGATIVE log-likelihood
            negated.setPrecision(precision);
            gradient_out->push_back(negated);
        }

        // du/ds = 2 u / (sigma^2 A)
        AV du_ds = *(*(u * inv_sigma2_row) / scale);
        du_ds = *(*(du_ds * inv_a) / scale);
        du_ds = *(du_ds * DataType(2));
        du_ds.setPrecision(0);

        // dA/ds = -2/sigma^2 + S2 * du/ds
        AV d_a_ds = *(*(s2 * du_ds) / scale);
        AV two_inv = *(inv_sigma2_row * DataType(2));
        d_a_ds -= two_inv;
        d_a_ds.setPrecision(0);

        // dl/ds = u^2/sigma^2 - 1 - (dA/ds) / (2 A)
        AV correction = *(*(d_a_ds * inv_a) / scale);
        AV half_correction = *(correction / DataType(2));
        AV row = pen - half_correction;  // pen is u^2/sigma^2
        row -= DataType(scale);
        row.setPrecision(0);

        AV masked = *(row * md.last_of_subject);
        AV summed = masked.chunkedSum(n);
        AV negated = -summed;
        negated.setPrecision(precision);
        gradient_out->push_back(negated);
    }

    AV neg = -total;
    neg.setPrecision(precision);
    return neg;
}

// Value-only wrapper, for the places that do not want a gradient.
AV FlatNegMarginalLogLik(const ModelData& md, const std::vector<AV>& params) {
    return FlatObjective(md, params, nullptr);
}

// BFGS iteration budget for the mixed models (task 0016).
//
// Was 12, a flat cap that suited the 3-parameter models and starved the
// 7-parameter ones. BFGS starts from an identity inverse-Hessian and applies one
// rank-2 update per iteration, so it needs roughly `dim` iterations before it has
// any usable curvature at all. At 12 the budget was 4.0x dim for 2a/2b, which
// converged at 9-12, but only 1.7x dim for 5a/5b, which all six exhausted it with
// gradient norms between 96 and 190 -- truncated mid-descent, not fitted.
//
// 60 is ~8.6x dim for 5a/5b. The easy models do not pay for it: they exit early
// on a failed line search well before the cap, so a larger cap only costs time
// where it actually binds.
constexpr int kGlmmBfgsIterations = 60;

// Step for the numerical observed-information Hessian below. A second difference
// divides by h^2, so it amplifies the objective's fixed-point noise by 1/h^2;
// the truncation error meanwhile grows as h^2. With an objective good to about
// 1e-4 at precision 16, the balance sits near h = eps^(1/4) ~ 0.1.
const double kHessianStep = 0.1;

// Above this the observed-information matrix is close enough to singular that
// its inverse should not be trusted at all. The threshold is deliberately high:
// with the Hessian built from differences of the ANALYTIC gradient, standard
// errors measured against an accurate double-precision reference stayed within
// 8% at condition numbers around 2e4, so warning there would be crying wolf.
// The earlier objective-difference Hessian was 17-46% off at the same
// conditioning, which is what the old, much lower threshold was compensating
// for.
const double kHessianConditionWarn = 1e6;

// Standard errors for the mixed models, from the observed information at the
// optimum.
//
// The BFGS inverse-Hessian approximation is NOT good enough for this. With a
// dozen iterations and up to eight parameters it can leave whole directions
// untouched, and its diagonal then reports a variance of almost exactly 1 --
// an artefact of the identity it was initialised with rather than a standard
// error. Instead the Hessian is formed by finite differences of the objective
// and inverted directly.
//
// The differences are opened and the small dense algebra is done in plaintext.
// That is consistent with the choice already made for p-values: the curvature of
// the log-likelihood at the optimum is exactly what a published standard error
// discloses, so evaluating it under MPC would protect nothing, and it avoids
// amplifying fixed-point noise through a second difference and a matrix inverse.
// `condition_out` receives a 1-norm condition-number estimate for the Hessian.
// It matters: in a near-collinear direction the Hessian is nearly singular, and
// inverting it amplifies the objective's fixed-point noise without limit. The
// standard errors on such terms are not trustworthy and the caller says so.
//
// The Hessian is built from CENTRAL DIFFERENCES OF THE ANALYTIC GRADIENT, not
// from second differences of the objective. That matters twice over. It costs
// 2*dim gradient evaluations instead of 1 + 2*dim + dim(dim-1)/2 objective
// evaluations -- about 2.5x less at dim = 7 -- and, more importantly, a first
// difference divides by h rather than h^2, so it amplifies the fixed-point noise
// by one factor of 1/h instead of two. That is exactly the term that made the
// standard errors on near-collinear directions unreliable.
//
// The differences are opened and the small dense algebra is done in plaintext,
// consistent with the choice already made for p-values: the curvature of the
// log-likelihood at the optimum is precisely what a published standard error
// discloses, so evaluating it under MPC would protect nothing.
// The gradient opens below steer nothing: the axes, the step and the trip count
// are all public and fixed, so unlike the line search inside MinimizeBFGS these
// values can go to one party without any other party needing to see them.
std::vector<double> ObservedInformationSE(const ValueGradFn& objective,
                                          const std::vector<AV>& optimum,
                                          size_t num_reported,
                                          double* condition_out = nullptr,
                                          int reveal_to = -1, int party_id = 0) {
    const size_t dim = optimum.size();
    const double h = kHessianStep;

    auto gradient_at = [&](size_t axis, double delta) {
        std::vector<AV> point;
        point.reserve(dim);
        for (size_t k = 0; k < dim; ++k) {
            AV v = Clone(optimum[k]);
            v.setPrecision(0);
            if (k == axis) v += static_cast<DataType>(std::llround(delta * scale));
            v.setPrecision(precision);
            point.push_back(v);
        }
        std::vector<AV> grad;
        objective(point, &grad);
        std::vector<double> out;
        out.reserve(dim);
        for (AV& g : grad) out.push_back(OpenScalarToParty(g, reveal_to, party_id));
        return out;
    };

    // Column k of the Hessian is d(grad)/d(x_k).
    std::vector<double> hess(dim * dim, 0.0);
    for (size_t k = 0; k < dim; ++k) {
        const std::vector<double> forward = gradient_at(k, h);
        const std::vector<double> backward = gradient_at(k, -h);
        for (size_t j = 0; j < dim; ++j)
            hess[j * dim + k] = (forward[j] - backward[j]) / (2.0 * h);
    }
    // The true Hessian is symmetric; averaging the two estimates of each
    // off-diagonal entry halves the noise for free.
    for (size_t j = 0; j < dim; ++j) {
        for (size_t k = j + 1; k < dim; ++k) {
            const double avg = 0.5 * (hess[j * dim + k] + hess[k * dim + j]);
            hess[j * dim + k] = hess[k * dim + j] = avg;
        }
    }

    // Invert by Gauss-Jordan with partial pivoting. Unlike the secure Cholesky,
    // this runs on public numbers, so pivoting costs nothing and the matrix does
    // not have to be positive definite for the routine to return something --
    // a non-positive diagonal in the result is reported as an unavailable
    // standard error rather than silently square-rooted.
    std::vector<double> m(hess), inv(dim * dim, 0.0);
    for (size_t k = 0; k < dim; ++k) inv[k * dim + k] = 1.0;
    for (size_t c = 0; c < dim; ++c) {
        size_t piv = c;
        for (size_t r = c + 1; r < dim; ++r)
            if (std::abs(m[r * dim + c]) > std::abs(m[piv * dim + c])) piv = r;
        for (size_t k = 0; k < dim; ++k) {
            std::swap(m[c * dim + k], m[piv * dim + k]);
            std::swap(inv[c * dim + k], inv[piv * dim + k]);
        }
        const double d = m[c * dim + c];
        if (std::abs(d) < 1e-12) {
            std::fill(inv.begin(), inv.end(), std::numeric_limits<double>::quiet_NaN());
            break;
        }
        for (size_t k = 0; k < dim; ++k) {
            m[c * dim + k] /= d;
            inv[c * dim + k] /= d;
        }
        for (size_t r = 0; r < dim; ++r) {
            if (r == c) continue;
            const double f = m[r * dim + c];
            for (size_t k = 0; k < dim; ++k) {
                m[r * dim + k] -= f * m[c * dim + k];
                inv[r * dim + k] -= f * inv[c * dim + k];
            }
        }
    }

    if (condition_out != nullptr) {
        // 1-norm condition estimate: max column sum of H times that of H^-1.
        double norm_h = 0.0, norm_inv = 0.0;
        for (size_t c = 0; c < dim; ++c) {
            double col_h = 0.0, col_inv = 0.0;
            for (size_t r = 0; r < dim; ++r) {
                col_h += std::abs(hess[r * dim + c]);
                col_inv += std::abs(inv[r * dim + c]);
            }
            norm_h = std::max(norm_h, col_h);
            norm_inv = std::max(norm_inv, col_inv);
        }
        *condition_out = std::isfinite(norm_h * norm_inv)
                             ? norm_h * norm_inv
                             : std::numeric_limits<double>::infinity();
    }

    std::vector<double> se;
    for (size_t k = 0; k < num_reported; ++k) {
        const double var = inv[k * dim + k];
        se.push_back(std::isfinite(var) && var > 0.0
                         ? std::sqrt(var)
                         : std::numeric_limits<double>::quiet_NaN());
    }
    return se;
}

// `reveal_to` restricts this fit's OUTPUT -- coefficients, standard errors and
// the variance component -- to one party. It does NOT make the whole fit
// single-party: MinimizeBFGS opens objective values and directional derivatives
// to steer its Armijo line search, and every party has to follow the same branch
// for the trip counts to stay public, so those scalars remain visible to all.
// That leak predates this parameter (tasks/0009 records it as a known follow-up);
// steps 6a/6b, fitted by IRLS, have no such intermediate.
FitResult FitGlmmLaplace(const ModelData& md, const ModelSpec& spec, int party_id,
                         int reveal_to = -1) {
    EngineRef engine = md.y.engine;
    const size_t p = md.p;
    const size_t dim = p + 1;  // beta plus s, where sigma = exp(s)

    // Start from beta = 0 and sigma = 1 (s = 0).
    std::vector<AV> x0;
    for (size_t k = 0; k < dim; ++k) {
        AV v(1, engine);
        v.setPrecision(precision);
        x0.push_back(v);
    }

    ValueGradFn objective = [&md](const std::vector<AV>& params,
                                  std::vector<AV>* gradient) -> AV {
        return FlatObjective(md, params, gradient);
    };

    OptResult opt = MinimizeBFGS(objective, x0, kGlmmBfgsIterations);

    FitResult r;
    r.step = spec.step;
    r.scope = spec.scope;
    r.terms = md.terms;
    r.has_random = true;
    r.iterations = opt.iterations;
    r.converged = opt.converged;
    r.rows_used = md.rows_used;
    r.rows_dropped_null_fu = md.rows_dropped_null_fu;

    for (size_t k = 0; k < p; ++k)
        r.estimate.push_back(OpenScalarToParty(opt.params[k], reveal_to, party_id));
    double condition = 0.0;
    r.se = ObservedInformationSE(objective, opt.params, p, &condition, reveal_to, party_id);
    r.hessian_condition = condition;
    const double s_hat = OpenScalarToParty(opt.params[p], reveal_to, party_id);
    r.sigma2 = std::exp(2.0 * s_hat);

    r.notes.push_back(
        "standard errors come from central differences of the analytic gradient at the "
        "optimum, not from the BFGS inverse-Hessian approximation");
    if (!opt.hessian_updated)
        r.notes.push_back(
            "no BFGS curvature update was accepted, so the optimiser may have stopped early");
    {
        // Always report the conditioning: it is the single most useful number
        // for judging how much to trust the standard errors on the weakly
        // determined terms, which here are the intercept and any near-collinear
        // dummy such as hispanic.
        std::ostringstream note;
        note << "observed-information 1-norm condition estimate " << std::scientific
             << std::setprecision(2) << condition;
        if (!(condition < kHessianConditionWarn))
            note << " -- close to singular; treat every standard error from this fit as "
                    "indicative only";
        r.notes.push_back(note.str());
    }
    return r;
}

}  // namespace cdough::regression
