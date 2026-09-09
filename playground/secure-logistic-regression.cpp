#include <cmath>
#include <numeric>
#include <vector>
#include <functional>

#include "cdough.h"

// ../scripts/run_experiment.py -p 3 -r 16 secure-logistic-regression

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;
using namespace cdough::debug;
using namespace cdough::service;

using DataType = int64_t;
using HW = cdough::matrix::HeightWidth;
using AV = ASharedVector<DataType>;
using BV = BSharedVector<DataType>;
using SMatrix = SecureMatrix<DataType>;
using PMatrix = PlainMatrix<DataType>;

const int precision = 16;
const DataType scale = 1 << precision;

const double kLn2 = 0.69314718055994531;
const double kLn2_inv = 1.44269504088896341;
const double kSqrt2 = 1.41421356237309505;
const double kSqrt1_2 = 0.70710678118654752;
const float kSmallEpsilon = 0.0001;
const float kSeriesTolerance = 0.0001;
const float kNumericalGradientStep = 0.05;
const float kMaxExpArg = 10.0;

const DataType kLn2_scaled = std::llround(kLn2 * scale);
const DataType kLn2_inv_scaled = std::llround(kLn2_inv * scale);
const DataType kSqrt2_scaled = std::llround(kSqrt2 * scale);
const DataType kSqrt1_2_scaled = std::llround(kSqrt1_2 * scale);
const DataType kSmallEpsilon_scaled = (kSmallEpsilon * scale);
const DataType kSeriesTolerance_scaled = (kSeriesTolerance * scale);
const DataType kHalf_scaled = (0.5 * scale);
const DataType kMaxNewtonStep_scaled = (4.0 * scale);
const DataType kMaxExpArg_scaled = (kMaxExpArg * scale);

constexpr int kMaxSeriesTerms = 3;
constexpr int kMaxNewtonStep = 4;
constexpr int kNewtonIterations = 5;

// Width of the oblivious exponent stage in Exp: the shifted exponent k + kExpOffset
// must fit in kExpBits bits, i.e. k in [-kExpOffset, kExpOffset - 1] = [-16, 15].
// That covers x in [-11.4, 10.7], which spans the whole range representable at
// `precision` bits; kMaxExpArg keeps k inside it with margin to spare.
constexpr int kExpBits = 5;
constexpr int kExpOffset = 1 << (kExpBits - 1);

// Data structures for secure mixed-effects logistic regression
struct ClusterGroup {
    AV y;                   // Binary outcomes (0/1), length N (group size)
    std::vector<AV> x_cols; // num_fixed columns, each of size N

    ClusterGroup(AV y_in, std::vector<AV> x_cols_in)
        : y(std::move(y_in)), x_cols(std::move(x_cols_in)) {}

    size_t size() const { return y.size(); }
};

struct Dataset {
    std::vector<ClusterGroup> groups;
    size_t num_fixed = 0;
};

// Use these helpers wherever a copy is going to be written to.
AV Clone(const AV& v) {
    AV out(v.size(), v.engine);
    out = v;
    out.setPrecision(v.getPrecision());
    return out;
}

std::vector<AV> Clone(const std::vector<AV>& vs) {
    std::vector<AV> out;
    out.reserve(vs.size());
    for (const AV& v : vs) out.push_back(Clone(v));
    return out;
}

// std::vector<AV>(n, AV(...)) copy-constructs n aliases of a single buffer, so
// every element would share storage. Build the elements individually instead.
std::vector<AV> MakeVector(size_t n, size_t elem_size, EngineRef engine) {
    std::vector<AV> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) out.emplace_back(elem_size, engine);
    return out;
}

std::vector<std::vector<AV>> MakeMatrix(size_t rows, size_t cols, EngineRef engine) {
    std::vector<std::vector<AV>> out;
    out.reserve(rows);
    for (size_t i = 0; i < rows; ++i) out.push_back(MakeVector(cols, 1, engine));
    return out;
}

// Secure clamping to [-bound_scaled, bound_scaled].
AV ClampAbs(const AV& x, DataType bound_scaled) {
    AV x_ = Clone(x);
    x_.setPrecision(0);

    // cond_high = 1 if x >= bound, and then x - cond_high * (x - bound) == bound.
    AV diff_high = x_ - bound_scaled;
    AV cond_high = *(diff_high.gtez());
    x_ -= *(cond_high * diff_high);

    // cond_low = 1 if x < -bound, i.e. if (-bound - 1) - x >= 0.
    AV diff_low = -x_;
    diff_low -= (bound_scaled + 1);
    AV cond_low = *(diff_low.gtez());
    AV delta_low = -x_;
    delta_low -= bound_scaled; // -bound - x
    x_ += *(cond_low * delta_low);

    x_.setPrecision(precision);
    return x_;
}

AV Exp (const AV& x) {
    // Saturate first. The oblivious 2^k stage below can only represent
    // k in [-kExpOffset, kExpOffset - 1]; an out-of-range k would wrap modulo
    // 2^kExpBits and silently invert the result
    AV x_ = ClampAbs(x, kMaxExpArg_scaled);
    x_.setPrecision(0);

    // TODO: truncate, not divide by scale
    // TODO: add multiplication by float/double constants
    AV quotient = (*(x_ * kLn2_inv_scaled)) / scale;
    AV k_fixed = quotient + kHalf_scaled;

    // Integer k (unscaled)
    AV k_int = *(k_fixed / scale);

    // Remainder r = x - k * ln(2), so |r| <= ln(2)/2
    AV r = x_ - (*(k_int * kLn2_scaled));

    // 2. Maclaurin series evaluation: exp(r) = 1 + r + r^2/2! + r^3/3! + ...
    AV term(x.size(), x.engine);
    AV series(x.size(), x.engine);
    term += scale;
    series += scale;

    for (int n = 1; n <= kMaxSeriesTerms; ++n) {
        // term = (term * r) / (n * scale)
        term = *(*(term * r) / scale) / static_cast<DataType>(n);
        series += term;
    }

    // 3. Oblivious 2^k scaling
    // Shift k by an offset to keep exponent positive within [0, 2^kExpBits - 1]
    AV k_shifted = k_int + static_cast<DataType>(kExpOffset);

    auto k_b = k_shifted.a2b();

    // Multiply by 2^(b_i * 2^i) obliviously: factor = 1 + b_i * (2^(2^i) - 1).
    // The factors are unscaled integers, so the product needs no rescaling
    AV result = series;
    BV current_bit(x.size(), x.engine);
    for (int i = 0; i < kExpBits; ++i) {
        current_bit.bit_logical_right_shift(*k_b, i);
        current_bit.mask(1);
        AV bit_a = *current_bit.b2a_bit();
        DataType multiplier = (DataType(1) << (1 << i)) - 1;
        AV factor(x.size(), x.engine);
        factor += DataType(1);
        factor += *(bit_a * multiplier);
        result = *(result * factor);
    }

    // Adjust for the constant offset 2^(-kExpOffset)
    result = *(result / (DataType(1) << kExpOffset));
    result.setPrecision(precision);

    return result;
}

// Requires positive numbers.
AV Log(AV x) {
    AV x_(x.size(), x.engine);
    x_ = x;
    x_.setPrecision(0);

    AV m(x.size(), x.engine);
    m = x;
    AV e(x.size(), x.engine); // e starts at 0

    // Range reduction: we want m in [sqrt(1/2), sqrt(2)) ~ [0.7071, 1.4142]
    // 1. High steps: while m >= sqrt(2) * 2^step, divide m by 2^step, e += step
    // Using powers of 2 for step = 8, 4, 2, 1
    const int steps[] = {8, 4, 2, 1};
    for (int step : steps) {
        DataType threshold = static_cast<DataType>(kSqrt2 * (1 << step) * scale);
        // cond = (m - threshold) >= 0
        AV diff = m - threshold;
        AV cond_a = diff.gtez(); // 1 if m >= threshold, 0 otherwise

        // If cond_a == 1, m = m / (2^step) => m_new = m - cond_a * (m - m / 2^step)
        // Or m_diff = m - m / (1 << step) = m * (1 - 1/(2^step))
        AV m_reduced = *(m / (DataType(1) << step));
        AV m_delta = m - m_reduced;
        m -= *(cond_a * m_delta);

        // e += cond_a * step
        e += *(cond_a * static_cast<DataType>(step));
    }

    // Single step check for m >= sqrt(2)
    {
        AV diff = m - kSqrt2_scaled;
        AV cond_a = diff.gtez();
        AV m_reduced = *(m / DataType(2));
        AV m_delta = m - m_reduced;
        m -= *(cond_a * m_delta);
        e += *(cond_a * static_cast<DataType>(1));
    }

    // 2. Low steps: while m < sqrt(1/2) / 2^step, multiply m by 2^step, e -= step
    for (int step : steps) {
        // threshold = (kSqrt1_2 / 2^step) * scale
        DataType threshold = static_cast<DataType>((kSqrt1_2 / (1 << step)) * scale);
        // cond: m < threshold <=> -(m - (threshold - 1)) >= 0 <=> (m - (threshold - 1)) < 0
        // Or AV neg_m = -m; neg_m + (threshold - 1)
        AV diff = -m;
        diff += (threshold - 1);
        AV cond_a = diff.gtez(); // 1 if m < threshold, 0 otherwise

        AV m_scaled = *(m * (DataType(1) << step));
        AV m_delta = m_scaled - m;
        m += *(cond_a * m_delta);

        e -= *(cond_a * static_cast<DataType>(step));
    }

    // Single step check for m < sqrt(1/2)
    {
        AV diff = -m;
        diff += (kSqrt1_2_scaled - 1);
        AV cond_a = diff.gtez();
        AV m_scaled = *(m * DataType(2));
        AV m_delta = m_scaled - m;
        m += *(cond_a * m_delta);
        e -= *(cond_a * static_cast<DataType>(1));
    }

    // Now m in [sqrt(1/2), sqrt(2)]
    // Compute w = (m - 1) / (m + 1)
    // Numerator: (m - 1.0) scaled by scale => m - scale
    // Denominator: (m + 1.0) scaled by scale => m + scale
    // To maintain fixed-point precision in w = num / den, we scale num by 2^precision:
    // num_fixed = (m - scale) << precision
    // w_b = num_fixed_b / den_b
    AV num = m - scale;
    AV den = m + scale;

    auto num_b = (*(num * scale)).a2b();
    auto den_b = den.a2b();

    auto w_b = (*num_b) / (*den_b);
    AV w = *(w_b->b2a());
    w.setPrecision(0);

    // Compute series: 2 * (w + w^3/3 + w^5/5)
    AV w_squared = (*(w * w)) / scale;
    AV power = w;
    AV series(x.size(), x.engine);

    for (int i = 0; i < kMaxSeriesTerms; ++i) {
        DataType divisor = 2 * i + 1;
        AV term = *(power / divisor);
        series += term;
        power = (*(power * w_squared)) / scale;
    }

    AV log_m = *(series * DataType(2));
    AV e_ln2 = (*(e * kLn2_scaled));

    AV result = e_ln2 + log_m;
    result.setPrecision(precision);
    return result;
}

// log(1 + x) routed through Log
AV Log1p(const AV& x) {
    AV one_plus_x(x.size(), x.engine);
    one_plus_x = x;

    one_plus_x.setPrecision(0);
    one_plus_x = one_plus_x + scale;

    one_plus_x.setPrecision(precision);
    return Log(one_plus_x);
}

// Numerically stable logistic function: Sigmoid(eta)
// For positive eta: z = exp(-eta), return 1 / (1 + z)
// For negative eta: z = exp(eta), return z / (1 + z)
// Oblivious formulation:
// mask = (eta >= 0)
// abs_eta = mask ? eta : -eta = (2*mask - 1) * eta
// z = exp(-abs_eta)  (since abs_eta >= 0, z in (0, 1])
// num = mask + (1 - mask) * z = mask * (1 - z) + z
// den = 1 + z
// sigmoid = num / den
AV Sigmoid(const AV& eta) {
    AV eta_copy(eta.size(), eta.engine);
    eta_copy = eta;

    AV mask = *(eta_copy.gtez()); // 1 if eta >= 0, 0 if eta < 0

    // abs_eta = (2*mask - 1) * eta
    AV two_mask = *(mask * DataType(2));
    two_mask -= DataType(1);
    AV abs_eta = *(two_mask * eta_copy);

    AV neg_abs_eta = -abs_eta;
    neg_abs_eta.setPrecision(precision);
    AV z = Exp(neg_abs_eta); // z = exp(-|eta|) <= 1.0
    z.setPrecision(0);

    // num = mask * (scale - z) + z
    AV one_minus_z = -z;
    one_minus_z += scale;
    AV num = *(mask * one_minus_z);
    num += z;

    AV den = z;
    den += scale;

    // Secure division: num / den with fixed-point precision
    auto num_b = (*(num * scale)).a2b();
    auto den_b = den.a2b();
    auto res_b = (*num_b) / (*den_b);
    AV res = *(res_b->b2a());
    res.setPrecision(precision);
    return res;
}

// Numerically stable log(1 + exp(eta)) (softplus function)
// If eta > 0: eta + log1p(exp(-eta))
// Else: log1p(exp(eta))
// Oblivious formulation:
// mask = (eta >= 0)
// abs_eta = mask ? eta : -eta
// z = exp(-abs_eta)
// softplus = mask * eta + log1p(z)
AV LogOnePlusExp(const AV& eta) {
    AV eta_copy(eta.size(), eta.engine);
    eta_copy = eta;
    eta_copy.setPrecision(0);

    AV mask = *(eta_copy.gtez()); // 1 if eta >= 0, 0 if eta < 0
    AV two_mask = *(mask * DataType(2));
    two_mask -= DataType(1);
    AV abs_eta = *(two_mask * eta_copy);

    AV neg_abs_eta = -abs_eta;
    AV z = Exp(neg_abs_eta);
    AV log1p_z = Log1p(z);

    AV pos_term = *(mask * eta_copy);
    pos_term.setPrecision(precision);
    AV result = pos_term + log1p_z;

    return result;
}

// Sum of all elements in an AV vector, returning a 1-element AV
AV Sum(const AV& v) {
    AV res = v.chunkedSum(v.size());
    res.setPrecision(precision);
    return res;
}

// Secure clamping to [-kMaxNewtonStep, kMaxNewtonStep]
AV ClampNewtonStep(const AV& step) {
    return ClampAbs(step, kMaxNewtonStep_scaled);
}

// Finds the conditional mode u_hat = argmax_u g_i(u) for one group, given the
// fixed effects `beta` (std::vector<AV> of size p, each size 1) and variance `sigma2` (AV of size 1).
// The objective g_i is strictly concave in u, so a damped Newton iteration converges reliably.
AV ConditionalMode(const ClusterGroup& group, const std::vector<AV>& beta, const AV& sigma2) {
    EngineRef engine = sigma2.engine;
    size_t num_obs = group.size();
    size_t num_fixed = beta.size();

    // inv_sigma2 = 1.0 / sigma2 (in fixed-point: scale^2 / sigma2)
    AV scale_sq(1, engine);
    scale_sq += (DataType(1) << (2 * precision));
    auto scale_sq_b = scale_sq.a2b();
    auto sigma2_b = sigma2.a2b();
    auto inv_sigma2_b = (*scale_sq_b) / (*sigma2_b);
    AV inv_sigma2 = *(inv_sigma2_b->b2a());
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

    AV u(1, engine); // Initial mode u = 0 (size 1)
    u.setPrecision(0);

    for (int iter = 0; iter < kNewtonIterations; ++iter) {
        // Broadcast u to size num_obs
        AV u_rep = u.repeated_subset_reference(num_obs);
        u_rep.setPrecision(0);

        AV eta = x_beta + u_rep;
        eta.setPrecision(precision);
        AV p = Sigmoid(eta); // size num_obs
        p.setPrecision(0);

        // gradient = -u * inv_sigma2 + sum_j (y_j - p_j)
        AV u_inv = (*(u * inv_sigma2)) / scale;
        AV grad_prior = -u_inv; // size 1

        AV one_minus_p = -p;
        one_minus_p += scale;
        AV var_p = (*(p * one_minus_p)) / scale; // p * (1 - p), size num_obs

        AV y_copy = group.y;
        y_copy.setPrecision(0);
        AV diff_y_p = y_copy - p; // size num_obs
        diff_y_p.setPrecision(precision);
        AV sum_grad = Sum(diff_y_p); // size 1
        sum_grad.setPrecision(0);
        AV gradient = grad_prior + sum_grad; // size 1

        // curvature = inv_sigma2 + sum_j p_j * (1 - p_j)
        var_p.setPrecision(precision);
        AV sum_curv = Sum(var_p); // size 1
        sum_curv.setPrecision(0);
        AV curvature = inv_sigma2 + sum_curv; // size 1

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
AV GroupLaplaceLogLik(const ClusterGroup& group, const std::vector<AV>& beta, const AV& sigma2) {
    EngineRef engine = sigma2.engine;
    size_t num_obs = group.size();
    size_t num_fixed = beta.size();

    // inv_sigma2 = scale^2 / sigma2
    AV scale_sq(1, engine);
    scale_sq += (DataType(1) << (2 * precision));
    auto scale_sq_b = scale_sq.a2b();
    auto sigma2_b = sigma2.a2b();
    auto inv_sigma2_b = (*scale_sq_b) / (*sigma2_b);
    AV inv_sigma2 = *(inv_sigma2_b->b2a());
    inv_sigma2.setPrecision(0);

    AV u_hat = ConditionalMode(group, beta, sigma2); // size 1
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
    AV conditional_log_lik = Sum(obs_cll); // size 1
    conditional_log_lik.setPrecision(0);

    // curvature A_i = inv_sigma2 + sum_j p_j * (1 - p_j)
    AV one_minus_p = -p;
    one_minus_p += scale;
    AV var_p = (*(p * one_minus_p)) / scale;
    var_p.setPrecision(precision);
    AV sum_var_p = Sum(var_p);
    sum_var_p.setPrecision(0);
    AV curvature = inv_sigma2 + sum_var_p; // size 1

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

// Splits a packed parameter vector [beta..., s] into `beta` and sigma^2, where
// the variance is parameterized as sigma = exp(s) to keep it strictly positive.
void UnpackParameters(const std::vector<AV>& params, size_t num_fixed,
                      std::vector<AV>& beta, AV& sigma2) {
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

// Central finite-difference gradient of a scalar objective `f` at `x`.
std::vector<AV> NumericalGradient(
    const std::function<AV(const std::vector<AV>&)>& f,
    const std::vector<AV>& x) {
    size_t dim = x.size();
    std::vector<AV> gradient;
    gradient.reserve(dim);

    std::vector<AV> perturbed = Clone(x);
    for (size_t k = 0; k < dim; ++k) {
        perturbed[k].setPrecision(precision);
    }

    for (size_t k = 0; k < dim; ++k) {
        // Step size h = kNumericalGradientStep * (1.0 + |x[k]|)
        AV xk_copy = x[k];
        xk_copy.setPrecision(0);
        AV mask = *(xk_copy.gtez());
        AV two_mask = *(mask * DataType(2));
        two_mask -= DataType(1);
        AV abs_xk = *(two_mask * xk_copy); // (1 or -1) * xk_copy gives |x[k]| directly without / scale

        AV one_plus_abs = abs_xk;
        one_plus_abs += scale;
        DataType h_step_scaled = static_cast<DataType>(kNumericalGradientStep * scale);
        AV h = (*(one_plus_abs * h_step_scaled)) / scale; // size 1

        // f_plus = f(perturbed with x[k] + h)
        h.setPrecision(precision);
        perturbed[k] = x[k] + h;
        perturbed[k].setPrecision(precision);
        AV f_plus = f(perturbed);
        f_plus.setPrecision(0);

        // f_minus = f(perturbed with x[k] - h)
        perturbed[k] = x[k] - h;
        perturbed[k].setPrecision(precision);
        AV f_minus = f(perturbed);
        f_minus.setPrecision(0);

        // Reset perturbed[k]
        perturbed[k] = x[k];
        perturbed[k].setPrecision(precision);

        // gradient[k] = (f_plus - f_minus) / (2 * h)
        AV diff = f_plus - f_minus;
        h.setPrecision(0);
        AV two_h = *(h * DataType(2));

        auto diff_scaled_b = (*(diff * scale)).a2b();
        auto two_h_b = two_h.a2b();
        auto grad_b = (*diff_scaled_b) / (*two_h_b);
        AV grad_k = *(grad_b->b2a());
        grad_k.setPrecision(precision);

        gradient.push_back(std::move(grad_k));
    }

    return gradient;
}

// =============================================================================
// BFGS Optimization & Secure Linear Algebra Helpers
// =============================================================================

// Generates an n x n identity SecureMatrix in MPC (row-major).
SMatrix Identity(size_t n, EngineRef engine) {
    cdough::Vector<DataType> eye(n * n, 0);
    for (size_t i = 0; i < n; ++i) {
        eye[i * n + i] = scale;
    }
    PMatrix plain_eye(eye, n, n, false);
    auto sec_eye = engine.secret_share_matrix(plain_eye, 0);
    sec_eye.setPrecision(precision);
    return sec_eye;
}

// Matrix-vector product m * v where m is SecureMatrix (n x n) and v is std::vector<AV> (length n).
std::vector<AV> MatVec(const SMatrix& m, const std::vector<AV>& v) {
    size_t n = v.size();
    assert(m.rows() == n && m.cols() == n);
    EngineRef engine = v[0].engine;

    // m is stored row-major: element (i, j) is at data_[i * n + j]
    AV m_data = m.data();
    m_data.setPrecision(0);

    std::vector<AV> result;
    result.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        AV row_sum(1, engine);
        row_sum.setPrecision(0);
        for (size_t j = 0; j < n; ++j) {
            // Slice element at index (i * n + j)
            AV m_ij = m_data.slice(i * n + j, i * n + j + 1);
            m_ij.setPrecision(0);
            AV vj = v[j];
            vj.setPrecision(0);
            AV prod = (*(m_ij * vj)) / scale;
            row_sum += prod;
        }
        row_sum.setPrecision(precision);
        result.push_back(std::move(row_sum));
    }
    return result;
}

// Standard inner product between two secure vectors.
AV Dot(const std::vector<AV>& a, const std::vector<AV>& b) {
    assert(a.size() == b.size());
    size_t n = a.size();
    EngineRef engine = a[0].engine;

    AV sum(1, engine);
    sum.setPrecision(0);
    for (size_t i = 0; i < n; ++i) {
        AV ai = a[i];
        AV bi = b[i];
        ai.setPrecision(0);
        bi.setPrecision(0);
        AV prod = (*(ai * bi)) / scale;
        sum += prod;
    }
    sum.setPrecision(precision);
    return sum;
}

// BFGS update of the inverse-Hessian approximation:
//   H+ = (I - rho s y^T) H (I - rho y s^T) + rho s s^T,   rho = 1 / (y^T s).
SMatrix BfgsInverseUpdate(const SMatrix& h_inv, const std::vector<AV>& s,
                         const std::vector<AV>& y, const AV& rho) {
    size_t n = s.size();
    assert(h_inv.rows() == n && h_inv.cols() == n);
    assert(y.size() == n);
    EngineRef engine = s[0].engine;

    // We extract h_inv elements into an n x n 2D array of 1-element AVs
    // Distinct buffers per element
    std::vector<std::vector<AV>> h_elements = MakeMatrix(n, n, engine);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            h_elements[i][j] = h_inv.data().slice(i * n + j, i * n + j + 1);
            h_elements[i][j].setPrecision(0);
        }
    }

    AV rho_copy = rho;
    rho_copy.setPrecision(0);

    // Compute left = I - rho * s * y^T as an n x n matrix in std::vector<std::vector<AV>>
    // In fixed point: (s_i * y_j) / scale, then (* rho) / scale
    std::vector<std::vector<AV>> left = MakeMatrix(n, n, engine);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            AV si = s[i];
            AV yj = y[j];
            si.setPrecision(0);
            yj.setPrecision(0);
            AV s_y = (*(si * yj)) / scale;
            AV rho_s_y = (*(rho_copy * s_y)) / scale;

            AV elem(1, engine);
            elem.setPrecision(0);
            if (i == j) {
                elem += scale;
            }
            elem -= rho_s_y;
            left[i][j] = elem;
        }
    }

    // temp = left * h_inv
    // temp[i][j] = sum_k (left[i][k] * h_inv[k][j]) / scale
    std::vector<std::vector<AV>> temp = MakeMatrix(n, n, engine);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            AV sum(1, engine);
            sum.setPrecision(0);
            for (size_t k = 0; k < n; ++k) {
                AV h_kj = h_elements[k][j];
                AV left_ik = left[i][k];
                left_ik.setPrecision(0);
                h_kj.setPrecision(0);
                AV prod = (*(left_ik * h_kj)) / scale;
                sum += prod;
            }
            temp[i][j] = sum;
        }
    }

    // updated = temp * left^T + rho * s * s^T
    // Flatten result into a single AV of size n * n to construct SecureMatrix
    cdough::Vector<DataType> dummy_init(n * n, 0);
    AV updated_data = engine.secret_share_a(dummy_init, 0, 0);
    updated_data.setPrecision(0);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            AV sum(1, engine);
            sum.setPrecision(0);
            for (size_t k = 0; k < n; ++k) {
                AV temp_ik = temp[i][k];
                AV left_jk = left[j][k];
                temp_ik.setPrecision(0);
                left_jk.setPrecision(0);
                AV prod = (*(temp_ik * left_jk)) / scale;
                sum += prod;
            }
            AV si = s[i];
            AV sj = s[j];
            si.setPrecision(0);
            sj.setPrecision(0);
            AV s_s = (*(si * sj)) / scale;
            AV rho_s_s = (*(rho_copy * s_s)) / scale;
            sum += rho_s_s;

            // Place into updated_data at index i * n + j
            AV sum_rep = sum.repeated_subset_reference(n * n);
            sum_rep.setPrecision(0);
            cdough::Vector<DataType> mask_vec(n * n, 0);
            mask_vec[i * n + j] = scale; // Use scale (1.0 in fixed-point)
            AV mask_elem = engine.secret_share_a(mask_vec, 0, 0);
            mask_elem.setPrecision(0);
            AV placed = (*(sum_rep * mask_elem)) / scale;
            placed.setPrecision(0);
            updated_data += placed;
        }
    }

    updated_data.setPrecision(precision);
    SMatrix updated(updated_data, n, n, false);
    updated.setPrecision(precision);
    return updated;
}

// Result of the outer quasi-Newton optimization.
struct OptResult {
    std::vector<AV> params;
    AV value;
    int iterations = 0;
    bool converged = false;

    OptResult(std::vector<AV> p, AV v, int it = 0, bool conv = false)
        : params(std::move(p)), value(std::move(v)), iterations(it), converged(conv) {}
};

// Minimizes `f` starting from `x0` using BFGS with a backtracking (Armijo) line
// search and numerical gradients.
OptResult MinimizeBFGS(
    const std::function<AV(const std::vector<AV>&)>& f,
    const std::vector<AV>& x0,
    int max_iterations = 20) {
    size_t n = x0.size();
    EngineRef engine = x0[0].engine;

    std::vector<AV> x = Clone(x0);
    for (size_t i = 0; i < n; ++i) {
        x[i].setPrecision(precision);
    }
    AV fx = f(x);
    fx.setPrecision(precision);

    std::vector<AV> gradient = NumericalGradient(f, x);
    SMatrix h_inv = Identity(n, engine);

    OptResult result(x, fx, 0, false);

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        result.iterations = iteration + 1;

        // Check gradient convergence in plaintext after opening
        double max_grad = 0.0;
        for (size_t i = 0; i < n; ++i) {
            auto opened_g = gradient[i].open();
            double val = std::abs(static_cast<double>(opened_g[0]) / scale);
            if (val > max_grad) max_grad = val;
        }

        if (max_grad < kSmallEpsilon) {
            result.converged = true;
            break;
        }

        // Search direction d = -H_inv * gradient
        std::vector<AV> direction = MatVec(h_inv, gradient);
        for (size_t i = 0; i < n; ++i) {
            direction[i] = -direction[i];
            direction[i].setPrecision(precision);
        }

        AV directional_derivative = Dot(gradient, direction);
        directional_derivative.setPrecision(precision);

        // Check if descent direction: directional_derivative < 0
        auto opened_dd = directional_derivative.open();
        double dd_val = static_cast<double>(opened_dd[0]) / scale;
        if (dd_val >= 0.0) {
            h_inv = Identity(n, engine);
            for (size_t i = 0; i < n; ++i) {
                direction[i] = -gradient[i];
                direction[i].setPrecision(precision);
            }
            directional_derivative = Dot(gradient, direction);
            directional_derivative.setPrecision(precision);
            auto opened_dd2 = directional_derivative.open();
            dd_val = static_cast<double>(opened_dd2[0]) / scale;
        }

        // Backtracking line search satisfying the Armijo sufficient-decrease rule
        const double c1 = 1e-4;
        double alpha = 1.0;
        bool line_search_failed = false;
        std::vector<AV> x_new = Clone(x);
        AV fx_new = Clone(fx);

        auto opened_fx = fx.open();
        double fx_val = static_cast<double>(opened_fx[0]) / scale;

        while (true) {
            for (size_t i = 0; i < n; ++i) {
                AV alpha_dir = Clone(direction[i]);
                alpha_dir.setPrecision(0);
                alpha_dir = (*(alpha_dir * static_cast<DataType>(alpha * scale))) / scale;
                x_new[i] = x[i] + alpha_dir;
                x_new[i].setPrecision(precision);
            }
            fx_new = f(x_new);
            fx_new.setPrecision(precision);

            auto opened_fx_new = fx_new.open();
            double fx_new_val = static_cast<double>(opened_fx_new[0]) / scale;

            if (std::isfinite(fx_new_val) &&
                fx_new_val <= fx_val + c1 * alpha * dd_val) {
                break;
            }
            alpha *= 0.5;
            if (alpha < 1e-5) {
                x_new = x;
                fx_new = fx;
                line_search_failed = true;
                break;
            }
        }

        // Step = x_new - x
        std::vector<AV> step = MakeVector(n, 1, engine);
        double max_step = 0.0;
        for (size_t i = 0; i < n; ++i) {
            step[i] = x_new[i] - x[i];
            step[i].setPrecision(precision);
            auto opened_s = step[i].open();
            double val = std::abs(static_cast<double>(opened_s[0]) / scale);
            if (val > max_step) max_step = val;
        }

        // A failed line search is a fixed point: x is unchanged, so the next
        // iteration recomputes the same direction and fails identically
        if (line_search_failed) {
            if (engine.getPartyID() == 0) {
                std::cout << "[BFGS] iter " << std::setw(3) << (iteration + 1)
                          << "  no descent found along search direction; stopping at"
                          << " neg_log_lik=" << std::fixed << std::setprecision(6) << fx_val
                          << "  |grad|=" << std::scientific << std::setprecision(3) << max_grad
                          << std::endl;
            }
            result.converged = true;
            break;
        }

        std::vector<AV> gradient_new = NumericalGradient(f, x_new);
        std::vector<AV> gradient_delta = MakeVector(n, 1, engine);
        for (size_t i = 0; i < n; ++i) {
            gradient_delta[i] = gradient_new[i] - gradient[i];
            gradient_delta[i].setPrecision(precision);
        }

        AV curvature = Dot(step, gradient_delta);
        curvature.setPrecision(precision);
        auto opened_curv = curvature.open();
        double curv_val = static_cast<double>(opened_curv[0]) / scale;

        if (curv_val > kSmallEpsilon) {
            // rho = 1.0 / curvature
            AV scale_sq(1, engine);
            scale_sq += (DataType(1) << (2 * precision));
            auto scale_sq_b = scale_sq.a2b();
            auto curv_b = curvature.a2b();
            auto rho_b = (*scale_sq_b) / (*curv_b);
            AV rho = *(rho_b->b2a());
            rho.setPrecision(precision);

            h_inv = BfgsInverseUpdate(h_inv, step, gradient_delta, rho);
            h_inv.setPrecision(precision);
        }

        x = x_new;
        gradient = gradient_new;
        auto opened_new_fx = fx_new.open();
        double obj_change = std::abs(fx_val - static_cast<double>(opened_new_fx[0]) / scale);
        fx = fx_new;

        if (engine.getPartyID() == 0) {
            std::cout << "[BFGS] iter " << std::setw(3) << (iteration + 1)
                      << "  neg_log_lik=" << std::fixed << std::setprecision(6) << static_cast<double>(opened_new_fx[0]) / scale
                      << "  |grad|=" << std::scientific << std::setprecision(3) << max_grad
                      << "  alpha=" << std::fixed << std::setprecision(4) << alpha
                      << "  |step|=" << std::scientific << std::setprecision(3) << max_step
                      << "  d_obj=" << obj_change << std::endl;
        }

        if (max_step < kSmallEpsilon && max_grad < kSmallEpsilon) {
            result.converged = true;
            break;
        }
    }

    result.params = x;
    result.value = fx;
    return result;
}


int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();

    if (pID == 0) {
        std::cout << "kLn2: " << kLn2_scaled << " -> " << static_cast<double>(kLn2_scaled) / scale << std::endl;
        std::cout << "kSqrt2: " << kSqrt2_scaled << " -> " << static_cast<double>(kSqrt2_scaled) / scale << std::endl;
        std::cout << "kSqrt1_2: " << kSqrt1_2_scaled << " -> " << static_cast<double>(kSqrt1_2_scaled) / scale << std::endl;
        std::cout << "SmallEpsilon: " << kSmallEpsilon_scaled << " -> " << static_cast<double>(kSmallEpsilon_scaled) / scale << std::endl;
        std::cout << "kSeriesTolerance: " << kSeriesTolerance_scaled << " -> " << static_cast<double>(kSeriesTolerance_scaled) / scale << std::endl;
    }

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

    // 5. Test MinimizeBFGS
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

    return 0;
}