#pragma once

#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

#include "cdough.h"

// ../scripts/run_experiment.py -p 3 -r 16 secure-logistic-regression

namespace cdough::regression {
using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;
using namespace cdough::debug;
using namespace cdough::service;

using DataType = int64_t;
using HW = cdough::matrix::HeightWidth;
using AV = COMPILED_MPC_PROTOCOL_NAMESPACE::ASharedVector<DataType>;
using BV = COMPILED_MPC_PROTOCOL_NAMESPACE::BSharedVector<DataType>;
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

// Use these helpers wherever a copy is going to be written to.
AV Clone(const AV& v) {
    AV out(v.size(), v.engine);
    out = v;
    out.setPrecision(v.getPrecision());
    return out;
}

// Secure reciprocal 1/x in fixed point, computed as scale^2 / x over a boolean
// division circuit. Requires x > 0: the non-restoring division circuit assumes
// non-negative operands.
AV SecureReciprocal(const AV& x) {
    AV numerator(x.size(), x.engine);
    numerator += (DataType(1) << (2 * precision));

    auto numerator_b = numerator.a2b();
    AV denominator = Clone(x);
    denominator.setPrecision(0);
    auto denominator_b = denominator.a2b();

    auto quotient_b = (*numerator_b) / (*denominator_b);
    AV quotient = *(quotient_b->b2a());
    quotient.setPrecision(precision);
    return quotient;
}

// Secure clamping to [-bound_scaled, bound_scaled].
AV ClampAbs(const AV& x, DataType bound_scaled) {
    AV x_(x.size(), x.engine);
    x_ = x;
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
    delta_low -= bound_scaled;  // -bound - x
    x_ += *(cond_low * delta_low);

    x_.setPrecision(precision);
    return x_;
}

// Sum of all elements in an arithmetic shared vector (returns size 1 AV).
AV Sum(const AV& x) {
    return x.chunkedSum(x.size());
}

// Width of the oblivious exponent stage in Exp: the shifted exponent k + kExpOffset
// must fit in kExpBits bits, i.e. k in [-kExpOffset, kExpOffset - 1] = [-16, 15].
// That covers x in [-11.4, 10.7], which spans the whole range representable at
// `precision` bits; kMaxExpArg keeps k inside it with margin to spare.
constexpr int kExpBits = 5;
constexpr int kExpOffset = 1 << (kExpBits - 1);

AV Exp(const AV& x) {
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
    AV e(x.size(), x.engine);  // e starts at 0

    // Range reduction: we want m in [sqrt(1/2), sqrt(2)) ~ [0.7071, 1.4142]
    // 1. High steps: while m >= sqrt(2) * 2^step, divide m by 2^step, e += step
    // Using powers of 2 for step = 8, 4, 2, 1
    const int steps[] = {8, 4, 2, 1};
    for (int step : steps) {
        DataType threshold = static_cast<DataType>(kSqrt2 * (1 << step) * scale);
        // cond = (m - threshold) >= 0
        AV diff = m - threshold;
        AV cond_a = diff.gtez();  // 1 if m >= threshold, 0 otherwise

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
        AV cond_a = diff.gtez();  // 1 if m < threshold, 0 otherwise

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

    AV mask = *(eta_copy.gtez());  // 1 if eta >= 0, 0 if eta < 0

    // abs_eta = (2*mask - 1) * eta
    AV two_mask = *(mask * DataType(2));
    two_mask -= DataType(1);
    AV abs_eta = *(two_mask * eta_copy);

    AV neg_abs_eta = -abs_eta;
    neg_abs_eta.setPrecision(precision);
    AV z = Exp(neg_abs_eta);  // z = exp(-|eta|) <= 1.0
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

    AV mask = *(eta_copy.gtez());  // 1 if eta >= 0, 0 if eta < 0
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

}  // namespace cdough::regression