#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#include "cdough.h"
#include "core/random/prg/prg_algorithm.h"

// Fixed-point kernels for the MPC analysis pipeline: the types and constants
// everything else is built on, plus the transcendental and division routines.
//
// Two conventions hold throughout this header set:
//   - Values are RAW SCALED INTEGERS held at precision 0, with an explicit
//     `/ scale` after every multiply. handle_precision (protocol.h:115) throws
//     on a precision mismatch, so mixing conventions is a runtime error.
//   - `AV a = b` is a SHALLOW copy sharing the underlying buffer, while
//     `operator=` is a deep element-wise copy. Use Clone() before mutating a
//     copy of anything you do not own.
//
// These headers define non-inline functions, so each belongs to exactly one
// translation unit. That is how the playground is built -- CMakeLists.txt globs
// playground/*.cpp into one binary per file -- and it matches the convention
// established by the logistic-regression branch.

namespace cdough::regression {

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;
using namespace cdough::debug;
using namespace cdough::service;

using DataType = int64_t;
using HW = cdough::matrix::HeightWidth;
// Qualified deliberately. This namespace sits inside `cdough`, which also
// declares ASharedVector/BSharedVector, so the unqualified names are ambiguous
// against the protocol namespace's aliases of the same name.
using AV = COMPILED_MPC_PROTOCOL_NAMESPACE::ASharedVector<DataType>;
using BV = COMPILED_MPC_PROTOCOL_NAMESPACE::BSharedVector<DataType>;
using SMatrix = COMPILED_MPC_PROTOCOL_NAMESPACE::SecureMatrix<DataType>;
using PMatrix = COMPILED_MPC_PROTOCOL_NAMESPACE::PlainMatrix<DataType>;

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

// Series lengths. Three terms was the original setting for both; with the Exp
// range-reduction floor fixed, the truncation error of the series is what
// limits accuracy, and two extra terms each (two extra multiplies per call,
// negligible beside the normalisation ladders) buys about two decimal digits.
constexpr int kExpSeriesTerms = 5;
constexpr int kLogSeriesTerms = 5;

// Superseded by the two constants above, which tune the Exp and Log series
// independently. Retained because it was part of this header's interface on the
// logistic-regression branch; it no longer controls anything.
constexpr int kMaxSeriesTerms = 3;
constexpr int kMaxNewtonStep = 4;
// Iterations of the inner damped Newton search for the conditional mode. The
// objective is strictly concave in u and the search starts at u = 0, so it
// converges almost immediately. Measured against a 12-iteration reference, 3
// iterations reproduce the fitted coefficients to 1e-6 while 2 leave 1e-4, so 3
// it is -- a 40% cut in the inner loop for no accuracy at all. Keep this in
// step with NEWTON_ITERATIONS in scripts/testing/validate_mpc_analysis.py,
// which mirrors the estimator.
constexpr int kNewtonIterations = 3;

// Width of the oblivious exponent stage in Exp: the shifted exponent k + kExpOffset
// must fit in kExpBits bits, i.e. k in [-kExpOffset, kExpOffset - 1] = [-16, 15].
// That covers x in [-11.4, 10.7], which spans the whole range representable at
// `precision` bits; kMaxExpArg keeps k inside it with margin to spare.
constexpr int kExpBits = 5;
constexpr int kExpOffset = 1 << (kExpBits - 1);

// Public bias that turns Exp's truncating shift into a floor. Must exceed
// |x/ln2 + 1/2| for any clamped x, i.e. 10/ln2 + 1/2 = 14.9.
constexpr DataType kExpRoundBias = 64;

// --- Div / Recip -------------------------------------------------------------
// Minimax linear seed for 1/m on m in [1, 2):  r0 = (24 - 8m)/17, max relative
// error 1/17 = 5.88%. Newton (r <- r(2 - m r)) squares that error each step:
// 3.5e-3, 1.2e-5, 1.4e-10 -- three steps land well under one fixed-point ulp.
const double kRecipSeedA = 24.0 / 17.0;
const double kRecipSeedB = 8.0 / 17.0;
const DataType kRecipSeedA_scaled = std::llround(kRecipSeedA * scale);
const DataType kRecipSeedB_scaled = std::llround(kRecipSeedB * scale);
constexpr int kRecipNewtonSteps = 3;

// Sigmoid's and Log's denominators are confined to known intervals, so they can
// skip Div's normalisation ladder entirely and go straight to a seed. The
// minimax linear seed for 1/d on [L, U] is a + b*d with
//     b = -2 / (L*U + (L+U)^2/4),   a = -b*(L+U),
// which equioscillates at L, (L+U)/2 and U.
//
// Sigmoid: den = 1 + exp(-|eta|), so den is in (1, 2]. Seed error 5.88%,
// three Newton steps reach 1.4e-10.
const DataType kRecipUnitA_scaled = std::llround((24.0 / 17.0) * scale);
const DataType kRecipUnitB_scaled = std::llround((8.0 / 17.0) * scale);
constexpr int kRecipUnitSteps = 3;

// Log: den = m + 1 with m range-reduced into [sqrt(1/2), sqrt(2)], so den is in
// [1.7071, 2.4142]. Seed error 1.49%, two Newton steps reach 5.0e-8.
const DataType kRecipLogA_scaled = std::llround(0.985061500 * scale);
const DataType kRecipLogB_scaled = std::llround(0.239015999 * scale);
constexpr int kRecipLogSteps = 2;

// Denominators are clamped into this band before normalisation. Below the low
// bound a fixed-point denominator is indistinguishable from zero anyway.
const double kDivDenMin = 1.0 / 4096.0;
const double kDivDenMax = 1.0 * (1 << 24);
const DataType kDivDenMin_scaled = std::llround(kDivDenMin * scale);
const DataType kDivDenMax_scaled = std::llround(kDivDenMax * scale);

// --- Sqrt / Rsqrt ------------------------------------------------------------
// Degree-2 minimax seed for m^{-1/2} on m in [0.5, 2), found by Remez exchange:
// max relative error 2.40%. Newton (g <- g(3 - m g^2)/2) is quadratic, giving
// 8.7e-4 then 1.1e-6 -- two steps, not one: after a single step the error is
// still ~57 ulp at precision 16, which is too coarse for a standard error.
const double kRsqrtSeedC0 = 1.8885658148542937;
const double kRsqrtSeedC1 = -1.1615489719711822;
const double kRsqrtSeedC2 = 0.2896606126125612;
const DataType kRsqrtSeedC0_scaled = std::llround(kRsqrtSeedC0 * scale);
const DataType kRsqrtSeedC1_scaled = std::llround(kRsqrtSeedC1 * scale);
const DataType kRsqrtSeedC2_scaled = std::llround(kRsqrtSeedC2 * scale);
constexpr int kRsqrtNewtonSteps = 2;

const double kSqrtArgMin = 1.0 / 65536.0;
const double kSqrtArgMax = 1.0 * (1 << 28);
const DataType kSqrtArgMin_scaled = std::llround(kSqrtArgMin * scale);
const DataType kSqrtArgMax_scaled = std::llround(kSqrtArgMax * scale);

// Normalisation ladders. {16,8,4,2,1} covers any exponent in [0, 31] for Div;
// Sqrt steps in powers of four, so {8,4,2,1} covers x in [4^-15, 4^16).
constexpr int kDivLadder[] = {16, 8, 4, 2, 1};
constexpr int kSqrtLadder[] = {8, 4, 2, 1};

// NOTE: the balanced ClusterGroup / Dataset representation that used to live
// here has been replaced by the flat, ragged-cluster layout further down
// (SecureCohort + ModelData + FlatNegMarginalLogLik). Real patients have a
// ragged number of encounters, which the balanced form could not express
// without padding every patient to the largest cluster.

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


// =============================================================================
// 1. Fixed-point kernels
// =============================================================================

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

// Oblivious clamp to a public [lo, hi]. ClampAbs is the symmetric special case.
AV ClampRange(const AV& x, DataType lo_scaled, DataType hi_scaled) {
    AV x_ = Clone(x);
    x_.setPrecision(0);

    // cond_high = 1 if x >= hi, and then x - cond_high * (x - hi) == hi.
    AV diff_high = x_ - hi_scaled;
    AV cond_high = *(diff_high.gtez());
    x_ -= *(cond_high * diff_high);

    // cond_low = 1 if lo - x >= 0, i.e. if x <= lo.
    AV diff_low = -x_;
    diff_low += lo_scaled;
    AV cond_low = *(diff_low.gtez());
    x_ += *(cond_low * diff_low);

    x_.setPrecision(precision);
    return x_;
}

// |x|. The (2*gtez(x) - 1) * x idiom, named once instead of inlined everywhere.
AV Abs(const AV& x) {
    AV x_ = Clone(x);
    x_.setPrecision(0);
    AV sign = *(x_.gtez());
    AV two_sign = *(sign * DataType(2));
    two_sign -= DataType(1);  // +1 when x >= 0, -1 otherwise
    AV out = *(two_sign * x_);
    out.setPrecision(precision);
    return out;
}

// Reciprocal of a denominator already known to lie in a fixed public interval.
//
// This is the fast path that matters. Div's normalisation ladder is ten
// conditional shifts and ten sign tests, and it exists only to bring an
// arbitrary denominator into [1, 2). Sigmoid and Log both know their
// denominator's range up front, so for them the whole ladder is dead work --
// and they are called on every row, several times per objective evaluation.
// Given a minimax linear seed for the interval, all that remains is
// r <- r(2 - d r), which is two multiplications per step.
AV RecipSeeded(const AV& den, DataType seed_a_scaled, DataType seed_b_scaled, int steps) {
    AV d_ = Clone(den);
    d_.setPrecision(0);

    AV r = -(*(*(d_ * seed_b_scaled) / scale));
    r += seed_a_scaled;
    for (int i = 0; i < steps; ++i) {
        AV dr = *(*(d_ * r) / scale);
        AV corr = -dr;
        corr += DataType(2) * DataType(scale);  // 2 - d*r
        r = *(*(r * corr) / scale);
    }
    r.setPrecision(0);
    return r;
}

// Secure fixed-point division num / den, for den > 0.
//
// This exists to keep BSharedVector::operator/ (circuits.h:39 -- 64 sequential
// non-restoring iterations, ~24k AND gates and ~700 rounds *per element*) off
// the hot path. It is the single biggest cost lever in this program.
//
// The denominator is normalised into [1, 2) by a public ladder of conditional
// power-of-two shifts, and the *same* shifts are applied to the numerator. The
// quotient is therefore invariant, and the numerator is held at the magnitude of
// the answer throughout -- which is what stops a large denominator from
// underflowing the result. A minimax linear seed plus three Newton steps
// (r <- r(2 - d r), multiplications only) then inverts the normalised
// denominator, to a relative error of ~1.4e-10, far under one ulp at
// precision 16.
AV Div(const AV& num, const AV& den) {
    AV n_ = Clone(num);
    n_.setPrecision(0);
    AV d_ = ClampRange(den, kDivDenMin_scaled, kDivDenMax_scaled);
    d_.setPrecision(0);

    // Down ladder: the condition "remaining exponent >= s" is exactly den >= 2^s,
    // so the greedy pass over {16,8,4,2,1} builds floor(log2 den) in binary.
    for (int s : kDivLadder) {
        AV diff = d_ - (DataType(scale) << s);
        AV cond = *(diff.gtez());  // 1 if den >= 2^s
        AV d_shift = *(d_ / (DataType(1) << s));
        d_ -= *(cond * (d_ - d_shift));
        AV n_shift = *(n_ / (DataType(1) << s));
        n_ -= *(cond * (n_ - n_shift));
    }

    // Up ladder: "at least s more doublings" is den < 2^(1-s).
    for (int s : kDivLadder) {
        const DataType threshold = (s >= precision + 1) ? DataType(1)
                                                        : (DataType(scale) >> (s - 1));
        AV diff = -d_;
        diff += (threshold - 1);
        AV cond = *(diff.gtez());  // 1 if den < 2^(1-s)
        AV d_shift = *(d_ * (DataType(1) << s));
        d_ += *(cond * (d_shift - d_));
        AV n_shift = *(n_ * (DataType(1) << s));
        n_ += *(cond * (n_shift - n_));
    }

    // d_ is now in [1, 2). Seed r0 = (24 - 8 d)/17, then Newton.
    AV r = -(*(*(d_ * kRecipSeedB_scaled) / scale));
    r += kRecipSeedA_scaled;
    for (int i = 0; i < kRecipNewtonSteps; ++i) {
        AV dr = *(*(d_ * r) / scale);
        AV corr = -dr;
        corr += DataType(2) * DataType(scale);  // 2 - d*r
        r = *(*(r * corr) / scale);
    }

    AV out = *(*(n_ * r) / scale);
    out.setPrecision(precision);
    return out;
}

// 1 / x for x > 0.
AV Recip(const AV& x) {
    AV one(x.size(), x.engine);
    one.setPrecision(0);
    one += DataType(scale);
    one.setPrecision(precision);
    return Div(one, x);
}

// sqrt(x) and 1/sqrt(x) for x > 0, computed together because they share the
// range reduction and the Newton iteration.
//
// Deliberately NOT Exp(0.5*Log(x)): Log carries a private division (~700 rounds)
// to seed an iteration that costs six multiplies, its reduction ladder is out of
// contract for x >= sqrt(2)*2^16 = 92682, and a saturated Exp seed makes
// g*(3 - x*g^2) wrap int64 silently. Instead the reduction is done here in
// powers of *four* -- so the recovered exponent is exact and needs no sqrt(2)
// correction -- and both the argument and the seed are clamped to public bounds,
// which guarantees x*g^2 stays inside the (0, 3) convergence basin by
// construction rather than by hope.
struct SqrtPair {
    AV root;     // sqrt(x)
    AV inv_root; // 1 / sqrt(x)
};

SqrtPair SqrtBoth(const AV& x) {
    AV m = ClampRange(x, kSqrtArgMin_scaled, kSqrtArgMax_scaled);
    m.setPrecision(0);

    // corr tracks 2^e and inv_corr tracks 2^-e, where x = m * 4^e with m in
    // [0.5, 2). Both stay in the magnitude of sqrt(x) and 1/sqrt(x), so neither
    // overflows nor underflows for x in the clamped band.
    AV corr(x.size(), x.engine);
    corr.setPrecision(0);
    corr += DataType(scale);
    AV inv_corr = Clone(corr);

    // Down ladder: "e >= s" is x >= 2^(2s-1).
    for (int s : kSqrtLadder) {
        AV diff = m - (DataType(scale) << (2 * s - 1));
        AV cond = *(diff.gtez());
        AV m_shift = *(m / (DataType(1) << (2 * s)));
        m -= *(cond * (m - m_shift));
        AV c_shift = *(corr * (DataType(1) << s));
        corr += *(cond * (c_shift - corr));
        AV i_shift = *(inv_corr / (DataType(1) << s));
        inv_corr -= *(cond * (inv_corr - i_shift));
    }

    // Up ladder: "e <= -s" is x < 2^(1-2s).
    for (int s : kSqrtLadder) {
        const int shift = 2 * s - 1;
        const DataType threshold = (shift >= precision) ? DataType(1)
                                                        : (DataType(scale) >> shift);
        AV diff = -m;
        diff += (threshold - 1);
        AV cond = *(diff.gtez());
        AV m_shift = *(m * (DataType(1) << (2 * s)));
        m += *(cond * (m_shift - m));
        AV c_shift = *(corr / (DataType(1) << s));
        corr -= *(cond * (corr - c_shift));
        AV i_shift = *(inv_corr * (DataType(1) << s));
        inv_corr += *(cond * (i_shift - inv_corr));
    }

    // m is now in [0.5, 2). Degree-2 minimax seed for m^{-1/2}, then Newton.
    AV m_sq = *(*(m * m) / scale);
    AV g = *(*(m_sq * kRsqrtSeedC2_scaled) / scale);
    g += *(*(m * kRsqrtSeedC1_scaled) / scale);
    g += kRsqrtSeedC0_scaled;

    for (int i = 0; i < kRsqrtNewtonSteps; ++i) {
        AV g_sq = *(*(g * g) / scale);
        AV h = *(*(m * g_sq) / scale);  // m * g^2, inside (0, 3) by construction
        AV three_minus_h = -h;
        three_minus_h += DataType(3) * DataType(scale);
        g = *(*(g * three_minus_h) / scale);
        g = *(g / DataType(2));
    }

    // sqrt(m) = m * g, then rescale by the tracked powers of two.
    AV root_m = *(*(m * g) / scale);
    SqrtPair out{*(*(root_m * corr) / scale), *(*(g * inv_corr) / scale)};
    out.root.setPrecision(precision);
    out.inv_root.setPrecision(precision);
    return out;
}

AV Sqrt(const AV& x) { return SqrtBoth(x).root; }
AV Rsqrt(const AV& x) { return SqrtBoth(x).inv_root; }

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

    // Integer k (unscaled). div_const_a truncates toward zero, not toward minus
    // infinity, so for -1 < x/ln2 + 1/2 < 0 the nearest integer came out as 0
    // instead of -1 and r escaped [-ln2/2, ln2/2] -- exp(-1) evaluated the
    // 3-term series at r = -1 and returned exactly 1/3. Adding a public bias
    // before the shift and removing it after turns the truncation into a floor.
    // The bias only has to exceed |x/ln2 + 1/2|, and x is clamped to +/-10.
    k_fixed += kExpRoundBias * DataType(scale);
    AV k_int = *(k_fixed / scale);
    k_int -= kExpRoundBias;

    // Remainder r = x - k * ln(2), so |r| <= ln(2)/2
    AV r = x_ - (*(k_int * kLn2_scaled));

    // 2. Maclaurin series evaluation: exp(r) = 1 + r + r^2/2! + r^3/3! + ...
    AV term(x.size(), x.engine);
    AV series(x.size(), x.engine);
    term += scale;
    series += scale;

    for (int n = 1; n <= kExpSeriesTerms; ++n) {
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

    // m has been reduced into [sqrt(1/2), sqrt(2)], so den is in
    // [1.7071, 2.4142] -- again no normalisation needed.
    AV inv_den = RecipSeeded(den, kRecipLogA_scaled, kRecipLogB_scaled, kRecipLogSteps);
    AV w = *(*(num * inv_den) / scale);
    w.setPrecision(0);

    // Compute series: 2 * (w + w^3/3 + w^5/5)
    AV w_squared = (*(w * w)) / scale;
    AV power = w;
    AV series(x.size(), x.engine);

    for (int i = 0; i < kLogSeriesTerms; ++i) {
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

    // den = 1 + z with z in (0, 1], so den is in (1, 2] and needs no
    // normalisation. This replaced BSharedVector::operator/ (circuits.h:39),
    // which was 79% of Sigmoid's cost and, since Sigmoid runs six times per
    // objective evaluation on every row, the largest single cost in the program.
    AV inv_den = RecipSeeded(den, kRecipUnitA_scaled, kRecipUnitB_scaled, kRecipUnitSteps);
    AV res = *(*(num * inv_den) / scale);
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

// Sum of all elements in an arithmetic shared vector (returns size 1 AV).
AV Sum(const AV& x) {
    return x.chunkedSum(x.size());
}

// Secret-share a vector of doubles from party 0 at the module precision.
AV ShareDoubles(EngineRef engine, const std::vector<double>& values) {
    cdough::Vector<DataType> plain(values.size(), precision);
    for (size_t i = 0; i < values.size(); ++i) {
        plain[i] = static_cast<DataType>(std::llround(values[i] * scale));
    }
    return engine.secret_share_a(plain, 0, precision);
}

// Open a secure vector and decode it back to doubles.
std::vector<double> OpenToDoubles(const AV& v) {
    auto opened = v.open();
    std::vector<double> out(opened.size());
    for (size_t i = 0; i < opened.size(); ++i) {
        out[i] = static_cast<double>(opened[i]) / scale;
    }
    return out;
}


// Open a 1-element vector. `scaled` distinguishes fixed-point values from raw
// counts, which are held as plain integers at precision 0.
double OpenScalar(const AV& v, bool scaled = true) {
    auto opened = v.open();
    const double raw = static_cast<double>(opened[0]);
    return scaled ? raw / scale : raw;
}

// =============================================================================
// Opening to ONE party
//
// SharedVector::open() (shared_vector.h:143) reveals to EVERY party. That is the
// right default for a result the parties jointly agreed to learn, and the wrong
// one when the result belongs to a single party -- the lone data owner of
// `--owner`, who contributed the whole input and is the only party entitled to
// anything derived from it.
//
// The recipient masks first:
//
//   1. the recipient draws r uniformly over the whole ring and secret-shares it;
//   2. every party opens x + r;
//   3. only the recipient subtracts r back off.
//
// x + r is uniform on Z_2^64 and independent of x, so step 2 discloses nothing
// to anyone else -- information-theoretically, not computationally. The cost is
// one extra secret-share and one extra open, and it uses nothing beyond
// secret_share and open, so it behaves identically under every protocol the
// framework compiles.
//
// These calls are COLLECTIVE. Every party must reach them, whatever it intends
// to do with the result: the open underneath is a synchronised exchange and a
// party that skips it hangs the rest.
//
// `reveal_to < 0` means "reveal to everybody", so a call site can carry one
// parameter instead of branching.
// =============================================================================

// A uniformly random ring element per slot, from /dev/urandom via the
// framework's own PRG rather than a seeded std:: generator: the mask is the only
// thing standing between the other parties and the recipient's output, so it has
// to be drawn from a source that is not reproducible from a captured state.
cdough::Vector<DataType> RandomRingVector(size_t n) {
    cdough::Vector<DataType> out(n, 0);
    if (n == 0) return out;
    cdough::random::DevUrandomPRGAlgorithm prg;
    std::vector<uint8_t> bytes(n * sizeof(DataType));
    prg.fillBytes(std::span<uint8_t>(bytes.data(), bytes.size()));
    for (size_t i = 0; i < n; ++i) {
        DataType word = 0;
        std::memcpy(&word, bytes.data() + i * sizeof(DataType), sizeof(DataType));
        out[i] = word;
    }
    return out;
}

// Reveal an arithmetic vector to `reveal_to` alone, as raw ring elements. Every
// other party gets an empty vector back.
std::vector<DataType> OpenRawToParty(const AV& v, int reveal_to, int party_id) {
    const size_t n = v.size();
    auto decode = [n](const auto& opened) {
        std::vector<DataType> out(n);
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<DataType>(opened[i]);
        return out;
    };
    if (reveal_to < 0) return decode(v.open());

    const bool mine = (party_id == reveal_to);
    cdough::Vector<DataType> mask = mine ? RandomRingVector(n) : cdough::Vector<DataType>(n, 0);
    EngineRef engine = v.engine;
    AV shared_mask = engine.template secret_share_a<DataType>(mask, reveal_to, 0);
    shared_mask.setPrecision(0);

    AV masked = Clone(v);
    masked.setPrecision(0);
    masked += shared_mask;

    const auto opened = masked.open();  // uniform, and independent of v
    if (!mine) return {};

    std::vector<DataType> out(n);
    for (size_t i = 0; i < n; ++i)
        out[i] = static_cast<DataType>(opened[i]) - mask[i];  // -fwrapv: wraps, exactly
    return out;
}

// The boolean-sharing counterpart. Same construction with XOR in place of
// addition, which is the group operation B-shares live in.
std::vector<DataType> OpenRawToParty(const BV& v, int reveal_to, int party_id) {
    const size_t n = v.size();
    auto decode = [n](const auto& opened) {
        std::vector<DataType> out(n);
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<DataType>(opened[i]);
        return out;
    };
    if (reveal_to < 0) return decode(v.open());

    const bool mine = (party_id == reveal_to);
    cdough::Vector<DataType> mask = mine ? RandomRingVector(n) : cdough::Vector<DataType>(n, 0);
    EngineRef engine = v.engine;
    BV shared_mask = engine.template secret_share_b<DataType>(mask, reveal_to);

    BV masked(n, engine);
    masked = v;
    masked ^= shared_mask;

    const auto opened = masked.open();
    if (!mine) return {};

    std::vector<DataType> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<DataType>(opened[i]) ^ mask[i];
    return out;
}

// OpenToDoubles, revealed to one party. Empty on every other party.
std::vector<double> OpenToPartyDoubles(const AV& v, int reveal_to, int party_id) {
    const std::vector<DataType> raw = OpenRawToParty(v, reveal_to, party_id);
    std::vector<double> out(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) out[i] = static_cast<double>(raw[i]) / scale;
    return out;
}

// OpenScalar, revealed to one party. Returns 0 on every other party, which is a
// placeholder and not a result -- callers must gate on `party_id == reveal_to`.
double OpenScalarToParty(const AV& v, int reveal_to, int party_id, bool scaled = true) {
    const std::vector<DataType> raw = OpenRawToParty(v, reveal_to, party_id);
    if (raw.empty()) return 0.0;
    const double value = static_cast<double>(raw[0]);
    return scaled ? value / scale : value;
}

}  // namespace cdough::regression
