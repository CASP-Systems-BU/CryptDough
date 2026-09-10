#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <fstream>
#include <random>
#include <sstream>
#include <set>
#include <string>
#include <vector>
#include <functional>

#include "cdough.h"

// Three patient-encounter tables in (any_system, umass_system, nonumass_system);
// nineteen terminal outputs out -- two printed descriptive reports, three
// one-row aggregate tables, and fourteen fitted regression models (six
// specifications across three populations, minus the four interaction fits that
// can only run on the pooled table). Nothing downstream consumes anything else:
// the pipeline is a wide fan-out, and every node reads a source table directly.
//
// Design and decisions: tasks/0009_mpc-analysis-pipeline.md
//
// Layout:
//   1. Fixed-point kernels      ClampAbs/ClampRange/Abs, Div/Recip, Sqrt/Rsqrt,
//                               Exp/Log/Log1p/Sigmoid/LogOnePlusExp
//   2. Segmented helpers        SegScan/SegTotal, First/LastOfGroup, CountDistinct
//   3. Dense linear algebra     Gram, Cholesky factor/solve, SymmetricInverse
//   4. BFGS                     numerical gradient, Armijo line search
//   5. Cohort + ingestion       synthetic generator, CSV reader/writer, sharing
//   6. Descriptive + aggregate  d1a, d1b, sisa_perct_cnt x3
//   7. Models                   design matrices, IRLS (6a/6b), flat Laplace GLMM
//   8. Reporting + oracle       coefficient tables, plaintext IRLS cross-check
//   9. main()                   stage selection and the pipeline driver
//
// Two conventions carried from the original code and relied on throughout:
//   - Values are RAW SCALED INTEGERS held at precision 0, with an explicit
//     `/ scale` after every multiply. handle_precision (protocol.h:115) throws
//     on a precision mismatch, so mixing conventions is a runtime error.
//   - `AV a = b` is a SHALLOW copy sharing the underlying buffer, while
//     `operator=` is a deep element-wise copy. Use Clone() before mutating a
//     copy of anything you do not own.
//
// Run:
//   ../scripts/run_experiment.py -p 3 -r 200 mpc-analysis
//   ./mpc-analysis -S kernels                 # accuracy harnesses only (fast)
//   ./mpc-analysis -S describe -r 200         # ingestion + descriptive nodes
//   ./mpc-analysis -S models -r 200           # the fourteen fits (slow)
//   ./mpc-analysis -D /data                   # read per-party CSVs instead
//   ./mpc-analysis -O /tmp/dump               # dump the synthetic cohort to CSV

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

// Series lengths. Three terms was the original setting for both; with the Exp
// range-reduction floor fixed, the truncation error of the series is what
// limits accuracy, and two extra terms each (two extra multiplies per call,
// negligible beside the normalisation ladders) buys about two decimal digits.
constexpr int kExpSeriesTerms = 5;
constexpr int kLogSeriesTerms = 5;
constexpr int kMaxNewtonStep = 4;
constexpr int kNewtonIterations = 5;

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

    auto num_b = (*(num * scale)).a2b();
    auto den_b = den.a2b();

    auto w_b = (*num_b) / (*den_b);
    AV w = *(w_b->b2a());
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

// =============================================================================
// Segmented (per-group) helpers
//
// Everything in this section assumes the rows are SORTED by `keys`, padded to a
// power of two (aggregate() asserts it, aggregation.h:198), that the pad rows
// carry a key sentinel which cannot collide with a real group, and that the pad
// rows of every value column are zero. aggregators::aggregate has no valid-bit
// handling of its own -- that all lives in EncodedTable -- so the caller owns
// those invariants.
// =============================================================================

// common.h:14 declares `enum class Direction { ... } Direction;` -- that trailing
// name is a VARIABLE which shadows the type, so the type can only be named with
// an elaborated specifier. aggregate() itself works around this the same way
// (`const enum Direction dir`).
using SegDirection = enum cdough::aggregators::Direction;

using AggSpecA = std::vector<std::tuple<AV, AV, void (*)(const AV&, AV&, const AV&)>>;
using AggSpecB = std::vector<std::tuple<BV, BV, void (*)(const BV&, BV&, const BV&)>>;

size_t NextPowerOfTwo(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Per-group inclusive scan. Forward leaves each row holding the sum of its
// group up to and including itself; Reverse does the same from the far end.
void SegScan(std::vector<BV>& keys, const std::vector<AV>& in, std::vector<AV>& out,
             SegDirection dir) {
    assert(in.size() == out.size());
    AggSpecB b_spec;
    AggSpecA a_spec;
    a_spec.reserve(in.size());
    // The aggregator computes `a + group * b`, where the group bit is an
    // arithmetic 0/1 at precision 0. handle_precision (protocol.h:115) throws on
    // a precision mismatch, so every column here is handled as a raw scaled
    // integer; inputs and outputs are precision 0 in and precision 0 out.
    std::vector<AV> in_p0, out_p0;
    in_p0.reserve(in.size());
    out_p0.reserve(in.size());
    for (size_t k = 0; k < in.size(); ++k) {
        in_p0.push_back(in[k]);
        in_p0.back().setPrecision(0);
        out_p0.push_back(out[k]);
        out_p0.back().setPrecision(0);
    }
    for (size_t k = 0; k < in.size(); ++k) {
        a_spec.push_back({in_p0[k], out_p0[k], &cdough::aggregators::sum<AV>});
    }
    cdough::aggregators::aggregate(keys, b_spec, a_spec, dir);
    for (size_t k = 0; k < out.size(); ++k) out[k].setPrecision(0);
}

// Broadcast each group's TOTAL to every row of that group.
//
// The obvious formulation -- scan forward, then scan the result backwards -- is
// wrong: a Reverse pass over an already-scanned column produces a
// suffix-of-prefixes, which is silently garbage rather than an error. Both
// passes must read the ORIGINAL column, and the row's own value is then double
// counted exactly once:
//
//     total = prefix + suffix - self
//
// The subtraction is local, so this costs two aggregate invocations and no
// extra communication.
void SegTotal(std::vector<BV>& keys, const std::vector<AV>& in, std::vector<AV>& out) {
    assert(in.size() == out.size());
    const size_t n = in[0].size();

    std::vector<AV> suffix;
    suffix.reserve(in.size());
    for (size_t k = 0; k < in.size(); ++k) suffix.emplace_back(n, in[k].engine);

    SegScan(keys, in, out, SegDirection::Forward);
    SegScan(keys, in, suffix, SegDirection::Reverse);

    for (size_t k = 0; k < in.size(); ++k) {
        AV self = Clone(in[k]);
        self.setPrecision(0);
        out[k].setPrecision(0);
        suffix[k].setPrecision(0);
        out[k] += suffix[k];
        out[k] -= self;
    }
}

// Single-column convenience wrappers.
AV SegTotal(std::vector<BV>& keys, const AV& in) {
    std::vector<AV> ins{in};
    std::vector<AV> outs;
    outs.emplace_back(in.size(), in.engine);
    SegTotal(keys, ins, outs);
    return outs[0];
}

AV SegScan(std::vector<BV>& keys, const AV& in, SegDirection dir) {
    std::vector<AV> ins{in};
    std::vector<AV> outs;
    outs.emplace_back(in.size(), in.engine);
    SegScan(keys, ins, outs, dir);
    return outs[0];
}

// 1 on the first row of each run of equal keys (operators::distinct marks
// exactly that, and always marks row 0).
BV FirstOfGroup(std::vector<BV>& keys) {
    BV uniq(keys[0].size(), keys[0].engine);
    cdough::operators::distinct(keys, uniq);
    return uniq;
}

// 1 on the last row of each run of equal keys: last[i] = first[i+1], with the
// final row always last. This is the same shift EncodedTable::aggregate uses to
// pick the surviving row of a group (encoded_table.h:1311).
BV LastOfGroup(std::vector<BV>& keys) {
    const size_t n = keys[0].size();
    BV uniq = FirstOfGroup(keys);

    BV last(n, keys[0].engine);
    last.zero();
    BV head = last.slice(0, n - 1);
    head = uniq.slice(1);

    cdough::Vector<DataType> one(1, 1);
    BV tail = last.slice(n - 1);
    tail = keys[0].engine.template public_share_b<DataType>(one);
    return last;
}

// Arithmetic 0/1 indicator of the last row of each group, which is the mask that
// makes a per-cluster quantity count exactly once.
AV LastOfGroupArith(std::vector<BV>& keys) {
    BV last = LastOfGroup(keys);
    AV out = *(last.b2a_bit());
    out.setPrecision(0);
    return out;
}

// COUNT(DISTINCT key) over the rows where `mask` is 1.
//
// The table is already sorted by key, so a group contributes iff the mask is set
// anywhere inside it. SegTotal gives the per-group mask count on every row;
// gtez on (count - 1) turns that into "seen at least once"; and multiplying by
// the first-of-group indicator counts each group exactly once.
AV CountDistinct(std::vector<BV>& keys, const AV& mask) {
    AV seen = SegTotal(keys, mask);
    seen.setPrecision(0);
    seen -= DataType(1);
    AV any = *(seen.gtez());  // 1 if the group has at least one masked row

    BV first_b = FirstOfGroup(keys);
    AV first = *(first_b.b2a_bit());
    first.setPrecision(0);

    AV contrib = *(any * first);
    AV total = contrib.chunkedSum(contrib.size());
    total.setPrecision(0);
    return total;
}

// =============================================================================
// Small dense secure linear algebra
//
// Matrices are flat AVs in row-major order; `p` is public and small (<= 8 for
// the widest model here), so every loop bound below is public and the control
// flow is completely data independent. All values are raw scaled integers
// (precision 0) with explicit rescaling after each multiply, matching the
// convention used by the kernels above.
// =============================================================================

// A single element as a 1-long view. Assigning to the returned value writes
// through to the underlying buffer.
AV Cell(const AV& m, size_t idx) { return m.slice(idx, idx + 1); }

AV ScalarZero(EngineRef engine) {
    AV z(1, engine);
    z.setPrecision(0);
    return z;
}

// A public p-vector, secret shared. Used for the unit vectors that turn a
// Cholesky solve into a matrix inverse.
AV PublicVector(EngineRef engine, const std::vector<double>& values) {
    cdough::Vector<DataType> v(values.size(), 0);
    for (size_t i = 0; i < values.size(); ++i) {
        v[i] = static_cast<DataType>(std::llround(values[i] * scale));
    }
    AV out = engine.template public_share_a<DataType>(v);
    out.setPrecision(0);
    return out;
}

// Cholesky factorisation A = L L^T of a symmetric positive definite A.
//
// No pivoting, for two independent reasons. Numerically it is unnecessary: for
// SPD matrices Cholesky is unconditionally backward stable, with a growth factor
// bounded by 1, so pivoting buys nothing. Cryptographically it is impossible:
// pivoting means comparing secret magnitudes and permuting rows on the result,
// which is data-dependent control flow. SPD is guaranteed here by the ridge term
// the callers add to the diagonal.
struct Cholesky {
    AV l;         // p*p, lower triangular, row-major
    AV inv_diag;  // p, reciprocals of L's diagonal (so the solves need no division)
};

Cholesky CholeskyFactor(const AV& a, size_t p) {
    EngineRef engine = a.engine;
    AV l(p * p, engine);
    l.setPrecision(0);
    AV inv_diag(p, engine);
    inv_diag.setPrecision(0);

    AV a_ = a;
    a_.setPrecision(0);

    for (size_t j = 0; j < p; ++j) {
        // d = A[j][j] - sum_{k<j} L[j][k]^2
        AV d = Clone(Cell(a_, j * p + j));
        d.setPrecision(0);
        for (size_t k = 0; k < j; ++k) {
            AV ljk = Cell(l, j * p + k);
            d -= *(*(ljk * ljk) / scale);
        }

        // r = 1/sqrt(d) and L[j][j] = d * r = sqrt(d), neither needing a division.
        AV r = Rsqrt(d);
        r.setPrecision(0);
        AV diag = *(*(d * r) / scale);
        AV diag_cell = Cell(l, j * p + j);
        diag_cell = diag;
        AV inv_cell = Cell(inv_diag, j);
        inv_cell = r;

        // L[i][j] = (A[i][j] - sum_{k<j} L[i][k] L[j][k]) / L[j][j]
        for (size_t i = j + 1; i < p; ++i) {
            AV v = Clone(Cell(a_, i * p + j));
            v.setPrecision(0);
            for (size_t k = 0; k < j; ++k) {
                AV lik = Cell(l, i * p + k);
                AV ljk = Cell(l, j * p + k);
                v -= *(*(lik * ljk) / scale);
            }
            AV cell = Cell(l, i * p + j);
            cell = *(*(v * r) / scale);
        }
    }
    l.setPrecision(0);
    inv_diag.setPrecision(0);
    return Cholesky{l, inv_diag};
}

// Solve A x = b from a precomputed factorisation, by forward then back
// substitution. Reusing the factorisation is what makes a full inverse cheap:
// the factorisation is most of the cost and is shared across all p solves.
AV CholeskySolveWith(const Cholesky& f, const AV& b, size_t p) {
    EngineRef engine = b.engine;
    AV b_ = b;
    b_.setPrecision(0);

    AV y(p, engine);
    y.setPrecision(0);
    for (size_t i = 0; i < p; ++i) {
        AV t = Clone(Cell(b_, i));
        t.setPrecision(0);
        for (size_t k = 0; k < i; ++k) {
            AV lik = Cell(f.l, i * p + k);
            AV yk = Cell(y, k);
            t -= *(*(lik * yk) / scale);
        }
        AV cell = Cell(y, i);
        cell = *(*(t * Cell(f.inv_diag, i)) / scale);
    }

    AV x(p, engine);
    x.setPrecision(0);
    for (size_t i = p; i-- > 0;) {
        AV t = Clone(Cell(y, i));
        t.setPrecision(0);
        for (size_t k = i + 1; k < p; ++k) {
            AV lki = Cell(f.l, k * p + i);
            AV xk = Cell(x, k);
            t -= *(*(lki * xk) / scale);
        }
        AV cell = Cell(x, i);
        cell = *(*(t * Cell(f.inv_diag, i)) / scale);
    }
    x.setPrecision(0);
    return x;
}

AV CholeskySolve(const AV& a, const AV& b, size_t p) {
    return CholeskySolveWith(CholeskyFactor(a, p), b, p);
}

// Full inverse of an SPD matrix: one factorisation, then p solves against the
// unit vectors. Returned row-major.
AV SymmetricInverse(const AV& a, size_t p) {
    EngineRef engine = a.engine;
    Cholesky f = CholeskyFactor(a, p);
    AV inv(p * p, engine);
    inv.setPrecision(0);
    for (size_t k = 0; k < p; ++k) {
        std::vector<double> e(p, 0.0);
        e[k] = 1.0;
        AV col = CholeskySolveWith(f, PublicVector(engine, e), p);
        for (size_t i = 0; i < p; ++i) {
            AV cell = Cell(inv, i * p + k);
            cell = Cell(col, i);
        }
    }
    inv.setPrecision(0);
    return inv;
}

// Weighted Gram matrix X^T W X for X stored row-major as n*p, W a length-n
// vector of weights. Returned p*p row-major.
//
// The p(p+1)/2 distinct entries are computed in a SINGLE dot_product call: the
// two operands are built as public gathers (mapping_reference is a free view,
// no communication) that lay the required column pairs out end to end, and the
// engine then contracts each length-n chunk locally. Only p(p+1)/2 ring
// elements cross the wire, independent of n.
AV Gram(const AV& x_rm, const AV& w, size_t n, size_t p) {
    EngineRef engine = x_rm.engine;

    AV x_ = x_rm;
    x_.setPrecision(0);
    AV w_ = w;
    w_.setPrecision(0);

    // Z = W * X, row-scaled. repeated_subset_reference repeats each weight p
    // times, which lines it up with the row-major layout of X.
    AV z = *(*(x_ * w_.repeated_subset_reference(p)) / scale);
    z.setPrecision(0);

    std::vector<size_t> pairs_k, pairs_l;
    for (size_t k = 0; k < p; ++k) {
        for (size_t l = k; l < p; ++l) {
            pairs_k.push_back(k);
            pairs_l.push_back(l);
        }
    }
    const size_t num_pairs = pairs_k.size();

    std::vector<cdough::VectorSizeType> lhs_map(num_pairs * n), rhs_map(num_pairs * n);
    for (size_t t = 0; t < num_pairs; ++t) {
        for (size_t i = 0; i < n; ++i) {
            lhs_map[t * n + i] = static_cast<cdough::VectorSizeType>(i * p + pairs_k[t]);
            rhs_map[t * n + i] = static_cast<cdough::VectorSizeType>(i * p + pairs_l[t]);
        }
    }

    AV lhs = z.mapping_reference(lhs_map);
    AV rhs = x_.mapping_reference(rhs_map);
    lhs.setPrecision(0);
    rhs.setPrecision(0);

    AV packed = *(*lhs.dot_product(rhs, n) / scale);
    packed.setPrecision(0);

    AV g(p * p, engine);
    g.setPrecision(0);
    for (size_t t = 0; t < num_pairs; ++t) {
        AV v = Cell(packed, t);
        AV up = Cell(g, pairs_k[t] * p + pairs_l[t]);
        up = v;
        AV lo = Cell(g, pairs_l[t] * p + pairs_k[t]);
        lo = v;
    }
    g.setPrecision(0);
    return g;
}

// Add a public ridge to the diagonal, in place.
//
// Not optional. At precision 16 the IRLS weight p(1-p) truncates to zero for
// |eta| >= 12, so under the quasi-separation that a 8-parameter model with an
// interaction term invites, X^T W X goes singular and the Cholesky diagonal
// stops being positive. The ridge keeps the matrix SPD, which is exactly the
// assumption that licenses factorising without pivoting.
void AddRidge(AV& g, size_t p, double lambda) {
    const DataType lam = static_cast<DataType>(std::llround(lambda * scale));
    g.setPrecision(0);
    for (size_t i = 0; i < p; ++i) {
        AV cell = Cell(g, i * p + i);
        cell += lam;
    }
}

// Secure clamping to [-kMaxNewtonStep, kMaxNewtonStep]
AV ClampNewtonStep(const AV& step) {
    return ClampAbs(step, kMaxNewtonStep_scaled);
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
    // False when no curvature update ever passed the y^T s > 0 test, in which
    // case h_inv is still the identity and its diagonal is NOT a variance.
    bool hessian_updated = false;

    OptResult(std::vector<AV> p, AV v, int it = 0, bool conv = false)
        : params(std::move(p)), value(std::move(v)), iterations(it), converged(conv) {}
};

// Minimizes `f` starting from `x0` using BFGS with a backtracking (Armijo) line
// search and numerical gradients.
// `h_inv_out`, when given, receives the converged inverse-Hessian approximation.
// For a negative log-likelihood objective that is the asymptotic covariance
// matrix of the estimates, so the standard errors come out of the optimisation
// for free. It is a quasi-Newton approximation rather than the exact observed
// information -- unlike the IRLS models, whose covariance is exact -- and the
// report says so.
OptResult MinimizeBFGS(
    const std::function<AV(const std::vector<AV>&)>& f,
    const std::vector<AV>& x0,
    int max_iterations = 20,
    SMatrix* h_inv_out = nullptr) {
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
            result.hessian_updated = true;
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
    if (h_inv_out != nullptr) *h_inv_out = h_inv;
    return result;
}


// =============================================================================
// The analysis cohort
//
// One row per patient per acute-care encounter, flat, SORTED BY subject_id and
// padded to a power of two -- the layout the segmented scans above require. The
// twelve mixed models cluster on subject_id, and patients have a ragged number
// of encounters, so nothing here is padded per patient.
// =============================================================================

// Integer codes for the two CLASS variables. gender is varchar(20) in the source
// schema; MPC has no strings, so the ETL must hand over an integer code, and the
// mapping has to be pinned here because it decides which level is the reference.
//
// PROC GLIMMIX CLASS makes the LAST sorted level the reference. patsy and lme4
// use the FIRST. Getting this backwards silently flips the sign of every
// categorical coefficient, so the reference is stated explicitly rather than
// inherited from whatever the library happens to do.
const std::vector<DataType> kGenderLevels = {1, 2, 3};  // 3 (unknown/other) is the reference
const std::vector<DataType> kHispanicLevels = {0, 1};   // 1 is the reference, per SAS
const char* const kGenderNames[] = {"female", "male", "other/unknown"};

// fu_month only ever takes these values; anything else is NULL upstream.
const std::vector<DataType> kFuWindows = {1, 3, 6};

// Key sentinel for pad rows. Must not collide with a real subject_id, and must
// sort last so the pad block forms its own segment under both scan directions.
const DataType kKeySentinel = std::numeric_limits<DataType>::max();

// Which of the three source tables a cohort represents.
enum class SystemScope { Any, UMass, NonUMass };

const char* ScopeName(SystemScope s) {
    switch (s) {
        case SystemScope::Any: return "any_system";
        case SystemScope::UMass: return "umass_system";
        default: return "nonumass_system";
    }
}

const char* ScopeLabel(SystemScope s) {
    switch (s) {
        case SystemScope::Any: return "all systems";
        case SystemScope::UMass: return "UMass only";
        default: return "non-UMass only";
    }
}

// Plaintext form of one input table. Used to build the synthetic cohort, to
// parse CSV, and as the oracle the secure results are checked against.
struct PlainCohort {
    SystemScope scope = SystemScope::Any;
    std::vector<DataType> subject_id;
    std::vector<DataType> newage;
    std::vector<DataType> gender;
    std::vector<DataType> hispanic;
    std::vector<DataType> index_visit;
    std::vector<DataType> visit_num;
    std::vector<DataType> final_visit;
    std::vector<DataType> fu_month;          // 0/1/3/6; meaningless where present = 0
    std::vector<DataType> fu_month_present;  // 0 encodes SQL NULL
    std::vector<DataType> data_source;       // 1 = UMass, 2 = non-UMass
    std::vector<DataType> sisa;              // suicide_acutecare_icd_narrow, the outcome
    std::vector<DataType> sa;                // sa_icd_narrow

    size_t rows() const { return subject_id.size(); }

    // Sort by subject_id, then by visit_num within a patient. The segmented
    // scans compare adjacent keys, so grouping is only correct once sorted.
    void SortBySubject() {
        std::vector<size_t> order(rows());
        std::iota(order.begin(), order.end(), size_t{0});
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            if (subject_id[a] != subject_id[b]) return subject_id[a] < subject_id[b];
            return visit_num[a] < visit_num[b];
        });
        auto permute = [&](std::vector<DataType>& v) {
            std::vector<DataType> out(v.size());
            for (size_t i = 0; i < order.size(); ++i) out[i] = v[order[i]];
            v.swap(out);
        };
        permute(subject_id); permute(newage); permute(gender); permute(hispanic);
        permute(index_visit); permute(visit_num); permute(final_visit);
        permute(fu_month); permute(fu_month_present); permute(data_source);
        permute(sisa); permute(sa);
    }
};

// Secret-shared form. Flags and small integers are held as arithmetic 0/1 (or
// small ints) at precision 0; everything that feeds arithmetic is scaled.
struct SecureCohort {
    SystemScope scope = SystemScope::Any;
    size_t n = 0;      // real rows
    size_t n_pad = 0;  // padded to a power of two for the segmented scans

    BV subject_key;  // B-shared sort/group key, sentinel on pad rows
    std::vector<BV> keys;  // {subject_key}, the form the scan helpers take

    AV valid;         // 1 on real rows, 0 on pad rows (arithmetic)
    AV newage;        // scaled
    AV visit_num;     // scaled
    AV fu_month;      // scaled
    AV fu_present;    // 0/1
    AV index_visit;   // 0/1
    AV final_visit;   // 0/1
    AV sisa;          // 0/1
    AV sa;            // 0/1
    AV umass;         // 0/1, 1 where data_source == 1
    AV hispanic;      // 0/1 raw value (also used for the frequency table)
    std::vector<AV> gender_is;    // one 0/1 indicator per level in kGenderLevels
    std::vector<AV> hispanic_is;  // one 0/1 indicator per level in kHispanicLevels

    AV last_of_subject;   // 1 on the last row of each patient (arithmetic)
    AV first_of_subject;  // 1 on the first row of each patient (arithmetic)

    // Shared vectors carry an engine reference and so have no default
    // constructor; everything is sized to n_pad up front and filled in later.
    SecureCohort(EngineRef engine, size_t rows, size_t padded)
        : n(rows), n_pad(padded),
          subject_key(padded, engine),
          valid(padded, engine),
          newage(padded, engine),
          visit_num(padded, engine),
          fu_month(padded, engine),
          fu_present(padded, engine),
          index_visit(padded, engine),
          final_visit(padded, engine),
          sisa(padded, engine),
          sa(padded, engine),
          umass(padded, engine),
          hispanic(padded, engine),
          last_of_subject(padded, engine),
          first_of_subject(padded, engine) {}

    // Number of patients. Published: it is already the first reported column of
    // both descriptive nodes, so treating it as secret would protect nothing
    // while turning every quantile into an oblivious linear scan.
    size_t num_subjects = 0;
};

// =============================================================================
// Synthetic cohort
//
// Generated from a FIXED seed so every party derives an identical plaintext
// table. That matters for more than reproducibility: the number of encounters
// per patient is itself random, so a per-party RNG would give the parties
// different row counts and the shares would not line up. Party 0's copy is the
// one that gets secret shared; the others are used only as the oracle.
// =============================================================================

struct SyntheticTruth {
    double beta0 = 0.0;
    double beta_visit = 0.0;
    double beta_fu = 0.0;
    double beta_age = 0.0;
    double sigma = 0.0;
    double slope_diff = 0.0;  // the step 6 estimand: UMass vs non-UMass time slope
};

PlainCohort MakeSyntheticCohort(size_t num_subjects, SyntheticTruth& truth, uint64_t seed = 20260910ull) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::normal_distribution<double> gauss(0.0, 1.0);

    truth.beta0 = -1.60;
    truth.beta_visit = 0.11;
    truth.beta_fu = 0.045;
    truth.beta_age = -0.012;
    truth.sigma = 0.70;
    truth.slope_diff = 0.06;

    PlainCohort c;
    c.scope = SystemScope::Any;

    for (size_t sid = 1; sid <= num_subjects; ++sid) {
        // Right-skewed visit count. A large share of patients have exactly one
        // encounter -- the lineage page notes those contribute no within-patient
        // information to any of the trend models, so the distribution matters.
        const double u = uni(rng);
        int visits = 1;
        if (u > 0.42) visits = 2;
        if (u > 0.66) visits = 3;
        if (u > 0.80) visits = 4;
        if (u > 0.88) visits = 5 + static_cast<int>(uni(rng) * 4);
        if (u > 0.97) visits = 9 + static_cast<int>(uni(rng) * 8);

        const DataType gender = kGenderLevels[static_cast<size_t>(uni(rng) * 3.0) % 3];
        const DataType hispanic = uni(rng) < 0.12 ? 1 : 0;
        const int base_age = 14 + static_cast<int>(uni(rng) * 55.0);
        const double u_subject = truth.sigma * gauss(rng);
        // A patient's encounters mostly stay inside one system.
        const double umass_propensity = uni(rng) < 0.55 ? 0.85 : 0.15;

        for (int v = 1; v <= visits; ++v) {
            const bool is_umass = uni(rng) < umass_propensity;
            const DataType data_source = is_umass ? 1 : 2;

            // fu_month is null outside the windows and on a same-day repeat.
            DataType fu = 0, fu_present = 1;
            if (v == 1) {
                fu = 0;
            } else {
                const double f = uni(rng);
                if (f < 0.28) fu = 1;
                else if (f < 0.58) fu = 3;
                else if (f < 0.82) fu = 6;
                else fu_present = 0;  // beyond 182 days, or a same-day repeat
            }

            const double time_visit = static_cast<double>(v);
            const double time_fu = static_cast<double>(fu);
            const double age = base_age + (v - 1) * 0.4;
            const double slope = truth.beta_visit + (is_umass ? truth.slope_diff : 0.0);
            const double eta = truth.beta0 + slope * time_visit +
                               truth.beta_fu * time_fu + truth.beta_age * (age - 40.0) +
                               (gender == 1 ? 0.18 : 0.0) + (hispanic == 1 ? 0.10 : 0.0) +
                               u_subject;
            const double prob = 1.0 / (1.0 + std::exp(-eta));
            const DataType sisa = uni(rng) < prob ? 1 : 0;
            const DataType sa = (sisa == 1 && uni(rng) < 0.35) ? 1 : 0;

            c.subject_id.push_back(static_cast<DataType>(sid));
            c.newage.push_back(static_cast<DataType>(std::llround(age)));
            c.gender.push_back(gender);
            c.hispanic.push_back(hispanic);
            c.index_visit.push_back(v == 1 ? 1 : 0);
            c.visit_num.push_back(v);
            c.final_visit.push_back(v == visits ? 1 : 0);
            c.fu_month.push_back(fu);
            c.fu_month_present.push_back(fu_present);
            c.data_source.push_back(data_source);
            c.sisa.push_back(sisa);
            c.sa.push_back(sa);
        }
    }
    c.SortBySubject();
    return c;
}

// Derive umass_system / nonumass_system from any_system, the way the upstream
// ETL does: filter on data_source, then RE-SEQUENCE the per-system columns,
// because a patient's third encounter overall may be their first in this system.
//
// The `reproduce_etl_bug` flag deliberately reproduces the fault documented on
// the lineage page: the ETL sets fu_month<sfx> = 0 by testing the GLOBAL
// index_visit rather than index_visit<sfx>, so a patient whose history in this
// system starts later than their overall history gets NULL on their first row
// here. It is on by default because the point of the port is to match the
// pipeline as it actually runs, not as it was meant to.
PlainCohort DeriveSystemCohort(const PlainCohort& any, SystemScope scope,
                               bool reproduce_etl_bug = true) {
    const DataType want = (scope == SystemScope::UMass) ? 1 : 2;
    PlainCohort c;
    c.scope = scope;

    // Rows for this system, in the order they already appear (sorted by subject).
    std::vector<size_t> rows;
    for (size_t i = 0; i < any.rows(); ++i)
        if (any.data_source[i] == want) rows.push_back(i);

    size_t i = 0;
    while (i < rows.size()) {
        size_t j = i;
        while (j < rows.size() && any.subject_id[rows[j]] == any.subject_id[rows[i]]) ++j;
        const size_t count = j - i;
        for (size_t k = 0; k < count; ++k) {
            const size_t src = rows[i + k];
            const int v = static_cast<int>(k) + 1;

            DataType fu = any.fu_month[src];
            DataType fu_present = any.fu_month_present[src];
            if (v == 1) {
                if (reproduce_etl_bug && any.index_visit[src] != 1) {
                    // First encounter in THIS system, but not the patient's first
                    // overall: the upstream guard misses it and leaves it null.
                    fu = 0;
                    fu_present = 0;
                } else {
                    fu = 0;
                    fu_present = 1;
                }
            }

            c.subject_id.push_back(any.subject_id[src]);
            c.newage.push_back(any.newage[src]);
            c.gender.push_back(any.gender[src]);
            c.hispanic.push_back(any.hispanic[src]);
            c.index_visit.push_back(v == 1 ? 1 : 0);
            c.visit_num.push_back(v);
            c.final_visit.push_back(k + 1 == count ? 1 : 0);
            c.fu_month.push_back(fu);
            c.fu_month_present.push_back(fu_present);
            c.data_source.push_back(any.data_source[src]);
            c.sisa.push_back(any.sisa[src]);
            c.sa.push_back(any.sa[src]);
        }
        i = j;
    }
    return c;
}

// =============================================================================
// Ingestion
//
// Each of the three tables has a designated input party that holds it in the
// clear, mirroring examples/ex6_three_party_private_input.cpp and
// docker/DEPLOYMENT.md: every party calls the loader, only the owner opens the
// file, and plaintext never crosses the wire.
//
// Because the owner holds the table, the sort by subject_id and the categorical
// indicator coding are done locally in plaintext before sharing. That reveals
// nothing the owner does not already know and avoids an oblivious sort over the
// whole table. A deployment in which NO party holds any_system in the clear
// would need an oblivious sort and merge instead; that is out of scope here.
// =============================================================================

// Column order of the CSV. Bracketed names mark the columns that would be
// B-shared under the EncodedTable convention; kept here for documentation and
// so the header check is exact.
const std::vector<std::string> kCsvHeader = {
    "[subject_id]", "newage",      "[gender]",       "[hispanic]",
    "[index_visit]", "visit_num",  "[final_visit]",  "[fu_month]",
    "[fu_month_present]", "[data_source]", "[sisa]", "[sa]"};

std::string CohortCsvName(SystemScope scope) {
    return std::string(ScopeName(scope)) + ".csv";
}

// Write a cohort out in the exact format the loader expects. Used to dump the
// synthetic cohort for the R / Python / SQLite cross-checks.
void WriteCohortCsv(const PlainCohort& c, const std::string& path) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "could not open " << path << " for writing" << std::endl;
        return;
    }
    for (size_t i = 0; i < kCsvHeader.size(); ++i)
        out << kCsvHeader[i] << (i + 1 == kCsvHeader.size() ? '\n' : ',');
    for (size_t i = 0; i < c.rows(); ++i) {
        out << c.subject_id[i] << ',' << c.newage[i] << ',' << c.gender[i] << ','
            << c.hispanic[i] << ',' << c.index_visit[i] << ',' << c.visit_num[i] << ','
            << c.final_visit[i] << ',' << c.fu_month[i] << ',' << c.fu_month_present[i] << ','
            << c.data_source[i] << ',' << c.sisa[i] << ',' << c.sa[i] << '\n';
    }
}

// Read a cohort CSV. Header order is free -- columns are matched by name, as
// EncodedTable::inputCSVTableData does -- but every expected name must be
// present. Values are integers only.
PlainCohort ReadCohortCsv(const std::string& path, SystemScope scope) {
    PlainCohort c;
    c.scope = scope;
    std::ifstream in(path);
    if (!in) {
        std::cerr << "FATAL: could not open " << path << std::endl;
        std::exit(1);
    }

    std::string line;
    if (!std::getline(in, line)) {
        std::cerr << "FATAL: " << path << " is empty" << std::endl;
        std::exit(1);
    }

    std::vector<std::string> header;
    {
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            while (!tok.empty() && (tok.back() == '\r' || tok.back() == ' ')) tok.pop_back();
            header.push_back(tok);
        }
    }
    std::map<std::string, size_t> pos;
    for (size_t i = 0; i < header.size(); ++i) pos[header[i]] = i;
    for (const std::string& want : kCsvHeader) {
        if (pos.find(want) == pos.end()) {
            std::cerr << "FATAL: column " << want << " missing from " << path << std::endl;
            std::exit(1);
        }
    }

    auto* const targets = new std::vector<DataType>*[kCsvHeader.size()]{
        &c.subject_id, &c.newage, &c.gender, &c.hispanic,
        &c.index_visit, &c.visit_num, &c.final_visit, &c.fu_month,
        &c.fu_month_present, &c.data_source, &c.sisa, &c.sa};

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) fields.push_back(tok);
        for (size_t k = 0; k < kCsvHeader.size(); ++k) {
            const size_t idx = pos[kCsvHeader[k]];
            targets[k]->push_back(idx < fields.size()
                                      ? static_cast<DataType>(std::stoll(fields[idx]))
                                      : DataType{0});
        }
    }
    delete[] targets;

    c.SortBySubject();
    return c;
}

// Secret share a plaintext cohort from `input_party`. Non-owning parties pass a
// PlainCohort of the same shape with zero contents; only the row count and the
// schema have to agree across parties.
SecureCohort ShareCohort(EngineRef engine, const PlainCohort& c, int input_party) {
    const size_t n = c.rows();
    const size_t np = NextPowerOfTwo(std::max<size_t>(n, 2));
    SecureCohort sc(engine, n, np);
    sc.scope = c.scope;

    {
        std::set<DataType> subjects(c.subject_id.begin(), c.subject_id.end());
        sc.num_subjects = subjects.size();
    }

    // Pad rows carry the key sentinel (so they form their own segment under both
    // scan directions) and zeros everywhere else (so they contribute nothing).
    auto share_int = [&](const std::vector<DataType>& src) {
        cdough::Vector<DataType> v(np, 0);
        for (size_t i = 0; i < np; ++i) v[i] = i < n ? src[i] : DataType{0};
        AV out = engine.template secret_share_a<DataType>(v, input_party, 0);
        out.setPrecision(0);
        return out;
    };
    auto share_scaled = [&](const std::vector<DataType>& src) {
        cdough::Vector<DataType> v(np, precision);
        for (size_t i = 0; i < np; ++i)
            v[i] = i < n ? static_cast<DataType>(src[i]) * DataType(scale) : DataType{0};
        AV out = engine.template secret_share_a<DataType>(v, input_party, precision);
        out.setPrecision(0);
        return out;
    };
    auto share_indicator = [&](const std::vector<DataType>& src, DataType level) {
        cdough::Vector<DataType> v(np, 0);
        for (size_t i = 0; i < np; ++i) v[i] = (i < n && src[i] == level) ? 1 : 0;
        AV out = engine.template secret_share_a<DataType>(v, input_party, 0);
        out.setPrecision(0);
        return out;
    };

    {
        cdough::Vector<DataType> kv(np, 0);
        for (size_t i = 0; i < np; ++i) kv[i] = i < n ? c.subject_id[i] : kKeySentinel;
        sc.subject_key = engine.template secret_share_b<DataType>(kv, input_party);
    }
    sc.keys.push_back(sc.subject_key);

    {
        cdough::Vector<DataType> vv(np, 0);
        for (size_t i = 0; i < np; ++i) vv[i] = i < n ? 1 : 0;
        sc.valid = engine.template secret_share_a<DataType>(vv, input_party, 0);
        sc.valid.setPrecision(0);
    }

    sc.newage = share_scaled(c.newage);
    sc.visit_num = share_scaled(c.visit_num);
    sc.fu_month = share_scaled(c.fu_month);
    sc.fu_present = share_int(c.fu_month_present);
    sc.index_visit = share_int(c.index_visit);
    sc.final_visit = share_int(c.final_visit);
    sc.sisa = share_int(c.sisa);
    sc.sa = share_int(c.sa);
    sc.hispanic = share_int(c.hispanic);
    sc.umass = share_indicator(c.data_source, 1);

    for (DataType lvl : kGenderLevels) sc.gender_is.push_back(share_indicator(c.gender, lvl));
    for (DataType lvl : kHispanicLevels)
        sc.hispanic_is.push_back(share_indicator(c.hispanic, lvl));

    // Group-boundary indicators, computed once and reused by every node. The
    // AND with `valid` is what keeps the sentinel pad block from contributing a
    // spurious final group.
    AV last = LastOfGroupArith(sc.keys);
    sc.last_of_subject = *(last * sc.valid);
    sc.last_of_subject.setPrecision(0);

    BV first_b = FirstOfGroup(sc.keys);
    AV first = *(first_b.b2a_bit());
    first.setPrecision(0);
    sc.first_of_subject = *(first * sc.valid);
    sc.first_of_subject.setPrecision(0);

    return sc;
}

// =============================================================================
// Descriptive nodes (d1a, d1b) and the aggregate tables (sisa_perct_cnt)
// =============================================================================

// Open a 1-element vector. `scaled` distinguishes fixed-point values from raw
// counts, which are held as plain integers at precision 0.
double OpenScalar(const AV& v, bool scaled = true) {
    auto opened = v.open();
    const double raw = static_cast<double>(opened[0]);
    return scaled ? raw / scale : raw;
}

// Sum of a 0/1 mask: a count.
AV MaskCount(const AV& mask) {
    AV c = mask.chunkedSum(mask.size());
    c.setPrecision(0);
    return c;
}

// Sum of a scaled column over the masked rows.
AV MaskedSum(const AV& x_scaled, const AV& mask) {
    AV prod = *(x_scaled * mask);  // mask is an unscaled 0/1, so no rescale
    AV s = prod.chunkedSum(prod.size());
    s.setPrecision(0);
    return s;
}

// COUNT(*) of masked rows whose integer value is <= threshold.
AV MaskedCountAtMost(const AV& x_scaled, const AV& mask, long threshold) {
    AV diff = -x_scaled;
    diff += static_cast<DataType>(threshold) * DataType(scale);
    AV le = *(diff.gtez());  // 1 where x <= threshold
    AV hit = *(le * mask);
    AV c = hit.chunkedSum(hit.size());
    c.setPrecision(0);
    return c;
}

// The k-th smallest masked value (0-based), by binary search on the CDF over a
// PUBLIC integer range.
//
// One comparison bit is opened per step. The search path is a function of the
// returned order statistic and nothing else, so this discloses exactly the
// quantile that the pipeline publishes anyway -- not the underlying values, and
// not the rest of the distribution. That is the whole reason for preferring this
// to an oblivious sort: it is cheaper AND it leaks strictly less than revealing
// a full histogram would.
long SecureOrderStat(const AV& x_scaled, const AV& mask, long k, long lo, long hi) {
    while (lo < hi) {
        const long mid = lo + (hi - lo) / 2;
        const double cnt = OpenScalar(MaskedCountAtMost(x_scaled, mask, mid), false);
        if (cnt >= static_cast<double>(k) + 0.5) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}

// PROC UNIVARIATE's moments and quantiles for one column over the masked rows.
struct Univariate {
    long n = 0;
    double mean = 0.0, sd = 0.0;
    double minimum = 0.0, q1 = 0.0, median = 0.0, q3 = 0.0, maximum = 0.0;
};

// Quantiles use PCTLDEF=4 (linear interpolation), which is what PERCENTILE_CONT
// computes. PROC UNIVARIATE defaults to PCTLDEF=5 (empirical with averaging), so
// on small samples the two can differ by one observation's worth; adding
// PCTLDEF=4 to the SAS call makes them agree.
Univariate SecureUnivariate(const AV& x_scaled, const AV& mask, long lo, long hi) {
    Univariate u;
    u.n = static_cast<long>(std::llround(OpenScalar(MaskCount(mask), false)));
    if (u.n <= 0) return u;

    AV n_sec = MaskCount(mask);
    n_sec.setPrecision(0);
    AV n_scaled = *(n_sec * DataType(scale));

    AV total = MaskedSum(x_scaled, mask);
    AV mean = Div(total, n_scaled);
    mean.setPrecision(0);
    u.mean = OpenScalar(mean);

    // Two-pass variance: subtract the mean first rather than forming sum(x^2).
    // Numerically safer, and it keeps the accumulator inside int64 for the
    // largest cohorts this program is meant to run on.
    AV centred = x_scaled - mean.repeated_subset_reference(x_scaled.size());
    centred.setPrecision(0);
    AV sq = *(*(centred * centred) / scale);
    AV ss = MaskedSum(sq, mask);
    if (u.n > 1) {
        AV denom(1, x_scaled.engine);
        denom.setPrecision(0);
        denom += static_cast<DataType>(u.n - 1) * DataType(scale);
        u.sd = std::sqrt(std::max(0.0, OpenScalar(Div(ss, denom))));
    }

    auto order_stat = [&](long k) {
        return static_cast<double>(SecureOrderStat(x_scaled, mask, k, lo, hi));
    };
    auto pctl = [&](double q) {
        const double h = (static_cast<double>(u.n) - 1.0) * q;
        const long lo_idx = static_cast<long>(std::floor(h));
        const double frac = h - static_cast<double>(lo_idx);
        const double a = order_stat(lo_idx);
        if (frac <= 0.0 || lo_idx + 1 >= u.n) return a;
        return a + frac * (order_stat(lo_idx + 1) - a);
    };

    u.minimum = order_stat(0);
    u.q1 = pctl(0.25);
    u.median = pctl(0.50);
    u.q3 = pctl(0.75);
    u.maximum = order_stat(u.n - 1);
    return u;
}

void PrintUnivariate(const std::string& label, const Univariate& u) {
    std::cout << "\n  " << label << "\n"
              << "    n       " << u.n << "\n"
              << "    mean    " << std::fixed << std::setprecision(4) << u.mean << "\n"
              << "    sd      " << u.sd << "\n"
              << "    min     " << u.minimum << "\n"
              << "    q1      " << u.q1 << "\n"
              << "    median  " << u.median << "\n"
              << "    q3      " << u.q3 << "\n"
              << "    max     " << u.maximum << std::defaultfloat << std::endl;
}

// A one-way frequency table over a PUBLIC level set. Cheaper than an oblivious
// group-by and it discloses nothing beyond the counts, which are the published
// output. Missing values are excluded from the denominator, matching PROC FREQ.
struct FreqRow {
    long value = 0;
    long frequency = 0;
    double percent = 0.0;
};

std::vector<FreqRow> SecureFrequency(const std::vector<AV>& indicators,
                                     const std::vector<DataType>& levels, const AV& mask) {
    std::vector<FreqRow> rows;
    long total = 0;
    for (size_t i = 0; i < levels.size(); ++i) {
        AV hit = *(indicators[i] * mask);
        const long f = static_cast<long>(std::llround(OpenScalar(MaskCount(hit), false)));
        rows.push_back(FreqRow{static_cast<long>(levels[i]), f, 0.0});
        total += f;
    }
    for (FreqRow& r : rows)
        r.percent = total > 0 ? 100.0 * static_cast<double>(r.frequency) / total : 0.0;
    return rows;
}

// The visit-count distribution: a frequency table over a public range of integer
// values. The counts ARE the published output of node d1a, so revealing them is
// the point.
std::vector<FreqRow> SecureValueHistogram(const AV& x_scaled, const AV& mask, long lo, long hi) {
    std::vector<FreqRow> rows;
    long total = 0;
    double prev_cdf = 0.0;
    for (long v = lo; v <= hi; ++v) {
        const double cdf = OpenScalar(MaskedCountAtMost(x_scaled, mask, v), false);
        const long f = static_cast<long>(std::llround(cdf - prev_cdf));
        prev_cdf = cdf;
        if (f > 0) rows.push_back(FreqRow{v, f, 0.0});
        total += f;
    }
    for (FreqRow& r : rows)
        r.percent = total > 0 ? 100.0 * static_cast<double>(r.frequency) / total : 0.0;
    return rows;
}

void PrintFrequency(const std::string& label, const std::string& value_header,
                    const std::vector<FreqRow>& rows) {
    std::cout << "\n  " << label << "\n    " << std::left << std::setw(22) << value_header
              << std::setw(12) << "frequency" << std::setw(10) << "percent" << std::endl;
    for (const FreqRow& r : rows)
        std::cout << "    " << std::left << std::setw(22) << r.value << std::setw(12)
                  << r.frequency << std::fixed << std::setprecision(2) << r.percent
                  << std::defaultfloat << std::endl;
}

// --- node d1a: visits per patient --------------------------------------------
//
// final_visit = 1 keeps exactly one row per patient -- their last encounter --
// and visit_num on that row is a running count, so it equals the patient total.
void ReportD1a(const SecureCohort& c, int party_id, long max_visits) {
    AV mask = *(c.final_visit * c.valid);
    Univariate u = SecureUnivariate(c.visit_num, mask, 1, max_visits);
    std::vector<FreqRow> hist = SecureValueHistogram(c.visit_num, mask, 1, max_visits);
    if (party_id != 0) return;

    std::cout << "\n=== d1a  visits per patient  [" << ScopeName(c.scope)
              << ", final_visit = 1] ===" << std::endl;
    PrintUnivariate("visit_num", u);
    PrintFrequency("distribution", "visits_per_patient", hist);
    std::cout << "\n  NOTE: mean and sd are the wrong summary here -- visit counts are bounded\n"
                 "        below at 1 and heavily right skewed. The distribution above says more,\n"
                 "        in particular how many patients have exactly one visit: those patients\n"
                 "        contribute no within-patient information to any of the trend models.\n"
                 "  NOTE: quantiles use PCTLDEF=4 (linear interpolation). PROC UNIVARIATE\n"
                 "        defaults to PCTLDEF=5 and can differ by one observation's worth."
              << std::endl;
}

// --- node d1b: demographics at the index visit -------------------------------
void ReportD1b(const SecureCohort& c, int party_id) {
    AV mask = *(c.index_visit * c.valid);
    Univariate u = SecureUnivariate(c.newage, mask, 0, 120);
    std::vector<FreqRow> gender = SecureFrequency(c.gender_is, kGenderLevels, mask);
    std::vector<FreqRow> hispanic = SecureFrequency(c.hispanic_is, kHispanicLevels, mask);
    if (party_id != 0) return;

    std::cout << "\n=== d1b  demographics  [" << ScopeName(c.scope)
              << ", index_visit = 1] ===" << std::endl;
    PrintUnivariate("newage", u);
    PrintFrequency("gender", "gender", gender);
    for (size_t i = 0; i < gender.size(); ++i)
        std::cout << "      " << gender[i].value << " = " << kGenderNames[i] << std::endl;
    PrintFrequency("hispanic", "hispanic", hispanic);
    std::cout << "\n  NOTE: two one-way tables, not a crosstab -- the SAS asks for\n"
                 "        `table gender hispanic`, which is two separate tables.\n"
                 "  NOTE: check the hispanic cell counts here before trusting any model that\n"
                 "        adjusts for it. Sparsity here is what justifies dropping it from\n"
                 "        model 5b on the UMass subset."
              << std::endl;
}

// --- nodes sisa_perct_cnt{,_umass,_nonumass} ---------------------------------
struct SisaCounts {
    long window = 0;
    long pts_with_sisa = 0;
    long total_pts = 0;
    double sisa_pct = 0.0;
    long total_sa = 0;
    long null_fu_rows = 0;
};

std::vector<SisaCounts> ReportSisaCounts(SecureCohort& c, int party_id) {
    std::vector<SisaCounts> out;

    // Rows the follow-up windows cannot see at all.
    AV not_present = -c.fu_present;
    not_present += DataType(1);
    AV dropped = *(not_present * c.valid);
    const long null_rows = static_cast<long>(std::llround(OpenScalar(MaskCount(dropped), false)));

    for (DataType w : kFuWindows) {
        // in_window = fu_month is non-null AND equals w.
        //
        // Clone first: `AV a = b` is a SHALLOW copy that shares the underlying
        // buffer, so mutating it in place would corrupt the cohort column for
        // every later window.
        AV diff_lo = Clone(c.fu_month);
        diff_lo.setPrecision(0);
        diff_lo -= static_cast<DataType>(w) * DataType(scale);  // fu - w
        AV ge = *(diff_lo.gtez());
        AV diff_hi = -diff_lo;                                  // w - fu
        AV le = *(diff_hi.gtez());
        AV eq = *(le * ge);
        AV in_window = *(*(eq * c.fu_present) * c.valid);

        AV with_sisa = *(in_window * c.sisa);

        // Percentages count DISTINCT PATIENTS; the attempt total counts
        // ENCOUNTERS, with no DISTINCT, so a patient with two coded attempts in a
        // window contributes two. That asymmetry is in the source query and is
        // preserved deliberately.
        AV total_pts = CountDistinct(c.keys, in_window);
        AV pts_with_sisa = CountDistinct(c.keys, with_sisa);
        AV total_sa = MaskCount(*(in_window * c.sa));

        SisaCounts r;
        r.window = static_cast<long>(w);
        r.total_pts = static_cast<long>(std::llround(OpenScalar(total_pts, false)));
        r.pts_with_sisa = static_cast<long>(std::llround(OpenScalar(pts_with_sisa, false)));
        r.total_sa = static_cast<long>(std::llround(OpenScalar(total_sa, false)));
        r.sisa_pct = r.total_pts > 0 ? 100.0 * static_cast<double>(r.pts_with_sisa) /
                                           static_cast<double>(r.total_pts)
                                     : 0.0;
        r.null_fu_rows = null_rows;
        out.push_back(r);
    }

    if (party_id != 0) return out;

    const std::string suffix = c.scope == SystemScope::Any
                                   ? ""
                                   : (c.scope == SystemScope::UMass ? "_umass" : "_nonumass");
    std::cout << "\n=== sisa_perct_cnt" << suffix << "  [" << ScopeLabel(c.scope)
              << "] ===" << std::endl;
    std::cout << "\n  " << std::left << std::setw(10) << "window" << std::setw(18)
              << "pts_with_sisa" << std::setw(14) << "total_pts" << std::setw(12) << "sisa_pct"
              << std::setw(12) << "total_sa" << std::endl;
    for (const SisaCounts& r : out)
        std::cout << "  " << std::left << std::setw(10) << (std::to_string(r.window) + "m")
                  << std::setw(18) << r.pts_with_sisa << std::setw(14) << r.total_pts
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.sisa_pct
                  << std::defaultfloat << std::setw(12) << r.total_sa << std::endl;

    std::cout << "\n  NOTE: the denominator is not the cohort. total_pts counts patients with a\n"
                 "        record in the window, so sisa_pct is a prevalence conditional on being\n"
                 "        observed then -- not cumulative incidence by that month.\n"
                 "  NOTE: total_sa counts ENCOUNTERS, not patients (no DISTINCT in the source).\n"
                 "  NOTE: " << null_rows
              << " rows have a null fu_month and are invisible to every line above,\n"
                 "        appearing in neither numerator nor denominator and in no exclusion count."
              << std::endl;
    return out;
}

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
    AV last_of_subject;
    std::vector<std::string> terms;
    long rows_used = 0;
    long rows_dropped_null_fu = 0;

    ModelData(EngineRef engine, size_t np, size_t width)
        : n_pad(np), p(width), x(np * width, engine), y(np, engine), row_mask(np, engine),
          last_of_subject(np, engine) {}
};

ModelData BuildDesign(const SecureCohort& c, const ModelSpec& spec) {
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

    md.rows_used = static_cast<long>(std::llround(OpenScalar(MaskCount(md.row_mask), false)));
    AV dropped = c.valid - md.row_mask;
    md.rows_dropped_null_fu =
        static_cast<long>(std::llround(OpenScalar(MaskCount(dropped), false)));
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
FitResult FitLogisticIrls(const ModelData& md, const ModelSpec& spec) {
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

    AV cov = SymmetricInverse(gram, p);

    FitResult r;
    r.step = spec.step;
    r.scope = spec.scope;
    r.terms = md.terms;
    r.has_random = false;
    r.iterations = kIrlsIterations;
    r.rows_used = md.rows_used;
    r.rows_dropped_null_fu = md.rows_dropped_null_fu;
    r.converged = true;

    const std::vector<double> beta_open = OpenToDoubles(beta);
    const std::vector<double> cov_open = OpenToDoubles(cov);
    for (size_t k = 0; k < p; ++k) {
        r.estimate.push_back(beta_open[k]);
        r.se.push_back(std::sqrt(std::max(0.0, cov_open[k * p + k])));
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

    std::vector<BV> keys = md.keys;

    for (int it = 0; it < kNewtonIterations; ++it) {
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

        // One call, both columns: the group bits are shared between them.
        std::vector<AV> in{resid, varp};
        std::vector<AV> out;
        out.emplace_back(n, engine);
        out.emplace_back(n, engine);
        SegTotal(keys, in, out);

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
AV FlatNegMarginalLogLik(const ModelData& md, const std::vector<AV>& params) {
    EngineRef engine = md.y.engine;
    const size_t n = md.n_pad, p = md.p;

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

    // Forward scan only: the totals are needed on the last row of each cluster,
    // not broadcast back, so one pass suffices.
    std::vector<BV> keys = md.keys;
    std::vector<AV> in{cll_row, varp_row};
    std::vector<AV> pre;
    pre.emplace_back(n, engine);
    pre.emplace_back(n, engine);
    SegScan(keys, in, pre, SegDirection::Forward);

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

    AV neg = -total;
    neg.setPrecision(precision);
    return neg;
}

constexpr int kGlmmBfgsIterations = 12;

// Step for the numerical observed-information Hessian below. A second difference
// divides by h^2, so it amplifies the objective's fixed-point noise by 1/h^2;
// the truncation error meanwhile grows as h^2. With an objective good to about
// 1e-4 at precision 16, the balance sits near h = eps^(1/4) ~ 0.1.
const double kHessianStep = 0.1;

// Above this, the observed-information matrix is ill-conditioned enough that the
// standard errors on its weakly determined directions should not be relied on.
const double kHessianConditionWarn = 1e4;

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
std::vector<double> ObservedInformationSE(
    const std::function<AV(const std::vector<AV>&)>& objective,
    const std::vector<AV>& optimum, size_t num_reported,
    double* condition_out = nullptr) {
    const size_t dim = optimum.size();
    EngineRef engine = optimum[0].engine;
    const double h = kHessianStep;

    auto shifted = [&](const std::vector<double>& deltas) {
        std::vector<AV> point;
        point.reserve(dim);
        for (size_t k = 0; k < dim; ++k) {
            AV v = Clone(optimum[k]);
            v.setPrecision(0);
            v += static_cast<DataType>(std::llround(deltas[k] * scale));
            v.setPrecision(precision);
            point.push_back(v);
        }
        return OpenScalar(objective(point));
    };

    const std::vector<double> zero(dim, 0.0);
    const double f0 = shifted(zero);

    std::vector<double> f_plus(dim), f_minus(dim);
    for (size_t k = 0; k < dim; ++k) {
        std::vector<double> d = zero;
        d[k] = h;
        f_plus[k] = shifted(d);
        d[k] = -h;
        f_minus[k] = shifted(d);
    }

    std::vector<double> hess(dim * dim, 0.0);
    for (size_t k = 0; k < dim; ++k)
        hess[k * dim + k] = (f_plus[k] - 2.0 * f0 + f_minus[k]) / (h * h);
    for (size_t j = 0; j < dim; ++j) {
        for (size_t k = j + 1; k < dim; ++k) {
            std::vector<double> d = zero;
            d[j] = h;
            d[k] = h;
            const double f_jk = shifted(d);
            const double v = (f_jk - f_plus[j] - f_plus[k] + f0) / (h * h);
            hess[j * dim + k] = v;
            hess[k * dim + j] = v;
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
    (void)engine;
    return se;
}

FitResult FitGlmmLaplace(const ModelData& md, const ModelSpec& spec, int party_id) {
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

    auto objective = [&md](const std::vector<AV>& params) -> AV {
        return FlatNegMarginalLogLik(md, params);
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

    for (size_t k = 0; k < p; ++k) r.estimate.push_back(OpenScalar(opt.params[k]));
    double condition = 0.0;
    r.se = ObservedInformationSE(objective, opt.params, p, &condition);
    r.hessian_condition = condition;
    const double s_hat = OpenScalar(opt.params[p]);
    r.sigma2 = std::exp(2.0 * s_hat);

    r.notes.push_back(
        "standard errors come from a finite-difference observed-information matrix at the "
        "optimum, not from the BFGS inverse-Hessian approximation");
    if (!opt.hessian_updated)
        r.notes.push_back(
            "no BFGS curvature update was accepted, so the optimiser may have stopped early");
    if (!(condition < kHessianConditionWarn)) {
        std::ostringstream note;
        note << "the observed-information matrix is ill-conditioned (1-norm condition estimate "
             << std::scientific << std::setprecision(2) << condition
             << "), so standard errors on the weakly determined terms -- typically the "
                "intercept and any near-collinear dummy such as hispanic -- are not reliable "
                "at this precision. The time coefficient is unaffected.";
        r.notes.push_back(note.str());
    }
    (void)party_id;
    return r;
}

// =============================================================================
// Reporting
//
// p-values, confidence limits and odds ratios are computed HERE, in plaintext,
// from the revealed (estimate, standard error) pair. That is deliberate: p is a
// deterministic public function of two numbers the pipeline publishes anyway, so
// evaluating the normal CDF under MPC would protect nothing -- and at precision
// 16 the smallest representable positive value is 1.5e-5, so a secure erf would
// floor out around p = 1e-4 regardless.
// =============================================================================

const double kZ975 = 1.959963985;

double TwoSidedNormalP(double z) { return std::erfc(std::abs(z) * M_SQRT1_2); }

std::string FormatP(double p_value) {
    std::ostringstream os;
    if (p_value < 1e-4) return "<0.0001";
    os << std::fixed << std::setprecision(4) << p_value;
    return os.str();
}

void PrintFit(const FitResult& r, const ModelSpec& spec) {
    std::cout << "\n=== model " << r.step << "  [" << ScopeLabel(r.scope) << "] ===\n"
              << "  outcome  suicide_acutecare_icd_narrow (event = 1), logit link\n"
              << "  time     " << TimeName(spec.time) << "\n"
              << "  fit      "
              << (r.has_random ? "random-intercept mixed model, Laplace (BFGS)"
                               : "fixed-effects logistic regression (IRLS)")
              << "\n"
              << "  rows     " << r.rows_used;
    if (r.rows_dropped_null_fu > 0)
        std::cout << "   (" << r.rows_dropped_null_fu << " dropped: fu_month is null)";
    std::cout << "\n  iters    " << r.iterations << (r.converged ? " (converged)" : "")
              << std::endl;

    const bool want_or = spec.covars || spec.interaction;
    std::cout << "\n  " << std::left << std::setw(34) << "Term" << std::setw(12) << "Estimate"
              << std::setw(12) << "StdErr" << std::setw(10) << "z" << std::setw(11) << "Pr>|z|";
    if (want_or) std::cout << std::setw(12) << "OddsRatio" << std::setw(22) << "95% CI (OR)";
    std::cout << std::endl;

    for (size_t k = 0; k < r.terms.size(); ++k) {
        const double est = r.estimate[k];
        const double se = r.se[k];
        const bool have_se = std::isfinite(se) && se > 0.0;
        const double z = have_se ? est / se : 0.0;
        std::ostringstream se_s, z_s, p_s;
        if (have_se) {
            se_s << std::fixed << std::setprecision(5) << se;
            z_s << std::fixed << std::setprecision(3) << z;
            p_s << FormatP(TwoSidedNormalP(z));
        } else {
            se_s << "n/a";
            z_s << "n/a";
            p_s << "n/a";
        }
        std::cout << "  " << std::left << std::setw(34) << r.terms[k] << std::fixed
                  << std::setprecision(5) << std::setw(12) << est << std::setw(12)
                  << se_s.str() << std::setw(10) << z_s.str() << std::setw(11) << p_s.str();
        if (want_or) {
            std::ostringstream ci;
            if (have_se) {
                ci << "(" << std::fixed << std::setprecision(3) << std::exp(est - kZ975 * se)
                   << ", " << std::exp(est + kZ975 * se) << ")";
            } else {
                ci << "n/a";
            }
            std::cout << std::setprecision(4) << std::setw(12) << std::exp(est) << std::setw(22)
                      << ci.str();
        }
        std::cout << std::defaultfloat << std::endl;
    }

    if (r.has_random)
        std::cout << "\n  random intercept variance (subject_id): " << std::fixed
                  << std::setprecision(5) << r.sigma2 << std::defaultfloat << std::endl;

    if (spec.covars)
        std::cout << "\n  NOTE: newage was centred at " << kAgeCentre
                  << " for conditioning, so the intercept is the\n"
                     "        log-odds at age "
                  << kAgeCentre << ". The age coefficient itself is unaffected." << std::endl;

    for (const std::string& n : r.notes) std::cout << "  NOTE: " << n << std::endl;

    // The faults carried over from the source script.
    if (spec.interaction) {
        std::cout
            << "  NOTE: this model has NO random effect. The SAS declares subject_id in CLASS\n"
               "        and never uses it -- there is no RANDOM statement -- so PROC GLIMMIX\n"
               "        treats every encounter as independent. With repeated encounters per\n"
               "        patient the standard errors are too small and the p-values too\n"
               "        optimistic, on precisely the comparison the script exists to make.\n"
               "        Reproduced here rather than corrected."
            << std::endl;
        std::cout << "  NOTE: the interaction coefficient IS the difference in "
                  << TimeName(spec.time)
                  << " slopes\n        between the two systems -- this is the estimand."
                  << std::endl;
    }
    if (spec.step == "6a")
        std::cout << "  NOTE: the SAS ESTIMATE statement references data_source*fu_month in a\n"
                     "        model containing data_source*visit_num, so SAS rejects it. The\n"
                     "        answer survives as the interaction coefficient above."
                  << std::endl;
    if (spec.step == "6b")
        std::cout << "  NOTE: method=laplace is omitted in the SAS, so that fit silently falls\n"
                     "        back to residual pseudo-likelihood and its estimates are not on the\n"
                     "        same footing as the other thirteen. This port uses the same IRLS as\n"
                     "        6a, so the two ARE comparable here -- unlike in the original."
                  << std::endl;
    if (spec.time == TimeAxis::FuMonth)
        std::cout << "  NOTE: fu_month takes only {0,1,3,6} but enters as a linear term, forcing\n"
                     "        a constant change in log-odds per month across an unevenly spaced\n"
                     "        grid. If it is a window label rather than elapsed time it belongs\n"
                     "        in CLASS. Rows where it is null are dropped from this fit entirely."
                  << std::endl;
    if (spec.drop_hispanic)
        std::cout << "  NOTE: hispanic is dropped from THIS model only (sparse data), so its\n"
                     "        odds ratios are not directly comparable with the non-UMass fit."
                  << std::endl;
    if (!spec.covars)
        std::cout << "  NOTE: unadjusted by design. Compare against model "
                  << (spec.step == "2a" ? "5a" : "5b")
                  << " before reading\n        anything causal into the coefficient." << std::endl;
    if (spec.scope != SystemScope::Any && spec.time == TimeAxis::FuMonth)
        std::cout << "  NOTE: the time axis inherits an upstream ETL fault -- fu_month is null on\n"
                     "        the first in-system encounter of any patient whose history here\n"
                     "        starts later than their overall history, because the ETL guarded it\n"
                     "        on the global index_visit. Those rows drop out of this fit."
                  << std::endl;
}

// =============================================================================
// Plaintext oracle
//
// A direct IRLS fit in double precision, used to score the secure fixed-effects
// models. The mixed models are scored against the parameters the synthetic
// cohort was generated from instead, since a plaintext IRLS is not their
// estimand.
// =============================================================================

struct PlainFit {
    std::vector<double> estimate;
    std::vector<double> se;
};

PlainFit PlainLogisticIrls(const std::vector<double>& x_rm, const std::vector<double>& y,
                           const std::vector<double>& mask, size_t n, size_t p,
                           double ridge, int iterations) {
    std::vector<double> beta(p, 0.0);
    std::vector<double> g(p * p, 0.0);

    for (int it = 0; it < iterations; ++it) {
        std::vector<double> w(n), r(n);
        for (size_t i = 0; i < n; ++i) {
            double eta = 0.0;
            for (size_t k = 0; k < p; ++k) eta += x_rm[i * p + k] * beta[k];
            eta = std::max(-10.0, std::min(10.0, eta));
            const double pr = 1.0 / (1.0 + std::exp(-eta));
            w[i] = mask[i] * pr * (1.0 - pr);
            r[i] = mask[i] * (y[i] - pr);
        }
        std::vector<double> rhs(p, 0.0);
        g.assign(p * p, 0.0);
        for (size_t i = 0; i < n; ++i) {
            for (size_t k = 0; k < p; ++k) {
                rhs[k] += x_rm[i * p + k] * r[i];
                for (size_t l = k; l < p; ++l) {
                    const double v = w[i] * x_rm[i * p + k] * x_rm[i * p + l];
                    g[k * p + l] += v;
                    if (l != k) g[l * p + k] += v;
                }
            }
        }
        for (size_t k = 0; k < p; ++k) g[k * p + k] += ridge;

        // Gauss-Jordan solve and inverse in one pass.
        std::vector<double> m(g), inv(p * p, 0.0), sol(rhs);
        for (size_t k = 0; k < p; ++k) inv[k * p + k] = 1.0;
        for (size_t c = 0; c < p; ++c) {
            size_t piv = c;
            for (size_t rr = c + 1; rr < p; ++rr)
                if (std::abs(m[rr * p + c]) > std::abs(m[piv * p + c])) piv = rr;
            for (size_t k = 0; k < p; ++k) {
                std::swap(m[c * p + k], m[piv * p + k]);
                std::swap(inv[c * p + k], inv[piv * p + k]);
            }
            std::swap(sol[c], sol[piv]);
            const double d = m[c * p + c];
            if (std::abs(d) < 1e-15) continue;
            for (size_t k = 0; k < p; ++k) {
                m[c * p + k] /= d;
                inv[c * p + k] /= d;
            }
            sol[c] /= d;
            for (size_t rr = 0; rr < p; ++rr) {
                if (rr == c) continue;
                const double f = m[rr * p + c];
                for (size_t k = 0; k < p; ++k) {
                    m[rr * p + k] -= f * m[c * p + k];
                    inv[rr * p + k] -= f * inv[c * p + k];
                }
                sol[rr] -= f * sol[c];
            }
        }
        for (size_t k = 0; k < p; ++k)
            beta[k] += std::max(-4.0, std::min(4.0, sol[k]));

        if (it + 1 == iterations) {
            PlainFit out;
            out.estimate = beta;
            for (size_t k = 0; k < p; ++k)
                out.se.push_back(std::sqrt(std::max(0.0, inv[k * p + k])));
            return out;
        }
    }
    PlainFit out;
    out.estimate = beta;
    out.se.assign(p, 0.0);
    return out;
}

// Build the same design matrix the secure path builds, but in the clear.
void PlainDesign(const PlainCohort& c, const ModelSpec& spec, std::vector<double>& x_rm,
                 std::vector<double>& y, std::vector<double>& mask, size_t& p) {
    std::vector<std::string> dummy_terms;
    p = 2;  // intercept + time
    if (spec.covars) {
        p += 1;                            // newage
        p += kGenderLevels.size() - 1;     // gender dummies
        if (!spec.drop_hispanic) p += kHispanicLevels.size() - 1;
    }
    if (spec.interaction) p += 2;

    const size_t n = c.rows();
    x_rm.assign(n * p, 0.0);
    y.assign(n, 0.0);
    mask.assign(n, 1.0);

    for (size_t i = 0; i < n; ++i) {
        const double t = spec.time == TimeAxis::VisitNum
                             ? static_cast<double>(c.visit_num[i])
                             : static_cast<double>(c.fu_month[i]);
        if (spec.time == TimeAxis::FuMonth && c.fu_month_present[i] == 0) mask[i] = 0.0;

        size_t k = 0;
        x_rm[i * p + k++] = 1.0;
        x_rm[i * p + k++] = t;
        if (spec.covars) {
            x_rm[i * p + k++] = static_cast<double>(c.newage[i]) - kAgeCentre;
            for (size_t j = 0; j + 1 < kGenderLevels.size(); ++j)
                x_rm[i * p + k++] = (c.gender[i] == kGenderLevels[j]) ? 1.0 : 0.0;
            if (!spec.drop_hispanic)
                for (size_t j = 0; j + 1 < kHispanicLevels.size(); ++j)
                    x_rm[i * p + k++] = (c.hispanic[i] == kHispanicLevels[j]) ? 1.0 : 0.0;
        }
        if (spec.interaction) {
            const double umass = (c.data_source[i] == 1) ? 1.0 : 0.0;
            x_rm[i * p + k++] = umass;
            x_rm[i * p + k++] = umass * t;
        }
        for (size_t kk = 0; kk < p; ++kk) x_rm[i * p + kk] *= mask[i];
        y[i] = static_cast<double>(c.sisa[i]) * mask[i];
    }
}

// =============================================================================
// Accuracy harness helpers
// =============================================================================

// Print a labelled expected-vs-actual table with absolute and relative error,
// and return the worst relative error seen. Party 0 only.
double ReportAccuracy(int party_id, const std::string& label,
                      const std::vector<double>& inputs,
                      const std::vector<double>& expected,
                      const std::vector<double>& actual) {
    double worst_rel = 0.0, worst_abs = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double denom = std::abs(expected[i]) > 1e-12 ? std::abs(expected[i]) : 1.0;
        worst_abs = std::max(worst_abs, std::abs(expected[i] - actual[i]));
        worst_rel = std::max(worst_rel, std::abs(expected[i] - actual[i]) / denom);
    }
    if (party_id != 0) return worst_rel;

    std::cout << "\n--- " << label << " ---\n"
              << std::left << std::setw(14) << "Input" << std::setw(18) << "Plaintext"
              << std::setw(18) << "MPC" << std::setw(14) << "Abs Error"
              << std::setw(14) << "Rel Error" << std::endl;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double abs_err = std::abs(expected[i] - actual[i]);
        const double denom = std::abs(expected[i]) > 1e-12 ? std::abs(expected[i]) : 1.0;
        std::cout << std::left << std::setw(14) << inputs[i]
                  << std::setw(18) << expected[i] << std::setw(18) << actual[i]
                  << std::setw(14) << abs_err << std::setw(14) << (abs_err / denom)
                  << std::endl;
    }
    // Report both, because relative error is misleading near the bottom of the
    // fixed-point range: 1 ulp at precision 16 is 1.5e-5, so an expected value
    // of a few ulps (exp(-10) = 4.5e-5, say) can only ever be a few percent
    // accurate no matter how good the algorithm is. Absolute error at or below
    // ~1e-5 means the kernel is at the representation floor, which is the best
    // available answer.
    std::cout << "worst absolute error: " << std::scientific << worst_abs
              << "   worst relative error: " << worst_rel << std::defaultfloat << std::endl;
    return worst_rel;
}

// Exercises Div / Recip / Sqrt / Rsqrt against the standard library. These are
// the kernels this program adds, so they are checked before anything is built
// on top of them.
void TestNewKernels(EngineRef engine, int party_id) {
    single_cout("\n================ new fixed-point kernels ================");

    // Div spans four orders of magnitude in the denominator, which is what the
    // normalisation ladder exists to handle.
    const std::vector<double> num = {1.0, 3.0, -7.5, 1.0,   250.0, 1.0,    9.0,   -1.0};
    const std::vector<double> den = {1.0, 2.0,  4.0, 0.125, 1000.0, 0.03125, 7.0, 3.0};
    std::vector<double> div_expected(num.size()), div_labels(num.size());
    for (size_t i = 0; i < num.size(); ++i) {
        div_expected[i] = num[i] / den[i];
        div_labels[i] = den[i];
    }
    AV div_res = Div(ShareDoubles(engine, num), ShareDoubles(engine, den));
    ReportAccuracy(party_id, "Div (label = denominator)", div_labels, div_expected,
                   OpenToDoubles(div_res));

    const std::vector<double> rec_in = {0.03125, 0.1, 0.5, 1.0, 1.5, 2.0, 7.0, 100.0, 4096.0};
    std::vector<double> rec_expected(rec_in.size());
    for (size_t i = 0; i < rec_in.size(); ++i) rec_expected[i] = 1.0 / rec_in[i];
    ReportAccuracy(party_id, "Recip", rec_in, rec_expected,
                   OpenToDoubles(Recip(ShareDoubles(engine, rec_in))));

    const std::vector<double> sq_in = {0.01, 0.25, 0.5, 1.0, 2.0, 3.0, 10.0, 100.0, 1024.0, 65536.0};
    std::vector<double> sq_expected(sq_in.size()), rsq_expected(sq_in.size());
    for (size_t i = 0; i < sq_in.size(); ++i) {
        sq_expected[i] = std::sqrt(sq_in[i]);
        rsq_expected[i] = 1.0 / std::sqrt(sq_in[i]);
    }
    SqrtPair sp = SqrtBoth(ShareDoubles(engine, sq_in));
    ReportAccuracy(party_id, "Sqrt", sq_in, sq_expected, OpenToDoubles(sp.root));
    ReportAccuracy(party_id, "Rsqrt", sq_in, rsq_expected, OpenToDoubles(sp.inv_root));

    const std::vector<double> exp_in = {-10.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0};
    std::vector<double> exp_expected(exp_in.size());
    for (size_t i = 0; i < exp_in.size(); ++i) exp_expected[i] = std::exp(exp_in[i]);
    ReportAccuracy(party_id, "Exp", exp_in, exp_expected,
                   OpenToDoubles(Exp(ShareDoubles(engine, exp_in))));

    const std::vector<double> log_in = {0.01, 0.1, 0.25, 0.5, 0.7071, 1.0, 1.4142, 2.0, 4.0, 10.0, 100.0, 1000.0};
    std::vector<double> log_expected(log_in.size());
    for (size_t i = 0; i < log_in.size(); ++i) log_expected[i] = std::log(log_in[i]);
    ReportAccuracy(party_id, "Log", log_in, log_expected,
                   OpenToDoubles(Log(ShareDoubles(engine, log_in))));

    const std::vector<double> sig_in = {-8.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 8.0};
    std::vector<double> sig_expected(sig_in.size()), sp_expected(sig_in.size());
    for (size_t i = 0; i < sig_in.size(); ++i) {
        sig_expected[i] = sig_in[i] >= 0.0 ? 1.0 / (1.0 + std::exp(-sig_in[i]))
                                           : std::exp(sig_in[i]) / (1.0 + std::exp(sig_in[i]));
        sp_expected[i] = sig_in[i] > 0.0 ? sig_in[i] + std::log1p(std::exp(-sig_in[i]))
                                         : std::log1p(std::exp(sig_in[i]));
    }
    ReportAccuracy(party_id, "Sigmoid", sig_in, sig_expected,
                   OpenToDoubles(Sigmoid(ShareDoubles(engine, sig_in))));
    ReportAccuracy(party_id, "LogOnePlusExp (softplus)", sig_in, sp_expected,
                   OpenToDoubles(LogOnePlusExp(ShareDoubles(engine, sig_in))));

    const std::vector<double> abs_in = {-3.5, -1.0, -0.25, 0.0, 0.25, 1.0, 3.5};
    std::vector<double> abs_expected(abs_in.size());
    for (size_t i = 0; i < abs_in.size(); ++i) abs_expected[i] = std::abs(abs_in[i]);
    ReportAccuracy(party_id, "Abs", abs_in, abs_expected,
                   OpenToDoubles(Abs(ShareDoubles(engine, abs_in))));
}

// Checks the segmented-scan layer against a plaintext group-by on a RAGGED key
// column. This is the piece most likely to be subtly wrong -- the forward /
// reverse composition has a trap in it -- so it is verified directly rather
// than only through the models that use it.
void TestSegmented(EngineRef engine, int party_id) {
    single_cout("\n================ segmented (per-group) helpers ================");

    // Ragged groups: sizes 3, 2, 1, 4. Padded to a power of two with a sentinel
    // key so the pad block is its own segment on the reverse pass, and with
    // zeroed values so it contributes nothing.
    const std::vector<DataType> plain_keys_real = {1, 1, 1, 2, 2, 3, 4, 4, 4, 4};
    const std::vector<double> plain_vals_real = {1.5, 2.0, -0.5, 4.0, 1.0,
                                                 7.25, 0.5, 0.5, 0.5, 0.5};
    const std::vector<DataType> plain_mask_real = {1, 0, 0, 0, 0, 1, 1, 1, 0, 0};

    const size_t n_real = plain_keys_real.size();
    const size_t n = NextPowerOfTwo(n_real);
    const DataType kKeySentinel = std::numeric_limits<DataType>::max();

    cdough::Vector<DataType> kv(n, 0), vv(n, precision), mv(n, 0);
    for (size_t i = 0; i < n; ++i) {
        kv[i] = i < n_real ? plain_keys_real[i] : kKeySentinel;
        vv[i] = i < n_real ? static_cast<DataType>(std::llround(plain_vals_real[i] * scale)) : 0;
        mv[i] = i < n_real ? plain_mask_real[i] : 0;
    }

    BV key_col = engine.template secret_share_b<DataType>(kv, 0);
    AV val_col = engine.template secret_share_a<DataType>(vv, 0, precision);
    AV mask_col = engine.template secret_share_a<DataType>(mv, 0, 0);
    std::vector<BV> keys{key_col};

    // --- expected values, in the clear ---
    std::map<DataType, double> group_sum;
    std::map<DataType, int> group_count;
    std::set<DataType> distinct_masked;
    for (size_t i = 0; i < n_real; ++i) {
        group_sum[plain_keys_real[i]] += plain_vals_real[i];
        group_count[plain_keys_real[i]]++;
        if (plain_mask_real[i]) distinct_masked.insert(plain_keys_real[i]);
    }

    AV total = SegTotal(keys, val_col);
    auto opened_total = total.open();
    AV prefix = SegScan(keys, val_col, SegDirection::Forward);
    auto opened_prefix = prefix.open();
    BV last_b = LastOfGroup(keys);
    auto opened_last = last_b.open();
    BV first_b = FirstOfGroup(keys);
    auto opened_first = first_b.open();
    AV n_distinct = CountDistinct(keys, mask_col);
    auto opened_nd = n_distinct.open();

    if (party_id != 0) return;

    std::cout << "\n--- SegTotal / SegScan / LastOfGroup (label = key) ---\n"
              << std::left << std::setw(6) << "row" << std::setw(6) << "key"
              << std::setw(10) << "value" << std::setw(14) << "total(exp)"
              << std::setw(14) << "total(mpc)" << std::setw(14) << "prefix(exp)"
              << std::setw(14) << "prefix(mpc)" << std::setw(7) << "first"
              << std::setw(6) << "last" << std::endl;

    bool ok = true;
    double running = 0.0;
    for (size_t i = 0; i < n_real; ++i) {
        if (i == 0 || plain_keys_real[i] != plain_keys_real[i - 1]) running = 0.0;
        running += plain_vals_real[i];
        const double exp_total = group_sum[plain_keys_real[i]];
        const double got_total = static_cast<double>(opened_total[i]) / scale;
        const double got_prefix = static_cast<double>(opened_prefix[i]) / scale;
        const bool exp_first = (i == 0 || plain_keys_real[i] != plain_keys_real[i - 1]);
        const bool exp_last =
            (i + 1 == n_real) || (plain_keys_real[i] != plain_keys_real[i + 1]);

        ok &= std::abs(exp_total - got_total) < 1e-3;
        ok &= std::abs(running - got_prefix) < 1e-3;
        ok &= (static_cast<int>(opened_first[i]) == static_cast<int>(exp_first));
        // The last real row is only "last" once the sentinel block is excluded,
        // which is what makes the pad key matter.
        ok &= (static_cast<int>(opened_last[i]) == static_cast<int>(exp_last));

        std::cout << std::left << std::setw(6) << i << std::setw(6) << plain_keys_real[i]
                  << std::setw(10) << plain_vals_real[i] << std::setw(14) << exp_total
                  << std::setw(14) << got_total << std::setw(14) << running
                  << std::setw(14) << got_prefix << std::setw(7)
                  << static_cast<int>(opened_first[i]) << std::setw(6)
                  << static_cast<int>(opened_last[i]) << std::endl;
    }

    const long got_nd = static_cast<long>(opened_nd[0]);
    const long exp_nd = static_cast<long>(distinct_masked.size());
    ok &= (got_nd == exp_nd);
    std::cout << "COUNT(DISTINCT key) WHERE mask=1: expected " << exp_nd << ", got " << got_nd
              << std::endl;
    std::cout << (ok ? "SEGMENTED HELPERS: PASS" : "SEGMENTED HELPERS: *** FAIL ***")
              << std::endl;
}

// Checks Gram / CholeskySolve / SymmetricInverse against plaintext linear
// algebra: the solve is scored by the residual ||Ax - b||_inf and the inverse by
// ||A A^-1 - I||_inf, both computed in the clear from the opened results.
void TestLinearAlgebra(EngineRef engine, int party_id) {
    single_cout("\n================ dense secure linear algebra ================");

    const size_t n = 64;
    const size_t p = 4;

    // A small design matrix and weights, chosen so X^T W X is comfortably SPD.
    std::vector<double> x_plain(n * p), w_plain(n), y_plain(n);
    for (size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(n);
        x_plain[i * p + 0] = 1.0;
        x_plain[i * p + 1] = t - 0.5;
        x_plain[i * p + 2] = (i % 3 == 0) ? 1.0 : 0.0;
        x_plain[i * p + 3] = std::sin(6.0 * t) * 0.5;
        w_plain[i] = 0.1 + 0.15 * ((i % 5) / 4.0);
        y_plain[i] = 0.3 + 0.8 * x_plain[i * p + 1] - 0.4 * x_plain[i * p + 2];
    }

    std::vector<double> gram_expected(p * p, 0.0);
    for (size_t k = 0; k < p; ++k)
        for (size_t l = 0; l < p; ++l)
            for (size_t i = 0; i < n; ++i)
                gram_expected[k * p + l] += w_plain[i] * x_plain[i * p + k] * x_plain[i * p + l];
    // A mild ridge, matching what the model code will use.
    const double lambda = 1e-3;
    for (size_t k = 0; k < p; ++k) gram_expected[k * p + k] += lambda;

    std::vector<double> b_plain(p, 0.0);
    for (size_t k = 0; k < p; ++k)
        for (size_t i = 0; i < n; ++i) b_plain[k] += x_plain[i * p + k] * y_plain[i];

    AV x_sec = ShareDoubles(engine, x_plain);
    AV w_sec = ShareDoubles(engine, w_plain);
    AV b_sec = ShareDoubles(engine, b_plain);

    AV g = Gram(x_sec, w_sec, n, p);
    AddRidge(g, p, lambda);
    AV sol = CholeskySolve(g, b_sec, p);
    AV inv = SymmetricInverse(g, p);

    const std::vector<double> g_open = OpenToDoubles(g);
    const std::vector<double> sol_open = OpenToDoubles(sol);
    const std::vector<double> inv_open = OpenToDoubles(inv);

    if (party_id != 0) return;

    double gram_err = 0.0;
    for (size_t i = 0; i < p * p; ++i)
        gram_err = std::max(gram_err, std::abs(g_open[i] - gram_expected[i]));

    // Residual of the secure solution, measured against the EXACT Gram matrix.
    double resid = 0.0;
    for (size_t k = 0; k < p; ++k) {
        double row = 0.0;
        for (size_t l = 0; l < p; ++l) row += gram_expected[k * p + l] * sol_open[l];
        resid = std::max(resid, std::abs(row - b_plain[k]));
    }

    double inv_err = 0.0;
    for (size_t k = 0; k < p; ++k) {
        for (size_t l = 0; l < p; ++l) {
            double acc = 0.0;
            for (size_t m = 0; m < p; ++m) acc += gram_expected[k * p + m] * inv_open[m * p + l];
            inv_err = std::max(inv_err, std::abs(acc - (k == l ? 1.0 : 0.0)));
        }
    }

    std::cout << "\n--- Gram / Cholesky (n=" << n << ", p=" << p << ") ---\n";
    std::cout << "max |Gram_mpc - Gram_plain|      : " << std::scientific << gram_err << "\n";
    std::cout << "solve residual ||Ax - b||_inf    : " << resid << "\n";
    std::cout << "inverse error ||A A^-1 - I||_inf : " << inv_err << std::defaultfloat
              << std::endl;

    std::cout << "\nsolution (mpc vs plaintext):\n";
    // Reference solution by plaintext Gaussian elimination on the exact Gram.
    std::vector<double> m(gram_expected), rhs(b_plain);
    for (size_t c = 0; c < p; ++c) {
        size_t piv = c;
        for (size_t r = c + 1; r < p; ++r)
            if (std::abs(m[r * p + c]) > std::abs(m[piv * p + c])) piv = r;
        for (size_t k = 0; k < p; ++k) std::swap(m[c * p + k], m[piv * p + k]);
        std::swap(rhs[c], rhs[piv]);
        const double d = m[c * p + c];
        for (size_t k = 0; k < p; ++k) m[c * p + k] /= d;
        rhs[c] /= d;
        for (size_t r = 0; r < p; ++r) {
            if (r == c) continue;
            const double f = m[r * p + c];
            for (size_t k = 0; k < p; ++k) m[r * p + k] -= f * m[c * p + k];
            rhs[r] -= f * rhs[c];
        }
    }
    bool ok = (gram_err < 1e-2) && (resid < 1e-2) && (inv_err < 1e-2);
    for (size_t k = 0; k < p; ++k) {
        std::cout << "  x[" << k << "] = " << std::setw(12) << sol_open[k]
                  << "   plaintext " << std::setw(12) << rhs[k]
                  << "   diff " << std::abs(sol_open[k] - rhs[k]) << std::endl;
        ok &= std::abs(sol_open[k] - rhs[k]) < 1e-2;
    }
    std::cout << (ok ? "LINEAR ALGEBRA: PASS" : "LINEAR ALGEBRA: *** FAIL ***") << std::endl;
}
// =============================================================================
// The pipeline
// =============================================================================

// The six specifications, fitted to the three populations. The two interaction
// models can only run on any_system, because that is the only table in which
// data_source varies -- which is exactly why the script is built around it.
std::vector<ModelSpec> AllModelSpecs() {
    std::vector<ModelSpec> out;
    const std::vector<SystemScope> scopes = {SystemScope::UMass, SystemScope::Any,
                                             SystemScope::NonUMass};

    struct Base {
        const char* step;
        TimeAxis time;
        bool covars;
        bool interaction;
    };
    const std::vector<Base> bases = {
        {"2a", TimeAxis::VisitNum, false, false}, {"2b", TimeAxis::FuMonth, false, false},
        {"5a", TimeAxis::VisitNum, true, false},  {"5b", TimeAxis::FuMonth, true, false},
        {"6a", TimeAxis::VisitNum, true, true},   {"6b", TimeAxis::FuMonth, true, true},
    };

    for (const Base& b : bases) {
        for (SystemScope sc : scopes) {
            if (b.interaction && sc != SystemScope::Any) continue;
            ModelSpec m;
            m.step = b.step;
            m.time = b.time;
            m.covars = b.covars;
            m.interaction = b.interaction;
            m.random_intercept = !b.interaction;
            m.scope = sc;
            // The SAS comments hispanic out of model 5b on the UMass subset,
            // citing sparse data.
            m.drop_hispanic = (m.step == "5b" && sc == SystemScope::UMass);
            out.push_back(m);
        }
    }
    return out;
}

const SecureCohort& PickCohort(const SecureCohort& any, const SecureCohort& umass,
                               const SecureCohort& nonumass, SystemScope s) {
    if (s == SystemScope::UMass) return umass;
    if (s == SystemScope::NonUMass) return nonumass;
    return any;
}

const PlainCohort& PickPlain(const PlainCohort& any, const PlainCohort& umass,
                             const PlainCohort& nonumass, SystemScope s) {
    if (s == SystemScope::UMass) return umass;
    if (s == SystemScope::NonUMass) return nonumass;
    return any;
}

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();

    // Which part of the program to run. The kernel and segmented-scan harnesses
    // are fast; the model fits are not, so they are separately selectable.
    //   kernels  - fixed-point, segmented-scan and linear-algebra accuracy only
    //   describe - ingestion plus the descriptive and aggregate nodes
    //   models   - the fourteen regression fits
    //   all      - everything (default)
    const std::string stage = engine.getArg<std::string>("stage", "S", "all");
    const bool run_kernels = (stage == "kernels" || stage == "all");
    const bool run_describe = (stage == "describe" || stage == "models" || stage == "all");
    const bool run_models = (stage == "models" || stage == "all");
    const bool print_describe = (stage == "describe" || stage == "all");

    // Number of patients in the synthetic cohort. Ignored in CSV mode.
    const int num_subjects = engine.getArg<int>("subjects", "r", 200);
    // Directory holding any_system.csv / umass_system.csv / nonumass_system.csv.
    // Empty means "use the synthetic generator".
    const std::string data_dir = engine.getArg<std::string>("data-dir", "D", "");
    // Where to dump the synthetic cohort for the R / Python / SQLite cross-checks.
    const std::string out_dir = engine.getArg<std::string>("out-dir", "O", "");
    // Which party holds each table in the clear.
    const int party_any = engine.getArg<int>("party-any", "pa", 0);
    const int party_umass = engine.getArg<int>("party-umass", "pu", 0);
    const int party_nonumass = engine.getArg<int>("party-nonumass", "pn", 0);

    if (run_kernels) {
        TestNewKernels(engine, pID);
        TestSegmented(engine, pID);
        TestLinearAlgebra(engine, pID);
    }
    if (!run_describe) return 0;

    // ---------------------------------------------------------------- ingest
    SyntheticTruth truth;
    PlainCohort any_plain, umass_plain, nonumass_plain;

    if (data_dir.empty()) {
        any_plain = MakeSyntheticCohort(static_cast<size_t>(num_subjects), truth);
        umass_plain = DeriveSystemCohort(any_plain, SystemScope::UMass);
        nonumass_plain = DeriveSystemCohort(any_plain, SystemScope::NonUMass);
        if (pID == 0 && !out_dir.empty()) {
            WriteCohortCsv(any_plain, out_dir + "/" + CohortCsvName(SystemScope::Any));
            WriteCohortCsv(umass_plain, out_dir + "/" + CohortCsvName(SystemScope::UMass));
            WriteCohortCsv(nonumass_plain,
                           out_dir + "/" + CohortCsvName(SystemScope::NonUMass));
            std::cout << "wrote the synthetic cohort to " << out_dir << std::endl;
        }
    } else {
        any_plain = ReadCohortCsv(data_dir + "/" + CohortCsvName(SystemScope::Any),
                                  SystemScope::Any);
        umass_plain = ReadCohortCsv(data_dir + "/" + CohortCsvName(SystemScope::UMass),
                                    SystemScope::UMass);
        nonumass_plain = ReadCohortCsv(data_dir + "/" + CohortCsvName(SystemScope::NonUMass),
                                       SystemScope::NonUMass);
    }

    SecureCohort any = ShareCohort(engine, any_plain, party_any);
    SecureCohort umass = ShareCohort(engine, umass_plain, party_umass);
    SecureCohort nonumass = ShareCohort(engine, nonumass_plain, party_nonumass);

    if (pID == 0) {
        std::cout << "\n################ MPC analysis pipeline ################\n"
                  << "source: " << (data_dir.empty() ? "synthetic cohort" : data_dir) << "\n"
                  << std::left << std::setw(20) << "table" << std::setw(12) << "rows"
                  << std::setw(12) << "padded" << std::setw(12) << "patients" << std::endl;
        for (const SecureCohort* c : {&any, &umass, &nonumass})
            std::cout << std::left << std::setw(20) << ScopeName(c->scope) << std::setw(12)
                      << c->n << std::setw(12) << c->n_pad << std::setw(12) << c->num_subjects
                      << std::endl;
        if (data_dir.empty())
            std::cout << "\ngenerating parameters: beta0=" << truth.beta0
                      << "  beta_visit=" << truth.beta_visit << "  beta_fu=" << truth.beta_fu
                      << "  beta_age=" << truth.beta_age << "  sigma=" << truth.sigma
                      << "  slope_diff(UMass-nonUMass)=" << truth.slope_diff << std::endl;
    }

    // ------------------------------------------------- descriptive and counts
    if (print_describe) {
        long max_visits = 1;
        for (size_t i = 0; i < any_plain.rows(); ++i)
            max_visits = std::max<long>(max_visits, static_cast<long>(any_plain.visit_num[i]));
        // A public upper bound for the histogram sweep. The counts themselves are
        // the published output of the node.
        ReportD1a(any, pID, max_visits);
        ReportD1b(any, pID);
        ReportSisaCounts(any, pID);
        ReportSisaCounts(umass, pID);
        ReportSisaCounts(nonumass, pID);
    }

    if (!run_models) return 0;

    // ------------------------------------------------------------- the models
    const std::vector<ModelSpec> specs = AllModelSpecs();
    if (pID == 0)
        std::cout << "\n################ " << specs.size()
                  << " regression models ################" << std::endl;

    std::vector<FitResult> fits;
    for (const ModelSpec& spec : specs) {
        const SecureCohort& c = PickCohort(any, umass, nonumass, spec.scope);
        ModelData md = BuildDesign(c, spec);

        FitResult r = spec.random_intercept ? FitGlmmLaplace(md, spec, pID)
                                            : FitLogisticIrls(md, spec);
        fits.push_back(r);
        if (pID == 0) PrintFit(r, spec);

        // Score against the plaintext oracle. For the fixed-effects models this
        // is the same estimator in double precision, so the two should agree to
        // the fixed-point floor. For the mixed models it is NOT the same
        // estimand -- it ignores the random intercept -- so it is reported as
        // context, with the generating parameters as the real reference.
        if (pID == 0) {
            const PlainCohort& pc = PickPlain(any_plain, umass_plain, nonumass_plain, spec.scope);
            std::vector<double> x_rm, y, mask;
            size_t pp = 0;
            PlainDesign(pc, spec, x_rm, y, mask, pp);
            PlainFit pf = PlainLogisticIrls(x_rm, y, mask, pc.rows(), pp, kIrlsRidge,
                                            kIrlsIterations);
            std::cout << "\n  " << (spec.random_intercept
                                        ? "plaintext IRLS (no random effect -- context only):"
                                        : "plaintext IRLS oracle (same estimator):")
                      << "\n  " << std::left << std::setw(34) << "Term" << std::setw(14)
                      << "MPC" << std::setw(14) << "Plaintext" << std::setw(12) << "Diff"
                      << std::endl;
            double worst = 0.0;
            for (size_t k = 0; k < r.terms.size() && k < pf.estimate.size(); ++k) {
                const double d = std::abs(r.estimate[k] - pf.estimate[k]);
                if (!spec.random_intercept) worst = std::max(worst, d);
                std::cout << "  " << std::left << std::setw(34) << r.terms[k] << std::fixed
                          << std::setprecision(5) << std::setw(14) << r.estimate[k]
                          << std::setw(14) << pf.estimate[k] << std::setw(12) << d
                          << std::defaultfloat << std::endl;
            }
            if (!spec.random_intercept)
                std::cout << "  max |MPC - plaintext| = " << std::scientific << worst
                          << std::defaultfloat
                          << (worst < 5e-3 ? "   OK" : "   *** CHECK ***") << std::endl;
        }
    }

    // --------------------------------------------------------------- summary
    if (pID == 0) {
        std::cout << "\n################ summary ################\n"
                  << std::left << std::setw(8) << "model" << std::setw(18) << "population"
                  << std::setw(32) << "time term" << std::setw(14) << "estimate"
                  << std::setw(12) << "SE" << std::setw(11) << "Pr>|z|" << std::endl;
        for (size_t i = 0; i < fits.size(); ++i) {
            const FitResult& r = fits[i];
            // The time coefficient is always the second term; for the
            // interaction models the estimand is the interaction, which is last.
            const size_t idx = specs[i].interaction ? r.terms.size() - 1 : 1;
            const bool have_se = std::isfinite(r.se[idx]) && r.se[idx] > 0.0;
            const double z = have_se ? r.estimate[idx] / r.se[idx] : 0.0;
            std::ostringstream se_s, p_s;
            if (have_se) {
                se_s << std::fixed << std::setprecision(5) << r.se[idx];
                p_s << FormatP(TwoSidedNormalP(z));
            } else {
                se_s << "n/a";
                p_s << "n/a";
            }
            std::cout << std::left << std::setw(8) << r.step << std::setw(18)
                      << ScopeLabel(r.scope) << std::setw(32) << r.terms[idx] << std::fixed
                      << std::setprecision(5) << std::setw(14) << r.estimate[idx]
                      << std::defaultfloat << std::setw(12) << se_s.str() << std::setw(11)
                      << p_s.str() << std::endl;
        }
        if (data_dir.empty())
            std::cout << "\ngenerating truth: visit_num slope " << truth.beta_visit
                      << ", fu_month slope " << truth.beta_fu << ", UMass slope difference "
                      << truth.slope_diff << " (models 6a/6b)" << std::endl;
        std::cout << "\nThe two step-6 models are the comparison the script is built to make:\n"
                     "their interaction coefficient IS the difference in time slopes between\n"
                     "the UMass and non-UMass systems. The paired subset models cannot answer\n"
                     "it -- data_source does not vary inside either subset."
                  << std::endl;

        // Machine-readable form, for scripts/testing/validate_mpc_analysis.py
        // --compare. One tab-separated line per coefficient, plus one per
        // variance component. Kept deliberately dull so it stays parseable.
        std::cout << "\n# RESULT step\tpopulation\tterm\testimate\tse" << std::endl;
        for (size_t i = 0; i < fits.size(); ++i) {
            const FitResult& fit = fits[i];
            const char* pop = fit.scope == SystemScope::Any
                                  ? "any"
                                  : (fit.scope == SystemScope::UMass ? "umass" : "nonumass");
            for (size_t k = 0; k < fit.terms.size(); ++k) {
                std::cout << "RESULT\t" << fit.step << '\t' << pop << '\t' << fit.terms[k]
                          << '\t' << std::setprecision(9) << fit.estimate[k] << '\t';
                if (std::isfinite(fit.se[k]))
                    std::cout << fit.se[k];
                else
                    std::cout << "nan";
                std::cout << std::endl;
            }
            if (fit.has_random)
                std::cout << "RESULT\t" << fit.step << '\t' << pop
                          << "\trandom_intercept_variance\t" << std::setprecision(9)
                          << fit.sigma2 << "\tnan" << std::endl;
        }
    }
    return 0;
}
