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

const int precision = 16;
const DataType scale = 1 << precision;

const float kLn2 = 0.6931;
const float kLn2_inv = 1.4427;
const float kSqrt2 = 1.4142;
const float kSqrt1_2 = 0.7071;
const float kSmallEpsilon = 0.0001;
const float kSeriesTolerance = 0.0001;
const float kNumericalGradientStep = 0.001;

const DataType kLn2_scaled = (kLn2 * scale);
const DataType kLn2_inv_scaled = (kLn2_inv * scale);
const DataType kSqrt2_scaled = (kSqrt2 * scale);
const DataType kSqrt1_2_scaled = (kSqrt1_2 * scale);
const DataType kSmallEpsilon_scaled = (kSmallEpsilon * scale);
const DataType kSeriesTolerance_scaled = (kSeriesTolerance * scale);
const DataType kHalf_scaled = (0.5 * scale);
const DataType kMaxNewtonStep_scaled = (4.0 * scale);

constexpr int kMaxSeriesTerms = 3;
constexpr int kMaxNewtonStep = 4;
constexpr int kNewtonIterations = 5;

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

AV Exp (const AV& x) {
    AV x_ = x;
    x_.setPrecision(0);

    // TODO: truncate, not divide by scale
    // TODO: add multiplication by float/double constants
    AV quotient = (*(x_ * kLn2_inv_scaled)) / scale;
    AV pos_quotient = quotient.gtez();

    // pos_quotient ? 0.5 : - 0.5 is equivalent to (pos_quotient - 0.5)
    // Since we do precision, we need to scale both
    // hence, pos_quotient * scale - ((DataType)(0.5 * scale))
    AV k_add = (*(pos_quotient * scale)) - ((DataType)(0.5 * scale));
    AV k_fixed = quotient + k_add;

    // Integer k (unscaled)
    AV k_int = *(k_fixed / scale);

    // Remainder r = x - k * ln(2)
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
    // Shift k by an offset to keep exponent positive within [0, 2^B - 1]
    constexpr int kOffset = 8;
    AV k_shifted = k_int + static_cast<DataType>(kOffset);

    auto k_b = k_shifted.a2b();

    // Multiply by 2^(b_i * 2^i) obliviously: factor = 1 + b_i * (2^(2^i) - 1)
    AV result = series;
    BV current_bit(x.size(), x.engine);
    for (size_t i = 0; i < 4; ++i) { // 4 bits cover offset range [0, 15]
        current_bit.bit_logical_right_shift(*k_b, i);
        current_bit.mask(1);
        AV bit_a = *current_bit.b2a_bit();
        DataType multiplier = (DataType(1) << (1 << i)) - 1;
        AV factor(x.size(), x.engine);
        factor += scale;
        factor += *(bit_a * (multiplier * scale));
        result = *(result * factor) / scale;
    }

    // Adjust for the constant offset 2^(-kOffset)
    result = *(result / (DataType(1) << kOffset));
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


    return 0;
}