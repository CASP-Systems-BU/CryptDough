#include <numeric>

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

const DataType kLn2_scaled = (kLn2 * scale);
const DataType kLn2_inv_scaled = (kLn2_inv * scale);
const DataType kSqrt2_scaled = (kSqrt2 * scale);
const DataType kSqrt1_2_scaled = (kSqrt1_2 * scale);
const DataType kSmallEpsilon_scaled = (kSmallEpsilon * scale);
const DataType kSeriesTolerance_scaled = (kSeriesTolerance * scale);

constexpr int kMaxSeriesTerms = 3;
constexpr int kMaxNewtonStep = 4;

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

    return 0;
}