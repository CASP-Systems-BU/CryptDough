#include <iostream>
#include <set>

#include "cdough.h"

using namespace cdough::debug;
using namespace cdough::service;
using namespace cdough::random;
using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

const int num_parties = PROTOCOL_NUM;

template <typename Engine>
void test_correctness(int test_size, Engine& engine) {
    int precision = 16;

    Vector<int64_t> x(test_size);
    for (int i = 0; i < test_size; i++) {
        x[i] = i;
    }

    ASharedVector<int64_t> a = engine.secret_share_a(x, 0, precision);
    BSharedVector<int64_t> b = engine.secret_share_b(x, 0, precision);

    // test multiplication
    auto product = a * a;
    auto product_opened = product->open();
    for (int i = 0; i < test_size; i++) {
        // get the actual and correct values of the square
        // convert to double first to avoid loss of information
        double actual = double(product_opened[i]) / (1 << precision);
        double correct = (double(x[i]) / (1 << precision)) * (double(x[i]) / (1 << precision));
        // get the difference and normalize it
        // normalized difference does not depend on precision
        double diff = actual - correct;
        double scaled_error = diff * (1 << precision);
        // assert that the normalized difference is within some threshold
        assert((scaled_error <= 2) && (scaled_error >= -2));
    }

#ifndef MPC_PROTOCOL_SPDZ_2K_NPC
    // test b2a_bit
    auto b2a_bit_result = b.b2a_bit();
    auto b2a_bit_result_opened = b2a_bit_result->open();
    for (int i = 0; i < test_size; i++) {
        assert(b2a_bit_result_opened[i] == (x[i] & 1));
    }
#endif
}

template <typename Engine>
void test_precision_consistency(int test_size, Engine& engine) {
    int precision = 16;

    Vector<int64_t> x(test_size);
    for (int i = 0; i < test_size; i++) {
        x[i] = i;
    }

    ASharedVector<int64_t> a = engine.secret_share_a(x, 0, precision);
    BSharedVector<int64_t> b = engine.secret_share_b(x, 0, precision);

    // test addition
    auto sum = a + a;
    assert(sum->getPrecision() == precision);

    // test subtraction
    auto difference = a - a;
    assert(difference->getPrecision() == precision);

    // test multiplication
    auto product = a * a;
    assert(product->getPrecision() == precision);

#ifndef MPC_PROTOCOL_SPDZ_2K_NPC
    // test AND
    auto and_result = b & b;
    assert(and_result->getPrecision() == precision);

    // test XOR
    auto xor_result = b ^ b;
    assert(xor_result->getPrecision() == precision);

#endif
    // test b2a_bit
    auto b2a_bit_result = b.b2a_bit();
    assert(b2a_bit_result->getPrecision() == precision);

#ifdef USE_DALSKOV_FANTASTIC_FOUR
    // test b2a
    auto b2a_result = b.b2a();
    assert(b2a_result->getPrecision() == precision);
#endif 

    // test a2b
    auto a2b_result = a.a2b();
    assert(a2b_result->getPrecision() == precision);
}

template <typename T, typename Engine>
void test_float_vectors(int test_size, Engine& engine) {
    int precision = 16;

    std::vector<T> x(test_size);
    for (int i = 0; i < test_size; i++) {
        // calculate square root of i
        x[i] = std::sqrt(i);
    }

    // floating point constructor
    Vector<int64_t> y(x, precision);

    ASharedVector<int64_t> y_a = engine.secret_share_a(y, 0, precision);
    BSharedVector<int64_t> y_b = engine.secret_share_b(y, 0, precision);

    auto y_a_opened = y_a.open();
    auto y_b_opened = y_b.open();

    for (int i = 0; i < test_size; i++) {
        assert(y_a_opened[i] == y_b_opened[i]);
        double actual = double(y_a_opened[i]) / (1 << precision);
        double diff = actual - x[i];
        double scaled_error = diff * (1 << precision);
        // assert that the value is within some threshold of correct value
        assert((scaled_error <= 2) && (scaled_error >= -2));
    }
}

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();

    test_correctness(1000, engine);
    single_cout("Fixed-point arithmetic... OK");

    test_precision_consistency(1000, engine);
    single_cout("Fixed-point precision consistency... OK");

    test_float_vectors<float>(1000, engine);
    test_float_vectors<double>(1000, engine);
    single_cout("Floating point vector construction... OK");

    return 0;
}