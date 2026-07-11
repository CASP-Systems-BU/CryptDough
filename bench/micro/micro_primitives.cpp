#include <iomanip>

#include "cdough.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"

using namespace cdough::debug;
using namespace cdough::service;
using namespace std::chrono;

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

#if DEFAULT_BITWIDTH == 64
using T = int64_t;
#else
using T = int32_t;
#endif

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();
    auto test_size = engine.getArg<size_t>("test-size", "r", 1 << 20);

    // TODO: deprecated constructor usage in public API
    BSharedVector<T> a(test_size, engine), b(test_size, engine);
    ASharedVector<T> x(test_size, engine), y(test_size, engine);
    SecureMatrix<T> m(x, 1, test_size);

    single_cout("Vector " << test_size << " x "
                          << std::numeric_limits<std::make_unsigned_t<T>>::digits << "b");

    stopwatch::timepoint("Start");

    engine.reserve_mul_triples<T>(test_size);
    stopwatch::timepoint("Reserve MUL");

    // estimate for now
    size_t and_triples = test_size * 14;

    engine.reserve_and_triples<T>(and_triples);
    stopwatch::timepoint("Reserve AND");

    auto c = a & b;
    stopwatch::timepoint("AND");

    auto z = x * y;
    stopwatch::timepoint("MULT");

    auto d = a == b;
    stopwatch::timepoint("EQ");

    auto e = a > b;
    stopwatch::timepoint("GR");

    auto d_a = x == y;
    stopwatch::timepoint("EQ_A");

    auto e_a = x > y;
    stopwatch::timepoint("GR_A");

    auto f = a + b;
    stopwatch::timepoint("Boolean +");

    auto f2 = a.rca(b);
    stopwatch::timepoint("RCA");

    auto f3 = a.ppa(b);
    stopwatch::timepoint("PPA");

    auto g = a.rca_compare(b);
    stopwatch::timepoint("RCA<");

    auto h = x.dot_product(y, 8);
    stopwatch::timepoint("Dot Product");

    auto m1 = m.reLUVectorized();
    stopwatch::timepoint("ReLU");

    engine.print_statistics();
    engine.print_communicator_statistics();

    return 0;
}

#pragma GCC diagnostic pop
