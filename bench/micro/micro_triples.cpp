#include "cdough.h"
#include "profiling/stopwatch.h"

using namespace cdough::debug;
using namespace cdough::service;
using namespace cdough::random;
using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

template <typename T>
void reserve_b(int up_to, EngineRef engine) {
    auto L = sizeof(T) * 8;
    for (int s = 128; s <= up_to; s *= 2) {
        stopwatch::get_elapsed();
        engine.reserve_and_triples<T>(s);
        auto reserve_time = stopwatch::get_elapsed();

        auto us_per = reserve_time / s * 1e6;

        single_cout("AND " << std::right << std::fixed << std::setprecision(5) << std::setw(3) << L
                           << "b x " << std::setw(10) << s << ": " << std::setw(10) << reserve_time
                           << " s; " << std::setw(10) << us_per << " us / triple = "
                           << std::setw(10) << us_per / L * 1e3 << " ns / bit");

        // use em up
        BSharedVector<T> a(s, engine), b(s, engine);
        a &= b;
    }
    single_cout("--");
}

template <typename T>
void reserve_a(int up_to, EngineRef engine) {
    auto L = sizeof(T) * 8;
    for (int s = 128; s <= up_to; s *= 2) {
        stopwatch::get_elapsed();
        engine.reserve_mul_triples<T>(s);
        auto reserve_time = stopwatch::get_elapsed();

        auto us_per = reserve_time / s * 1e6;

        single_cout("MUL " << std::right << std::fixed << std::setprecision(5) << std::setw(3) << L
                           << "b x " << std::setw(10) << s << ": " << std::setw(10) << reserve_time
                           << " s; " << std::setw(10) << us_per << " us / triple = "
                           << std::setw(10) << us_per / L * 1e3 << " ns / bit");

        // use em up
        ASharedVector<T> a(s, engine), b(s, engine);
        a *= b;
    }
    single_cout("--");
}

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
#ifndef MPC_PROTOCOL_BEAVER_TWO
    single_cout("Skipping micro_triples for non-2PC");
#else

    auto pID = engine.getPartyID();
    auto test_size = engine.getArg<size_t>("test-size", "r", 1 << 20);

    reserve_b<int8_t>(test_size, engine);
    reserve_b<int16_t>(test_size, engine);
    reserve_b<int32_t>(test_size, engine);
    reserve_b<int64_t>(test_size, engine);
    reserve_b<__int128_t>(test_size, engine);

    reserve_a<int8_t>(test_size, engine);
    reserve_a<int16_t>(test_size, engine);
    reserve_a<int32_t>(test_size, engine);
    reserve_a<int64_t>(test_size, engine);
    reserve_a<__int128_t>(test_size, engine);

    engine.print_communicator_statistics();

#endif

    return 0;
}
