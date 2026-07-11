#include "cdough.h"
#include "profiling/stopwatch.h"

using namespace cdough::debug;
using namespace cdough::service;
using namespace cdough::random;
using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();
    auto test_size = engine.getArg<size_t>("test-size", "r", 1 << 20);
    auto num_threads = engine.get_num_threads();

#ifdef USE_LIBOTE
    // Create input vector for OPRF evaluation
    cdough::Vector<__int128_t> input(test_size * num_threads);
    for (int i = 0; i < test_size; i++) {
        input[i] = i;
    }
    cdough::Vector<__int128_t> output(test_size * num_threads);

    // start timer
    stopwatch::timepoint("Start");

    // Parallel OPRF evaluation using the new runtime function
    bool is_sender = (pID == 0);
    engine.evaluate_oprf(input, output, is_sender);

    stopwatch::timepoint("OPRF (Role 0)");

    engine.evaluate_oprf(input, output, !is_sender);

    stopwatch::timepoint("OPRF (Role 1)");
#else
    single_cout("LibOTE not enabled, skipping OPRF microbenchmark.");
#endif

    return 0;
}