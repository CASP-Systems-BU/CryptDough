#include "cdough.h"
#include "profiling/memory.h"

using namespace cdough::debug;
using namespace cdough::service;
using namespace cdough::random;
using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

/*
 * Run microbenchmark for DPF.
 *
 * We specify the number of DPF points and their domain.
 * We run distributed key generation and (local)expansion.
 */

int main(int argc, char** argv) {
    // Initialize cdough runtime [executable - threads_num - p_factor - batch_size]
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();
    int test_size = 1 << 20;
    int domain = 1 << 5;
    if (argc >= 5) {
        test_size = atoi(argv[4]);
    }
    if (argc >= 6) {
        domain = 1 << atoi(argv[5]);
    }

#ifdef USE_LIBOTE
    single_cout("Inputs: " << test_size << ", domain: " << domain);

    cdough::random::DPF<int64_t> dpf(pID, 0, engine.comm0());
    cdough::Vector<int64_t> input(test_size);

    // start timer
    stopwatch::timepoint("Start");
    memory::mempoint("Start");

    dpf.keyGen(input, domain);

    stopwatch::timepoint("Distributed Key Generation");

    auto output = dpf.expand();

    stopwatch::timepoint("Expansion");
    memory::mempoint("Peak Memory");
#else
    single_cout("LibOTE not enabled, skipping DPF microbenchmark.");
#endif

    return 0;
}