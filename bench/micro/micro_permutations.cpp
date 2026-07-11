#include "cdough.h"

// enforce header ordering
#include "core/random/permutations/permutation_manager.h"
#include "profiling/stopwatch.h"

using namespace cdough::debug;
using namespace cdough::service;
using namespace cdough::random;
using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();

    auto test_size = engine.getArg<size_t>("test-size", "r", 1 << 20);
    auto num_permutations = engine.getArg<size_t>("num-permutations", "p", 1);

    auto manager = PermutationManager::get();

    // start timer
    stopwatch::timepoint("Start");

    manager->getNext<int64_t>(test_size, engine);

    // stop timer
    stopwatch::timepoint("Single Thread - 1 Permutation");

    for (int i = 0; i < num_permutations; i++) {
        manager->getNext<int64_t>(test_size, engine);
    }

    // stop timer
    stopwatch::timepoint("Single Thread - N Permutation");
    manager->reserve(test_size, num_permutations, engine);

    // stop timer
    stopwatch::timepoint("Multi Thread - N Permutations");

    return 0;
}