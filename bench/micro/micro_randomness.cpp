#include "cdough.h"
#include "profiling/stopwatch.h"

using namespace cdough::debug;
using namespace cdough::service;
using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

#include <unistd.h>

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();
    auto test_size = engine.getArg<size_t>("test-size", "r", 1 << 20);

    cdough::Vector<int> local(test_size);
    cdough::Vector<int> common(test_size);

    /*
        local randomness
    */

    // start timer
    stopwatch::timepoint("Start");

    engine.populateLocalRandom(local);

    // stop timer
    stopwatch::timepoint("Local Randomness");

    /*
        common randomness
    */

    std::set<int> group = engine.getGroups()[0];
    if (group.contains(pID)) {
        engine.populateCommonRandom(common, group);
    }

    // stop timer
    stopwatch::timepoint("Common Randomness");

    return 0;
}