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
    auto num_columns = engine.getArg<int>("num-columns", "nc", 2);

    cdough::Vector<int> v(test_size);
    for (int i = 0; i < test_size; i++) {
        v[i] = i;
    }
    BSharedVector<int> b = engine.secret_share_b(v, 0);

    // start timer
    stopwatch::timepoint("Start");
    stopwatch::profile_init();

    b.shuffle();

    // stop timer
    stopwatch::timepoint("Shuffle");
    stopwatch::profile_done();

    engine.print_statistics();
    engine.print_communicator_statistics();

    return 0;
}