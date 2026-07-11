#include "cdough.h"

using namespace cdough::debug;
using namespace cdough::service;

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

#define MAX_COLUMNS 8
#define MIN_ROW_EXPONENT 5
#define MAX_ROW_EXPONENT 20

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();
    auto test_size = engine.getArg<size_t>("test-size", "r", 1 << 20);

    cdough::Vector<int64_t> v(test_size);
    for (int i = 0; i < test_size; i++) {
        v[i] = i;
    }
    BSharedVector<int64_t> b = engine.secret_share_b(v, 0);

    stopwatch::profile_init();

    // start timer
    stopwatch::timepoint("Start");
    cdough::operators::quicksort(b);
    stopwatch::timepoint("Quicksort");
    stopwatch::profile_done();

    engine.mark_statistics();
    cdough::operators::bitonic_sort(b);
    stopwatch::timepoint("Bitonic");
    engine.print_statistics();

    engine.mark_statistics();
    cdough::operators::pairwise_sort(b);
    stopwatch::timepoint("Pairwise");
    engine.print_statistics();

    stopwatch::profile_init();
    cdough::operators::radix_sort(b);
    stopwatch::timepoint("Radix Sort");
    stopwatch::profile_done();
    stopwatch::done();

    // thread_stopwatch::write(pID);

    engine.print_statistics();
    engine.print_communicator_statistics();

    return 0;
}
