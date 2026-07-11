#include "cdough.h"

using namespace cdough::debug;
using namespace cdough::service;

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

#define MAX_COLUMNS 8
#define MIN_ROW_EXPONENT 5
#define MAX_ROW_EXPONENT 20

#if DEFAULT_BITWIDTH == 64
using T = int64_t;
#else
using T = int32_t;
#endif

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();
    auto test_size = engine.getArg<size_t>("test-size", "r", 1 << 20);

    single_cout("Using bitwidth: " << sizeof(T) * 8 << " bits");

    cdough::Vector<T> v(test_size);
    for (int i = 0; i < test_size; i++) {
        v[i] = i;
    }
    BSharedVector<T> b = engine.secret_share_b(v, 0);

    stopwatch::profile_init();

    stopwatch::timepoint("Start");
    cdough::operators::radix_sort(b);
    stopwatch::timepoint("Radix Sort");
    stopwatch::profile_done();

    // thread_stopwatch::write(pID);

    engine.print_statistics();

    return 0;
}
