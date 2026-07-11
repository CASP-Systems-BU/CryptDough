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
    int test_size = engine.getArg<size_t>("test-size", "r", 1 << 20);

    cdough::Vector<int> v(test_size);
    for (int i = 0; i < test_size; i++) {
        v[i] = i;
    }
    BSharedVector<int> b = engine.secret_share_b(v, 0);

    // our radix sort
    stopwatch::timepoint("Start");
    cdough::operators::radix_sort(b);
    stopwatch::timepoint("Ours");

    // AHI+22 radix sort
    cdough::operators::radix_sort_ccs(b, 32, true);
    stopwatch::timepoint("AHI+22");

    engine.print_statistics();

    return 0;
}