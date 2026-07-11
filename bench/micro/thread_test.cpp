/* thread_test
 *
 * This test allows for easy benchmarking of *individual* operations without
 * having to recompile. This is useful for automated data collection.
 *
 * Specify which op on the command line, like:
 *
 *  thread_test 1 1 8192 AND
 *
 * Other operations can be easily added below.
 */
#include "cdough.h"

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

    // 8M elements
    const size_t test_size = 1 << 23;

    auto op = engine.getArg<std::string>("op", "o", "AND");

    // TODO: deprecated constructor usage in public API
    BSharedVector<T> a(test_size, engine), b(test_size, engine);

    const int bw = std::numeric_limits<std::make_unsigned_t<T>>::digits;

    single_cout("Test " << test_size << " x " << bw << " " << op << "\n");

    stopwatch::timepoint("Start");

    if (op == "AND") {
        auto c = a & b;
    } else if (op == "EQ") {
        auto c = a == b;
    } else if (op == "GR") {
        auto c = a > b;
    } else if (op == "RCA") {
        auto c = a + b;
    } else if (op == "QS") {
        cdough::operators::quicksort(a);
    } else if (op == "RS") {
        cdough::operators::radix_sort(a);
    } else {
        std::cerr << "Unknown operator!\n";
        exit(-1);
    }

    stopwatch::timepoint("Exec " + op);

    engine.print_statistics();
    // thread_stopwatch::write(pID);
}
