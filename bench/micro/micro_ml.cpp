#include "cdough.h"
#include "profiling/output.h"
#include "profiling/stopwatch.h"

using namespace cdough::debug;
using namespace cdough::service;
using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

using T = __int128_t;

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();
    auto test_size = engine.getArg<size_t>("test-size", "r", 1 << 20);

    int precision = 16;

    cdough::Vector<T> x(test_size);
    for (size_t i = 0; i < test_size; i++) {
        x[i] = i;
    }

    ASharedVector<T> x1 = engine.secret_share_a(x, 0);
    ASharedVector<T> x2 = engine.secret_share_a(x, 0);

    ASharedVector<T> y1 = engine.secret_share_a(x, 0, precision);
    ASharedVector<T> y2 = engine.secret_share_a(x, 0, precision);

    stopwatch::timepoint("Start");

    auto prod = x1 * x2;

    stopwatch::timepoint("Integer Multiplication");

    ASharedVector<T> prod_before_div = x1 * x2;
    int divisor = 1 << precision;

    auto div_result = prod_before_div / divisor;

    stopwatch::timepoint("Multiplication with Public Division");

    auto prod_fixed = y1 * y2;

    stopwatch::timepoint("Fixed-Point Multiplication");

    cdough::benchmarking::output("Dummy Accuracy", 0.51);

    stopwatch::done();

    return 0;
}
