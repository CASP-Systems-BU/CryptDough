#include <span>

#include "cdough.h"
#include "profiling/stopwatch.h"

using namespace cdough::debug;
using namespace cdough::service;
using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

#define REPEAT(n, expr)           \
    for (int i = 0; i < n; i++) { \
        expr;                     \
    }

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();

    if (pID != 0) {
        return 0;
    }

    auto test_size = engine.getArg<size_t>("test-size", "r", 1 << 20);

    std::vector<int> v1 = std::vector<int>(test_size, 0);
    std::iota(v1.begin(), v1.end(), 0);

    std::vector<int> v2 = std::vector<int>(test_size, 0);
    std::iota(v2.begin(), v2.end(), 0);

    std::span<int> s1 = std::span<int>(v2);

    stopwatch::timepoint("Start");

    cdough::Vector<int> V1 = cdough::Vector<int>(std::move(v1));
    stopwatch::timepoint("Move constructor");

    cdough::Vector<int> V2 = cdough::Vector<int>(v2);
    stopwatch::timepoint("Copy constructor");

    cdough::Vector<int> V3 = cdough::Vector<int>(test_size, 5);
    stopwatch::timepoint("Repeated value constructor");

    cdough::Vector<int> V4 = cdough::Vector<int>(s1);
    stopwatch::timepoint("Range copy constructor (using span)");

    stopwatch::done();

    return 0;
}
