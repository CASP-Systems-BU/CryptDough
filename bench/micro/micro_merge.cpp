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

    cdough::Vector<int> v(test_size);
    for (int i = 0; i < test_size; i++) {
        v[i] = i;
    }
    // secret share and shuffle the vector
    BSharedVector<int> b = engine.secret_share_b(v, 0);
    b.shuffle();

    // sort each half of the list
    cdough::Vector<int> shuffled = b.open();
    std::vector<int> first_half(test_size / 2);
    std::vector<int> second_half(test_size / 2);
    for (int i = 0; i < test_size / 2; i++) {
        first_half[i] = shuffled[i];
        second_half[i] = shuffled[test_size / 2 + i];
    }
    std::sort(first_half.begin(), first_half.end());
    std::sort(second_half.begin(), second_half.end());
    // recombine and share
    cdough::Vector<int> vec_to_merge(test_size);
    for (int i = 0; i < test_size / 2; i++) {
        vec_to_merge[i] = first_half[i];
        vec_to_merge[test_size / 2 + i] = second_half[i];
    }
    BSharedVector<int> b2 = engine.secret_share_b(vec_to_merge, 0);

    // start timer
    stopwatch::timepoint("Start");

    cdough::operators::odd_even_merge(b2);

    // stop timer
    stopwatch::timepoint("Merge");

    return 0;
}