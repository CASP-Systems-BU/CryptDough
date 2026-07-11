#include "cdough.h"

using namespace cdough::debug;
using namespace cdough::service;

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();
    int test_size = engine.getArg<size_t>("test-size", "r", 1 << 20);

    // TODO: deprecated constructor usage in public API
    BSharedVector<int> a(test_size, engine), b(test_size, engine);

    // start timer
    struct timeval begin, end;
    long seconds, micro;
    double elapsed;
    gettimeofday(&begin, 0);

    BSharedVector<int> c_1 = a.rca(b);

    // stop timer
    gettimeofday(&end, 0);
    seconds = end.tv_sec - begin.tv_sec;
    micro = end.tv_usec - begin.tv_usec;
    elapsed = seconds + micro * 1e-6;
    if (pID == 0) {
        std::cout << "RCA_QUERY:\t\t\t" << test_size << "\t\telapsed\t\t" << elapsed << std::endl;
    }

    gettimeofday(&begin, 0);

    BSharedVector<int> c_2 = a.ppa(b);

    gettimeofday(&end, 0);
    seconds = end.tv_sec - begin.tv_sec;
    micro = end.tv_usec - begin.tv_usec;
    elapsed = seconds + micro * 1e-6;
    if (pID == 0) {
        std::cout << "CLA_QUERY:\t\t\t" << test_size << "\t\telapsed\t\t" << elapsed << std::endl;
    }

    return 0;
}