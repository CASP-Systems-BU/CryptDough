#include "cdough.h"

using namespace cdough::debug;
using namespace cdough::service;

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();
    auto test_size = engine.getArg<size_t>("test-size", "r", 1 << 20);

    std::vector<std::string> schema = {"[TIMESTAMP]", "[ID]", "[FUNC]", "[THRESHOLD_WINDOW]"};
    std::vector<cdough::Vector<int>> data(schema.size(), cdough::Vector<int>(test_size));
    EncodedTable<int> table = engine.secret_share_table(data, schema);

    // start timer
    struct timeval begin, end;
    long seconds, micro;
    double elapsed;
    gettimeofday(&begin, 0);

    table.threshold_session_window({"[ID]"}, "[FUNC]", "[TIMESTAMP]", "[THRESHOLD_WINDOW]", 5,
                                   false);
    // stop timer
    gettimeofday(&end, 0);
    seconds = end.tv_sec - begin.tv_sec;
    micro = end.tv_usec - begin.tv_usec;
    elapsed = seconds + micro * 1e-6;
    if (pID == 0) {
        std::cout << "MICRO_SESSION_THRESHOLD:\t\t\t" << test_size << "\t\telapsed\t\t" << elapsed
                  << std::endl;
    }

    return 0;
}