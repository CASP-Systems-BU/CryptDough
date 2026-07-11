#include "cdough.h"

using namespace cdough::debug;
using namespace cdough::service;

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();
    int test_size = engine.getArg<size_t>("test-size", "r", 1 << 20);

    std::vector<std::string> schema = {"[SEL]", "DATA", "[DATA]", "SUM", "[MAX]", "[MIN]"};
    std::vector<cdough::Vector<int>> data(schema.size(), cdough::Vector<int>(test_size));
    EncodedTable<int> table = engine.secret_share_table(data, schema);

    // start timer
    struct timeval begin, end;
    long seconds, micro;
    double elapsed;
    gettimeofday(&begin, 0);

    using A = ASharedVector<int>;

    table.aggregate({"[SEL]"}, {{"DATA", "SUM", cdough::aggregators::sum<A>}});

    // stop timer
    gettimeofday(&end, 0);
    seconds = end.tv_sec - begin.tv_sec;
    micro = end.tv_usec - begin.tv_usec;
    elapsed = seconds + micro * 1e-6;
    if (pID == 0) {
        std::cout << "SUM_AGGREGATION:\t\t\t" << test_size << "\t\telapsed\t\t" << elapsed
                  << std::endl;
    }

    gettimeofday(&begin, 0);

    using B = BSharedVector<int>;

    table.aggregate({"[SEL]"}, {
                                   {"[DATA]", "[MIN]", cdough::aggregators::min<B>},
                               });

    gettimeofday(&end, 0);
    seconds = end.tv_sec - begin.tv_sec;
    micro = end.tv_usec - begin.tv_usec;
    elapsed = seconds + micro * 1e-6;
    if (pID == 0) {
        std::cout << "MIN_AGGREGATION:\t\t\t" << test_size << "\t\telapsed\t\t" << elapsed
                  << std::endl;
    }

    gettimeofday(&begin, 0);

    table.aggregate({"[SEL]"}, {{"[DATA]", "[MAX]", cdough::aggregators::max<B>}});

    gettimeofday(&end, 0);
    seconds = end.tv_sec - begin.tv_sec;
    micro = end.tv_usec - begin.tv_usec;
    elapsed = seconds + micro * 1e-6;
    if (pID == 0) {
        std::cout << "MAX_AGGREGATION:\t\t\t" << test_size << "\t\telapsed\t\t" << elapsed
                  << std::endl;
    }

    stopwatch::done();          // print wall clock time
    stopwatch::profile_done();  // print profiling data

    engine.print_statistics();
    engine.print_communicator_statistics();

    return 0;
}