#include "cdough.h"

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;
using namespace cdough::debug;
using namespace cdough::service;

// ../scripts/run_experiment.py -p 3 -r 4 energy

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();
    int test_size = engine.getArg<int>("test-size", "r", 8);  // default for startmpc/mpirun

    std::vector<std::string> schema = {"[TIMESTAMP]",
                                       "TIMESTAMP",
                                       "[DEVICE_ID]",
                                       "TUMBLING_WINDOW_PER_HOUR",
                                       "[TUMBLING_WINDOW_PER_HOUR]",
                                       "ENERGY_CONSUMPTION",
                                       "TOTAL_CONSUMPTION"};

    std::vector<cdough::Vector<int32_t>> energy_data(schema.size(), Vector<int32_t>(test_size));

    EncodedTable<int32_t> energy_table = engine.secret_share_table(energy_data, schema);
    ASharedVector<int64_t> timestamp_a(test_size, engine);

    // start timer
    stopwatch::timepoint("Start");

    // energy_table.tumbling_window("TIMESTAMP", 3600, "TUMBLING_WINDOW_PER_HOUR");
    ASharedVector<int64_t> window_id = timestamp_a / 3600;
    (*energy_table["TUMBLING_WINDOW_PER_HOUR"].contents.get()) = window_id;
    energy_table.convert_a2b("TUMBLING_WINDOW_PER_HOUR", "[TUMBLING_WINDOW_PER_HOUR]");
    energy_table.sort({{"[TUMBLING_WINDOW_PER_HOUR]", ASC}}, {"ENERGY_CONSUMPTION"});

    using A = ASharedVector<int>;
    energy_table.aggregate({"[TUMBLING_WINDOW_PER_HOUR]"},
                           {{"ENERGY_CONSUMPTION", "TOTAL_CONSUMPTION", cdough::aggregators::sum<A>}});

    stopwatch::done();
    return 0;
}