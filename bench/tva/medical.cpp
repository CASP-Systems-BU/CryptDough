#include "cdough.h"

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;
using namespace cdough::debug;
using namespace cdough::service;

// ../scripts/run_experiment.py -p 3 -r 4 medical

#define GLUCOSE_THRESHOLD 5

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();
    int test_size = engine.getArg<int>("test-size", "r", 8);  // default for startmpc/mpirun

    std::vector<std::string> schema = {"[TIMESTAMP]", "[PATIENT_ID]", "[GLUCOSE]", "INSULIN"};

    std::vector<cdough::Vector<int32_t>> medical_data(schema.size(), Vector<int32_t>(test_size));
    EncodedTable<int32_t> medical_table = engine.secret_share_table(medical_data, schema);

    // start timer
    stopwatch::timepoint("Start");

    medical_table.sort({"[PATIENT_ID]", "[TIMESTAMP]"}, ASC);

    medical_table.addColumns({"[THRESHOLD_WINDOW]", "TOTAL_EVENTS"}, test_size);

    medical_table.threshold_session_window({"[PATIENT_ID]"}, "[GLUCOSE]", "[TIMESTAMP]",
                                           "[THRESHOLD_WINDOW]", GLUCOSE_THRESHOLD, false);

    /* This call WILL mark final result rows valid. If further computation
     * was occurring, we might want to sort on VALID + trim, but there's no need
     * here. We will simply send the entire shared table to the frontend.
     */
    using A = ASharedVector<int>;
    medical_table.aggregate({"[PATIENT_ID]", "[THRESHOLD_WINDOW]"},
                            {{"INSULIN", "TOTAL_EVENTS", cdough::aggregators::sum<A>}});

    // Remove intermediate columns
    medical_table.deleteColumns({"INSULIN", "[GLUCOSE]", "[TIMESTAMP]"});

    // Mask out invalid rows + shuffle for privacy.
    medical_table.finalize();

    stopwatch::done();
    return 0;
}