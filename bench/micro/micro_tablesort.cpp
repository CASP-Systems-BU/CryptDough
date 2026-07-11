#include "cdough.h"
#include "profiling/stopwatch.h"

using namespace cdough::debug;
using namespace cdough::service;
using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

#include <unistd.h>

#include <cmath>

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();

    auto test_size = engine.getArg<size_t>("test-size", "r", 1 << 20);
    auto num_columns = engine.getArg<int>("num-columns", "nc", 4);
    auto num_sort_columns = engine.getArg<int>("num-sort-columns", "ns", 1);

    auto localPRG = engine.rand0()->localPRG.get();

    // generate a table
    std::vector<cdough::Vector<int>> table_data;
    std::vector<std::string> schema;
    for (int i = 0; i < num_columns; i++) {
        table_data.push_back(cdough::Vector<int>(test_size));
        schema.push_back("[" + std::to_string(i) + "]");
        localPRG->getNext(table_data[i]);
    }
    EncodedTable<int> table1 = engine.secret_share_table(table_data, schema);
    EncodedTable<int> table2 = engine.secret_share_table(table_data, schema);
    EncodedTable<int> table3 = engine.secret_share_table(table_data, schema);

    std::vector<std::pair<std::string, SortOrder>> spec;
    for (int i = 0; i < num_sort_columns; i++) {
        spec.push_back(std::make_pair("[" + std::to_string(i) + "]", ASC));
    }
    spec.push_back(std::make_pair(ENC_TABLE_VALID, ASC));

    stopwatch::timepoint("Start");
    stopwatch::profile_init();

    table1.sort(spec, cdough::SortingProtocol::NETWORK);
    stopwatch::timepoint("Table Sorting Network");

    table2.sort(spec, cdough::SortingProtocol::QUICKSORT);
    stopwatch::timepoint("Table Quicksort");

    table3.sort(spec, cdough::SortingProtocol::RADIXSORT);
    stopwatch::timepoint("Table Radixsort");

    stopwatch::profile_done();

    return 0;
}