/**
 * @file ex6_three_party_private_input.cpp
 * @brief Example 6: three organizations, each contributing its own private table.
 *
 * This is the data model for a real cross-organizational deployment: every party holds
 * a plaintext CSV that must never leave its machine, and the computation runs over the
 * union without any party seeing another's rows.
 *
 * The mechanism is `inputCSVTableData(path, input_party)`. Only the party whose ID
 * matches `input_party` opens the file; every other party allocates zero-filled columns
 * and receives secret shares over the wire. So each organization can mount only its own
 * data directory, and the plaintext is never transmitted.
 *
 * Run it with docker/run-party-external.sh, one party per organization. Each party
 * mounts ITS OWN data at /data:
 *
 *   party 0:  --data-dir /srv/org-a/private     (holds /data/party0.csv)
 *   party 1:  --data-dir /srv/org-b/private     (holds /data/party1.csv)
 *   party 2:  --data-dir /srv/org-c/private     (holds /data/party2.csv)
 *
 * Every party constructs the SAME schema even though only one of them reads each file:
 * the table layout is public, only the contents are private. Keep the schema in the run
 * manifest so the parties cannot drift.
 *
 * Expected CSV layout (header line names the columns, in schema order):
 *
 *   [ID],[ZIP],[SCORE]
 *   1,02215,71
 *   2,02139,63
 *
 * Columns are parsed as integers, so a value like 02215 loads as 2215. Anything where
 * leading zeros or non-numeric characters matter (ZIP codes, identifiers) has to be
 * encoded numerically by the parties beforehand, as part of the schema they agree on.
 */

#include "cdough.h"

// Tell cdough to use the selected protocol & communicator
using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();

    // Public metadata: identical on every party.
    const std::string tableName = "Contributions";
    const std::vector<std::string> schema = {"[ID]", "[ZIP]", "[SCORE]"};
    const int rowsPerParty = 8;

    // The directory each organization mounts read-only at /data. Only the owning party
    // ever opens its own file.
    const std::string dataDir = "/data/";

    single_cout("Three-party private input: each party contributes " +
                std::to_string(rowsPerParty) + " rows");

    // One table per contributing organization. All three tables exist on all three
    // parties -- what differs is who can actually read the underlying file.
    std::vector<EncodedTable<int32_t>> contributions;
    for (int owner = 0; owner < 3; ++owner) {
        contributions.push_back(
            engine.secret_share_table<int32_t>(tableName + "_" + std::to_string(owner), schema,
                                               rowsPerParty));
    }

    for (int owner = 0; owner < 3; ++owner) {
        const std::string path = dataDir + "party" + std::to_string(owner) + ".csv";
        // Every party calls this; only party `owner` opens the file. The others
        // allocate zeros and receive shares, so nothing plaintext crosses the wire.
        contributions[owner].inputCSVTableData(path, owner);
        single_cout("  loaded contribution from party " + std::to_string(owner));
    }

    // From here the three tables are ordinary secret-shared tables and every operator
    // in the framework applies. Opening the result reveals it to all parties, so in a
    // real deployment open only the aggregate the parties agreed to learn -- never the
    // input tables themselves.
    auto opened = contributions[0].open_with_schema();
    print_table(opened, pID);

    return 0;
}
