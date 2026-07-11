#include "cdough.h"

using namespace cdough::debug;
using namespace cdough::service;
using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

const int test_size = 1 << 16;

template <typename T, typename Engine>
void TestBasicCommunication(int testSize, Engine& engine) {
    auto n = engine.getNumParties();
    auto othersCount = n - 1;

    // Exchanging one vector
    {
        cdough::Vector<T> x(testSize), y(testSize), z(testSize);
        engine.populateLocalRandom(x);

        // Exchange Shares with party +1
        engine.comm0()->exchangeShares(x, y, +1, -1);

        // Exchange Shares with party -1
        engine.comm0()->exchangeShares(y, z, -1, +1);

        // I should get the vector that I have just sent
        if (engine.getPartyID() == 0) {
            assert(x.same_as(z));
        }
    }

    // Exhange multiple vectors
    {
        std::vector<cdough::Vector<T>> x, y, z;
        for (int i = 0; i < othersCount; i++) {
            x.push_back(cdough::Vector<T>(testSize));
            y.push_back(cdough::Vector<T>(testSize));
            z.push_back(cdough::Vector<T>(testSize));
            engine.populateLocalRandom(x[i]);
        }

        std::vector<cdough::PartyID> to(othersCount);
        std::vector<cdough::PartyID> from(othersCount);
        for (int i = 0; i < othersCount; i++) {
            to[i] = i + 1;
            from[i] = -i - 1;
        }

        // Exchange Shares with parties +1, +2, +3
        engine.comm0()->exchangeShares(x, y, to, from);

        // Exchange Shares with parties -1, -2, -3
        engine.comm0()->exchangeShares(y, z, from, to);

        // I should get the vectors that I have just sent
        if (engine.getPartyID() == 0) {
            for (int i = 0; i < othersCount; i++) {
                assert(x[i].same_as(z[i]));
            }
        }
    }
}

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);

    TestBasicCommunication<int8_t>(test_size, engine);
    single_cout("int8_t communication: OK");

    TestBasicCommunication<int16_t>(test_size, engine);
    single_cout("int16_t communication: OK");

    TestBasicCommunication<int32_t>(test_size, engine);
    single_cout("int32_t communication: OK");

    TestBasicCommunication<int64_t>(test_size, engine);
    single_cout("int64_t communication: OK");

    TestBasicCommunication<__int128_t>(test_size, engine);
    single_cout("__int128_t communication: OK");

    // Tear down communication

    return 0;
}
