#include "cdough.h"

// Include other files if not already included
#include "core/math/primes.h"

// ../scripts/run_experiment.py -p 3 -s same -c mpi -r 1 -T 1 test_correlated

using namespace cdough;
using namespace cdough::service;
using namespace cdough::random;
using namespace cdough::math::primes;
using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

const size_t test_size = 1 << 12;

template <typename T, template <typename> class C, typename Engine>
void checkCorrelation(std::string label, Engine& engine) {
    auto L = std::numeric_limits<std::make_unsigned_t<T>>::digits;
    single_cout_nonl("Checking length " << test_size << " " << L << "-bit " << label << "... ");

    auto gen = engine.rand0()->template getCorrelation<T, C<T>>();
    auto corr = gen->getNext(test_size);

    gen->assertCorrelated(corr);

    single_cout("OK");
}

template <typename T, typename Engine>
void test_permutation_correlations(int test_size_, Engine& engine) {
#ifdef MPC_PROTOCOL_BEAVER_TWO
    // get the generator and interpret it as a dishonest-majority generator
    // otherwise it won't have the assertCorrelated function
    auto base_generator = engine.rand0()->template getCorrelation<T, ShardedPermutationGenerator>();
    auto generator = dynamic_pointer_cast<DMShardedPermutationGenerator<T>>(base_generator);

    // check for nullptr
    if (generator == nullptr) {
        throw std::runtime_error("Failed to get generator of type DMShardedPermutationGenerator");
    }

    cdough::random::PermutationManager::get()->reserve(test_size_, 2, engine);

    // check individual permutation correlations
    auto result_a = cdough::random::PermutationManager::get()->getNext<T>(test_size_, engine,
                                                                       cdough::Encoding::AShared);
    auto result_b = cdough::random::PermutationManager::get()->getNext<T>(test_size_, engine,
                                                                       cdough::Encoding::BShared);

    std::shared_ptr<DMShardedPermutation<T>> perm_corr_a =
        std::dynamic_pointer_cast<DMShardedPermutation<T>>(result_a);
    std::shared_ptr<DMShardedPermutation<T>> perm_corr_b =
        std::dynamic_pointer_cast<DMShardedPermutation<T>>(result_b);
    // check for nullptr
    if ((perm_corr_a == nullptr) || (perm_corr_b == nullptr)) {
        throw std::runtime_error("Failed to get permutation");
    }

    generator->assertCorrelated(perm_corr_a);
    generator->assertCorrelated(perm_corr_b);

    // check pairs of permutation correlations
    auto [first, second] =
        cdough::random::PermutationManager::get()->getNextPair<T, T>(test_size_, engine);
    auto pair_first = std::dynamic_pointer_cast<DMShardedPermutation<T>>(first);
    auto pair_second = std::dynamic_pointer_cast<DMShardedPermutation<T>>(second);
    if ((pair_first == nullptr) || (pair_second == nullptr)) {
        throw std::runtime_error("Failed to get permutation in pair");
    }
    // make sure each permutation individually is correct
    generator->assertCorrelated(pair_first);
    generator->assertCorrelated(pair_second);
#endif
}

template <typename InT, typename OutT>
void test_shprg(EngineRef& engine) {
#ifdef MPC_PROTOCOL_BEAVER_TWO
    // short seed length for testing
    int seed_length = 1024;
    // output length is 2x the seed length
    int n = seed_length * 2;

    // a function to test the almost homomorphism of two inputs
    auto test_almost_homomorphism = [](const std::vector<OutT>& input1,
                                       const std::vector<OutT>& input2) {
        int n = input1.size();
        for (int i = 0; i < n; i++) {
            auto diff = input1[i] - input2[i];
            assert((diff == -1) || (diff == 0) || (diff == 1));
        }
    };

    auto shprg = SeedHomomorphicPRG<InT, OutT>(engine, seed_length, seed_length * 2);

    // sample two full domain seeds
    auto seed1 = shprg.sample_seed(engine);
    auto seed2 = shprg.sample_seed(engine);
    auto seed_sum = shprg.add_seeds(seed1, seed2);

    // expand the seeds
    auto expanded1 = shprg.expand(seed1);
    auto expanded2 = shprg.expand(seed2);
    auto expanded_sum = shprg.expand(seed_sum);
    auto results_sum = shprg.add_results(expanded1, expanded2);

    test_almost_homomorphism(expanded_sum, results_sum);

    // sample two binary seeds
    auto binary_seed1 = shprg.sample_binary_seed(engine);
    auto binary_seed2 = shprg.sample_binary_seed(engine);
    // make sure the sum of seeds is still binary
    for (int i = 0; i < seed_length / 2; i++) {
        binary_seed2[i] = 0;
    }
    for (int i = seed_length / 2; i < seed_length; i++) {
        binary_seed1[i] = 0;
    }
    auto binary_seed_sum = shprg.add_seeds(binary_seed1, binary_seed2);

    // expand the binary seeds
    auto expanded_binary1 = shprg.expand_binary_seed(binary_seed1);
    auto expanded_binary2 = shprg.expand_binary_seed(binary_seed2);
    auto expanded_binary_sum = shprg.expand_binary_seed(binary_seed_sum);
    auto results_binary_sum = shprg.add_results(expanded_binary1, expanded_binary2);

    test_almost_homomorphism(expanded_binary_sum, results_binary_sum);

    // check that compressed seed expansion is the same as uncompressed seed expansion
    auto compressed_seed = shprg.compress_binary_seed(binary_seed1);
    auto expanded_compressed = shprg.expand_compressed_binary_seed(compressed_seed);
    for (int i = 0; i < n; i++) {
        assert(expanded_compressed[i] == expanded_binary1[i]);
    }
#endif
}

template <typename T, typename Engine>
void TestDummyAuthTriplesGenerator(const size_t testSize, Engine& engine) {
    // Party Information
    auto pID = engine.getPartyID();
    auto pNum = engine.getNumParties();
    auto othersCount = pNum - 1;

    // Helpers
    auto communicator = engine.comm0();
    auto randManager = engine.rand0();
    auto zeroSharingGenerator = randManager->zeroSharingGenerator;
    auto localPRG = randManager->localPRG;

    // Generating the key
    T partyKey;
    localPRG->getNext(partyKey);

    // Creating the dummy generator
    auto dummyGenerator = cdough::random::DummyAuthTripleGenerator<T, cdough::EVector<T, 2>>(
        pNum, partyKey, pID, localPRG, zeroSharingGenerator, communicator);

    // Getting some dummy triples
    auto triple = dummyGenerator.getNext(testSize);
    dummyGenerator.assertCorrelated(triple);
}

template <typename T, typename Engine>
void TestDummyAuthRandomGenerator(const size_t testSize, Engine& engine) {
    // Party Information
    auto pID = engine.getPartyID();
    auto pNum = engine.getNumParties();
    auto othersCount = pNum - 1;

    // Helpers
    auto communicator = engine.comm0();
    auto randManager = engine.rand0();
    auto zeroSharingGenerator = randManager->zeroSharingGenerator;
    auto localPRG = randManager->localPRG;

    // Generating the key
    T partyKey;
    localPRG->getNext(partyKey);

    // Creating the dummy generator
    auto dummyGenerator = cdough::random::DummyAuthRandomGenerator<T, cdough::EVector<T, 2>>(
        pNum, partyKey, pID, localPRG, zeroSharingGenerator, communicator);

    // Getting some dummy random numbers
    auto a = dummyGenerator.getNext(testSize);
    dummyGenerator.assertCorrelated(a);
}

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pid = engine.getPartyID();

#ifndef MPC_PROTOCOL_BEAVER_TWO
    single_cout("Skipping test_correlated for non-2PC");
#else
    checkCorrelation<int8_t, OLEGenerator>("OLE", engine);
    checkCorrelation<int32_t, OLEGenerator>("OLE", engine);
    checkCorrelation<int64_t, OLEGenerator>("OLE", engine);
    checkCorrelation<__int128_t, OLEGenerator>("OLE", engine);

    checkCorrelation<int8_t, OTGenerator>("rOT", engine);
    checkCorrelation<int32_t, OTGenerator>("rOT", engine);
    checkCorrelation<int64_t, OTGenerator>("rOT", engine);
    checkCorrelation<__int128_t, OTGenerator>("rOT", engine);

    checkCorrelation<int8_t, BeaverAndGenerator>("Beaver AND Triples", engine);
    checkCorrelation<int32_t, BeaverAndGenerator>("Beaver AND Triples", engine);
    checkCorrelation<int64_t, BeaverAndGenerator>("Beaver AND Triples", engine);
    checkCorrelation<__int128_t, BeaverAndGenerator>("Beaver AND Triples", engine);

    checkCorrelation<int8_t, BeaverMulGenerator>("Beaver Triples", engine);
    checkCorrelation<int32_t, BeaverMulGenerator>("Beaver Triples", engine);
    checkCorrelation<int64_t, BeaverMulGenerator>("Beaver Triples", engine);
    checkCorrelation<__int128_t, BeaverMulGenerator>("Beaver Triples", engine);

    // we only generate 128-bit permutation correlations and cut them down
    // so we only have a 128-bit generator object to run assertCorrelated
    test_permutation_correlations<__int128_t>(1000, engine);
    single_cout("Permutation Correlations... OK");

    test_shprg<int64_t, int32_t>(engine);
    test_shprg<__int128_t, int64_t>(engine);
    single_cout("SHPRG... OK");
#endif

    ///////////////////////////////////////////
    // Testing the DummyAuthTriplesGenerator //
    ///////////////////////////////////////////
    single_cout("Testing DummyAuthTriplesGenerator...");
    TestDummyAuthTriplesGenerator<int8_t>(test_size, engine);
    single_cout("int8_t dummy authenticated triples generation: OK");

    TestDummyAuthTriplesGenerator<int16_t>(test_size, engine);
    single_cout("int16_t dummy authenticated triples generation: OK");

    TestDummyAuthTriplesGenerator<int32_t>(test_size, engine);
    single_cout("int32_t dummy authenticated triples generation: OK");

    TestDummyAuthTriplesGenerator<int64_t>(test_size, engine);
    single_cout("int64_t dummy authenticated triples generation: OK");

    TestDummyAuthTriplesGenerator<__int128_t>(test_size, engine);
    single_cout("__int128_t dummy authenticated triples generation: OK");

    single_cout("");

    ///////////////////////////////////////////
    // Testing the DummyAuthRandomGenerator ///
    ///////////////////////////////////////////
    single_cout("Testing DummyAuthRandomGenerator...");
    TestDummyAuthRandomGenerator<int8_t>(test_size, engine);
    single_cout("int8_t dummy authenticated random generation: OK");

    TestDummyAuthRandomGenerator<int16_t>(test_size, engine);
    single_cout("int16_t dummy authenticated random generation: OK");

    TestDummyAuthRandomGenerator<int32_t>(test_size, engine);
    single_cout("int32_t dummy authenticated random generation: OK");

    TestDummyAuthRandomGenerator<int64_t>(test_size, engine);
    single_cout("int64_t dummy authenticated random generation: OK");

    TestDummyAuthRandomGenerator<__int128_t>(test_size, engine);
    single_cout("__int128_t dummy authenticated random generation: OK");

    engine.malicious_check();
}
