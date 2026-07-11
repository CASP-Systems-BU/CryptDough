#include <iostream>
#include <type_traits>

#include "cdough.h"
// explicit include for testing functionality
#include "core/random/permutations/permutation_manager.h"

using namespace cdough::debug;
using namespace cdough::service;
using namespace cdough::random;
using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

const size_t test_size_1 = 1 << 12;
const size_t test_size_2 = 1 << 11;
const size_t test_size_3 = 1 << 10;

template <typename Generator, typename Engine>
void test_pooled(Engine& engine) {
    auto pID = engine.getPartyID();
    auto comm = engine.comm0();
    auto r = engine.rand0()->commonPRGManager;

    auto pooled = make_pooled<Generator>(pID, r, comm, 0);

    // reserve a batch and check that it's been reserved
    pooled->reserve(test_size_1);
    pooled->reserve(test_size_2);
    assert(pooled->size() == test_size_1 + test_size_2);

    // get the batch and check that the correlation is correct
    auto batch = pooled->getNext(test_size_3);
    assert(pooled->size() == test_size_1 + test_size_2 - test_size_3);
    auto batch_2 = pooled->getNext(test_size_1 + test_size_2 - test_size_3);
    assert(pooled->size() == 0);

    pooled->assertCorrelated(batch);
}

#ifdef MPC_PROTOCOL_BEAVER_TWO
template <typename Generator, typename T, cdough::Encoding E, typename Engine>
void test_pooled_triples(Engine& engine) {
    auto pID = engine.getPartyID();
    auto comm = engine.comm0();
    auto r = engine.rand0()->commonPRGManager;

    auto pooled = make_pooled<Generator>(pID, r, comm, 0);

    auto btgen = std::make_shared<BeaverTripleGenerator<T, E>>(pooled, comm);

    btgen->reserve(test_size_1);

    auto batch = btgen->getNext(test_size_3);
    btgen->assertCorrelated(batch);

    engine.template reserve_mul_triples<int32_t>(test_size_1);
    engine.template reserve_and_triples<int32_t>(test_size_1);
}
#endif

template <typename Engine>
void test_pooled_permutations(size_t num_permutations, size_t permutation_size, Engine& engine) {
    auto pID = engine.getPartyID();
    auto commonPRGManager = std::shared_ptr<CommonPRGManager>(engine.rand0()->commonPRGManager);
    auto groups = engine.getGroups();

    // start timer
    stopwatch::get_elapsed();

    auto generator =
        engine.rand0()->template getCorrelation<int32_t, ShardedPermutationGenerator>();
    for (int i = 0; i < num_permutations; i++) {
        generator->getNext(permutation_size);
    }

    // stop timer
    auto unthreaded_time = stopwatch::get_elapsed();

    auto manager = PermutationManager::get();
    manager->reserve(permutation_size, num_permutations, engine);

    // stop timer
    auto threaded_time = stopwatch::get_elapsed();

    if (pID == 0) {
        assert(threaded_time < unthreaded_time);
    }

    auto permutation = manager->getNext<int32_t>(permutation_size, engine);
}

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);

#if !defined(MPC_PROTOCOL_BEAVER_TWO) && !defined(MPC_PROTOCOL_SPDZ_2K_NPC)
    if (engine.get_num_threads() > 1) {
        test_pooled_permutations(10, 100000, engine);
        single_cout("Parallel Permutations...OK");
    } else {
        single_cout("Parallel Permutations...SKIPPED");
    }
#elif defined(USE_LIBOTE)
    test_pooled<GilboaOLE<int8_t>>(engine);
    test_pooled<GilboaOLE<int32_t>>(engine);
    test_pooled<GilboaOLE<int64_t>>(engine);
    single_cout("Pooled Gilboa OLE...OK");

    test_pooled<SilentOT<int8_t>>(engine);
    test_pooled<SilentOT<int32_t>>(engine);
    test_pooled<SilentOT<int64_t>>(engine);
    single_cout("Pooled Silent OT...OK");

    test_pooled_triples<GilboaOLE<int8_t>, int8_t, cdough::Encoding::AShared>(engine);
    test_pooled_triples<GilboaOLE<int32_t>, int32_t, cdough::Encoding::AShared>(engine);
    test_pooled_triples<GilboaOLE<int64_t>, int64_t, cdough::Encoding::AShared>(engine);

    test_pooled_triples<SilentOT<int8_t>, int8_t, cdough::Encoding::BShared>(engine);
    test_pooled_triples<SilentOT<int32_t>, int32_t, cdough::Encoding::BShared>(engine);
    test_pooled_triples<SilentOT<int64_t>, int64_t, cdough::Encoding::BShared>(engine);
    single_cout("Pooled Beaver Triples...OK");
#else
    single_cout("skipped");
#endif

    return 0;
}