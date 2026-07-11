#include "cdough.h"

using namespace cdough::debug;
using namespace cdough::service;
using namespace cdough::random;
using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

void test_setup(int max_threads) {
    int batch_size = 8192;

    // create runtime objects with variable number of threads and make sure they successfully create
    // if this test hangs, there is a deadlock in setup
    for (int num_threads = 1; num_threads < max_threads; num_threads <<= 1) {
        RunTime<cdough::service::null_service::plaintext_1pc::ProtocolFactory> rt(
            batch_size, num_threads, {}, true);
        rt.setup_workers(0);
    }
}

template <typename Engine>
void test_modify_parallel(int test_size, Engine& engine) {
    cdough::Vector<int> v(test_size);
    std::iota(v.begin(), v.end(), 0);

    BSharedVector<int> b = engine.secret_share_b(v, 0);

    // masking calls engine.modify_parallel
    b.mask(1);

    auto opened = b.open();

    v.mask(1);

    assert(opened.same_as(v));
}

template <typename Engine>
void test_parallel_generation(size_t test_size, Engine& engine) {
    int pID = engine.getPartyID();

    /*
        common randomness
    */
    cdough::Vector<int> common(test_size);
    stopwatch::get_elapsed();
    std::set<int> group = engine.getPartySet();

    engine.rand0()->commonPRGManager->get(group)->getNext(common);
    auto unthreaded_time = stopwatch::get_elapsed();

    engine.populateCommonRandom(common, group);
    auto threaded_time = stopwatch::get_elapsed();

    if (pID == 0) {
        assert(threaded_time < unthreaded_time);
    }
}

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);

    test_setup(32);
    single_cout("Runtime Setup...OK");

// Test uses boolean secret sharing
#ifndef MPC_PROTOCOL_SPDZ_2K_NPC
    test_modify_parallel(1000000, engine);
    single_cout("Modify Parallel...OK");
#else
    single_cout(
        "MPC_PROTOCOL_SPDZ_2K_NPC is defined. Skipping test_modify_parallel using B Shares.");
#endif

    auto T = engine.get_num_threads();
    if (T > 1) {
        test_parallel_generation(T * (1 << 24), engine);
        single_cout("Parallel Generation...OK");
    } else {
        single_cout("Parallel Generation...SKIPPED");
    }

    // Create test vector
    cdough::Vector<int> vec_1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    return 0;
}