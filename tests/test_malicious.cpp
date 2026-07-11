// Do not remove. This define tells `malicious_check` to not abort.
#define MAL_TEST_MODE

#include "cdough.h"

using namespace cdough::service;

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

/**
 * @brief Parties broadcast their checks, and then take AND of all received.
 * This ensure tests pass regardless of which party actually detected the
 * cheating.
 *
 */
template <typename Engine>
bool joint_malicious_check(Engine& engine) {
    bool my_check = engine.malicious_check();
    int r = true;

    for (int p = 1; p < engine.getNumParties(); p++) {
        engine.comm0()->sendShare(my_check, p);
    }
    for (int p = 1; p < engine.getNumParties(); p++) {
        engine.comm0()->receiveShare(r, p);
        my_check &= r;
    }

    // Allow us to run more tests.
    engine.reset_malicious_state();
    return my_check;
}

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);

// TODO: make optimized fantastic 4PC API match that of `dalskov_4pc`.
#if defined(MALICIOUS_PROTOCOL) && defined(USE_DALSKOV_FANTASTIC_FOUR)
    auto pid = engine.getPartyID();

    const int test_size = 1000;

    Vector<int> x(test_size), y(test_size);

    ASharedVector<int> a1 = engine.secret_share_a(x, 0);
    ASharedVector<int> a2 = engine.secret_share_a(y, 1);

    if (pid == 1) {
        // P1 cheats on one of its shares
        a1.vector(0)[test_size / 2] += 1;
    }

    // Call to `open()` will detect cheating
    a1.open();
    assert(!joint_malicious_check(engine));

    // Hashes should reset after a failed (non-abort) check. Should pass because
    // only local operations.
    auto c = a1 + a2;
    assert(joint_malicious_check(engine));

    // But open will catch it.
    c->open();
    assert(!joint_malicious_check(engine));

    // Check passes if we don't use manipulated data
    auto d = a2 * a2;
    assert(joint_malicious_check(engine));

    // But fails if we do
    auto e = a1 * a2;
    assert(!joint_malicious_check(engine));

#ifndef MPC_PROTOCOL_SPDZ_2K_NPC
    // Check boolean
    BSharedVector<int> b1 = engine.secret_share_b(x, 0);
    BSharedVector<int> b2 = engine.secret_share_b(y, 1);

    if (pid == 1) {
        // flip some bits
        b1.vector(0)[test_size / 3] ^= 0xffff;
    }

    auto f = b1 ^ b2;
    assert(joint_malicious_check(engine));

    auto g = b1 & b2;
    assert(!joint_malicious_check(engine));
#else
    single_cout("MPC_PROTOCOL_SPDZ_2K_NPC is defined. Skipping malicious boolean tests.");
#endif

    single_cout("Malicious primitives... OK");

#ifndef MPC_PROTOCOL_SPDZ_2K_NPC
    a1.shuffle();
    if (pid == 2) {
        a1.vector(0)[1] += 1;
    }
    assert(!joint_malicious_check(engine));
    single_cout("Malicious arithmetic shuffle... OK");
#else
    single_cout("MPC_PROTOCOL_SPDZ_2K_NPC is defined. Skipping malicious shuffle tests.");
#endif

#else
    single_cout("Malicious checks... skipped");
#endif

    return 0;
}