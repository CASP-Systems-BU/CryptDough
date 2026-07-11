#pragma once

#include <sodium.h>
#include <unistd.h>

#include <algorithm>
#include <set>
#include <typeindex>

#ifdef USE_LIBOTE
#include "backend/common/libote_io.h"
#endif

#include "core/random/permutations/dm_dummy.h"
#include "core/random/prg/random_generator.h"

namespace cdough::service {

/**
 * Utility function to setup a CommonPRG among a group.
 * @param rank The absolute rank of the current party.
 * @param group The group that shares the CommonPRG.
 * @param commonPRGManager The CommonPRGManager to add the new CommonPRG to.
 */
template <typename Engine>
void create_common_prg_among_group(int rank, std::set<int> group,
                                   std::shared_ptr<cdough::random::CommonPRGManager> commonPRGManager,
                                   Engine& engine) {
    // only members of the group should participate
    // if not in the group, just return
    if (!group.contains(rank)) {
        return;
    }

    unsigned char seed[crypto_aead_aes256gcm_KEYBYTES];

    // lowest rank party in the group determines the seed
    int lowestRank = *group.begin();  // this is deterministic as sets are sorted

    if (rank == lowestRank) {
        // generate a seed
        cdough::random::AESPRGAlgorithm::aesKeyGen(seed);
        // convert unsigned char to uint8_t vector to be sent
        std::vector<int8_t> seed_bytes;
        for (int byte = 0; byte < crypto_aead_aes256gcm_KEYBYTES; byte++) {
            seed_bytes.push_back((int8_t)seed[byte]);
        }
        cdough::Vector<int8_t> seed_to_send(seed_bytes);

        // send to all parties
        for (int other_rank : group) {
            if (rank == other_rank) continue;
            int relative_rank = other_rank - rank;
            cdough::Vector<int8_t> empty(crypto_aead_aes256gcm_KEYBYTES);
            engine.comm0()->exchangeShares(seed_to_send, empty, relative_rank, relative_rank);
        }
    } else {
        // all other parties receive
        cdough::Vector<int8_t> remote(crypto_aead_aes256gcm_KEYBYTES);
        int relative_rank = lowestRank - rank;

        cdough::Vector<int8_t> empty(crypto_aead_aes256gcm_KEYBYTES);
        engine.comm0()->exchangeShares(empty, remote, relative_rank, relative_rank);

        // convert uint8_t vector back to unsigned char for key
        for (int byte = 0; byte < crypto_aead_aes256gcm_KEYBYTES; byte++) {
            seed[byte] = (unsigned char)remote[byte];
        }
    }

    std::vector<unsigned char> seed_vec;
    for (int byte = 0; byte < crypto_aead_aes256gcm_KEYBYTES; byte++) {
        seed_vec.push_back(seed[byte]);
    }

    // by now, all parties agree on a seed
    std::unique_ptr<DeterministicPRGAlgorithm> prg_algorithm =
        std::make_unique<cdough::random::AESPRGAlgorithm>(seed_vec);
    auto commonPRG = std::make_shared<cdough::random::CommonPRG>(std::move(prg_algorithm), rank);
    commonPRGManager->add(commonPRG, group);
}

// NOTE: maybe these setup subroutines should be a part of the protocol?
// Or, common functionality can be broken out into a function here, and protocol
// specific stuff moved into the protocol itself.

/**
 * @brief Setup the Common PRGs for replicated 3PC
 *
 * @param rank
 * @param commonPRGManager
 */
template <typename Engine>
void setup_3pc_common_prgs(int rank,
                           std::shared_ptr<cdough::random::CommonPRGManager> commonPRGManager,
                           Engine& engine) {
    /*
        generate and exchange seeds
    */
    // generate a seed
    unsigned char local_key[crypto_aead_aes256gcm_KEYBYTES];
    cdough::random::AESPRGAlgorithm::aesKeyGen(local_key);
    // convert unsigned char to uint8_t vector to be exchanged
    // TODO: this conversion can probably be done more easily
    std::vector<int8_t> local_key_bytes;
    for (int byte = 0; byte < crypto_aead_aes256gcm_KEYBYTES; byte++) {
        local_key_bytes.push_back((int8_t)local_key[byte]);
    }
    // exchange keys
    cdough::Vector<int8_t> local(local_key_bytes);
    cdough::Vector<int8_t> remote(crypto_aead_aes256gcm_KEYBYTES);
    engine.comm0()->exchangeShares(local, remote, +1, -1);
    // convert uint8_t vector back to unsigned char for key
    unsigned char remote_key[crypto_aead_aes256gcm_KEYBYTES];
    for (int byte = 0; byte < crypto_aead_aes256gcm_KEYBYTES; byte++) {
        remote_key[byte] = (unsigned char)remote[byte];
    }

    /*
        create the CommonPRG objects
    */
    for (int j = 0; j < 3; j++) {
        if (j == rank) continue;

        int relative_rank = (3 + j - rank) % 3;

        unsigned char seed[crypto_aead_aes256gcm_KEYBYTES];
        if (relative_rank == 1) {
            // seed = local_key
            for (int byte = 0; byte < crypto_aead_aes256gcm_KEYBYTES; byte++) {
                seed[byte] = local_key[byte];
            }
        } else if (relative_rank == 2) {
            // seed = remote_key
            for (int byte = 0; byte < crypto_aead_aes256gcm_KEYBYTES; byte++) {
                seed[byte] = remote_key[byte];
            }
        }

        std::vector<unsigned char> seed_vec;
        for (int byte = 0; byte < crypto_aead_aes256gcm_KEYBYTES; byte++) {
            seed_vec.push_back(seed[byte]);
        }

        std::unique_ptr<DeterministicPRGAlgorithm> prg_algorithm =
            std::make_unique<cdough::random::AESPRGAlgorithm>(seed_vec);
        auto commonPRG = std::make_shared<cdough::random::CommonPRG>(std::move(prg_algorithm), rank);
        commonPRGManager->add(commonPRG, relative_rank);
    }
}

/**
 * @brief Setup the Common PRGs for Fantastic 4PC
 *
 * @param rank
 * @param commonPRGManager
 */
template <typename Engine>
void setup_4pc_common_prgs(int rank,
                           std::shared_ptr<cdough::random::CommonPRGManager> commonPRGManager,
                           Engine& engine) {
    /*
        generate and exchange seeds
    */
    // generate a seed
    unsigned char local_key[crypto_aead_aes256gcm_KEYBYTES];
    cdough::random::AESPRGAlgorithm::aesKeyGen(local_key);
    // convert unsigned char to uint8_t vector to be exchanged
    // TODO: this conversion can probably be done more easily
    std::vector<int8_t> local_key_bytes;
    for (int byte = 0; byte < crypto_aead_aes256gcm_KEYBYTES; byte++) {
        local_key_bytes.push_back((int8_t)local_key[byte]);
    }

    /*
        create the CommonPRG objects
    */
    for (int j = 0; j < 4; j++) {
        if (j == rank) continue;

        // prepare to exchange keys
        cdough::Vector<int8_t> local(local_key_bytes);
        cdough::Vector<int8_t> remote(crypto_aead_aes256gcm_KEYBYTES);

        // this is just a way of identifying which parties the CommonPRG is shared with
        // unlike in 3PC, this is now the relative_rank of the party the CommonPRG is NOT shared
        // with
        int relative_rank = (4 + j - rank) % 4;

        unsigned char seed[crypto_aead_aes256gcm_KEYBYTES];

        // party for which relative_rank = 1 shares key with other 2 parties
        if (relative_rank == 1) {
            // seed = local_key
            // send to the two other parties, who have relative ranks of -1 and -2
            engine.comm0()->sendShares(local, -1);
            engine.comm0()->sendShares(local, -2);
            for (int byte = 0; byte < crypto_aead_aes256gcm_KEYBYTES; byte++) {
                seed[byte] = local_key[byte];
            }
        } else if (relative_rank == 2) {
            // seed = remote_key
            // get from party with relative rank +1
            engine.comm0()->receiveShares(remote, +1);
            for (int byte = 0; byte < crypto_aead_aes256gcm_KEYBYTES; byte++) {
                seed[byte] = (unsigned char)remote[byte];
            }
        } else if (relative_rank == 3) {
            // seed = remote_key
            // get from party with relative rank +2
            engine.comm0()->receiveShares(remote, +2);
            for (int byte = 0; byte < crypto_aead_aes256gcm_KEYBYTES; byte++) {
                seed[byte] = (unsigned char)remote[byte];
            }
        }

        std::vector<unsigned char> seed_vec;
        for (int byte = 0; byte < crypto_aead_aes256gcm_KEYBYTES; byte++) {
            seed_vec.push_back(seed[byte]);
        }

        std::unique_ptr<DeterministicPRGAlgorithm> prg_algorithm =
            std::make_unique<cdough::random::AESPRGAlgorithm>(seed_vec);
        auto commonPRG = std::make_shared<cdough::random::CommonPRG>(std::move(prg_algorithm), rank);
        commonPRGManager->add(commonPRG, relative_rank);
    }
}

/**
 * @brief Setup the Common PRGs for Beaver 2PC
 *
 * @param rank
 * @param commonPRGManager
 */
void setup_2pc_common_prgs(int rank,
                           std::shared_ptr<cdough::random::CommonPRGManager> commonPRGManager) {
    auto common_prg = commonPRGManager->get({0, 1});

    // The commonprg held by both parties is also the _relative_ common prg with
    // its neighbor (used for zero sharing)
    commonPRGManager->add(common_prg, 1);
}

#ifdef MPC_PROTOCOL_BEAVER_TWO

/**
 * @brief Type for arithmetic OLE generators. We can either use quadratic-bandwidth Gilboa, or the
 * CRT-based optimization. Use the flag in debug.h to change between these.
 *
 * @tparam T
 */
template <typename T>
using ole_generator_t = std::conditional_t<USE_GILBOA_CRT, GilboaCRT<T>, GilboaOLE<T>>;

using OLEmodP_t = cdough::random::OLEmodPrimeInterface<uint8_t>;

/**
 * @brief Setup correlation generators for 2PC
 *
 * @tparam T
 * @param rank of this node
 * @param PRGm CommonPRGManager
 * @param comm communicator
 * @param thread thread index
 * @return Typed correlations for type T
 */
template <typename T>
cdough::random::TypedCorrelations_t<T> setup_2pc_correlations(
    int rank, std::shared_ptr<cdough::random::CommonPRGManager> PRGm, cdough::Communicator* comm,
    std::shared_ptr<OLEmodP_t> mod_ptr, int thread) {
    using namespace cdough::random;

#if defined USE_DUMMY_TRIPLES
    auto vg_a = std::make_shared<DummyOLE<T>>(rank, PRGm, comm);
    auto vg_b = std::make_shared<DummyOT<T>>(rank, PRGm, comm);
    auto btg_a = std::make_shared<BeaverTripleGenerator<T, cdough::Encoding::AShared>>(vg_a);
    auto btg_b = std::make_shared<BeaverTripleGenerator<T, cdough::Encoding::BShared>>(vg_b);
    return std::make_tuple(vg_b, vg_a, btg_a, btg_b, nullptr);
#elif defined USE_ZERO_TRIPLES
    auto vg_a = std::make_shared<ZeroOLE<T>>(rank, comm);
    auto vg_b = std::make_shared<ZeroOLE<T>>(rank, comm);
    auto btg_a = std::make_shared<BeaverTripleGenerator<T, cdough::Encoding::AShared>>(vg_a);
    auto btg_b = std::make_shared<BeaverTripleGenerator<T, cdough::Encoding::BShared>>(vg_b);
    return std::make_tuple(vg_b, vg_a, btg_a, btg_b, nullptr);
#else

#if defined(USE_LIBOTE) && defined(USE_SECURE_JOIN)
    std::shared_ptr<cdough::random::OPRF> oprf;
    if constexpr (std::is_same_v<T, __int128_t>) {
        oprf = std::make_shared<cdough::random::OPRF>(rank, thread, comm->host_prefix);
    }
#else
    auto oprf = nullptr;
#endif

    std::shared_ptr<ole_generator_t<T>> vg_a;

    if constexpr (USE_GILBOA_CRT) {
        assert(mod_ptr != nullptr);
        // All Gilboa CRT instantiations use the same underlying small Gilboa
        vg_a = std::make_shared<ole_generator_t<T>>(mod_ptr, PRGm);
    } else {
        // GILBOA_CRT is false, so we're using the regular (full-width) Gilboa
        vg_a = std::make_shared<ole_generator_t<T>>(rank, PRGm, comm, thread);
    }

    // Use real generators - gilboa OLE and silent OT
    auto vg_b = std::make_shared<SilentOT<T>>(rank, PRGm, comm, thread);
    // pooled variants
    auto pooled_ole = make_pooled<ole_generator_t<T>>(vg_a);
    auto pooled_ot = make_pooled<SilentOT<T>>(vg_b);
    auto btg_a =
        std::make_shared<BeaverTripleGenerator<T, cdough::Encoding::AShared>>(pooled_ole, comm);
    auto btg_b =
        std::make_shared<BeaverTripleGenerator<T, cdough::Encoding::BShared>>(pooled_ot, comm);
    return std::make_tuple(vg_b, vg_a, btg_a, btg_b, oprf);
#endif
}
#endif

/**
 * @brief General randomness-generation setup
 *
 * @param num_parties
 * @param rank
 * @param groups
 * @param thread
 * @return auto
 */
template <typename Engine>
auto setup_random_generation(int num_parties, int rank, std::vector<std::set<int>> groups,
                             int thread, Engine& engine) {
    // Ensure the global Boost.Asio io_context is initialised once with the
    // number of worker threads that the runtime was configured with.
    int num_threads = engine.template getArg<int>("threads", "t", 1);
    int num_libote_threads = engine.template getArg<int>("comm-threads", "n", num_threads);
#ifdef USE_LIBOTE
    cdough::libote_io::libOTeContext::getContext(num_libote_threads);
#endif

    // create the CommonPRGManager
    auto commonPRGManager = std::make_shared<cdough::random::CommonPRGManager>(num_parties);

    // add to the CommonPRGManager for each group
    for (std::set<int> group : groups) {
        create_common_prg_among_group(rank, group, commonPRGManager, engine);
    }

    // Add the everyone group
    std::set<int> _everyone;
    for (int i = 0; i < num_parties; i++) {
        _everyone.insert(i);
    }
    create_common_prg_among_group(rank, _everyone, commonPRGManager, engine);

    CorrRegistry_t registry;
    std::shared_ptr<cdough::random::ShardedPermutationGenerator> sharded_perm_gen;

#ifdef MPC_PROTOCOL_SPDZ_2K_NPC
    // Pairwise adjacent common PRGs for n-party SPDZ
    // (Needed for zero sharing generator)
    for (int i = 0; i < num_parties; i++) {
        create_common_prg_among_group(rank, {i, (i + 1) % num_parties}, commonPRGManager, engine);
    }

    auto c_next = commonPRGManager->get({rank, (rank + 1) % num_parties});
    commonPRGManager->add(c_next, +1);
    auto c_prev = commonPRGManager->get({rank, (num_parties + rank - 1) % num_parties});
    commonPRGManager->add(c_prev, -1);

#else  // Beaver / Replicated
    // create the relative CommonPRG objects
    if (num_parties == 2) {
#ifdef MPC_PROTOCOL_BEAVER_TWO
        setup_2pc_common_prgs(rank, commonPRGManager);

        std::shared_ptr<OLEmodP_t> modPptr = nullptr;
        auto comm = engine.workers[thread].getCommunicator();

#ifdef USE_LIBOTE
        if constexpr (USE_GILBOA_CRT) {
            if (thread == 0) {
                single_cout("[cdough] NOTE: Using Gilboa CRT");
            }

            // Make a single GilboaModPrime per thread. We'll pass this into the GilboaCRT for each
            // datatype. Different data types cannot be running at the same time, so this won't
            // cause any contention.
            auto GmP = new cdough::random::GilboaModPrime(rank, commonPRGManager, comm, thread);
            modPptr = std::shared_ptr<OLEmodP_t>(GmP);
        }
#endif

        auto g8 = setup_2pc_correlations<int8_t>(rank, commonPRGManager, comm, modPptr, thread);
        auto g16 = setup_2pc_correlations<int16_t>(rank, commonPRGManager, comm, modPptr, thread);
        auto g32 = setup_2pc_correlations<int32_t>(rank, commonPRGManager, comm, modPptr, thread);
        auto g64 = setup_2pc_correlations<int64_t>(rank, commonPRGManager, comm, modPptr, thread);
        auto g128 =
            setup_2pc_correlations<__int128_t>(rank, commonPRGManager, comm, modPptr, thread);

        registry = std::make_tuple(std::move(g8), std::move(g16), std::move(g32), std::move(g64),
                                   std::move(g128));

        // permutation generator
#if defined USE_DUMMY_TRIPLES || defined USE_ZERO_TRIPLES || !defined USE_SECURE_JOIN
        sharded_perm_gen = std::make_shared<cdough::random::DMDummyGenerator<__int128_t>>(
            rank, thread, commonPRGManager, comm);
#else
        // Use real permutation generator
        sharded_perm_gen =
            std::make_shared<cdough::random::DMPermutationCorrelationGenerator<__int128_t>>(
                rank, thread, commonPRGManager, comm);
#endif

#endif
    } else if (num_parties == 3) {
        setup_3pc_common_prgs(rank, commonPRGManager, engine);
    } else if (num_parties == 4) {
        setup_4pc_common_prgs(rank, commonPRGManager, engine);
    }

    if ((num_parties == 3) || (num_parties == 4)) {
        sharded_perm_gen = std::make_shared<cdough::random::HMShardedPermutationGenerator>(
            rank, commonPRGManager, groups);
    }
#endif  // end ifdef SPDZ

    // create the zero sharing generator
    auto zeroSharingGenerator =
        std::make_shared<cdough::random::ZeroSharingGenerator>(num_parties, commonPRGManager, rank);

    // create the randomness manager to access all generators
    return std::make_unique<cdough::random::RandomnessManager>(commonPRGManager, zeroSharingGenerator,
                                                            std::move(registry), sharded_perm_gen);
}

}  // namespace cdough::service
