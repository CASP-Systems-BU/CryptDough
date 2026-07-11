#include <iostream>

#include "cdough.h"
// explicit include for testing functionality
#include "core/random/permutations/permutation_manager.h"

using namespace cdough::debug;
using namespace cdough::service;
using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

using Group = std::set<int>;
using LocalPermutation = std::vector<int>;

// **************************************** //
//          Test Shuffle GenPerm            //
// **************************************** //
// this test checks that the output is actually a sorting permutation of a single bit
template <typename Engine>
void test_shuffle_gen_perm(int test_size, Engine& engine) {
    auto rank = engine.getPartyID();

    // generate a permutation for each group
    for (std::set<int> group : engine.getGroups()) {
        if (!group.contains(rank)) continue;

        auto common_prg = engine.rand0()->commonPRGManager->get(group);

        // generate the permutation
        std::vector<int> permutation = cdough::operators::gen_perm(test_size, common_prg);

        int lowestRank = *group.begin();
        if (rank == lowestRank) {
            // lowest rank receives every other party's vector
            for (int otherRank : group) {
                if (rank == otherRank) continue;
                int relative_rank = otherRank - rank;
                Vector<int> remote(test_size);
                engine.comm0()->exchangeShares(permutation, remote, relative_rank, relative_rank);

                // check correctness
                for (int j = 0; j < test_size; j++) {
                    assert(permutation[j] == remote[j]);
                }
            }
        } else {
            // just exchange with lowest rank, check equality
            int relative_rank = lowestRank - rank;
            Vector<int> remote(test_size);
            engine.comm0()->exchangeShares(permutation, remote, relative_rank, relative_rank);

            // check correctness
            for (int j = 0; j < test_size; j++) {
                assert(permutation[j] == remote[j]);
            }
        }
    }
}

// **************************************** //
//      Test Shuffle Local ApplyPerm        //
// **************************************** //
// this test checks that the local permutation passed is correctly applied
template <typename Engine>
void test_shuffle_local_apply_perm(int test_size, Engine& engine) {
    auto rank = engine.getPartyID();

    // generate a test vector
    cdough::Vector<int> x(test_size);
    for (int i = 0; i < test_size; i++) {
        x[i] = i;
    }

    // generate and apply a permutation for each group
    for (std::set<int> group : engine.getGroups()) {
        BSharedVector<int> b = engine.secret_share_b(x, 0);
        BSharedVector<int> b_inv = engine.secret_share_b(x, 0);
        std::vector<int> permutation;
        if (group.contains(rank)) {
            auto common_prg = engine.rand0()->commonPRGManager->get(group);

            // generate the permutation
            permutation = cdough::operators::gen_perm(test_size, common_prg);

            // apply the permutation
            cdough::operators::local_apply_perm(b, permutation);

            // apply the inverse permutation
            cdough::operators::local_apply_perm(b_inv, permutation);
            cdough::operators::local_apply_inverse_perm(b_inv, permutation);

            b_inv.vector.materialize_inplace();
            b.vector.materialize_inplace();
        }
        // reshare
        engine.reshare(b.vector, group, true);
        engine.reshare(b_inv.vector, group, true);

        cdough::Vector<int> y_b = b.open();
        cdough::Vector<int> y_b_inv = b_inv.open();

        if (group.contains(rank)) {
            // check that the permutation has been applied correctly
            for (int i = 0; i < test_size; i++) {
                assert(y_b[permutation[i]] == i);
                assert(y_b_inv[i] == i);
            }
        }
    }
}

// **************************************** //
//    Test Oblivious Apply Sharded Perm     //
// **************************************** //
template <typename Engine>
void test_shuffle_oblivious_apply_sharded_perm(int test_size, Engine& engine) {
    auto rank = engine.getPartyID();

    // generate a test vector
    cdough::Vector<int> x(test_size);
    for (int i = 0; i < test_size; i++) {
        x[i] = i;
    }
    ASharedVector<int> a = engine.secret_share_a(x, 0);
    BSharedVector<int> b = engine.secret_share_b(x, 0);

    auto groups = engine.getGroups();

    std::shared_ptr<cdough::random::ShardedPermutation> perm_a =
        cdough::random::PermutationManager::get()->getNext<int>(test_size, engine,
                                                             cdough::Encoding::AShared);
    std::shared_ptr<cdough::random::ShardedPermutation> perm_b =
        cdough::random::PermutationManager::get()->getNext<int>(test_size, engine,
                                                             cdough::Encoding::BShared);

    // apply the permutation
    oblivious_apply_sharded_perm(a, perm_a);
    oblivious_apply_sharded_perm(b, perm_b);
    // check that the vector was actually permuted by checking that at least 90% of the values are
    // different
    auto permuted_a = a.open();
    auto permuted_b = b.open();
    int count_identical_a = 0;
    int count_identical_b = 0;
    int threshold_identical = test_size / 10;
    for (int i = 0; i < test_size; i++) {
        if (permuted_a[i] == i) {
            count_identical_a++;
        }
        if (permuted_b[i] == i) {
            count_identical_b++;
        }
    }
    assert(count_identical_a <= threshold_identical);
    assert(count_identical_b <= threshold_identical);

    auto opened_a = a.open();
    auto opened_b = b.open();
    for (int i = 0; i < test_size; i++) {
        assert(opened_a[i] < test_size);
        assert(opened_b[i] < test_size);
    }

    // apply the inverse permutation
    oblivious_apply_inverse_sharded_perm(a, perm_a);
    oblivious_apply_inverse_sharded_perm(b, perm_b);
    // check that the result is the original vector
    auto unpermuted_a = a.open();
    auto unpermuted_b = b.open();
    for (int i = 0; i < test_size; i++) {
        assert(unpermuted_a[i] == i);
        assert(unpermuted_b[i] == i);
    }
}

// **************************************** //
//  Test Oblivious Apply Elementwise Perm   //
// **************************************** //
template <typename Engine>
void test_oblivious_apply_elementwise_perm(int test_size, Engine& engine) {
    // generate a test vector
    cdough::Vector<int> x(test_size);
    for (int i = 0; i < test_size; i++) {
        x[i] = i;
    }

    // apply binary to arithmetic
    ASharedVector<int> a1 = engine.secret_share_a(x, 0);
    ASharedVector<int> a2 = engine.secret_share_a(x, 0);
    cdough::ElementwisePermutation<cdough::EVector<int, a1.vector.replicationNumber>, Engine> perm1(
        test_size, cdough::Encoding::BShared, engine);
    perm1.shuffle();

    // check correctness against local
    Vector<int> local_perm1 = perm1.open();
    for (int i = 0; i < test_size; i++) {
        assert(local_perm1[i] < test_size);
    }

    cdough::operators::oblivious_apply_elementwise_perm(a1, perm1);
    cdough::operators::local_apply_perm(a2, local_perm1);

    auto a1_opened = a1.open();
    auto a2_opened = a2.open();

    for (int i = 0; i < a1_opened.size(); i++) {
        assert(a1_opened[i] == a2_opened[i]);
    }

    // apply arithmetic to binary
    BSharedVector<int> b1 = engine.secret_share_b(x, 0);
    BSharedVector<int> b2 = engine.secret_share_b(x, 0);
    cdough::ElementwisePermutation<cdough::EVector<int, b1.vector.replicationNumber>, Engine> perm2(
        test_size, cdough::Encoding::AShared, engine);
    perm2.shuffle();

    // check correctness against local
    Vector<int> local_perm2 = perm2.open();

    cdough::operators::oblivious_apply_elementwise_perm(b1, perm2);
    cdough::operators::local_apply_perm(b2, local_perm2);

    auto b1_opened = b1.open();
    auto b2_opened = b2.open();

    for (int i = 0; i < b1_opened.size(); i++) {
        assert(b1_opened[i] == b2_opened[i]);
    }
}

// **************************************** //
//           Test Reverse Perm              //
// **************************************** //
template <typename Engine>
void test_reverse_elementwise_permutation(int test_size, Engine& engine) {
    // generate a test vector
    cdough::Vector<int> x(test_size);
    for (int i = 0; i < test_size; i++) {
        x[i] = i;
    }
    ASharedVector<int> a = engine.secret_share_a(x, 0);

    // begin test
    cdough::ElementwisePermutation<cdough::EVector<int, a.vector.replicationNumber>, Engine> e(test_size,
                                                                                         engine);

    e.shuffle();
    auto opened_before = e.open();

    e.reverse();
    auto opened_after = e.open();

    for (int i = 0; i < test_size; i++) {
        assert((opened_before[i] + opened_after[i]) == (test_size - 1));
    }
}

// **************************************** //
//    Test Elementwise Permutation B2A      //
// **************************************** //
template <typename Engine>
void test_elementwise_permutation_b2a_conversion(int test_size, Engine& engine) {
    // generate a test vector
    cdough::Vector<int> x(test_size);
    for (int i = 0; i < test_size; i++) {
        x[i] = i;
    }
    BSharedVector<int> b = engine.secret_share_b(x, 0);

    cdough::ElementwisePermutation<cdough::EVector<int, b.vector.replicationNumber>, Engine> perm(
        test_size, cdough::Encoding::BShared, engine);

    perm.shuffle();

    auto opened_b = perm.open();

    perm.b2a();

    auto opened_a = perm.open();

    for (int i = 0; i < test_size; i++) {
        assert(opened_a[i] == opened_b[i]);
    }
}

// **************************************** //
//   Test Invert Elementwise Permutation    //
// **************************************** //
template <typename Engine>
void test_invert_elementwise_permutation(int test_size, Engine& engine) {
    // generate a test vector
    cdough::Vector<int> x(test_size);
    for (int i = 0; i < test_size; i++) {
        x[i] = i;
    }
    BSharedVector<int> b = engine.secret_share_b(x, 0);

    cdough::ElementwisePermutation<cdough::EVector<int, b.vector.replicationNumber>, Engine> perm(
        test_size, cdough::Encoding::BShared, engine);

    perm.shuffle();

    // apply the permutation to the vector
    cdough::operators::oblivious_apply_elementwise_perm(b, perm);

    // get the inverse permutation and apply it to the vector
    perm.invert();
    cdough::operators::oblivious_apply_elementwise_perm(b, perm);

    // check that the permutation and its inverse cancel out to the identity
    auto opened = b.open();
    for (int i = 0; i < test_size; i++) {
        assert(opened[i] == i);
    }
}

// **************************************** //
//             Test Resharing               //
// **************************************** //
template <typename Engine>
void test_resharing(int test_size, Engine& engine) {
    auto pID = engine.getPartyID();

    // generate a test vector
    cdough::Vector<int> x(test_size);
    for (int i = 0; i < test_size; i++) {
        x[i] = i;
    }
    ASharedVector<int> a = engine.secret_share_a(x, 0);
    BSharedVector<int> b = engine.secret_share_b(x, 0);

    cdough::Vector<int> old_a(test_size);
    cdough::Vector<int> old_b(test_size);
    old_a = a.vector(0);
    old_b = b.vector(0);

    for (std::set<int> group : engine.getGroups()) {
        engine.reshare(a.vector, group, false);
        engine.reshare(b.vector, group, true);
    }

    cdough::Vector<int> new_a(test_size);
    cdough::Vector<int> new_b(test_size);
    new_a = a.vector(0);
    new_b = b.vector(0);

#ifndef MPC_PROTOCOL_PLAINTEXT_ONE
    // check that randomization actually occurred
    assert(!old_a.same_as(new_a, false));
    assert(!old_b.same_as(new_b, false));
#endif

    // check that the result still opens to the correct values
    auto result_a = a.open();
    auto result_b = b.open();

    assert(result_a.same_as(x));
    assert(result_b.same_as(x));
}

// **************************************** //
//        Test Shuffle Correctness          //
// **************************************** //
template <typename T, typename Engine>
void test_shuffle_correctness(int test_size, Engine& engine) {
    // generate a test vector
    cdough::Vector<T> x(test_size);
    for (int i = 0; i < test_size; i++) {
        x[i] = i;
    }
    ASharedVector<T> a = engine.secret_share_a(x, 0);
    BSharedVector<T> b = engine.secret_share_b(x, 0);

    a.shuffle();
    b.shuffle();

    cdough::Vector<T> opened_a = a.open();
    cdough::Vector<T> opened_b = b.open();

    // check that the permutations are shared between parties correctly
    // this is fine for any number of parties because it just makes sure everybody agrees with their
    // successor and predecessor
    //      and if everybody does, then all parties must be in agreement
    cdough::Vector<T> shared_perm_prev_a(test_size);
    cdough::Vector<T> shared_perm_next_a(test_size);
    engine.comm0()->exchangeShares(opened_a, shared_perm_next_a, -1, +1);
    engine.comm0()->exchangeShares(opened_a, shared_perm_prev_a, +1, -1);
    cdough::Vector<T> shared_perm_prev_b(test_size);
    cdough::Vector<T> shared_perm_next_b(test_size);
    engine.comm0()->exchangeShares(opened_b, shared_perm_next_b, -1, +1);
    engine.comm0()->exchangeShares(opened_b, shared_perm_prev_b, +1, -1);

    for (int i = 0; i < test_size; i++) {
        assert(opened_a[i] == shared_perm_prev_a[i]);
        assert(opened_a[i] == shared_perm_next_a[i]);
        assert(opened_b[i] == shared_perm_prev_b[i]);
        assert(opened_b[i] == shared_perm_next_b[i]);
    }

    // check that it is still a permutation of [n] after shuffling
    std::vector<T> opened_vec_a(test_size);
    std::vector<T> opened_vec_b(test_size);
    for (int i = 0; i < test_size; i++) {
        opened_vec_a[i] = opened_a[i];
        opened_vec_b[i] = opened_b[i];
    }
    std::sort(opened_vec_a.begin(), opened_vec_a.end());
    std::sort(opened_vec_b.begin(), opened_vec_b.end());
    for (int i = 0; i < test_size; i++) {
        assert(opened_vec_a[i] == i);
        assert(opened_vec_b[i] == i);
    }
}

// **************************************** //
//      Test Permutation Composition        //
// **************************************** //
template <typename Engine>
void test_permutation_composition(int test_size, Engine& engine) {
    auto pID = engine.getPartyID();

    // generate a test vector
    cdough::Vector<int> x(test_size);
    for (int i = 0; i < test_size; i++) {
        x[i] = i;
    }

    BSharedVector<int> v1 = engine.secret_share_b(x, 0);
    BSharedVector<int> v2 = engine.secret_share_b(x, 0);
    BSharedVector<int> v3 = engine.secret_share_b(x, 0);

    // generate the permutation
    cdough::ElementwisePermutation<cdough::EVector<int, v1.vector.replicationNumber>, Engine> sigma(
        test_size, cdough::Encoding::BShared, engine);
    cdough::ElementwisePermutation<cdough::EVector<int, v1.vector.replicationNumber>, Engine> rho(
        test_size, cdough::Encoding::BShared, engine);

    sigma.shuffle();
    rho.shuffle();

    cdough::Vector<int> sigma_opened = sigma.open();
    cdough::Vector<int> rho_opened = rho.open();

    // apply the permutations sequentially locally
    cdough::operators::local_apply_perm(v1, sigma_opened);
    cdough::operators::local_apply_perm(v1, rho_opened);

    // apply the permutations sequentially obliviously
    cdough::operators::oblivious_apply_elementwise_perm(v2, sigma);
    cdough::operators::oblivious_apply_elementwise_perm(v2, rho);

    // check equivalence of local and oblivious sequential composition
    auto v1_opened = v1.open();
    auto v2_opened = v2.open();
    for (int i = 0; i < v1.size(); i++) {
        assert(v1_opened[i] == v2_opened[i]);
    }

    auto composition = cdough::operators::compose_permutations(sigma, rho);
    cdough::operators::oblivious_apply_elementwise_perm(v3, composition);

    auto v3_opened = v3.open();
    for (int i = 0; i < v3.size(); i++) {
        assert(v1_opened[i] == v3_opened[i]);
    }
}

// **************************************** //
//           Test Table Shuffle             //
// **************************************** //
template <typename Engine>
void test_table_shuffle(int num_rows, int num_columns, Engine& engine) {
    // generate a table
    std::vector<cdough::Vector<int>> table_data;
    std::vector<std::string> schema;
    for (int i = 0; i < num_columns; i++) {
        table_data.push_back(cdough::Vector<int>(num_rows));
        // make even columns binary and odd columns arithmetic
        if (i % 2 == 0) {
            schema.push_back("[" + std::to_string(i) + "]");
        } else {
            schema.push_back(std::to_string(i));
        }
        for (int j = 0; j < num_rows; j++) {
            table_data[i][j] = j;
        }
    }
    EncodedTable<int> table = engine.secret_share_table(table_data, schema);

    table.shuffle();

    // open
    std::vector<cdough::Vector<int>> opened = table.open();
    // make sure all columns have the same value for each row
    for (int i = 0; i < num_rows; i++) {
        int value = opened[0][i];
        for (int j = 1; j < num_columns; j++) {
            assert(opened[j][i] == value);
        }
    }
}

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();

    int DEFAULT_TEST_SIZE = 100;

#if !defined(MPC_PROTOCOL_BEAVER_TWO) && !defined(MPC_PROTOCOL_SPDZ_2K_NPC)
    DEFAULT_TEST_SIZE = 10000;

    test_resharing(DEFAULT_TEST_SIZE, engine);
    single_cout("Resharing...OK");
#endif

#ifndef MPC_PROTOCOL_SPDZ_2K_NPC
    // test shuffle permutation generation correctness
    test_shuffle_gen_perm(100, engine);   // non-power of 2
    test_shuffle_gen_perm(1024, engine);  // power of 2
    single_cout("Shuffle Permutation Generation...OK");

    test_shuffle_local_apply_perm(DEFAULT_TEST_SIZE, engine);
    single_cout("Local Permutation Application...OK");

    test_shuffle_oblivious_apply_sharded_perm(DEFAULT_TEST_SIZE, engine);
    single_cout("Oblivious Sharded Permutation Application...OK");

    test_oblivious_apply_elementwise_perm(DEFAULT_TEST_SIZE, engine);
    single_cout("Oblivious Elementwise Permutation Application...OK");

    test_reverse_elementwise_permutation(DEFAULT_TEST_SIZE, engine);
    single_cout("Reverse Elementwise Permutation...OK");

    test_elementwise_permutation_b2a_conversion(DEFAULT_TEST_SIZE, engine);
    single_cout("Convert Elementwise Permutation B2A...OK");

    test_invert_elementwise_permutation(DEFAULT_TEST_SIZE, engine);
    single_cout("Invert Elementwise Permutation...OK");

    test_shuffle_correctness<int>(DEFAULT_TEST_SIZE, engine);
    test_shuffle_correctness<int64_t>(DEFAULT_TEST_SIZE, engine);
    single_cout("Shuffle Correctness...OK");

    test_permutation_composition(DEFAULT_TEST_SIZE, engine);
    single_cout("Permutation Composition...OK");

    test_table_shuffle(DEFAULT_TEST_SIZE, 4, engine);
    test_table_shuffle(1024, 15, engine);
    single_cout("Table Shuffle...OK");
#else
    single_cout("MPC_PROTOCOL_SPDZ_2K_NPC is defined. Skipping test_shuffle.");
#endif

    return 0;
}