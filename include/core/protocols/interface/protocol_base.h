#pragma once

#include <numeric>
#include <set>
#include <vector>

#include "core/communication/communicator.h"

// Temporary workaround until we make protocol classes templated.
#define TYPE_BITS_int8_t 8
#define TYPE_BITS_int16_t 16
#define TYPE_BITS_int32_t 32
#define TYPE_BITS_int64_t 64
#define TYPE_BITS___int128_t 128

#define TYPE_BITS(T) TYPE_BITS_##T
#define GLUE_(a, b) a##b
#define GLUE(a, b) GLUE_(a, b)

// map PROTO_OBJ_NAME(int32_t) => proto_32 for Worker member lookup
// Eventually, this should be something like worker.proto<int32_t>
#define PROTO_OBJ_NAME(T) GLUE(proto_, TYPE_BITS(T))

namespace cdough {

namespace service {
    class Worker;
}

/**
 * @brief Actions for resharing operations.
 */
enum class ReshareAction { Send, Receive };

/**
 * @brief Assignment for resharing operations.
 *
 * Defines whether a party sends or receives shares during resharing,
 * along with the target parties and share indices.
 */
struct ReshareAssignment {
    // whether this party sends (in group) or receives (not in group)
    ReshareAction action;

    // Who to send to, or receive from
    std::vector<int> ranks;

    // if sending, which shares to send
    std::vector<int> shareIdx;
};

/**
 * @brief Base class for all secure multi-party computation protocols.
 */
// The protocol base
class ProtocolBase {
    // party id -> shares held by that party
    std::vector<std::vector<int>> partyShareMap = {};
    // share id -> parties holding that share
    std::vector<std::vector<int>> sharePartyMap = {};

    /**
     * @brief Defines the mapping from rank to the secret shares held by that rank.
     *
     * Index i in the vector is the set of shares held by rank i.
     * For a replication k, party i holds shares [i, i+1, ..., i+k-1].
     * This function should be called during construction.
     */
    void generatePartyShareMappings() {
        partyShareMap.clear();
        for (int i = 0; i < numParties; i++) {
            std::vector<int> party_shares;
            for (int j = 0; j < replicationNumber; j++) {
                party_shares.push_back((i + j) % numParties);
            }
            partyShareMap.push_back(party_shares);
        }
    }

    /**
     * @brief Generate a mapping from shares to parties holding that share.
     *
     * Called during construction to populate the mapping.
     */
    void generateSharePartyMappings() {
        sharePartyMap.clear();
        sharePartyMap.resize(numParties);
        auto psm = getPartyShareMappings();

        for (int p = 0; p < numParties; p++) {
            for (auto s : psm[p]) {
                sharePartyMap[s].push_back(p);
            }
        }

        for (auto v : sharePartyMap) {
            std::sort(v.begin(), v.end());
        }
    }

    /**
     * @brief Generate send assignment for resharing shares to other parties.
     *
     * @param p Party identifier.
     * @param group Set of parties in the resharing group.
     * @return ReshareAssignment for sending shares.
     */
    auto generate_send_assignment(int p, std::set<int> group) {
        ReshareAssignment ra;
        ra.action = ReshareAction::Send;

        auto s2p_map = getSharePartyMappings();
        auto p2s_map = getPartyShareMappings();

        auto my_shares = p2s_map[p];
        std::vector<bool> is_canonical;

        std::vector<int> ranks, share_idx;

        for (int si = 0; si < replicationNumber; si++) {
            auto s = my_shares[si];
            auto s_parties = s2p_map[s];

            bool canonical = true;

            // Check other parties with this share (sorted list)
            for (auto q : s_parties) {
                // Are they in the group?
                if (!group.contains(q)) {
                    continue;
                }

                // Someone else in group has lower rank.
                if (q != p) {
                    canonical = false;
                }

                break;
            }

            is_canonical.push_back(canonical);
        }

        // Check all parties not in the group and see what shares they want
        for (int q = 0; q < numParties; q++) {
            if (group.contains(q)) {
                continue;
            }

            auto q_shares = p2s_map[q];
            for (auto qs : q_shares) {
                bool should_send = false;
                int si = 0;

                // Do I have this share & am I the canonical holder?
                for (; si < replicationNumber; si++) {
                    if (my_shares[si] == qs && is_canonical[si]) {
                        should_send = true;
                        break;
                    }
                }

                if (should_send) {
                    ranks.push_back(q - p);
                    share_idx.push_back(si);
                }
            }
        }

        ra.ranks = ranks;
        ra.shareIdx = share_idx;

        return ra;
    }

    /**
     * @brief Generate receive assignment for resharing shares from other parties.
     *
     * @param p Party identifier.
     * @param group Set of parties in the resharing group.
     * @return ReshareAssignment for receiving shares.
     */
    auto generate_recv_assignment(int p, std::set<int> group) {
        ReshareAssignment ra;
        ra.action = ReshareAction::Receive;

        auto s2p_map = getSharePartyMappings();
        auto p2s_map = getPartyShareMappings();

        auto my_shares = p2s_map[p];

        // *relative* ranks
        std::vector<int> send_parties;

        for (auto s : my_shares) {
            // Receive from the lowest party in the group
            auto s_parties = s2p_map[s];

            for (auto q : s_parties) {
                if (group.contains(q)) {
                    send_parties.push_back(q - p);
                    break;
                }
            }
        }

        ra.ranks = send_parties;
        return ra;
    }

    /**
     * @brief Internal code to check for malicious behavior. Semihonest protocols should not
     * implement. Malicious protocols should implement their checks here.
     *
     * @return true no malicious behavior occurred
     * @return false malicious behavior occurred
     */
    virtual bool malicious_check_internal() { return true; }

    /**
     * @brief Reset the internal malicious-detection state
     *
     */
    virtual void reset_malicious_state() {}

    // This allows us to call reset_malicious_state, which is a private method
    friend class service::Worker;

   protected:
    // The unique id of the party that created the Protocol instance
    PartyID partyID;
    // The total number of computing parties participating in the protocol execution
    const int numParties;
    // The replication factor
    const int replicationNumber;
    // The number of allowed corruptions
    // const int corruptionsNumber;
    // The number of parties required to reconstruct the shares.
    // const int reconstructNumber;

    /**
     * @brief Map from party-groups to resharing assignment.
     *
     * `reshare` will retrieve a list of assignments from this map given
     * the current group, and then each party will execute its respective
     * instructions.
     *
     */
    std::map<std::set<int>, ReshareAssignment> reshareMap = {};

   public:
    size_t send_calls = 0;
    size_t recv_calls = 0;

    /**
     * @brief Constructor for ProtocolBase.
     *
     * @param pID Party identifier for this protocol instance.
     * @param partiesNum Total number of parties in the protocol.
     * @param replicationNum Replication factor for the protocol.
     */
    ProtocolBase(PartyID pID, int partiesNum, int replicationNum)
        : partyID(pID), numParties(partiesNum), replicationNumber(replicationNum) {
        assert(pID < partiesNum);

        generatePartyShareMappings();
        generateSharePartyMappings();

        // Iterate through all groups and generate resharing assignments.
        for (auto g : getGroups()) {
            if (g.contains(partyID)) {
                reshareMap[g] = generate_send_assignment(partyID, g);
            } else {
                reshareMap[g] = generate_recv_assignment(partyID, g);
            }
        }
    };

    virtual ~ProtocolBase() {
        // Final malicious check: this catches any malicious behavior which happened on the _last_
        // open, if one was not performed manually by the user. In test mode, does nothing, but in
        // regular programs, may cause an abort.
        malicious_check();
    }

    /**
     * Generates all combinations of a given size of an input size
     * (recursively).
     * @param set the set of ints to find combinations of.
     * @param partial_set the partially complete combination at any given
     * point in the recursion.
     * @param size the size of the combinations to search for, decremented
     * with each recursive call.
     * @param combinations a vector of combinations to be filled throughout
     * the algorithm.
     */
    static void generateAllCombinations(std::set<int> set, std::set<int> partial_set, int size,
                                        std::vector<std::set<int>>& combinations) {
        // base case
        if (size == 0) {
            combinations.push_back(partial_set);
            return;
        }
        // recursive case
        for (int element : set) {
            std::set<int> new_set(set);
            // remove all elements in new_set that are less than the current
            // element
            for (int other_element : set) {
                if (other_element < element) {
                    new_set.erase(other_element);
                }
            }
            new_set.erase(element);
            // add current element to the partial set and recurse
            std::set<int> new_partial_set(partial_set);
            new_partial_set.insert(element);
            generateAllCombinations(new_set, new_partial_set, size - 1, combinations);
        }
    }

    /**
     * @brief Defines the groups for the protocol.
     *
     * This is a STATIC function to be callable during setup before protocol objects exist.
     *
     * @param num_parties The number of parties in the protocol.
     * @param parties_to_reconstruct Number of parties needed to reconstruct.
     * @param num_adversaries Number of adversarial parties.
     * @return The vector of groups.
     */
    static std::vector<std::set<int>> generateGroups(int num_parties, int parties_to_reconstruct,
                                                     int num_adversaries) {
        if (num_parties == 1) {
            // special handling.
            return {{0}};
        }

        if ((num_parties >= 4) && (num_adversaries == 1)) {
            // the one adversary optimization
            std::vector<std::set<int>> groups;
            for (int i = 0; i < 2; i++) {
                std::set<int> group;
                for (int j = 0; j < parties_to_reconstruct; j++) {
                    group.insert(i * parties_to_reconstruct + j);
                }
                groups.push_back(group);
            }
            return groups;
        }
        int group_size = num_parties - num_adversaries;

        // Don't return singleton groups.
        if (group_size == 1) {
            return {};
        }

        std::set<int> parties;
        for (int i = 0; i < num_parties; i++) {
            parties.insert(i);
        }
        std::set<int> partial_combination;
        std::vector<std::set<int>> groups;
        generateAllCombinations(parties, partial_combination, group_size, groups);

        return groups;
    }

    /**
     * @brief Get the groups for shuffling operations.
     *
     * @return Vector of party groups for shuffling.
     */
    virtual std::vector<std::set<int>> getGroups() const {
        return generateGroups(numParties, 2, 1);
    }

    /**
     * @brief Generate randomness groups (includes the group of everyone).
     *
     * @param n Number of parties.
     * @param n_reconstruct Number of parties needed for reconstruction.
     * @param n_adversary Number of adversarial parties.
     * @return Vector of randomness groups.
     */
    static std::vector<std::set<int>> generateRandomnessGroups(int n, int n_reconstruct,
                                                               int n_adversary) {
        // Get the regular groups
        auto g = generateGroups(n, n_reconstruct, n_adversary);

        // Add the set of everyone
        std::vector<int> _everyone(n);
        std::iota(_everyone.begin(), _everyone.end(), 0);

        g.push_back(std::set<int>(_everyone.begin(), _everyone.end()));

        return g;
    }

    /**
     * @brief Get a mapping from parties to share numbers
     *
     * @return std::vector<std::vector<int>>
     */
    std::vector<std::vector<int>> getPartyShareMappings() const {
        assert(partyShareMap.size() > 0);
        return partyShareMap;
    }

    /**
     * @brief Get a mapping from shares to parties holding that share.
     *
     * @return Vector mapping share IDs to party lists.
     */
    std::vector<std::vector<int>> getSharePartyMappings() const {
        assert(sharePartyMap.size() > 0);
        return sharePartyMap;
    }

    /**
     * @brief Get the party ID for this protocol instance.
     *
     * @return Party identifier.
     */
    int getPartyID() const { return partyID; }

    /**
     * @brief Get the replication number for this protocol.
     *
     * @return Replication factor.
     */
    const int getRepNumber() const { return replicationNumber; }

    /**
     * @brief Get the total number of parties in this protocol.
     *
     * @return Number of parties.
     */
    const int getNumParties() const { return numParties; }

    /**
     * @brief Print statistics for this protocol. The default is to do
     * nothing, but protocols may choose to override this function to
     * print operator counts, network statistics, or other information.
     *
     */
    virtual void print_statistics() {}

    /**
     * @brief Mark statistics. This can be used to take relative
     * measurements of a section of code.
     *
     */
    virtual void mark_statistics() {}

    /**
     * @brief Clear all accumulated statistics (counters, marks, etc.).
     *
     */
    virtual void clear_statistics() {}

    /**
     * @brief Check for malicious behavior. Aborts if malicious behavior occurred.
     *
     * In MAL_TEST_MODE, does not abort, and resets internal state. MAL_TEST_MODE should
     * only be used for explicitly testing malicious behavior, and never on benchmarks or real
     * queries.
     */
    virtual bool malicious_check() {
        auto ok = malicious_check_internal();

#ifndef MAL_TEST_MODE
        // Normal programs will abort on a failed test.
        if (!ok) {
            abort();
        }
#endif

        return ok;
    }
};

}  // namespace cdough