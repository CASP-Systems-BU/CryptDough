#pragma once

#include <math.h>
#include <sodium.h>
#include <stdlib.h>

#include <queue>
#include <vector>

#include "common_prg.h"

namespace cdough::random {

/**
 * @brief Manages a queue of committed seeds.
 *
 * This class implements the commitment protocol for fresh seed generation
 * that prevents malicious parties from pre-computing attack strategies.
 * Seeds are committed in batches to reduce communication overhead.
 */
class CommittedSeedsQueue {
   private:
    // Used vector, for easy refactoring,
    template <typename T>
    using Vector = cdough::Vector<T>;

    // The seed type
    using Seed = Vector<unsigned char>;

    // This is the type used for commitments
    // It is the result of a hash function
    using Commitment = Vector<unsigned char>;

    // Key type for hashing commitments
    using Key = Vector<unsigned char>;

    // PartyID type
    using PartyID = cdough::PartyID;

    // Vector of committed seeds pending opening
    // Each vector belongs to one party.
    // indices are relative with 0 being self
    // This is a flattened vector for easy communication
    std::vector<Commitment> commitments;

    // local seeds, Flattened for efficient storage
    Seed localSeeds;

    // local keys, Flattened for efficient storage
    Key localKeys;

    // Communication and randomness infrastructure
    Communicator& communicator;
    std::shared_ptr<random::CommonPRG> localPRG;

    // Protocol parameters
    PartyID partyID;
    PartyID numParties;
    std::vector<PartyID> toPartyIDs;
    std::vector<PartyID> fromPartyIDs;

    // Configuration constants
    static constexpr int BATCH_SIZE = 1024;  // Number of seeds to commit in one batch
    static constexpr int SEED_BYTES = crypto_aead_aes256gcm_KEYBYTES;
    static constexpr int COMMITMENT_BYTES = crypto_generichash_BYTES;
    static constexpr int HASH_KEY_BYTES = crypto_generichash_KEYBYTES;

    // Current seed index
    // Default value set such that it implies
    // that we need to repopulate the queue before using any seeds.
    size_t currentSeedIndex = BATCH_SIZE;

   public:
    /**
     * @brief Constructor for CommittedSeedsQueue.
     *
     * @param _communicator Reference to the communicator for network operations.
     * @param _localPRG Pointer to local PRG for generating seeds and commitments.
     * @param _partyID The ID of this party.
     * @param _numParties Total number of parties in the protocol.
     * @param _toPartyIDs Vector of party IDs for outgoing communication.
     * @param _fromPartyIDs Vector of party IDs for incoming communication.
     */
    CommittedSeedsQueue(Communicator& _communicator, std::shared_ptr<random::CommonPRG> _localPRG,
                        PartyID _partyID, PartyID _numParties,
                        const std::vector<PartyID>& _toPartyIDs,
                        const std::vector<PartyID>& _fromPartyIDs)
        : communicator(_communicator),
          localPRG(_localPRG),
          partyID(_partyID),
          numParties(_numParties),
          toPartyIDs(_toPartyIDs),
          fromPartyIDs(_fromPartyIDs),
          localSeeds(BATCH_SIZE * SEED_BYTES),
          localKeys(BATCH_SIZE * HASH_KEY_BYTES) {}

    /**
     * @brief Repopulate the queue with new committed seeds.
     *
     * Generates a batch of fresh seeds, commits to them, and exchanges
     * commitments with all other parties in a single communication round.
     */
    void repopulateQueue() {
        // make sure the queue does not have any more seeds.
        if (currentSeedIndex < BATCH_SIZE) {
            return;
        }

        //////////////////////////////////////////////////
        // Step 1: Generate local seeds and commitments
        //////////////////////////////////////////////////
        currentSeedIndex = 0;

        // clear previous commitments and vector for self
        commitments.clear();
        commitments.push_back(Commitment(BATCH_SIZE * COMMITMENT_BYTES));

        // generate new seeds shares and new keys for commitment hashing
        localPRG->getNext(localSeeds);
        localPRG->getNext(localKeys);

        for (int i = 0; i < BATCH_SIZE; ++i) {
            // Create commitment using hash
            crypto_generichash((uint8_t*)&commitments[0][i * COMMITMENT_BYTES], COMMITMENT_BYTES,
                               (const uint8_t*)(&localSeeds[i * SEED_BYTES]), SEED_BYTES,
                               (const uint8_t*)(&localKeys[i * HASH_KEY_BYTES]), HASH_KEY_BYTES);
        }

        //////////////////////////////////////////////////
        // Step 2: Exchange commitments with all parties
        //////////////////////////////////////////////////
        std::vector<Vector<unsigned char>> commitmentsLocal;
        std::vector<Vector<unsigned char>> commitmentsRemote;

        for (int i = 1; i < numParties; ++i) {
            commitmentsLocal.push_back(commitments[0]);
            commitmentsRemote.push_back(Vector<unsigned char>(BATCH_SIZE * COMMITMENT_BYTES));
        }
        communicator.exchangeShares(commitmentsLocal, commitmentsRemote, toPartyIDs, fromPartyIDs);

        //////////////////////////////////////////////////
        // Step 3: Store all commitments from other parties
        //////////////////////////////////////////////////
        for (int i = 1; i < numParties; ++i) {
            commitments.push_back(commitmentsRemote[i - 1]);
        }
    }

    std::shared_ptr<CommonPRG> getNextPRG() {
        // If we've used all seeds in the current batch,
        // and we do not have already committed seeds for the next batch,
        // we can not open next seeds.
        // Note: if we reach a check without pre-committmed seeds, maybe
        // some party will commit malicious seeds.
        if (currentSeedIndex >= BATCH_SIZE) {
            std::cerr << "Party " << partyID
                      << ": No more pre-committed seeds available. Needs to abort." << std::endl;
            abort();
        }

        ////////////////////////////////////////////////////////
        // Step 1: Extract seed and key from flattened vectors
        ////////////////////////////////////////////////////////
        // we extract the current seed and key for this party from the localSeeds vector
        Seed currentSeedKey(SEED_BYTES + HASH_KEY_BYTES);
        for (int i = 0; i < SEED_BYTES; ++i) {
            currentSeedKey[i] = localSeeds[currentSeedIndex * SEED_BYTES + i];
        }
        for (int i = 0; i < HASH_KEY_BYTES; ++i) {
            currentSeedKey[SEED_BYTES + i] = localKeys[currentSeedIndex * HASH_KEY_BYTES + i];
        }

        ////////////////////////////////////////////////////////
        // Step 2: Exchange seeds and keys with all parties
        ////////////////////////////////////////////////////////
        std::vector<Seed> sentSeeds;
        std::vector<Seed> receivedSeeds;
        for (int i = 1; i < numParties; ++i) {
            sentSeeds.push_back(currentSeedKey);
            receivedSeeds.push_back(Seed(SEED_BYTES + HASH_KEY_BYTES));
        }
        communicator.exchangeShares(sentSeeds, receivedSeeds, toPartyIDs, fromPartyIDs);

        ////////////////////////////////////////////////////////
        // Step 3: Verify commitments for all parties
        ////////////////////////////////////////////////////////
        for (int i = 1; i < numParties; ++i) {
            Commitment expectedCommitment(COMMITMENT_BYTES);
            crypto_generichash((uint8_t*)&expectedCommitment[0], COMMITMENT_BYTES,
                               (const uint8_t*)(&receivedSeeds[i - 1][0]), SEED_BYTES,
                               (const uint8_t*)(&receivedSeeds[i - 1][SEED_BYTES]), HASH_KEY_BYTES);

            // iterate over the corresponding commitment for this party and check that it matches
            // the expected commitment
            for (int byte = 0; byte < COMMITMENT_BYTES; ++byte) {
                if (expectedCommitment[byte] !=
                    commitments[i][currentSeedIndex * COMMITMENT_BYTES + byte]) {
                    std::cerr << "Party " << partyID
                              << ": Commitment verification failed for seed from party "
                              << (partyID + i) % numParties << std::endl;
                    abort();
                }
            }
        }

        ////////////////////////////////////////////////////////
        // Step 4: Reconstruct the seed
        ////////////////////////////////////////////////////////
        // After verification, we can reconstruct the seed
        // by doing XOR of all received seeds (including our own)
        std::vector<unsigned char> seed_vec(SEED_BYTES);
        for (int byte = 0; byte < SEED_BYTES; byte++) {
            unsigned char byte_val = currentSeedKey[byte];
            for (int i = 0; i < numParties - 1; ++i) {
                byte_val ^= receivedSeeds[i][byte];
            }
            seed_vec[byte] = byte_val;
        }

        ////////////////////////////////////////////////////////
        // Step 5: Increment seed index and return new CommonPRG
        ////////////////////////////////////////////////////////
        // Increment the index for the next call
        currentSeedIndex++;

        // Return a new CommonPRG initialized with the next seed
        std::unique_ptr<DeterministicPRGAlgorithm> prg_algorithm =
            std::make_unique<AESPRGAlgorithm>(seed_vec);
        return std::make_shared<CommonPRG>(std::move(prg_algorithm), partyID);
    }
};

}  // namespace cdough::random