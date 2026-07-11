#pragma once

#include "core/protocols/interface/protocol.h"

namespace cdough {

/**
 * @brief Base factory class for creating protocol instances using CRTP.
 *
 * @tparam InnerFactory The derived factory class.
 */
template <typename InnerFactory>
class ProtocolFactory {
   public:
    virtual ~ProtocolFactory() = default;

    /**
     * @brief Create a protocol instance for the given data type.
     *
     * @tparam T The data type for the protocol.
     * @param thread_id
     * @param _communicator Pointer to the communicator.
     * @param _randomnessManager Pointer to the randomness manager.
     * @return Unique pointer to the created protocol instance.
     */
    template <typename T>
    std::unique_ptr<ProtocolBase> create(int thread_id, Communicator* _communicator,
                                         random::RandomnessManager* _randomnessManager) {
        // Delegate implementation to the derived class
        return static_cast<InnerFactory&>(*this).template create<T>(thread_id, _communicator,
                                                                    _randomnessManager);
    }
};

/**
 * @brief Default protocol factory implementation.
 *
 * @tparam Protocol The protocol class template.
 * @tparam S Share type template.
 * @tparam V Vector type template.
 * @tparam E Encoding vector type template.
 */
template <template <typename, typename, typename, typename> class Protocol,
          template <typename> class S, template <typename> class V, template <typename> class E>
class DefaultProtocolFactory : public ProtocolFactory<DefaultProtocolFactory<Protocol, S, V, E>> {
   public:
    template <typename T>
    using ProtocolInstance = Protocol<T, S<T>, V<T>, E<T>>;

    template <typename T>
    using AInnerContainer = E<T>;  // The inner container type for A-shared vectors

    template <typename T>
    using BInnerContainer = E<T>;  // The inner container type for B-shared vectors

   public:
    /**
     * @brief Constructor for DefaultProtocolFactory.
     *
     * @param partyID The party identifier.
     * @param partiesNumber The total number of parties.
     */
    DefaultProtocolFactory(int partyID, int partiesNumber)
        : partyID_(partyID), partiesNumber_(partiesNumber) {
        if (partiesNumber != DefaultNumParties) {
            std::cerr << "ERROR: Compiled for " << DefaultNumParties << " parties but run with "
                      << partiesNumber << "\n";
            abort();
        }
    }

    /**
     * @brief Create a protocol instance for the given data type.
     *
     * @tparam T The data type for the protocol.
     * @param thread_id
     * @param communicator Pointer to the communicator.
     * @param randomnessManager Pointer to the randomness manager.
     * @return Unique pointer to the created protocol instance.
     */
    template <typename T>
    std::unique_ptr<ProtocolBase> create(int thread_id, Communicator* communicator,
                                         random::RandomnessManager* randomnessManager) {
        // Create and return a new instance of the protocol
        return std::make_unique<ProtocolInstance<T>>(partyID_, thread_id, communicator,
                                                     randomnessManager);
    }

    static const int DefaultNumParties = ProtocolInstance<int>::parties_num;

   private:
    const int partyID_;
    const int partiesNumber_;
};

}  // namespace cdough