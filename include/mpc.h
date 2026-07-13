#pragma once

// Core - Containers
#include "backend/common/runtime.h"
#include "backend/common/util.h"
#include "core/containers/e_vector.h"
#include "core/containers/matrix/matrix.h"
#include "core/containers/vector.h"

// Core - Communication
#include "core/communication/communicator.h"
#include "core/communication/communicator_factory.h"
#include "core/communication/mpi_communicator.h"
#include "core/communication/no_copy_communicator/no_copy_communicator.h"
#include "core/communication/no_copy_communicator/no_copy_communicator_factory.h"
#include "core/communication/null_communicator.h"

// Core - Random
#include "core/random/correlation/dummy_auth_random_generator.h"
#include "core/random/correlation/dummy_auth_triple_generator.h"
#include "core/random/manager.h"
#include "core/random/permutations/hm_sharded_permutation_generator.h"
#include "core/random/permutations/zero_permutation_generator.h"
#include "core/random/prg/committed_seeds_queue.h"
#include "core/random/prg/zero_rg.h"

#ifdef MPC_PROTOCOL_BEAVER_TWO
#include "core/random/correlation/beaver_triple_generator.h"
#include "core/random/correlation/dummy_ole.h"
#include "core/random/correlation/gilboa_crt.h"
#include "core/random/correlation/gilboa_mod_p.h"
#include "core/random/correlation/gilboa_ole.h"
#include "core/random/correlation/libsecjoin.h"
#include "core/random/correlation/registry.h"
#include "core/random/correlation/silent_ot.h"
#include "core/random/correlation/zero_ole.h"
#endif

#include "core/random/pooled/pooled_generator.h"
#include "core/random/prg/common_prg.h"

// Core - Protocols
#include "core/protocols/dummy_0pc.h"
#include "core/protocols/interface/interface.h"
#include "core/protocols/plaintext_1pc.h"
#include "core/protocols/replicated_3pc.h"
#include "core/protocols/spdz2k_npc.h"
#include "core/protocols/eda_bits.h"

#ifdef USE_DALSKOV_FANTASTIC_FOUR
#include "core/protocols/dalskov_4pc.h"
#else
#include "core/protocols/custom_4pc.h"
#endif

#ifdef MPC_PROTOCOL_BEAVER_TWO
#include "core/protocols/beaver_2pc.h"
#endif

// Core - Vectors
#include "core/containers/a_shared_vector.h"
#include "core/containers/b_shared_vector.h"
#include "core/containers/encoded_vector.h"
#include "core/containers/permutation.h"
#include "core/containers/shared_vector.h"

// Tabular
#include "core/containers/tabular/encoded_table.h"
#include "core/containers/tabular/shared_column.h"

// Operators
#include "core/operators/operators.h"

// Core - Math
#include "core/math/math.h"

// Machine Learning
#include "core/operators/machine_learning.h"

// Macros Section
#define init_mpc_types(_Vector_, _ReplicatedShare_, _EVector_, _Replication_) \
    constexpr int Replication = _Replication_;                                \
                                                                              \
    template <typename T>                                                     \
    using ReplicatedShare = _ReplicatedShare_<T>;                             \
                                                                              \
    template <typename T>                                                     \
    using Vector = _Vector_<T>;                                               \
                                                                              \
    template <typename T>                                                     \
    using DataTable = std::vector<_Vector_<T>>;                               \
                                                                              \
    template <typename T>                                                     \
    using EVector = _EVector_<T, _Replication_>;

#define init_mpc_system(_Communicator_, _ProtocolFactory_)                                    \
    typedef _Communicator_ Communicator;                                                      \
    typedef _ProtocolFactory_<ReplicatedShare, Vector, EVector> ProtocolFactory;              \
    typedef ProtocolFactory::template ProtocolInstance<int8_t> Protocol_8;                    \
    typedef ProtocolFactory::template ProtocolInstance<int16_t> Protocol_16;                  \
    typedef ProtocolFactory::template ProtocolInstance<int32_t> Protocol_32;                  \
    typedef ProtocolFactory::template ProtocolInstance<int64_t> Protocol_64;                  \
    typedef ProtocolFactory::template ProtocolInstance<__int128_t> Protocol_128;              \
                                                                                              \
    using Engine = service::RunTime<ProtocolFactory>;                                         \
    using EngineRef = Engine&;                                                                \
                                                                                              \
    template <typename T>                                                                     \
    using ASharedVector = Engine::ASVector<T>;                                                \
    template <typename T>                                                                     \
    using BSharedVector = Engine::BSVector<T>;                                                \
    using EncodedColumn = Engine::EColumn;                                                    \
    template <typename T>                                                                     \
    using SharedColumn = Engine::SColumn<T>;                                                  \
    template <typename T>                                                                     \
    using PlainMatrix = Engine::PlainMatrix<T>;                                               \
    template <typename T>                                                                     \
    using SecureMatrix = Engine::SecureMatrix<T>;                                             \
                                                                                              \
    template <typename T>                                                                     \
    using EncodedTable =                                                                      \
        cdough::relational::EncodedTable<T, SharedColumn<T>, ASharedVector<T>, BSharedVector<T>, \
                                      cdough::EncodedVector, DataTable<T>>;                      \
    namespace {                                                                               \
    static std::vector<std::shared_ptr<Engine>> engine_ptr;                                   \
    }
