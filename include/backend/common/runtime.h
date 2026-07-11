#pragma once

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <mutex>
#include <queue>
#include <thread>

#include "core/containers/matrix/matrix.h"
#include "cryptoTools/Common/CLP.h"
#include "profiling/thread_profiling.h"
#include "setting.h"
#include "task.h"
#include "worker.h"

#ifdef MPC_USE_MPI_COMMUNICATOR
#include "mpi.h"
#endif

using namespace cdough::instrumentation;

namespace cdough::random {
class ShardedPermutation;
class OPRF;
}  // namespace cdough::random

/**
 * @brief Helper macro to get the current execution's inner class of type `T`
 *  A container is what protocols operate on, vector, multi-vector, matrix.
 *  They are wrapped inside AsharedVector and BSharedVector classes.
 *
 */
#define AInnerContainerClass(T) AInnerContainer<T>
#define BInnerContainerClass(T) BInnerContainer<T>
#define GenericInnerContainerClass(T) GenericInnerContainer<T>

/**
 * @brief Macro to generate evaluator for `reshare`
 *
 */
#define define_reshare(S)                                                 \
    template <typename... T>                                              \
    void reshare(GenericInnerContainerClass(S) & x, const T&... args) {   \
        eval_protocol_reshare<RepProto<S>, &Worker::PROTO_OBJ_NAME(S),    \
                              GenericInnerContainerClass(S)>(x, args...); \
    }

/**
 * @brief Macro to generate evaluator for single-argument functions which allocate their own output
 * storage, like `secret_share`
 *
 */
#define define_1_alloc(S, F, InT, OutT)                                                        \
    template <typename... T>                                                                   \
    OutT F(InT x, const T&... args) {                                                          \
        return eval_protocol_1arg_alloc<                                                       \
            RepProto<S>, &Worker::PROTO_OBJ_NAME(S),                                           \
            static_cast<OutT (RepProto<S>::*)(const InT&, const T&...)>(&RepProto<S>::F), InT, \
            OutT>(x, args...);                                                                 \
    }

/**
 * @brief Macro to generate evaluator for single-argument functions which return a pair, like
 * `div_const_a`
 *
 */
#define define_1_pair(S, F, InT, OutT)                                                           \
    template <typename... T>                                                                     \
    std::pair<OutT, OutT> F(InT x, const T&... args) {                                           \
        return eval_protocol_1arg_pair<RepProto<S>, &Worker::PROTO_OBJ_NAME(S), &RepProto<S>::F, \
                                       InT, OutT>(x, args...);                                   \
    }

/**
 * @brief Macro to generate evaluator for single-argument functions
 *
 */
#define define_1_arg(S, F, InT, OutT, BatchInput)                                                \
    template <typename... T>                                                                     \
    void F(InT x, OutT& r, const T&... args) {                                                   \
        eval_protocol_1arg<RepProto<S>, &Worker::PROTO_OBJ_NAME(S), &RepProto<S>::F, InT, OutT>( \
            x, r, BatchInput, args...);                                                          \
    }

/**
 * @brief Macro to generate evaluator for two-argument functions
 *
 */
#define define_2_arg(S, F, InT, OutT)                                                            \
    template <typename... T>                                                                     \
    void F(InT x, InT y, OutT& r, const T&... args) {                                            \
        eval_protocol_2arg<RepProto<S>, &Worker::PROTO_OBJ_NAME(S), &RepProto<S>::F, InT, OutT>( \
            x, y, r, args...);                                                                   \
    }

/**
 * @brief Macro to generate evaluator for two-argument functions which return two outputs
 *
 */
#define define_2_arg_2_rtr(S, F, InT, OutT)                                                     \
    template <typename... T>                                                                    \
    void F(InT x, InT y, OutT& r1, OutT& r2, const T&... args) {                                \
        eval_protocol_2arg_2_rtr<RepProto<S>, &Worker::PROTO_OBJ_NAME(S), &RepProto<S>::F, InT, \
                                 OutT>(x, y, r1, r2, args...);                                  \
    }

/**
 * @brief Macro to generate evaluator for two-argument functions which return a differently-sized
 * output
 *
 */
#define define_2_arg_aggregator(S, F, InT, OutT)                                                \
    template <typename... T>                                                                    \
    void F(InT x, InT y, OutT& r, size_t agg, const T&... args) {                               \
        eval_protocol_aggregator_2arg<RepProto<S>, &Worker::PROTO_OBJ_NAME(S), &RepProto<S>::F, \
                                      InT, OutT>(x, y, r, agg, args...);                        \
    }

#define define_2_arg_matrix_primitive(S, F, InT, OutT)                                           \
    template <typename... T>                                                                     \
    void F(InT x, InT y, OutT& r, size_t instancesCount, const T&... args) {                     \
        eval_protocol_matrix_2arg<RepProto<S>, &Worker::PROTO_OBJ_NAME(S), &RepProto<S>::F, InT, \
                                  OutT>(x, y, r, instancesCount, args...);                       \
    }



/**
 * @brief Macro to define all runtime-protocol functionalities
 *
 */
#define runtime_declare_protocol_functions(T)                                                     \
    define_2_arg(T, add_a, AInnerContainerClass(T), AInnerContainerClass(T));                     \
    define_2_arg(T, sub_a, AInnerContainerClass(T), AInnerContainerClass(T));                     \
    define_2_arg(T, multiply_a, AInnerContainerClass(T), AInnerContainerClass(T));                \
    define_1_arg(T, neg_a, AInnerContainerClass(T), AInnerContainerClass(T), true);               \
    define_2_arg(T, xor_b, BInnerContainerClass(T), BInnerContainerClass(T));                     \
    define_2_arg_aggregator(T, dot_product_a, AInnerContainerClass(T), AInnerContainerClass(T));  \
    define_2_arg_matrix_primitive(T, matrix_right_multiply_with_column_matrix_vectorized_a,       \
                                  AInnerContainerClass(T), AInnerContainerClass(T));              \
    define_2_arg_matrix_primitive(T, conv_2d_vectorized_a, AInnerContainerClass(T),               \
                                  AInnerContainerClass(T));                                       \
    define_2_arg_matrix_primitive(T, sumPoolingVectorized, AInnerContainerClass(T),               \
                                  AInnerContainerClass(T));                                       \
    define_2_arg(T, and_b, BInnerContainerClass(T), BInnerContainerClass(T));                     \
    define_2_arg(T, or_b, BInnerContainerClass(T), BInnerContainerClass(T));                      \
    define_2_arg(T, equal_b, BInnerContainerClass(T), BInnerContainerClass(T));                   \
    define_2_arg(T, ripple_carry_adder_b, BInnerContainerClass(T), BInnerContainerClass(T));      \
    define_2_arg(T, ripple_carry_adder_packed_sign_b, BInnerContainerClass(T), BInnerContainerClass(T));   \
    define_2_arg(T, parallel_prefix_adder_b, BInnerContainerClass(T), BInnerContainerClass(T));   \
    define_2_arg(T, rca_compare_b, BInnerContainerClass(T), BInnerContainerClass(T));   \
    define_2_arg_2_rtr(T, compare_b, BInnerContainerClass(T), BInnerContainerClass(T));           \
    define_2_arg_2_rtr(T, linear_compare_b, BInnerContainerClass(T), BInnerContainerClass(T));    \
    define_1_arg(T, gtez_a, AInnerContainerClass(T), AInnerContainerClass(T), true);              \
    define_1_arg(T, a2b_packed_sign_a_b, AInnerContainerClass(T), BInnerContainerClass(T), true);              \
    define_1_arg(T, not_b, BInnerContainerClass(T), BInnerContainerClass(T), true);               \
    define_2_arg(T, and_b_1, BInnerContainerClass(T), BInnerContainerClass(T));                   \
    define_1_arg(T, not_b_1, BInnerContainerClass(T), BInnerContainerClass(T), true);             \
    define_2_arg(T, or_b_1, BInnerContainerClass(T), BInnerContainerClass(T));                    \
    define_2_arg(T, xor_b_1, BInnerContainerClass(T), BInnerContainerClass(T));                   \
    define_1_arg(T, ltz, BInnerContainerClass(T), BInnerContainerClass(T), true);                 \
    define_1_arg(T, pack_from, BInnerContainerClass(T), BInnerContainerClass(T), false);          \
    define_1_arg(T, unpack_from, BInnerContainerClass(T), BInnerContainerClass(T), false);        \
    define_1_arg(T, bit_arithmetic_right_shift, BInnerContainerClass(T), BInnerContainerClass(T), \
                 true);                                                                           \
    define_1_arg(T, bit_logical_right_shift, BInnerContainerClass(T), BInnerContainerClass(T),    \
                 true);                                                                           \
    define_1_arg(T, bit_left_shift, BInnerContainerClass(T), BInnerContainerClass(T), true);      \
    define_1_arg(T, bit_xor, BInnerContainerClass(T), BInnerContainerClass(T), true);             \
    define_1_arg(T, extend_lsb, BInnerContainerClass(T), BInnerContainerClass(T), true);          \
    define_1_arg(T, b2a_bit, BInnerContainerClass(T), AInnerContainerClass(T), true);             \
    define_reshare(T);                                                                            \
    define_1_arg(T, reconstruct_from_a, cdough::Vector<T>, std::vector<AInnerContainerClass(T)>,     \
                 true);                                                                           \
    define_1_arg(T, reconstruct_from_b, cdough::Vector<T>, std::vector<BInnerContainerClass(T)>,     \
                 true);                                                                           \
    define_1_alloc(T, open_shares_a, AInnerContainerClass(T), cdough::Vector<T>);                    \
    define_1_alloc(T, open_shares_b, BInnerContainerClass(T), cdough::Vector<T>);                    \
    define_1_alloc(T, secret_share_a_internal, cdough::Vector<T>, AInnerContainerClass(T));          \
    define_1_alloc(T, secret_share_b_internal, cdough::Vector<T>, BInnerContainerClass(T));          \
    define_1_alloc(T, public_share, cdough::Vector<T>, GenericInnerContainerClass(T));               \
    define_1_pair(T, div_const_a, AInnerContainerClass(T), AInnerContainerClass(T));              \
    define_1_pair(T, redistribute_shares_b, BInnerContainerClass(T), BInnerContainerClass(T));

// For malicious protocols, we also have unchecked versions (not publicly-accessible)
#define unchecked_open_protocol_functions(T)                                      \
    define_1_alloc(T, unchecked_open_a, AInnerContainerClass(T), cdough::Vector<T>); \
    define_1_alloc(T, unchecked_open_b, BInnerContainerClass(T), cdough::Vector<T>);

// Forward declarations
namespace cdough {

class EncodedVector;

template <typename Share, typename EVector, typename Engine>
class ASharedVector;

template <typename Share, typename EVector, typename Engine>
class BSharedVector;

template <typename Share, typename EVector, typename Engine>
class SharedVector;

namespace relational {
    class EncodedColumn;

    template <typename Share, typename EVector, typename Engine>
    class SharedColumn;

    template <typename Share, typename SharedColumn, typename A, typename B, typename EncodedVector,
              typename DataTable>
    class EncodedTable;
}  // namespace relational
}  // namespace cdough

namespace cdough::service {

/**
 * @brief Configure the command line interface for a given long/short argument pair
 *
 * @param cmd
 * @param long
 * @param short
 * @param default default-initialized to T's zero value.
 */
template <typename T = std::string>
T register_cli(oc::CLP& cmd, std::string _long, std::string _short, T _default = T{}) {
    if (_long.empty() || _short.empty()) {
        throw std::runtime_error("Must provide non-null argument to CLP");
    }

    // Long-form options take precedence over abbreviated
    // If only short entered, use that. Otherwise use default.
    if (cmd.hasValue(_long)) {
        return cmd.get<T>(_long);
    }
    return cmd.getOr<T>(_short, _default);
}

// TODO: Use CommunicatorFactory as well after refactoring no-copy communicator
//    to remove engine dependency.
template <typename ProtocolFactory>
class RunTime {
   public:
    // Delete copy and move constructors and assignment operators
    // DO NOT REMOVE; they protect users from engine misuse.
    RunTime(const RunTime&) = delete;
    RunTime(RunTime&&) = delete;
    RunTime& operator=(const RunTime&) = delete;
    RunTime& operator=(RunTime&&) = delete;

    template <typename T>
    using AInnerContainer = ProtocolFactory::template AInnerContainer<T>;
    template <typename T>
    using BInnerContainer = ProtocolFactory::template BInnerContainer<T>;

    template <typename T>  // TODO: all functions using this assumption should be modified.
    using GenericInnerContainer = ProtocolFactory::template AInnerContainer<T>;

    //     -- most probably after making reuntime local for Operators and table.
    template <typename T>
    using ASVector = ASharedVector<T, AInnerContainer<T>, RunTime>;

    template <typename T>
    using BSVector = BSharedVector<T, BInnerContainer<T>, RunTime>;

    using EColumn = relational::EncodedColumn;

    template <typename T>
    using SColumn = relational::SharedColumn<T, GenericInnerContainer<T>, RunTime>;

    template <typename T>
    using DataTable = std::vector<Vector<T>>;

    template <typename T>
    using ETable = relational::EncodedTable<T, SColumn<T>, ASVector<T>, BSVector<T>, EncodedVector,
                                            DataTable<T>>;

    template <typename T>
    using PlainMatrix = cdough::matrix::PlainMatrix<T, Vector>;

    template <typename T>
    using SecureMatrix = cdough::matrix::SecureMatrix<T, ASVector>;

    template <typename T>
    using RepProto = Protocol<T, std::vector<T>, Vector<T>, GenericInnerContainer<T>>;

   protected:
    unchecked_open_protocol_functions(int8_t);
    unchecked_open_protocol_functions(int16_t);
    unchecked_open_protocol_functions(int32_t);
    unchecked_open_protocol_functions(int64_t);
    unchecked_open_protocol_functions(__int128_t);

   private:
    /**
     * @brief Use a barrier to synchronization workers with main thread
     *
     */
    std::shared_ptr<std::barrier<>> barrier;

    int num_threads;
    ssize_t batch_size;

    bool terminate_ = false;

    Setting setting;

    /**
     * @brief Handle for command-line parsing
     *
     */
    oc::CLP cmd;

    /**
     * @brief Rank of this party
     *
     */
    int rank_;

    /**
     * @brief if this runtime is being used for testing (true) or real execution (false)
     *
     */
    bool testing = false;

    /**
     * @brief Number of auxiliary clusters for multinode operation (0 = single-node mode)
     */
    int multinode_num_aux_ = 0;

    /**
     * @brief This auxiliary's index in alphabetical prefix order (0-based)
     */
    int multinode_aux_index_ = 0;

    /**
     * @brief We may have a different number of communication threads
     *
     */
    std::vector<std::thread> socket_comm_threads;

    /**
     * @brief Main thread waits for all workers to arrive at the barrier.
     *
     */
    void main_thread_wait() {
        thread_stopwatch::InstrumentBlock _ib{"wait"};
        barrier->arrive_and_wait();
    }

    /**
     * @brief Determine which batches each thread will be assigned
     *
     * @param total_size
     * @param batch_size_override
     * @return std::vector<std::pair<size_t, size_t>>
     */
    std::vector<std::pair<size_t, size_t>> getThreadBatchBoundaries(
        size_t total_size, std::optional<long> batch_size_override = std::nullopt) {
        std::vector<std::pair<size_t, size_t>> boundaries;
        // uses member variable batch size with an
        // optional override argument
        auto _batch_size = batch_size_override.value_or(getBatchSize());
        if (_batch_size < 0) {
            // If we want equal-division batching,
            // should have a much smaller chunk
            _batch_size = MINIMUM_CHUNK_SIZE;
        }

        //// original batching algorithm ////
        // how many total batches do we need?
        // (floored)
        size_t whole_batches = total_size / _batch_size;
        // every thread will get at least this many
        // batches
        size_t batches_per_thread = whole_batches / num_threads;
        // but there a few whole batches left over
        size_t remaining_batches = whole_batches % num_threads;
        // and then there is one partial batch
        // remaining
        size_t partial_batch = total_size % _batch_size;

        size_t start_index = 0;
        size_t end_index = 0;

        for (int i = 0; i < num_threads; i++) {
            size_t n = batches_per_thread * _batch_size;
            if (remaining_batches > 0) {
                // one extra batch for this thread
                n += _batch_size;
                remaining_batches -= 1;
            }
            // all other threads just get the base
            // amount

            // end_index is exclusive
            end_index = start_index + n;
            boundaries.push_back({start_index, end_index});

            start_index = end_index;
        }

        // last thread gets the remainder
        boundaries[num_threads - 1].second += partial_batch;

        return boundaries;
    }

    /**
     * @brief Add a task to all workers.
     *
     * TODO: make this a WorkerGroup method?
     *
     * @tparam F
     * @param size total size of this operation (usually the size of the input
     * vector)
     * @param task_factory lambda which returns a task for each worker.
     * @param batch_size optional adjusted batch size
     */
    template <typename F>
        requires std::invocable<F&, size_t, size_t> &&
                 std::convertible_to<std::invoke_result_t<F&, size_t, size_t>,
                                     std::unique_ptr<Task>>
    void addTask(size_t size, F&& task_factory, std::optional<long> batch_size = std::nullopt) {
        auto boundaries = getThreadBatchBoundaries(size, batch_size);

        for (int t = 0; t < num_threads; t++) {
            auto [start, end] = boundaries[t];

            // call the task factory
            workers[t].addTask(task_factory(start, end));
        }
    }

    /**
     * @brief Add a task to all workers; also pass a reference to the worker
     * into the task. This is used for thread-specific functionality like MPC
     * protocol primitives and randomness generation
     *
     * TODO: some clean way to combine these two versions?
     *
     * @tparam F
     * @param size
     * @param task_factory
     * @param batch_size optional adjusted batch size
     */
    template <typename F>
        requires std::invocable<F&, size_t, size_t, Worker&> &&
                 std::convertible_to<std::invoke_result_t<F&, size_t, size_t, Worker&>,
                                     std::unique_ptr<Task>>
    void addTask(size_t size, F&& task_factory, std::optional<long> batch_size = std::nullopt) {
        auto boundaries = getThreadBatchBoundaries(size, batch_size);

        for (int t = 0; t < num_threads; t++) {
            auto [start, end] = boundaries[t];
            auto& w = workers[t];

            // call the task factory
            w.addTask(task_factory(start, end, w));
        }
    }

   public:
    /**
     * @brief The worker threads
     *
     */
    std::vector<Worker> workers;

    /**
     * @brief Temporary pointer to worker zero. (Eventually we would like to get rid of this.)
     * Provides external callers a way to access a given worker. Currently only used in the
     * shuffle cost model, so probably an easy way to redesign this.
     *
     */
    Worker* worker0;

    /**
     * @brief Get worker zero's communicator
     *
     * @return cdough::Communicator*
     */
    cdough::Communicator* comm0() { return worker0->getCommunicator(); }

    /**
     * @brief Get worker zero's randomness manager
     *
     * @return cdough::random::RandomnessManager*
     */
    cdough::random::RandomnessManager* rand0() { return worker0->getRandManager(); }

    /**
     * @brief Check if this runtime is terminated
     *
     * @return true
     * @return false
     */
    bool terminated() { return terminate_; }

    /**
     * @brief Destructor. Set `terminate_` to true, and wait for all socket communicator threads to
     * be joined.
     *
     */
    ~RunTime() {
        terminate_ = true;

        // Check for SocketCommunicator thread
        // termination
        for (int i = 0; i < socket_comm_threads.size(); ++i) {
            socket_comm_threads[i].join();
        }

#ifdef MPC_USE_MPI_COMMUNICATOR
        // TODO: engine should not be responsible for finalizing MPI;
        // ideally should be handled by the communicator/CommunicatorFactory.
        static bool mpi_finalized = false;
        if (!testing && !mpi_finalized) {
            MPI_Finalize();
            mpi_finalized = true;
        }
#endif
    }

    /**
     * @brief Construct a new Runtime object
     *
     * @param _batch_size
     * @param _num_threads
     * @param testing change to `true` if creating additional `RunTime` objects inside tests
     */
    RunTime(const long _batch_size, const int _num_threads, std::optional<oc::CLP> cmd_,
            bool testing = false)
        : batch_size(_batch_size),
          num_threads(_num_threads),
          terminate_(false),
          testing(testing),
          // initialize synchronization barrier
          barrier(std::make_shared<std::barrier<>>(_num_threads + 1)) {
        if (cmd_) {
            cmd = *cmd_;
            setting = parse_setting(register_cli<std::string>(cmd, "setting", "s", "same"));
            multinode_num_aux_ = register_cli<int>(cmd, "multinode", "M", 0);
            multinode_aux_index_ = register_cli<int>(cmd, "aux-index", "m", 0);
        }
    }

    /**
     * @brief Setup the thread workers for this party
     *
     * @param rank
     */
    void setup_workers(int rank) {
        rank_ = rank;
        // prevent reallocation
        workers.reserve(num_threads);
        for (int i = 0; i < num_threads; i++) {
            // Create each worker - need to use emplace to construct directly
            // inside vector.
            workers.emplace_back(rank, barrier, i);
            // and start it
            workers.back().start();
        }

        worker0 = &workers.front();

        // wait until all thread have been initialized to return control from
        // this constructor
        barrier->arrive_and_wait();
    }

    /**
     * @brief Create a socket communicator thread and it to the runtime's internal store
     *
     * @tparam T
     * @param t
     */
    template <typename... T>
    void emplace_socket_thread(T&&... t) {
        socket_comm_threads.emplace_back(std::forward<T>(t)...);
    }

    template <typename T = std::string>
    T getArg(std::string _long, std::string _short, T _default = T{}) {
        return register_cli(cmd, _long, _short, _default);
    }

    /**
     * @brief Declare protocol functionalities for each supported type
     *
     */
    runtime_declare_protocol_functions(int8_t);
    runtime_declare_protocol_functions(int16_t);
    runtime_declare_protocol_functions(int32_t);
    runtime_declare_protocol_functions(int64_t);
    runtime_declare_protocol_functions(__int128_t);

    template <typename T1, typename... T2>
    auto secret_share_a(cdough::Vector<T1> data, const T2&... args) {
        return ASVector<T1>(secret_share_a_internal(data, args...), *this);
    }

    template <typename T1, typename... T2>
    auto secret_share_b(cdough::Vector<T1> data, const T2&... args) {
        return BSVector<T1>(secret_share_b_internal(data, args...), *this);
    }

    template <typename T1, typename... T2>
    auto public_share_a(cdough::Vector<T1> data, const T2&... args) {
        return ASVector<T1>(public_share(data, args...), *this);
    }

    template <typename T1, typename... T2>
    auto public_share_b(cdough::Vector<T1> data, const T2&... args) {
        // TODO: Should be removed; A/B public shares are not equivalent.
        return BSVector<T1>(public_share(data, args...), *this);
    }

    /**
     * Secret-shares the plaintext columns according to the encodings in the `schema`.
     * @tparam T The plaintext data type.
     * @tparam R The replication factor of the secret-shared columns.
     * @param columns The plaintext columns.
     * @param schema The encoded column names.
     * @param _party_id data owner
     * @return A vector of EncodedColumns that can be used to instantiate an EncodedTable.
     *
     * NOTE: This method can be used to construct an EncodedTable from a plaintext one as
     * follows: EncodedTable t = secret_share_columns(plaintext_table, schema);
     */
    template <typename T>
    std::vector<std::shared_ptr<EColumn>> secret_share_columns(
        const std::vector<cdough::Vector<T>>& columns, const std::vector<std::string>& schema,
        const int& _party_id = 0) {
        assert(columns.size() == schema.size());
        using namespace service;
        std::vector<std::shared_ptr<EColumn>> cols;
        for (int i = 0; i < columns.size(); i++) {
            std::unique_ptr<EncodedVector> v;
            int prec = static_cast<int>(columns[i].getPrecision());
            if (ETable<T>::isBShared(schema[i])) {
                v = std::make_unique<BSVector<T>>(
                    this->secret_share_b(columns[i], _party_id, prec));
            } else {
                v = std::make_unique<ASVector<T>>(
                    this->secret_share_a(columns[i], _party_id, prec));
            }
            cols.push_back(std::make_shared<SColumn<T>>(std::move(v), schema[i]));
        }
        return cols;
    }

    /**
     * Constructs a table and initializes it with zeros.
     *
     * @param _tableName The name of the table.
     * @param _columns The column names.
     * @param _rows The number of rows to allocate.
     */
    template <typename T>
    ETable<T> secret_share_table(const std::string& _tableName,
                                 const std::vector<std::string>& _columns, int _rows) {
        return ETable<T>(_tableName, _columns, _rows, *this);
    }

    /**
     * Constructs a table from encoded columns.
     * @param contents The table columns.
     */
    template <typename T>
    ETable<T> secret_share_table(std::vector<std::shared_ptr<EColumn>>&& contents) {
        return ETable<T>(std::move(contents), *this);
    }

    template <typename T>
    ETable<T> secret_share_table(const std::vector<cdough::Vector<T>>& columns,
                                 const std::vector<std::string>& schema, const int& _party_id = 0) {
        auto cols = secret_share_columns(columns, schema, _party_id);
        return ETable<T>(std::move(cols), *this);
    }

    /**
     * @brief Secret-share a plaintext matrix into a secure matrix
     * @tparam T
     *
     * @param mat The plaintext matrix
     * @param _party_id data owner
     */
    template <typename T>
    SecureMatrix<T> secret_share_matrix(const PlainMatrix<T>& mat, int _party_id = 0) {
        auto shared_data = secret_share_a(mat.data(), _party_id);
        return SecureMatrix<T>(shared_data, mat.rows(), mat.cols(), mat.isColumnWise());
    }

    /**
     * @brief Secret-share a matrix of zeros into a secure matrix
     * @tparam T
     *
     * @param rows Number of rows
     * @param cols Number of columns
     * @param colWise Whether the data is stored in column-wise order
     */
    template <typename T>
    SecureMatrix<T> public_share_zero_matrix(int rows, int cols, const bool& colWise = false) {
        cdough::Vector<T> data(rows * cols);
        auto shared_data = public_share_a(data);
        return SecureMatrix<T>(shared_data, rows, cols, colWise);
    }

    /**
     * @brief Execute a non-MPC parallel functionality over a vector which
     * puts the result into a new vector.
     *
     * @tparam ObjectType
     * @tparam T
     * @param x
     * @param res
     * @param func
     * @param args
     */
    template <typename ObjectType, typename... T>
    void execute_parallel(const ObjectType& x, ObjectType& res,
                          ObjectType (ObjectType::*func)(const T&...) const, const T&... args) {
        thread_stopwatch::InstrumentBlock _ib{};
        assert(x.total_size() == res.total_size());

        addTask(x.total_size(), [&](const size_t start, const size_t end) {
            return std::make_unique<Task_1_ref<ObjectType, ObjectType>>(
                x, res, start, end, batch_size, true,
                [&, this](ObjectType& _x, ObjectType& _res) { _res = (_x.*func)(args...); });
        });

        main_thread_wait();
    }

    /**
     * @brief Execute a non-MPC parallel functionality over a vector which
     * modifies the vector in place.
     *
     * @tparam ObjectType
     * @tparam T1
     * @tparam T2
     * @param x
     * @param func
     * @param args1
     * @param args2
     */
    template <typename ObjectType, typename... T1, typename... T2>
    void modify_parallel(ObjectType& x, void (ObjectType::*func)(const T1&..., const T2&...),
                         const T1&... args1, const T2&... args2) {
        thread_stopwatch::InstrumentBlock _ib{};

        addTask(x.total_size(), [&](const size_t start, const size_t end) {
            return std::make_unique<Task_1_void<ObjectType>>(
                x, start, end, batch_size,
                [&, this](ObjectType& _x) { (_x.*func)(args1..., args2...); });
        });

        main_thread_wait();
    }

    /**
     * @brief Execute a non-MPC member function over two batched elements.
     * Useful for parallelizing vector copies, for example. Multithreaded
     * equivalent to
     *   x.func(y)
     * where x is modified but y is not.
     *
     * This is similar to modify_parallel except that it applies batching to
     * both input and output vectors, whereas `modify_parallel` assumes only the
     * output vector `x` should have batching applied. A similar effect could be
     * achieved with `execute_parallel`, but this would incur an unecessary
     * copy.
     *
     * This function has two separate template arguments to account for the fact
     * that some functions (such as the copy-and-cast assignment operator) may
     * accept EVectors of different types.
     *
     * @tparam E1 RHS (x) EVector type
     * @tparam E2 LHS (y) EVector type
     * @param x
     * @param y
     * @param func
     */
    template <typename E1, typename E2>
    void modify_parallel_2arg(E1& x, const E2& y, E1& (E1::*func)(const E2&)) {
        thread_stopwatch::InstrumentBlock _ib{};

        // Task_1_ref args are (input, output) so we have to swap order of x,y
        addTask(x.total_size(), [&](const size_t start, const size_t end) {
            return std::make_unique<Task_1_ref<E2, E1>>(
                y, x, start, end, batch_size, true, [&, this](E2& _y, E1& _x) { (_x.*func)(_y); });
        });

        main_thread_wait();
    }

    /**
     * @brief Execute an arbitrary function (probably passed as a lambda) in a multithreaded
     * argument. Passed function should take two arguments representing the start and end
     * indices of its batch. Optional third argument represents the thread ID.
     *
     * @tparam F The function type.
     * @param range_size The number of elements to process.
     * @param func The function to execute.
     */
    template <typename F>
    void execute_parallel_unsafe(int range_size, F&& func) {
        thread_stopwatch::InstrumentBlock _ib{};

        addTask(
            range_size,
            [&](const size_t start, const size_t end, Worker& w) {
                size_t tid = w.getId();
                return std::make_unique<Task_0_void>(
                    start, end, static_cast<ssize_t>(end - start),
                    [func, tid](const size_t s, const size_t e) {
                        // three argument function case
                        if constexpr (std::is_invocable_v<F, size_t, size_t, size_t>) {
                            func(s, e, tid);
                        } else {
                            func(s, e);
                        }
                    });
            },
            1L);

        main_thread_wait();
    }

    /**
     * @brief Generates randomness in parallel using all available threads
     *
     * @tparam InputType
     * @tparam T
     * @param func a function of the RandomnessManager class that will be
     * called by each thread. The RandomnessManager class has functions
     * generate_local and generate_common that can be passed as input to
     * generate randomness from either the localPRG or a commonPRG.
     * @param input vector to generate randomness into
     * @param args additional args to pass to the generator
     */
    template <typename InputType, typename... T>
    void generate_parallel(void (cdough::random::RandomnessManager::*func)(InputType&, T...),
                           InputType& input, const T&... args) {
        thread_stopwatch::InstrumentBlock _ib{};

        addTask(input.total_size(), [&](const size_t start, const size_t end, Worker& w) {
            return std::make_unique<Task_1_void<InputType>>(
                input, start, end, batch_size,
                [&, this](InputType& _x) { (w.getRandManager()->*func)(_x, args...); });
        });

        main_thread_wait();
    }

    /**
     * @brief Parallel evaluate `reshare` over a vector.
     *
     * @tparam Proto
     * @tparam ProtoObj
     * @tparam EVector
     * @tparam T
     * @param x
     * @param args
     */
    template <typename Proto, auto ProtoObj, typename EVector, typename... T>
    void eval_protocol_reshare(EVector& x, const T&... args) {
        thread_stopwatch::InstrumentBlock _ib{};

        addTask(x.total_size(), [&](const size_t start, const size_t end, Worker& w) {
            return std::make_unique<Task_1_void<EVector>>(
                x, start, end, batch_size, [&, this](EVector& _x) {
                    static_cast<Proto*>((w.*ProtoObj).get())->reshare(_x, args...);
                });
        });

        main_thread_wait();
    }

    /**
     * @brief Evaluate a batched unary protocol function producing a
     * freshly-allocated OutT (e.g. secret sharing)
     *
     * @tparam Proto
     * @tparam ProtoObj
     * @tparam F
     * @tparam InT
     * @tparam OutT
     * @tparam T
     * @param x
     * @param args
     */
    template <typename Proto, auto ProtoObj, auto F, typename InT, typename OutT, typename... T>
        requires std::is_member_function_pointer_v<decltype(F)>
    OutT eval_protocol_1arg_alloc(InT& x, const T&... args) {
        thread_stopwatch::InstrumentBlock _ib{};

        OutT res(x.total_size());

        // thread-safe way to set precision inside the workers
        std::once_flag precision_flag;
        int precision = 0;

        addTask(x.total_size(), [&](const size_t start, const size_t end, Worker& w) {
            return std::make_unique<Task_1_ref<InT, OutT>>(
                x, res, start, end, batch_size, true, [&, this](InT& _x, OutT& _res) {
                    _res = (static_cast<Proto*>((w.*ProtoObj).get())->*F)(_x, args...);
                    std::call_once(precision_flag, [&]() { precision = _res.getPrecision(); });
                });
        });

        main_thread_wait();
        res.setPrecision(precision);
        return res;
    }

    /**
     * @brief Evaluate a batched unary protocol function producing a pair of
     * `<OutT, OutT>`
     *
     * @tparam Proto
     * @tparam ProtoObj
     * @tparam F
     * @tparam InT
     * @tparam OutT
     * @tparam T
     * @param x
     * @param args
     */
    template <typename Proto, auto ProtoObj, auto F, typename InT, typename OutT, typename... T>
        requires std::is_member_function_pointer_v<decltype(F)>
    std::pair<OutT, OutT> eval_protocol_1arg_pair(InT& x, const T&... args) {
        thread_stopwatch::InstrumentBlock _ib{};

        auto r = std::make_pair<OutT, OutT>(OutT(x.total_size()), OutT(x.total_size()));

        addTask(x.total_size(), [&](const size_t start, const size_t end, Worker& w) {
            return std::make_unique<Task_1_pair<InT, OutT>>(
                x, r, start, end, batch_size, [&, this](InT& _x, OutT& r1, OutT& r2) {
                    std::tie(r1, r2) = (static_cast<Proto*>((w.*ProtoObj).get())->*F)(_x, args...);
                });
        });

        main_thread_wait();
        return r;
    }

    /**
     * @brief Evaluated a batched unary protocol function producing an OutT.
     *
     * @tparam Proto
     * @tparam ProtoObj
     * @tparam F
     * @tparam InT
     * @tparam OutT
     * @tparam T
     * @param x
     * @param r
     * @param args
     */
    template <typename Proto, auto ProtoObj, auto F, typename InT, typename OutT, typename... T>
        requires std::is_member_function_pointer_v<decltype(F)>
    void eval_protocol_1arg(InT& x, OutT& r, const bool& batch_input, const T&... args) {
        thread_stopwatch::InstrumentBlock _ib{};

        addTask(x.total_size(), [&](const size_t start, const size_t end, Worker& w) {
            return std::make_unique<Task_1_ref<InT, OutT>>(
                x, r, start, end, batch_size, batch_input, [&, this](InT& _x, OutT& _r) {
                    (static_cast<Proto*>((w.*ProtoObj).get())->*F)(_x, _r, args...);
                });
        });

        main_thread_wait();
    }

    /**
     * @brief Evaluate a batched binary protocol function producing an OutT.
     *
     * @tparam Proto
     * @tparam ProtoObj
     * @tparam F
     * @tparam InT
     * @tparam OutT
     * @tparam T
     * @param x
     * @param y
     * @param r
     * @param args
     */
    template <typename Proto, auto ProtoObj, auto F, typename InT, typename OutT, typename... T>
        requires std::is_member_function_pointer_v<decltype(F)>
    void eval_protocol_2arg(InT& x, InT& y, OutT& r, const T&... args) {
        thread_stopwatch::InstrumentBlock _ib{};

        addTask(x.total_size(), [&](const size_t start, const size_t end, Worker& w) {
            return std::make_unique<Task_2_ref<InT, OutT>>(
                x, y, r, start, end, batch_size, [&, this](InT& _x, InT& _y, OutT& _r) {
                    (static_cast<Proto*>((w.*ProtoObj).get())->*F)(_x, _y, _r, args...);
                });
        });

        main_thread_wait();
    }

    /**
     * @brief Evaluate a batched binary protocol function producing a pair of OutTs.
     *
     * @tparam Proto
     * @tparam ProtoObj
     * @tparam F
     * @tparam InT
     * @tparam OutT
     * @tparam T
     * @param x
     * @param y
     * @param r1
     * @param r2
     * @param args
     */
    template <typename Proto, auto ProtoObj, auto F, typename InT, typename OutT, typename... T>
        requires std::is_member_function_pointer_v<decltype(F)>
    void eval_protocol_2arg_2_rtr(InT& x, InT& y, OutT& r1, OutT& r2, const T&... args) {
        thread_stopwatch::InstrumentBlock _ib{};

        addTask(x.total_size(), [&](const size_t start, const size_t end, Worker& w) {
            return std::make_unique<Task_2_ref_2_rtr<InT, OutT>>(
                x, y, r1, r2, start, end, batch_size,
                [&, this](InT& _x, InT& _y, OutT& _r1, OutT& _r2) {
                    (static_cast<Proto*>((w.*ProtoObj).get())->*F)(_x, _y, _r1, _r2, args...);
                });
        });

        main_thread_wait();
    }

    /**
     * @brief Extends protocol functionality for when input size is different than output size.
     *
     * @tparam Proto
     * @tparam ProtoObj
     * @tparam F
     * @tparam InT
     * @tparam OutT
     * @tparam T For the aggregation size.
     * @tparam T2
     * @param x
     * @param y
     * @param r
     * @param agg the aggregation size, the ration between input sizes and output sizes.
     * @param args additional arguments to the function.
     */
    template <typename Proto, auto ProtoObj, auto F, typename InT, typename OutT, typename... T>
        requires std::is_member_function_pointer_v<decltype(F)>
    void eval_protocol_aggregator_2arg(InT& x, InT& y, OutT& r, const size_t agg,
                                       const T&... args) {
        thread_stopwatch::InstrumentBlock _ib{};

        // This is required for both (1) avoiding accessing out of bounds
        // and (2) ensuring that the dot product is computed in batches of
        // size divisible by `aggSize` elements inside the runtime.
        assert(x.size() % agg == 0);
        ssize_t old_batch_size;
        old_batch_size = makeBatchSizeDivisibleBy(x.size(), agg);

        addTask(x.total_size(), [&](const size_t start, const size_t end, Worker& w) {
            return std::make_unique<Task_2_Agg_ref<InT, OutT>>(
                x, y, r, start, end, batch_size, agg, [&, agg, this](InT& _x, InT& _y, OutT& _r) {
                    (static_cast<Proto*>((w.*ProtoObj).get())->*F)(_x, _y, _r, agg, args...);
                });
        });

        // Restore the old batch size
        setBatchSize(old_batch_size);

        main_thread_wait();
    }

    /**
     * @brief Extends protocol functionality for when there are multiple instances
     * of matrices being processed in a batched manner.
     * Note:
     * 1. This function batches only the left hand matrix. The right hand matrix is not
     * batched but exposed to protocol functionalities as is.
     * 2. The output matrix is batched in ratio to the left hand matrix.
     *
     * @tparam Proto
     * @tparam ProtoObj
     * @tparam F
     * @tparam InT
     * @tparam OutT
     * @tparam T
     * @param x
     * @param y
     * @param r
     * @param instancesCount number of instances being processed in a batched manner.
     * @param args additional arguments to the function.
     */
    template <typename Proto, auto ProtoObj, auto F, typename InT, typename OutT, typename... T>
        requires std::is_member_function_pointer_v<decltype(F)>
    void eval_protocol_matrix_2arg(InT& x, InT& y, OutT& r, const size_t instancesCount,
                                   const T... args) {
        thread_stopwatch::InstrumentBlock _ib{};

        assert(x.size() % instancesCount == 0);
        assert(r.size() % instancesCount == 0);

        // Ensuring that batching is per input matrix
        // We send the same ratio of input to output sizes
        // for each thread/batch.
        const size_t inputSize = x.size();
        const size_t outputSize = r.size();
        const size_t inputInstanceSize = inputSize / instancesCount;
        const size_t outputInstanceSize = outputSize / instancesCount;
        ssize_t old_batch_size;
        old_batch_size = makeBatchSizeDivisibleBy(x.size(), inputInstanceSize);

        addTask(x.total_size(), [&](const size_t start, const size_t end, Worker& w) {
            return std::make_unique<Task_2_Matrix_ref<InT, OutT>>(
                x, y, r, start, end, batch_size, inputInstanceSize, outputInstanceSize,
                [&, inputInstanceSize, this](InT& _x, InT& _y, OutT& _r) {
                    const size_t newInstancesCount = _x.size() / inputInstanceSize;
                    (static_cast<Proto*>((w.*ProtoObj).get())->*F)(_x, _y, _r, newInstancesCount,
                                                                   args...);
                });
        });

        // Restore the old batch size
        setBatchSize(old_batch_size);

        main_thread_wait();
    }

    /**
     * Generate a set of sharded permutations in parallel using all available threads.
     *
     * @param ret The set of sharded permutations to generate.
     */
    template <typename T>
    void generate_permutations(std::vector<std::shared_ptr<cdough::random::ShardedPermutation>>& ret) {
        thread_stopwatch::InstrumentBlock _ib{};
        int num_permutations = ret.size();

        // calculate the number of permutations each
        // thread will generate
        std::vector<int> perms_per_thread(num_threads);
        int basePerms = num_permutations / num_threads;
        int remainder = num_permutations % num_threads;
        for (int i = 0; i < num_threads; ++i) {
            perms_per_thread[i] = basePerms + (i < remainder ? 1 : 0);
        }

        using permVec_t = std::vector<std::shared_ptr<cdough::random::ShardedPermutation>>;

        permVec_t perms(basePerms + 1);

        int perm_index = 0;
        for (int t = 0; t < num_threads; ++t) {
            // get this thread's generator
            auto& w = workers[t];
            auto generator =
                w.getRandManager()
                    ->template getCorrelation<T, random::ShardedPermutationGenerator>();

            perms.resize(perms_per_thread[t]);

            for (int i = 0; i < perms.size(); ++i) {
                perms[i] = ret[perm_index++];
            }

            w.addTask(std::make_unique<Task_1_void_nobatch<permVec_t>>(
                perms, [generator, this](permVec_t& _x) { generator->generateBatch(_x); }));
        }

        main_thread_wait();
    }

#if defined(USE_LIBOTE) && defined(USE_SECURE_JOIN)
    /**
     * @brief Evaluates OPRF on an input vector in parallel using all
     * available threads.
     *
     * @param input The input vector to evaluate OPRF on.
     * @param output The output vector to store OPRF results.
     * @param sender True if this party is the sender, false if receiver.
     */
    void evaluate_oprf(const Vector<__int128_t>& input, Vector<__int128_t>& output, int sender) {
        thread_stopwatch::InstrumentBlock _ib{};

        assert(input.size() == output.size());

        // Generate the OPRF key once (used only by the sender).
        // The receiver executes this as well, but they throw away the key.
        // The sender and receiver generate different keys.
        auto first_oprf = this->rand0()->template getCorrelation<__int128_t, cdough::random::OPRF>();
        cdough::random::OPRF::key_t key = first_oprf->keyGen();

        // Split the input into a single contiguous chunk per thread.  We use
        // an explicit batch-size override so that each thread receives at
        // most one chunk and therefore one call to the OPRF primitive.
        size_t per_thread = std::max<size_t>(1, (input.size() + num_threads - 1) / num_threads);
        auto boundaries = getThreadBatchBoundaries(input.size(), static_cast<long>(per_thread));

        for (int t = 0; t < num_threads; ++t) {
            auto [start, end] = boundaries[t];

            // Skip threads that have no work (possible when |input| < num_threads).
            if (start >= end) {
                continue;
            }

            const size_t chunk_size = end - start;

            auto& w = workers[t];
            auto oprf =
                w.getRandManager()->template getCorrelation<__int128_t, cdough::random::OPRF>();

            // Each task executes exactly once on its chunk (batch_size = chunk_size).
            w.addTask(std::make_unique<Task_0_void>(
                start, end, static_cast<ssize_t>(chunk_size),
                [&, oprf, start, end](const size_t /*s*/, const size_t /*e*/) {
                    const size_t local_size = end - start;

                    if (sender == 1) {
                        // Sender: evaluate using the shared key.
                        auto out_chunk = oprf->template evaluate_sender<__int128_t>(
                            key, static_cast<int>(local_size));

                        // Copy results back to the global output vector.
                        for (size_t i = 0; i < local_size; ++i) {
                            output[start + i] = out_chunk[i];
                        }
                    } else {
                        // Receiver: prepare the input slice for this chunk.
                        Vector<__int128_t> in_chunk(local_size);
                        for (size_t i = 0; i < local_size; ++i) {
                            in_chunk[i] = input[start + i];
                        }

                        auto out_chunk = oprf->template evaluate_receiver<__int128_t>(in_chunk);

                        // Copy results back to the global output vector.
                        for (size_t i = 0; i < local_size; ++i) {
                            output[start + i] = out_chunk[i];
                        }
                    }
                }));
        }

        main_thread_wait();
    }
#endif

    /**
     * @brief Get this node's party ID (rank)
     *
     * @return int
     */
    int getPartyID() const { return rank_; }

    /**
     * @brief Get the number of auxiliary clusters for multinode operation
     *
     * @return int Number of auxiliary clusters (0 = single-node mode)
     */
    int getMultinodeNumAux() const { return multinode_num_aux_; }

    /**
     * @brief Get this auxiliary's index in alphabetical prefix order
     *
     * @return int Auxiliary index (0-based)
     */
    int getMultinodeAuxIndex() const { return multinode_aux_index_; }

    /**
     * @brief Get the replication number for the current protocol
     *
     * @return int
     */
    int getReplicationNumber() const { return workers[0].proto_32->getRepNumber(); }

    /**
     * @brief Get all randomness groups for this protocol
     *
     * @return std::vector<std::set<int>>
     */
    std::vector<std::set<int>> getGroups() const { return workers[0].proto_32->getGroups(); }

    /**
     * @brief Get the current batch size
     *
     * @return ssize_t
     */
    ssize_t getBatchSize() const { return batch_size; }

    /**
     * @brief Get the number of batches which would be required for a vector of
     * length \f$n\f$.
     *
     * @param n
     * @return size_t
     */
    size_t numBatches(size_t n) {
        auto b = getBatchSize();
        if (b < 0) {
            // negative = equal division into that many batches
            return -b;
        }

        // ceiling division
        size_t total_batch = (n + b - 1) / b;
        size_t batch_per_thread = (total_batch + num_threads - 1) / num_threads;
        return batch_per_thread;
    }

    /**
     * @brief Update the batch size
     *
     * @param new_batch_size
     */
    void setBatchSize(const ssize_t& new_batch_size) { batch_size = new_batch_size; }

    /**
     * @brief Adjust the batch size to be divisible by some other divisor. Necessary for e.g.
     * partial dot product operation, which must assign batches of the proper size to each thread.
     *
     * @param total
     * @param divisor
     * @return ssize_t
     */
    ssize_t makeBatchSizeDivisibleBy(const size_t total, const size_t divisor) {
        const ssize_t old_batch_size = batch_size;

        // If batch size is negative, we want to compute equal batches
        if (batch_size < 0) {
            const size_t batches_num = std::abs(batch_size);
            batch_size = compute_equal_batches(0, total - 1, batches_num * num_threads);
        }

        // Regardless of whether we computed a new batch size or not,
        // we want to make sure it is divisible by the divisor
        if (batch_size % divisor != 0) {
            batch_size += divisor - (batch_size % divisor);
        }

        return old_batch_size;
    }

    /**
     * @brief Get the number of worker threads
     *
     * @return int
     */
    int get_num_threads() { return num_threads; }

    /**
     * @brief Get the batch size
     *
     * @return int
     */
    int get_batch_size() { return batch_size; }

    /**
     * @brief Get the number of parties
     *
     * @return const int
     */
    const int getNumParties() const { return workers[0].proto_32->getNumParties(); }

    const Setting getSetting() const { return setting; }

    /**
     * @brief Get a set of all party indices
     *
     * @return std::set<int>
     */
    std::set<int> getPartySet() const {
        std::set<int> partySet;
        for (int i = 0; i < getNumParties(); i++) {
            partySet.insert(i);
        }
        return partySet;
    }

    /**
     * @brief Get a mapping of parties to shares
     *
     * @return std::vector<std::vector<int>>
     */
    std::vector<std::vector<int>> getPartyShareMappings() const {
        return workers[0].proto_32->getPartyShareMappings();
    }

    /**
     * Populates the vector with locally generated pseudorandom shares.
     * @param v vector to populate
     */
    template <typename T>
    void populateLocalRandom(Vector<T>& v) {
        generate_parallel(&cdough::random::RandomnessManager::generate_local, v);
    }

    /**
     * Populates the vector with commonly generated pseudorandom shares.
     * @param v Vector to populate
     * @param group The group that shares a CommonPRG.
     */
    template <typename T>
    void populateCommonRandom(Vector<T>& v, std::set<int> group) {
        generate_parallel(&cdough::random::RandomnessManager::generate_common, v, group);
    }

    /**
     * Generates and pools Beaver triples in parallel.
     * @param func The function to call within the RandomnessManager to generate.
     * @param n The number of triples to generate.
     *
     * NOTE - This is very similar to generate_parallel except it doesn't take a
     * pre-allocated vector as input. We should eventually be able to merge
     * these functions by making one slightly more general one.
     *
     * TODO: use runtime::addtask. Custom batching here.
     */
    template <typename T>
    void reserve_triples(void (cdough::random::RandomnessManager::*func)(size_t n), size_t n) {
        thread_stopwatch::InstrumentBlock _ib{};

        // batch size here is -1: each thread generates a single batch of
        // triples
        addTask(
            n,
            [&](const size_t start, const size_t end, Worker& w) {
                const size_t triple_size = end - start;
                return std::make_unique<Task_0_void>(start, end, triple_size,
                                                     [&, triple_size, this](int _start, int _end) {
                                                         (w.getRandManager()->*func)(triple_size);
                                                     });
            },
            -1);

        main_thread_wait();
    }

    /**
     * Calls reserve_triples() to generate arithmetic triples.
     * @param n The number of triples to generate.
     */
    template <typename T>
    void reserve_mul_triples(size_t n) {
#ifdef MPC_PROTOCOL_BEAVER_TWO
        reserve_triples<T>(&cdough::random::RandomnessManager::reserve_mul_triples<T>, n);
#endif
    }

    /**
     * Calls reserve_triples() to generate binary triples.
     * @param n The number of triples to generate.
     */
    template <typename T>
    void reserve_and_triples(size_t n) {
#ifdef MPC_PROTOCOL_BEAVER_TWO
        reserve_triples<T>(&cdough::random::RandomnessManager::reserve_and_triples<T>, n);
#endif
    }

    bool malicious_check() {
        // Check all threads, all protocols
        bool ok = true;
        for (auto& w : workers) {
            ok &= w.malicious_check();
        }

        if (!ok) {
            std::cout << "P" << getPartyID() << ": Malicious Check FAILED!\n";
        }

        return ok;
    }

    void reset_malicious_state() {
        for (auto& w : workers) {
            w.reset_malicious_state();
        }
    }

    /**
     * Print the number of bytes sent by each communicator.
     */
    void print_communicator_statistics() {
#ifdef PRINT_COMMUNICATOR_STATISTICS
        const std::string _spacer = " ";
        const std::string title = "Communicator (Bytes sent):";
        const std::string lhs_prefix = "Thread ";

        int lhs_width = lhs_prefix.size() + std::ceil(std::log10(num_threads)) + 1;
        size_t total_bytes_approximate = comm0()->getBytesSent() * num_threads;
        int max_bytes_width =
            (total_bytes_approximate > 0 ? std::ceil(std::log10(total_bytes_approximate)) : 0) + 1;
        auto total_width = std::max(lhs_width + 2 * _spacer.size() + max_bytes_width, title.size());

        std::cout << "\n" << std::string(total_width, '=') << "\n" << title << "\n";

        size_t total_bytes_sent = 0;
        for (int i = 0; i < num_threads; ++i) {
            size_t bytes_sent = workers[i].getCommunicator()->getBytesSent();
            total_bytes_sent += bytes_sent;

            std::cout << _spacer << std::setw(lhs_width) << std::left
                      << lhs_prefix + std::to_string(i);
            std::cout << _spacer << std::setw(max_bytes_width) << std::right << bytes_sent << "\n";
        }
        std::cout << "\n"
                  << "P" << getPartyID() << std::setw(lhs_width) << std::left << " Total";
        std::cout << _spacer << std::setw(max_bytes_width) << std::right << total_bytes_sent
                  << "\n";

        std::cout << std::string(total_width, '=') << "\n";
#endif
    }

    /**
     * @brief Print statistics from worker 0. This is meant for cost modeling,
     * operation counting, etc. so currently only executes on worker 0. If
     * needed, can be extended to multithreaded in the future.
     */
    void print_statistics() { workers[0].print_statistics(); }

    void mark_statistics() { workers[0].mark_statistics(); }

    void clear_statistics() { workers[0].clear_statistics(); }

    /**
     * @brief Friend definitions for unchecked_open protection.
     */
    template <typename T, typename U, typename V>
    friend class cdough::SharedVector;
};

}  // namespace cdough::service
