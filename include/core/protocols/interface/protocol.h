#pragma once

#include "core/communication/communicator.h"
#include "core/protocols/interface/protocol_base.h"
#include "core/protocols/interface/protocol_circuits.h"
#include "core/random/manager.h"
#include "debug/cdough_debug.h"
#include "profiling/stopwatch.h"

using namespace cdough::benchmarking;

namespace cdough {

/**
 * @brief Abstract class defining primitive methods for secure protocols.
 *
 * This is the abstract class that defines the primitive methods each secure
 * protocol must implement.
 *
 * @tparam Data Plaintext data type.
 * @tparam Share Share type (e.g., a 32-bit integer, a pair of 64-bit integers, etc.).
 * @tparam Vector Data container type.
 * @tparam EVector Share container type.
 *
 * Primitive operations are grouped as follows:
 *  1. Arithmetic operations on arithmetic shares.
 *  2. Boolean operations on boolean shares.
 *  3. Primitives for sending and receiving shares.
 *  4. Primitives for constructing and opening shares to learners.
 */
template <typename Data, typename Share, typename Vector, typename EVector>
class Protocol : public ProtocolBase {
   public:
    // The communicator
    Communicator* communicator;
    // The randomness manager, in place of the old random generator
    random::RandomnessManager* randomnessManager;

    /**
     * @brief Protocol constructor.
     *
     * @param _communicator A pointer to the communicator.
     * @param _randomnessManager A pointer to the randomness manager.
     * @param _partyID The (globally) unique identifier of the party that calls this constructor.
     * @param _numParties The total number of computing parties participating in the protocol.
     * @param _replicationNumber The protocol's replication factor.
     */
    Protocol(Communicator* _communicator, random::RandomnessManager* _randomnessManager,
             PartyID _partyID, int _numParties, int _replicationNumber)
        : ProtocolBase(_partyID, _numParties, _replicationNumber) {
        this->communicator = _communicator;
        this->randomnessManager = _randomnessManager;
    }
    /// Destructor
    virtual ~Protocol() {}

    /**
     * @brief Reshare shares with other parties.
     *
     * The group rerandomizes the vector v and sends shares to all parties that are not in the
     * group.
     *
     * Operates in place.
     *
     * TODO: fix runtime to support this better.
     *
     * @param v The EVector representing each party's view of the vector to be
     * @param group The group of parties that perform the resharing.
     * @param binary whether this is a binary (true) or arithmetic (false) resharing
     * reshared.
     */
    virtual void reshare(EVector& v, const std::set<int> group, bool binary) {
        auto ra = reshareMap[group];

        if (ra.action == ReshareAction::Send) {
            assert(ra.ranks.size() == ra.shareIdx.size());

            {
                // Scope rand for garbage collection
                std::vector<Vector> rand;
                for (int i = 0; i < numParties; i++) {
                    rand.push_back(Vector(v.size()));
                }

                // Randomize
                if (binary) {
                    this->randomnessManager->zeroSharingGenerator->groupGetNextBinary(rand, group);
                } else {
                    this->randomnessManager->zeroSharingGenerator->groupGetNextArithmetic(rand,
                                                                                          group);
                }

                auto my_shares = getPartyShareMappings()[partyID];
                // Generating too many random values here: each party only needs
                // RepNum random vectors, but is generating PartyNum
                for (int i = 0; i < replicationNumber; i++) {
                    if (binary) {
                        v(i) ^= rand[my_shares[i]];
                    } else {
                        v(i) += rand[my_shares[i]];
                    }
                }
            }

            for (int i = 0; i < ra.ranks.size(); i++) {
                this->communicator->sendShares(v(ra.shareIdx[i]), ra.ranks[i]);
            }
        } else if (ra.action == ReshareAction::Receive) {
            this->communicator->receiveBroadcast(v.contents, ra.ranks);
        } else {
            throw new std::runtime_error("Invalid reshare action.");
        }
    }

    virtual void handle_precision(const EVector& x, const EVector& y, EVector& z) {
        if (x.getPrecision() != y.getPrecision()) {
            throw std::runtime_error("Precision mismatch between multiplication inputs");
        }
        z.matchPrecision(x);
    }

    // **************************************** //
    //          Arithmetic operations           //
    // **************************************** //

    /**
     * @brief Defines vectorized arithmetic addition.
     *
     * @param x The first shared vector of size S.
     * @param y The second shared vector of size S.
     * @param z The output shared vector of size S.
     */
    virtual void add_a(const EVector& x, const EVector& y, EVector& z) { z = x + y; }

    /**
     * @brief Defines vectorized arithmetic subtraction.
     *
     * @param x The first shared vector of size S.
     * @param y The second shared vector of size S.
     * @param z The output shared vector of size S.
     */
    virtual void sub_a(const EVector& x, const EVector& y, EVector& z) { z = x - y; }

    /**
     * @brief Defines vectorized arithmetic multiplication.
     *
     * @param first The first shared vector of size S.
     * @param second The second shared vector of size S.
     * @param result The output shared vector of size S.
     */
    virtual void multiply_a(const EVector& first, const EVector& second, EVector& result) = 0;

    /**
     * @brief Defines vectorized arithmetic negation.
     *
     * @param in The input shared vector of size S.
     * @param out The output shared vector of size S.
     */
    virtual void neg_a(const EVector& in, EVector& out) { out = -in; }

    /**
     * @brief Defines vectorized arithmetic division by constant.
     *
     * @param input The input shared vector.
     * @param c The constant divisor.
     * @return Pair of shared vectors representing the division result.
     */
    virtual std::pair<EVector, EVector> div_const_a(const EVector& input, const Data& c) = 0;

    /**
     * @brief Defines the vectorized dot product operation for consecutive elements.
     *
     * @param x The first shared vector of size S.
     * @param y The second shared vector of size S.
     * @param z The output shared vector.
     * @param aggSize The number of consecutive pairs of elements to aggregate.
     */
    virtual void dot_product_a(const EVector& x, const EVector& y, EVector& z, size_t aggSize) {
        EVector res(x.size());
        multiply_a(x, y, res);
        z = res.chunkedSum(aggSize);
    }

    /**
     * @brief Secure matrix right multiplication with a column matrix, vectorized implementation.
     * Expects the left-hand side matrix to be in row-major order.
     * Expects the right-hand side matrix to be in column-major order.
     *
     * @param x The left-hand side matrix as a shared vector.
     * @param y The right-hand side column matrix as a shared vector.
     * @param z The output shared vector.
     * @param lhs_rows Number of rows in the left-hand side matrix.
     * @param lhs_cols Number of columns in the left-hand side matrix.
     * @param rhs_rows Number of rows in the right-hand side matrix.
     * @param rhs_cols Number of columns in the right-hand side matrix.
     */
    virtual void matrix_right_multiply_with_column_matrix_vectorized_a(
        const EVector& x, const EVector& y, EVector& z, const size_t lhs_rows,
        const size_t lhs_cols, const size_t rhs_rows, const size_t rhs_cols) {
        auto x_size = x.size();
        auto y_size = y.size();

        // Copying inputs to use data views.
        EVector xCopy(x_size);
        EVector yCopy(y_size);

        xCopy = x;
        yCopy = y;

        auto xRef = xCopy.repeated_subset_reference(lhs_cols, rhs_cols);
        auto yRef = yCopy.cyclic_subset_reference(lhs_rows);

        dot_product_a(xRef, yRef, z, lhs_cols);
    }

    /**
     *  @brief Secure 2D convolution.
     *
     * Input layout:
     * The input consists of mutiple instances concatenated after each other.
     * Hence, the input size = instancesCount * inputHeight * inputWidth.
     * Each instance has multiple channels interleaved per spatial location.
     *
     * For example, for 2x2 input with 2 channels:
     * [ch1(0,0), ch2(0,0), ch1(0,1), ch2(0,1),
     * ch1(1,0), ch2(1,0), ch1(1,1), ch2(1,1)]
     *
     * Filter layout: the filter is expected to have multiple channels.
     * Hence, the filter size = channels * filterHeight * filterWidth.
     * For example, the the physical layout for 2x2 filter with 2 channels:
     *
     * [f_ch1(0,0), f_ch2(0,0), f_ch1(1,0), f_ch2(1,0),
     * g_ch1(0,1), g_ch2(0,1), g_ch1(1,1), g_ch2(1,1)]
     *
     * Output layout: (same layout as input but different height and width).
     * The output consists of mutiple instances concatenated after each other.
     * Hence, the output size = instancesCount * outputHeight * outputWidth * channels.
     * Each instance has multiple channels interleaved per spatial location.
     *
     * For example, for 2x2 output with 2 channels:
     * [f(0,0), g(0,0), f(0,1), g(0,1),
     * f(1,0), g(1,0), f(1,1), g(1,1)]
     *
     *
     * @param x The input shared vector.
     * @param y The filter shared vector.
     * @param z The output shared vector.
     * @param instancesCount Number of instances in the input.
     * @param inputHeight Height of each input instance.
     * @param inputWidth Width of each input instance.
     * @param filterHeight Height of each filter channel.
     * @param filterWidth Width of each filter channel.
     * @param strideHeight Stride height.
     * @param strideWidth Stride width.
     * @param paddingHeight Padding height.
     * @param paddingWidth Padding width.
     */
    virtual void conv_2d_vectorized_a(const EVector& x, const EVector& y, EVector& z,
                                      const size_t instancesCount, const size_t inputHeight,
                                      const size_t inputWidth, const size_t filterHeight,
                                      const size_t filterWidth, const size_t strideHeight,
                                      const size_t strideWidth, const size_t paddingHeight,
                                      const size_t paddingWidth) {
        auto size_x = x.size();
        auto size_y = y.size();
        EVector xCopy(size_x);
        EVector yCopy(size_y);

        xCopy = x;
        yCopy = y;

        auto xMaterialized = xCopy.conv2DLeftVectorization(
            instancesCount, inputHeight, inputWidth, filterHeight, filterWidth, strideHeight,
            strideWidth, paddingHeight, paddingWidth);
        auto xMaterializedSize = xMaterialized.size();
        auto yCopySize = yCopy.size();

        auto xCols = filterHeight * filterWidth;
        auto xRows = xMaterializedSize / xCols;
        auto yCols = yCopySize / xCols;

        matrix_right_multiply_with_column_matrix_vectorized_a(xMaterialized, yCopy, z, xRows, xCols,
                                                              xCols, yCols);
    }

    // TODO: remove y
    /**
     * @brief Secure sum pooling.
     * Aggregates values within pooling windows by summing them.
     * Aggregation is performed over multiple instances with multiple channels.
     * Each layer is pooled independently.
     *
     * Input layout:
     * The input consists of mutiple instances concatenated after each other.
     * Hence, the input size = instancesCount * inputHeight * inputWidth * channels.
     * Each instance has multiple channels interleaved per spatial location.
     *
     * For example, for 2x2 input with 2 channels:
     * [ch1(0,0), ch2(0,0), ch1(0,1), ch2(0,1),
     * ch1(1,0), ch2(1,0), ch1(1,1), ch2(1,1)]
     *
     *
     * Output layout: same layout as input but with different height and width.
     *
     *
     * @param x The input shared vector.
     * @param y (unused) Placeholder for compatibility with other protocols.
     * @param z The output shared vector.
     * @param instancesCount Number of instances in the input.
     * @param channels Number of channels in each instance.
     * @param inputHeight Height of each input instance.
     * @param inputWidth Width of each input instance.
     * @param poolHeight Height of the pooling window.
     * @param poolWidth Width of the pooling window.
     * @param strideHeight Stride height.
     * @param strideWidth Stride width.
     * @param paddingHeight Padding height.
     * @param paddingWidth Padding width.
     */
    virtual void sumPoolingVectorized(const EVector& x, const EVector& y, EVector& z,
                                      const size_t instancesCount, const size_t channels,
                                      const size_t inputHeight, const size_t inputWidth,
                                      const size_t poolHeight, const size_t poolWidth,
                                      const size_t strideHeight, const size_t strideWidth,
                                      const size_t paddingHeight, const size_t paddingWidth) {
        z = x.sumPoolingVectorized(instancesCount, channels, inputHeight, inputWidth, poolHeight,
                                   poolWidth, strideHeight, strideWidth, paddingHeight,
                                   paddingWidth);
    }

    /**
     * Defines vectorized arithmetic truncation.
     * This method must take one input vector with arithmetic shares and return
     * a new vector that contains arithmetic shares of the truncated input.
     * The default implementation invokes the protocol's public division protocol.
     *
     * @param x - The shared vector to truncate.
     *
     * @note This default protocol assumes that public division is still secure when
     * the error correction is not applied. This holds for all protocols currently in the system.
     */
    virtual void truncate(EVector& x) {
        if constexpr (!std::integral<Data>) {
            // Only integral types support fixed point (no NTL types)
            return;
        } else {
            int precision = x.getPrecision();
            if (precision == 0) {
                return;
            }

            // compute the public divisor
            Data divisor = 1 << precision;

            // run public division and discard the error
            // by discarding the error, we avoid incurring the a2b cost
            //   at the expense of one bit of error
            // ignoring error correction is thus a good default for truncation
            std::pair<EVector, EVector> ret = this->div_const_a(x, divisor);
            x = ret.first;

            // preserve precision
            x.setPrecision(precision);
        }
    }

    // **************************************** //
    //            Boolean operations            //
    // **************************************** //

    /**
     * @brief Defines vectorized bitwise XOR (^).
     *
     * @param x The first shared vector of size S.
     * @param y The second shared vector of size S.
     * @param z The output shared vector of size S.
     */
    virtual void xor_b(const EVector& x, const EVector& y, EVector& z) { z = x ^ y; }
    virtual void xor_b_1(const EVector& x, const EVector& y, EVector& z) {
        this->xor_b(x, y, z);
    }

    /**
     * @brief Defines vectorized bitwise AND (&).
     *
     * @param first The first shared vector of size S.
     * @param second The second shared vector of size S.
     * @param result The output shared vector of size S.
     */
    virtual void and_b(const EVector& first, const EVector& second, EVector& result) = 0;

    /**
     * @brief Defines vectorized boolean complement (~).
     *
     * @param in The input shared vector of size S.
     * @param out The output shared vector of size S.
     */
    virtual void not_b(const EVector& in, EVector& out) = 0;

    /**
     * @brief Defines vectorized boolean NOT (!).
     *
     * @param in The input shared vector of size S.
     * @param out The output shared vector of size S.
     */
    virtual void not_b_1(const EVector& in, EVector& out) = 0;

    virtual void gtez_a(const EVector& in, EVector& out) {
        // #ifndef USE_OPTIMIZED_A2B_SIGN_FOR_GTZE
        //         B conv = this->a2b();
        //         B sign = !conv.ltz();
        // #else
        static const int MAX_BITS_NUMBER = std::numeric_limits<std::make_unsigned_t<Data>>::digits;
        size_t compressed_size = in.size() / MAX_BITS_NUMBER + ((in.size() % MAX_BITS_NUMBER) > 0);

        EVector packed_sign(compressed_size);
        this->a2b_packed_sign_a_b(in, packed_sign);
        this->not_b(packed_sign, packed_sign);

        auto sign = in.construct_like(in.size());
        this->unpack_from(packed_sign, sign, 0);
        // #endif
        this->b2a_bit(sign, out);
    }

    // **************************************** //
    //       Default boolean operations         //
    // **************************************** //

    /**
     * @brief Defines vectorized logical AND (&&).
     *
     * Assumes data in range between 0 and 1.
     * Otherwise, it returns incorrect results.
     *
     * @param first The first shared vector of size S.
     * @param second The second shared vector of size S.
     * @param result The output shared vector of size S.
     */
    virtual void and_b_1(const EVector& first, const EVector& second, EVector& result) {
        this->and_b(first, second, result);
    }

    /**
     * @brief Defines vectorized bitwise OR (|).
     *
     * @param first The first shared vector of size S.
     * @param second The second shared vector of size S.
     * @param result The output shared vector of size S.
     */
    virtual void or_b(const EVector& first, const EVector& second, EVector& result) {
        EVector tmp1(first.size());
        EVector tmp2(first.size());
        this->not_b(first, tmp1);
        this->not_b(second, tmp2);
        this->and_b(tmp1, tmp2, tmp1);
        this->not_b(tmp1, result);
    }

    /**
     * @brief Defines vectorized logical OR (||).
     *
     * Assumes data in range between 0 and 1.
     * Otherwise, it returns incorrect results.
     *
     * @param first The first shared vector of size S.
     * @param second The second shared vector of size S.
     * @param result The output shared vector of size S.
     */
    virtual void or_b_1(const EVector& first, const EVector& second, EVector& result) {
        this->or_b(first, second, result);
    }

    // operator==
    virtual void equal_b(const EVector& first, const EVector& second, EVector& result) {
        result =
            protocols::circuits::bit_same<Data>(first, second, std::make_optional(result), *this);

        // If the LSB is 1, it means that the respective elements from `this` and `other` are the
        // same
        this->mask(result, (Data)1);
    }

    // compare
    virtual void compare_b(const EVector& first, const EVector& second, EVector& eq_bits,
                           EVector& gt_bits) {
        protocols::circuits::compare<Data>(first, second, eq_bits, gt_bits, *this);
    }

    // rca_compare
    virtual void rca_compare_b(const EVector& first, const EVector& second, EVector& gt_bits) {
        protocols::circuits::rca_compare<Data>(first, second, gt_bits, *this);
    }

    // linear_compare
    virtual void linear_compare_b(const EVector& x, const EVector& y, EVector& eq_bits,
                                  EVector& gt_bits) {
        protocols::circuits::linear_compare<Data>(x, y, eq_bits, gt_bits, *this);
    }

    // ripple_carry_adder
    virtual void ripple_carry_adder_b(const EVector& x, const EVector& y, EVector& res,
                                      const bool carry_in) {
        protocols::circuits::ripple_carry_adder<Data>(x, y, res, carry_in, *this);
    }

    // ripple_carry_adder_packed_sign_b
    virtual void ripple_carry_adder_packed_sign_b(const EVector& x, const EVector& y, EVector& res,
                                                  const bool carry_in) {
        protocols::circuits::ripple_carry_adder_packed_sign<Data>(x, y, res, carry_in, *this);
    }

    // parallel_prefix_adder
    virtual void parallel_prefix_adder_b(const EVector& x, const EVector& y, EVector& res,
                                         const bool& carry_in) {
        protocols::circuits::parallel_prefix_adder<Data>(x, y, res, carry_in, *this);
    }

    virtual void parallel_prefix_adder_packed_sign_b(const EVector& x, const EVector& y, EVector& res,
                                                  const bool carry_in) {
        protocols::circuits::parallel_prefix_adder_packed_sign<Data>(x, y, res, carry_in, *this);
    }

    // TODO: operator::/

    virtual void inplace_invert_b(EVector& x) { this->not_b(x, x); }

    /**
     * @brief Defines vectorized less-than-zero comparison.
     *
     * @param in The input shared vector of size S.
     * @param out The output shared vector of size S.
     */
    virtual void ltz(const EVector& in, EVector& out) { out = in.ltz(); }

    // **************************************** //
    //            Bit manipulation              //
    // **************************************** //
    // TODO: these functions should be default for replication only
    // TODO: they should be tested with the primitives

    // Note: source has more elements than destination
    virtual void pack_from(const EVector& source, EVector& destination, const int& pos) {
        destination.pack_from(source, pos);
    }

    // Note: source has less elements than destination
    virtual void unpack_from(const EVector& source, EVector& destination, const int& pos) {
        destination.unpack_from(source, pos);
    }

    virtual void bit_arithmetic_right_shift(const EVector& in, EVector& out,
                                            const int& shift_size) {
        out = in.bit_arithmetic_right_shift(shift_size);
    }

    virtual void bit_logical_right_shift(const EVector& in, EVector& out, const int& shift_size) {
        out = in.bit_logical_right_shift(shift_size);
    }

    virtual void bit_left_shift(const EVector& in, EVector& out, const int& shift_size) {
        out = in.bit_left_shift(shift_size);
    }

    virtual void bit_xor(const EVector& in, EVector& out) { out = in.bit_xor(); }

    virtual void extend_lsb(const EVector& in, EVector& out) { out = in.extend_lsb(); }

    virtual void mask(EVector& in, const Data mask_value) { in.mask(mask_value); }

    // **************************************** //
    //          Conversion operations           //
    // **************************************** //

    /**
     * @brief Defines vectorized boolean-to-arithmetic single bit conversion.
     *
     * @param in A B-shared vector of S single-bit elements.
     * @param out The output A-shared vector of size S.
     */
    virtual void b2a_bit(const EVector& in, EVector& out) = 0;

    virtual void a2b_packed_sign_a_b(const EVector& in, EVector& out) {
        auto v = this->redistribute_shares_b(in);


        // Getting only the MSB using ripple-carry adder circuit.
        this->ripple_carry_adder_packed_sign_b(v.first, v.second, out, false);
        // this->parallel_prefix_adder_packed_sign_b(v.first, v.second, out, false);
    }

    /**
     * @brief Defines a redistribution of arithmetic secret shares into boolean secret shares.
     *
     * @param x The input arithmetic shared vector.
     * @return Pair of boolean shared vectors representing the redistribution.
     */
    virtual std::pair<EVector, EVector> redistribute_shares_b(const EVector& x) = 0;

    // **************************************** //
    //          Reconstruction operations       //
    // **************************************** //

    /**
     * @brief Defines how to reconstruct a single data value by adding its arithmetic shares.
     *
     * @param shares The input vector containing arithmetic shares of the secret value.
     * @return The plaintext value of type Data.
     *
     * NOTE: This method is useful when a computing party also acts as learner that receives
     * arithmetic shares from other parties and needs to reconstruct a true value.
     */
    virtual Data reconstruct_from_a(const std::vector<Share>& shares) = 0;

    /**
     * @brief Vectorized version of the reconstruct_from_a() method.
     *
     * @param shares A vector of shared vectors (each one of size n) that contain arithmetic
     shares.
     * @return A new vector that contains n plaintext values of type Data.
     *
     * NOTE: This method is useful when a computing party also acts as learner that receives
     * arithmetic shared vectors from other parties and needs to reconstruct the original vector.
     */
    virtual Vector reconstruct_from_a(const std::vector<EVector>& shares) = 0;

    /**
     * @brief Defines how to reconstruct a single data value by XORing its boolean shares.
     *
     * @param shares The input vector containing boolean shares of the secret value.
     * @return The plaintext value of type Data.
     *
     * NOTE: This method is useful when a computing party also acts as learner that receives
     * boolean shares from other parties and needs to reconstruct a true value.
     */
    virtual Data reconstruct_from_b(const std::vector<Share>& shares) = 0;

    /**
     * @brief Vectorized version of the reconstruct_from_b() method.
     *
     * @param shares A vector of shared vectors (each one of size n) that contain boolean shares.
     * @return A new vector that contains n plaintext values of type Data.
     *
     * NOTE: This method is useful when a computing party also acts as learner that receives
     * boolean shared vectors from other parties and needs to reconstruct the original vector.
     */
    virtual Vector reconstruct_from_b(const std::vector<EVector>& shares) = 0;

    // **************************************** //
    //            Opening operations            //
    // **************************************** //
    /**
     * @brief Opens arithmetic shares to reveal plaintext values. For malicious protocols, run a
     * check before (to prevent leakage) and after (to confirm proper opening). In test mode,
     * returns empty vectors instead of aborting.
     *
     * @param shares A shared vector that contains arithmetic shares of the secret values.
     * @return A new vector that contains the plaintext values of type Data.
     */
    Vector open_shares_a(const EVector& shares) {
        // In normal executions, check aborts on error
        // But in MAL_TEST_MODE, does not abort, so we call open with a fresh (zero) vector.
        // We need to still run the opening protocol because other parties might not know about the
        // check failure, and will deadlock.
        // Also run a malicious check after, to confirm the opening succeeded.
        auto r = this->malicious_check() ? shares : EVector(shares.size());
        auto op = unchecked_open_a(r);
        return this->malicious_check() ? op : Vector(shares.size());
    }

    /**
     * @brief Opens boolean shares to reveal plaintext values. For malicious protocols, run a check
     * first.
     *
     * @param shares A shared vector that contains boolean shares of the secret values.
     * @return A new vector that contains the plaintext values of type Data.
     *
     * NOTE: This method is useful when computing parties need to reveal a secret-shared vector
     * to each other.
     */
    Vector open_shares_b(const EVector& shares) {
        auto r = this->malicious_check() ? shares : EVector(shares.size());
        auto op = unchecked_open_b(r);
        return this->malicious_check() ? op : Vector(shares.size());
    }

    virtual Vector unchecked_open_a(const EVector& shares) = 0;
    virtual Vector unchecked_open_b(const EVector& shares) = 0;

    // **************************************** //
    //        Share generation operations       //
    // **************************************** //

    /**
     * @brief Compute secret shares for a vector of plaintext values.
     *
     * @param data A vector of input values of type Data.
     * @return A vector of shared vectors containing arithmetic shares.
     */
    virtual std::vector<EVector> get_shares_a(const Vector& data) = 0;

    /**
     * @brief Compute secret shares for a vector of plaintext values.
     *
     * @param data A vector of input values of type Data.
     * @return A vector of shared vectors containing boolean shares.
     */
    virtual std::vector<EVector> get_shares_b(const Vector& data) = 0;

    /**
     * @brief Compute secret shares for a vector of plaintext values.
     *
     * @param data The plaintext vector that must be secret-shared among computing parties.
     * @param data_party The party that owns the data.
     * @return The boolean shared vector
     *
     * NOTE: This method is useful for secret-sharing plaintext data in cdough programs.
     */
    virtual EVector secret_share_b_internal(const Vector& data, const PartyID& data_party) = 0;

    /**
     * @brief Compute secret shares for a vector of plaintext values. Overloaded
     * secret_share_b_internal function that establishes a non-zero fixed-point precision.
     *
     * @param data The plaintext vector that must be secret-shared among computing parties.
     * @param data_party The party that owns the data.
     * @param fixed_point_precision precision of the input values
     * @return The boolean shared fixed-point vector
     */
    virtual EVector secret_share_b_internal(const Vector& data, const PartyID& data_party,
                                            const int& fixed_point_precision) {
        EVector ret = secret_share_b_internal(data, data_party);
        ret.setPrecision(fixed_point_precision);
        return ret;
    }

    /**
     * Defines how to A-share a plaintext vector according to a protocol.
     * @param data - The plaintext vector that must be secret-shared among
     * computing parties.
     * @param data_party data ownner
     * @return The arithmetic shared vector of the party that calls this method.
     *
     * NOTE: This method is useful for secret-sharing plaintext data in cdough programs.
     */
    virtual EVector secret_share_a_internal(const Vector& data, const PartyID& data_party) = 0;

    /**
     * @brief Compute secret shares for a vector of plaintext values. Overloaded
     * secret_share_a_internal function that establishes a non-zero fixed-point precision.
     *
     * @param data The plaintext vector that must be secret-shared among computing parties.
     * @param data_party The party that owns the data.
     * @param fixed_point_precision precision of the input values
     * @return The arithmetic shared fixed-point vector
     */
    virtual EVector secret_share_a_internal(const Vector& data, const PartyID& data_party,
                                            const int& fixed_point_precision) {
        EVector ret = secret_share_a_internal(data, data_party);
        ret.setPrecision(fixed_point_precision);
        return ret;
    }

    /**
     * @brief Create a "secret" share of a public value, `x`. This
     * implemented by setting one share to `x` and all others to zero;
     * this gives a valid sharing under both arithmetic and boolean.
     *
     * @param x The public vector to share.
     * @param who_knows a party who knows the value
     * @return The shared vector.
     */
    virtual EVector public_share(const Vector& x, const std::set<PartyID>& who_knows) = 0;

    /**
     * @brief Convenience overload that defaults to an empty `who_knows` set.
     */
    virtual EVector public_share(const Vector& x) { return public_share(x, std::set<PartyID>{}); }

    /**
     * An overloaded public_share function that establishes a non-zero
     * fixed-point precision.
     * @param x
     * @param fixed_point_precision
     * @return EVector
     */
    virtual EVector public_share(const Vector& x, const std::set<PartyID>& who_knows,
                                 const int& fixed_point_precision) {
        EVector ret = public_share(x, who_knows);
        ret.setPrecision(fixed_point_precision);
        return ret;
    }
};
}  // namespace cdough
