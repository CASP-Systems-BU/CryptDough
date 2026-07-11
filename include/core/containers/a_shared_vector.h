#pragma once

#include "core/math/util.h"
#include "core/operators/circuits.h"
#include "shared_vector.h"

// // Note: another approach that WORKS ONLY IF WE `other - (*this)` do not overflow.
// // we need (this > other)
// // which is (0 > other - this)
// // which is (other - this < 0)
// ASharedVector z = other - (*this);
// B z_b = z.a2b();
// B res = z_b.ltz();
#define define_boolean_based_Arth_Comparison(_op_)                                   \
    std::unique_ptr<ASharedVector> operator _op_(const ASharedVector& other) const { \
        B a_ = this->a2b();                                                          \
        B b_ = other.a2b();                                                          \
        B res = a_ _op_ b_;                                                          \
                                                                                     \
        return std::make_unique<ASharedVector>(res.b2a_bit());                       \
    }

namespace cdough {

/**
 * A SharedVector that contains arithmetic shares and supports secure arithmetic operations.
 * @tparam Share Share data type.
 * @tparam EVector Share container type.
 * @tparam Engine Secure computation engine type.
 */
template <typename Share, typename EVector, typename Engine>
class ASharedVector : public SharedVector<Share, EVector, Engine> {
   public:
    using EngineType = Engine;

    /**
     * Creates an ASharedVector of size `_size` and initializes it with zeros.
     *
     * Note: This constructor is deprecated from public API [last resort].
     *
     * @param _size The size of the ASharedVector.
     * @param eng The secure computation engine.
     */
    explicit ASharedVector(size_t _size, Engine& eng)
        : SharedVector<Share, EVector, Engine>(_size, Encoding::AShared, eng) {}

    /**
     * Creates an ASharedVector of size `_size` and initializes it with secret shares in the given
     * file.
     * @param _size The size of the ASharedVector.
     * @param _input_file The file that contains the secret shares.
     * @param eng The secure computation engine.
     */
    explicit ASharedVector(size_t _size, const std::string& _input_file, Engine& eng)
        : SharedVector<Share, EVector, Engine>(_size, _input_file, Encoding::AShared, eng) {}

    /**
     * This is a shallow copy constructor from EVector contents.
     * @param _shares The EVector whose contents will be pointed by the ASharedVector.
     * @param eng The secure computation engine.
     */
    explicit ASharedVector(EVector& _shares, Engine& eng)
        : SharedVector<Share, EVector, Engine>(_shares, Encoding::AShared, eng) {}

    /**
     * This is a move constructor from EVector contents.
     * @param _shares The EVector whose contents will be moved to the new ASharedVector.
     * @param eng The secure computation engine.
     */
    ASharedVector(EVector&& _shares, Engine& eng)
        : SharedVector<Share, EVector, Engine>(_shares, Encoding::AShared, eng) {}

    /**
     * This is a move constructor from another ASharedVector.
     * @param other The ASharedVector whose contents will be moved to the new ASharedVector.
     */
    ASharedVector(ASharedVector&& other) noexcept
        : SharedVector<Share, EVector, Engine>(other.vector, other.encoding, other.engine) {}

    /**
     * This is a copy constructor from another ASharedVector.
     * @param other The ASharedVector whose contents will be moved to the new ASharedVector.
     */
    ASharedVector(const ASharedVector& other)
        : SharedVector<Share, EVector, Engine>(other.vector, other.encoding, other.engine) {}

    /**
     * Copy constructor from SharedVector contents.
     * @param _shares The SharedVector object whose contents will be copied to the new
     * ASharedVector.
     */
    explicit ASharedVector(SharedVector<Share, EVector, Engine>& _shares)
        : SharedVector<Share, EVector, Engine>(_shares.vector, _shares.encoding, _shares.engine) {
        assert(_shares.encoding == Encoding::AShared);
        auto secretShares_ = reinterpret_cast<ASharedVector*>(&_shares);
    }

    /**
     * Move constructor that creates an ASharedVector from a unique pointer to an EncodedVector
     * object.
     * @param base The pointer to the SharedVector object whose contents will be moved to the new
     * ASharedVector.
     */
    ASharedVector(std::unique_ptr<ASharedVector>&& base)
        : ASharedVector((ASharedVector*)base.get()) {}

    /**
     * Shallow copy constructor that creates an ASharedVector from a unique pointer to an
     * EncodedVector object.
     * @param base The SharedVector object whose contents will be pointed by the new
     * ASharedVector.
     */
    ASharedVector(std::unique_ptr<ASharedVector>& base)
        : ASharedVector((ASharedVector*)base.get()) {}

    /**
     * Move constructor that creates an ASharedVector from a pointer to another ASharedVector
     * object.
     * @param _base The ASharedVector that will be moved as a whole (contents + state) to the new
     * ASharedVector.
     *
     * NOTE: This constructor is implicitly called by the two constructors above.
     */
    explicit ASharedVector(ASharedVector* _base) : ASharedVector(std::move(*_base)) {}

    /**
     * @brief Construct a new ASharedVector object like another ASharedVector but with a different
     * size.
     *
     * @param size The size of the new ASharedVector.
     * @return ASharedVector The newly constructed ASharedVector.
     */
    ASharedVector construct_like(std::optional<size_t> size = {}) const {
        auto new_size = size.value_or(this->size());
        return ASharedVector(this->vector.construct_like(new_size), this->engine);
    }

    /**
     * @brief Use the underlying SharedVector's implementation of `operator=`.
     *
     */
    using SharedVector<Share, EVector, Engine>::operator=;

    ASharedVector& operator=(const ASharedVector&) = default;
    ASharedVector& operator=(ASharedVector&&) = default;

    // Destructor
    virtual ~ASharedVector() {}

    /**
     * @brief Reuse underlying SharedVector implementation for access patterns.
     *
     */
    svector_reference(ASharedVector, simple_subset_reference);
    svector_reference(ASharedVector, alternating_subset_reference);
    svector_reference(ASharedVector, reversed_alternating_subset_reference);
    svector_reference(ASharedVector, repeated_subset_reference);
    svector_reference(ASharedVector, cyclic_subset_reference);
    svector_reference(ASharedVector, directed_subset_reference);
    svector_reference(ASharedVector, included_reference);
    svector_reference(ASharedVector, mapping_reference);
    svector_reference(ASharedVector, slice);
    svector_reference(ASharedVector, matrixSeparateChannels);
    svector_reference(ASharedVector, matrixInterleaveChannels);
    svector_reference(ASharedVector, conv2DLeftVectorization);

    /**
     * @brief Type alias for an equivalent BSharedVector.
     *
     */
    using B = BSharedVector<Share, EVector, Engine>;

    /**
     * @brief Convert from ASharedVector to BSharedVector. Each party redistributes boolean shares
     * of their additive shares, then uses a boolean addition circuit to "add" those shares back
     * together. In the end we are left with an XOR-sharing of the same value.
     *
     * Other protocols, such as using preprocessed shared bits, are possible.
     *
     */
    std::unique_ptr<B> a2b() const {
        if constexpr (CONFIDENTIAL_1PC) {
            // No point doing conversion!
            return std::make_unique<B>(this->asContainer(), this->engine);
        }

        if constexpr (SPDZ2k_NPC_PROTOCOL) {
            // Same protocol.
            return std::make_unique<B>(this->asContainer(), this->engine);
        }

        auto v = this->engine.redistribute_shares_b(this->vector);
        B v1(std::move(v.first), this->engine);
        B v2(std::move(v.second), this->engine);

        // This `+` here is actually a call to our binary adder circuit.
        auto res = v1 + v2;
        res->setPrecision(this->getPrecision());
        return res;
    }

    /**
     * @brief Convert from ASharedVector to BSharedVector, but only for the most significant bit
     * of each element (sign).
     *
     * Note: size of the resulting BSharedVector is n/MAX_BITS_NUMBER.
     *
     * @return std::unique_ptr<B> the resulting BSharedVector containing only the MSBs.
     */
    std::unique_ptr<B> a2b_packed_sign() const {
        static const int MAX_BITS_NUMBER = std::numeric_limits<std::make_unsigned_t<Share>>::digits;
        size_t compressed_size = this->vector.size() / MAX_BITS_NUMBER + ((this->vector.size() % MAX_BITS_NUMBER) > 0);

        // Getting only the MSB using ripple-carry adder circuit.
        auto res = std::make_unique<B>(compressed_size, this->engine);
        this->engine.a2b_packed_sign_a_b(this->vector, res->vector);

        res->setPrecision(this->getPrecision());
        return res;
    }

    // **************************************** //
    //           Arithmetic operators           //
    // **************************************** //

    /**
     * Elementwise secure arithmetic addition. Returns a unique ptr.
     */
    binary_op(+, ASharedVector, add_a, this, other);

    /**
     * Elementwise secure arithmetic subtraction. Returns a unique ptr.
     */
    binary_op(-, ASharedVector, sub_a, this, other);

    /**
     * Elementwise secure arithmetic multiplication. Returns a unique ptr.
     */
    binary_op(*, ASharedVector, multiply_a, this, other);

    /**
     * Elementwise secure arithmetic negation. Returns a unique ptr.
     */
    unary_op(-, ASharedVector, neg_a, this);

    binary_element_op(+, add_a, ASharedVector, Share);
    binary_element_op(-, add_a, ASharedVector, Share);
    binary_element_op(*, add_a, ASharedVector, Share);

    binary_element_op(>, add_a, ASharedVector, Share);
    binary_element_op(<, add_a, ASharedVector, Share);
    binary_element_op(>=, add_a, ASharedVector, Share);
    binary_element_op(<=, add_a, ASharedVector, Share);
    binary_element_op(==, add_a, ASharedVector, Share);
    binary_element_op(!=, add_a, ASharedVector, Share);

    /**
     * @brief Multiply by a public floating-point constant (local, no MPC protocol).
     * Scales the float to a fixed-point integer using this vector's precision,
     * multiplies each share locally, and truncates.
     */
    template <std::floating_point FP>
    std::unique_ptr<ASharedVector> operator*(FP y) const {
        int p = this->getPrecision();
        Share scaled =
            static_cast<Share>(std::llround(static_cast<long double>(y) * std::ldexp(1.0L, p)));
        auto result_ev = this->vector * scaled;
        if (p > 0) {
            result_ev = result_ev / (Share(1) << p);
        }
        return std::make_unique<ASharedVector>(std::move(result_ev), this->engine);
    }

    compound_assignment_op(+=, add_a, ASharedVector);
    compound_assignment_op(-=, sub_a, ASharedVector);
    compound_assignment_op(*=, multiply_a, ASharedVector);

    compound_assignment_element_op(+=, add_a, ASharedVector, Share);
    compound_assignment_element_op(-=, sub_a, ASharedVector, Share);
    compound_assignment_element_op(*=, multiply_a, ASharedVector, Share);

    /**
     * @brief Division by public constant. Call the underlying protocol's `div_const_a`
     * functionality, then perform error correction (if configured to do so using the compiler
     * directive USE_DIVISION_CORRECTION).
     *
     * @param c
     * @return std::unique_ptr<ASharedVector>
     */
    std::unique_ptr<ASharedVector> operator/(const Share& c) const {
        // compute division and error correction shares
        auto out = this->engine.div_const_a(this->vector, c);
        out.first.setPrecision(this->getPrecision());
        out.second.setPrecision(this->getPrecision());

#if defined(USE_DIVISION_CORRECTION) && !defined(MPC_PROTOCOL_SPDZ_2K_NPC)
        auto out_res = ASharedVector(std::move(out.first), this->engine);
        auto out_err = ASharedVector(std::move(out.second), this->engine);

        // Convert a2b, check < 0, and convert back
        auto correction = (!(out_err.a2b()->ltz()))->b2a_bit();

        // These are AShares, so this addition is local.
        return out_res + correction;
#else
        // If correct is disabled, ignore the error term.
        return std::make_unique<ASharedVector>(std::move(out.first), this->engine);
#endif
    }

    /**
     * @brief Auto-conversion private elementwise division. Since our current integer division
     * algorithm only supports BSharedVector inputs, convert `this` and `other` to binary before
     * calling `BSharedVector::operator/`. Do not convert the result back.
     *
     * @param other
     * @return std::unique_ptr<BSharedVector>
     */
    std::unique_ptr<BSharedVector<Share, EVector, Engine>> operator/(
        const ASharedVector& other) const {
        return this->a2b() / other.a2b();
    }

    /**
     * Computes the dot product of this vector with another vector.
     * Each `aggSize` consecutive elements contribute to an exactly on dot product element in the
     * result. The size of the resulting vector is determined by the `aggSize` parameter.
     *
     * NOTE: This function is efficient when doing dot products with small `aggSize` values,
     * For larger `aggSize` values including the entire vector in the dot product, we will need to
     * to do batching different in the engine. Currently, whole vector dot product will be executed
     * single-threaded because each threads takes multiple.
     *
     * @param other The second operand of the dot product.
     * @param aggSize The size of each dot product.
     * @return A unique pointer to a new ASharedVector that contains the result of the dot product.
     */
    std::unique_ptr<ASharedVector> dot_product(const ASharedVector& other,
                                               const size_t aggSize) const {
        // Number of elements in the dot product must match.
        assert(this->vector.size() == other.vector.size());

        // Compute the size of the resulting vector.
        auto newSize = math::div_ceil(this->vector.size(), aggSize);
        auto res = std::make_unique<ASharedVector>(newSize, this->engine);
        res->setPrecision(this->getPrecision());

        // Compute the dot product
        this->engine.dot_product_a(this->vector, other.vector, res->vector, aggSize);

        // Return the result as a new ASharedVector
        return res;
    }

    /**
     * @brief Secure matrix right multiplication with a column matrix, vectorized implementation.
     * Expects the left-hand side matrix to be in row-major order.
     * Expects the right-hand side matrix to be in column-major order.
     *
     * @param other The right-hand side column matrix.
     * @param lhs_rows Number of rows in the left-hand side matrix.
     * @param lhs_cols Number of columns in the left-hand side matrix.
     * @param rhs_rows Number of rows in the right-hand side matrix.
     * @param rhs_cols Number of columns in the right-hand side matrix.
     * @return std::unique_ptr<ASharedVector>
     */
    std::unique_ptr<ASharedVector> matrixRightMultiplyWithColumnMatrixVectorized(
        const ASharedVector& other, const size_t lhs_rows, const size_t lhs_cols,
        const size_t rhs_rows, const size_t rhs_cols) const {
        assert(lhs_cols == rhs_rows);

        // Compute the size of the resulting vector.
        const size_t output_rows = lhs_rows;
        const size_t output_cols = rhs_cols;
        const size_t outputSize = output_rows * output_cols;
        auto res = std::make_unique<ASharedVector>(outputSize, this->engine);
        res->setPrecision(this->getPrecision());

        // Compute the matrix multiplication
        this->engine.matrix_right_multiply_with_column_matrix_vectorized_a(
            this->vector, other.vector, res->vector, lhs_rows, lhs_cols, rhs_rows, rhs_cols);

        // Return the result as a new ASharedVector
        return res;
    }

    /**
     * @brief Secure 2D convolution, vectorized implementation.
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
     * @param other The filter matrix.
     * @param instancesCount Number of instances in the input batch.
     * @param inputHeight Height of each input instance.
     * @param inputWidth Width of each input instance.
     * @param filterHeight Height of each filter channel.
     * @param filterWidth Width of each filter channel.
     * @param strideHeight Stride height.
     * @param strideWidth Stride width.
     * @param paddingHeight Padding height.
     * @param paddingWidth Padding width.
     * @return std::unique_ptr<ASharedVector>
     */
    std::unique_ptr<ASharedVector> conv2DVectorized(
        const ASharedVector& other, const size_t instancesCount, const size_t inputHeight,
        const size_t inputWidth, const size_t filterHeight, const size_t filterWidth,
        const size_t strideHeight, const size_t strideWidth, const size_t paddingHeight,
        const size_t paddingWidth) const {
        // We have at least one instance
        assert(instancesCount > 0);

        // We have valid input dimensions
        assert(this->size() == instancesCount * inputHeight * inputWidth);

        // Filter size is divisible not equal because it has multiple channels
        assert((other.size() % (filterHeight * filterWidth)) == 0);

        // Compute the size of the resulting vector.
        const auto outputInstanceRows =
            (inputHeight + 2 * paddingHeight - filterHeight) / strideHeight + 1;
        const auto outputInstanceCols =
            (inputWidth + 2 * paddingWidth - filterWidth) / strideWidth + 1;
        const auto channelsSize = other.size() / (filterHeight * filterWidth);
        auto newSize = instancesCount * outputInstanceRows * outputInstanceCols * channelsSize;
        auto res = std::make_unique<ASharedVector>(newSize, this->engine);
        res->setPrecision(this->getPrecision());

        // Compute the 2D convolution
        this->engine.conv_2d_vectorized_a(this->vector, other.vector, res->vector, instancesCount,
                                          inputHeight, inputWidth, filterHeight, filterWidth,
                                          strideHeight, strideWidth, paddingHeight, paddingWidth);

        // Return the result as a new ASharedVector
        return res;
    }

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
     * @param instancesCount Number of instances in the input batch.
     * @param channels Number of channels in each instance.
     * @param inputHeight Height of each input instance.
     * @param inputWidth Width of each input instance.
     * @param poolHeight Height of the pooling window.
     * @param poolWidth Width of the pooling window.
     * @param strideHeight Stride height.
     * @param strideWidth Stride width.
     * @param paddingHeight Padding height.
     * @param paddingWidth Padding width.
     *
     * @return std::unique_ptr<ASharedVector> The resulting pooled ASharedVector.
     *
     */
    std::unique_ptr<ASharedVector> sumPoolingVectorized(
        const size_t instancesCount, const size_t channels, const size_t inputHeight,
        const size_t inputWidth, const size_t poolHeight, const size_t poolWidth,
        const size_t strideHeight, const size_t strideWidth, const size_t paddingHeight,
        const size_t paddingWidth) const {
        // Compute the size of the resulting vector.
        const auto outputInstanceRows =
            (inputHeight + 2 * paddingHeight - poolHeight) / strideHeight + 1;
        const auto outputInstanceCols =
            (inputWidth + 2 * paddingWidth - poolWidth) / strideWidth + 1;
        auto newSize = instancesCount * outputInstanceRows * outputInstanceCols * channels;
        auto res = std::make_unique<ASharedVector>(newSize, this->engine);
        res->setPrecision(this->getPrecision());

        // Compute the sum pooling
        this->engine.sumPoolingVectorized(this->vector, this->vector, res->vector, instancesCount,
                                          channels, inputHeight, inputWidth, poolHeight, poolWidth,
                                          strideHeight, strideWidth, paddingHeight, paddingWidth);

        // Return the result
        return res;
    }

    /**
     * @brief Negate this vector in place.
     *
     */
    void inplace_invert() { this->engine.neg_a(this->vector, this->vector); }

    /**
     * @brief Secure comparison: greater than.
     *  Note: If you do not need secure multiplication/addition after comparison, consider using
     *  BSharedVector instead. It uses conversion and BSharedVector's comparison operator under the
     *  hood. It keeps the results in the ASharedVector world after comparison.
     *
     * @param other The other ASharedVector to compare against.
     * @return std::unique_ptr<ASharedVector>
     */
    define_boolean_based_Arth_Comparison(>);
    define_boolean_based_Arth_Comparison(<);
    define_boolean_based_Arth_Comparison(>=);
    define_boolean_based_Arth_Comparison(<=);
    define_boolean_based_Arth_Comparison(==);
    define_boolean_based_Arth_Comparison(!=);

    /**
     * @brief Secure comparison: greater than or equal to zero.
     *  Note: If you do not need secure multiplication/addition after comparison, consider using
     *  BSharedVector instead. It uses conversion and BSharedVector's comparison operator under the
     *  hood. It keeps the results in the ASharedVector world after comparison.
     *
     * @return std::unique_ptr<ASharedVector>
     */
    std::unique_ptr<ASharedVector> gtez() const {
        // #ifndef USE_OPTIMIZED_A2B_SIGN_FOR_GTZE
        //         B conv = this->a2b();
        //         B res = !conv.ltz();
        // #else
        //         B conv = ~(*this->a2b_packed_sign());
        //         B res(this->size(), this->engine);
        //         res.unpack_from(conv, 0);
        // #endif
        //         auto result = std::make_unique<ASharedVector>(res.b2a_bit());
        //         result->setPrecision(this->getPrecision());
        //         return result;
        auto sign = std::make_unique<ASharedVector>(this->size(), this->engine);
        this->engine.gtez_a(this->vector, sign->vector);
        sign->setPrecision(this->getPrecision());
        return sign;
    }

    /**
     * @brief Compute chunked sums over groups of consecutive elements.
     * Simply calls down to the underlying EVector's chunkedSum method.
     *
     * @param aggSize The number of elements to aggregate in each sum.
     * @return ASharedVector containing the chunked sums.
     */
    ASharedVector chunkedSum(const size_t aggSize = 0) const {
        return ASharedVector(this->vector.chunkedSum(aggSize), this->engine);
    }
};

}  // namespace cdough
