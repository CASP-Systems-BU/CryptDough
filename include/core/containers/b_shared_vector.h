#pragma once

#include <bit>
#include <cmath>
#include <limits>

#include "core/operators/aggregation.h"
#include "core/operators/common.h"
#include "debug/cdough_debug.h"
#include "shared_vector.h"

namespace cdough {
// Forward class declarations
template <typename EVector, typename Engine>
class ElementwisePermutation;


/**
 * @brief Alias for a unique pointer to a BSharedVector.
 *
 * @tparam S
 * @tparam E
 */
template <typename S, typename E, typename Eng>
using unique_B = std::unique_ptr<BSharedVector<S, E, Eng>>;

/**
 * A SharedVector that contains boolean shares and supports secure boolean operations.
 * @tparam Share Share data type.
 * @tparam EVector Share container type.
 * @tparam Engine Secure computation engine type.
 */
template <typename Share, typename EVector, typename Engine>
class BSharedVector : public SharedVector<Share, EVector, Engine> {
    using unique_B = std::unique_ptr<BSharedVector>;

    double model_rca(double latency, double bandwidth, size_t n) const {
        // each logical round is split into batches
        auto rounds = (MAX_BITS_NUMBER - 1);
        rounds *= this->engine.numBatches(n);

        return rounds * (latency + n / bandwidth);
    }

    double model_ppa(double latency, double bandwidth, size_t n) const {
        auto elm_per_sec = bandwidth / MAX_BITS_NUMBER;
        // 1 + 3 lg(n)
        auto rounds = 1 + 3 * std::bit_width(MAX_BITS_NUMBER - 1);
        rounds *= this->engine.numBatches(n);

        // No scaling here, because no bit packing
        return rounds * (latency + n / elm_per_sec);
    }

public:
    unique_B rca (const BSharedVector<Share, EVector, Engine>& b, const bool carry_in = false) const {
        auto res = std::make_unique<BSharedVector<Share, EVector, Engine>>(this->size(), this->engine);
        res->setPrecision(this->getPrecision());
        this->engine.ripple_carry_adder_b(this->vector, b.vector, res->vector, carry_in);
        return res;
    }

    unique_B rca_packed_sign(const BSharedVector<Share, EVector, Engine>& b, bool carry_in = false) const {
        static const int MAX_BITS_NUMBER = std::numeric_limits<std::make_unsigned_t<Share>>::digits;
        size_t compressed_size = this->size() / MAX_BITS_NUMBER + ((this->size() % MAX_BITS_NUMBER) > 0);

        auto res = std::make_unique<BSharedVector<Share, EVector, Engine>>(compressed_size, this->engine);
        this->engine.ripple_carry_adder_packed_sign_b(this->vector, b.vector, res->vector, carry_in);
        return res;
    }

    unique_B ppa(const BSharedVector<Share, EVector, Engine>& b, const bool carry_in = false) const {
        auto res = std::make_unique<BSharedVector<Share, EVector, Engine>>(this->size(), this->engine);
        res->setPrecision(this->getPrecision());
        this->engine.parallel_prefix_adder_b(this->vector, b.vector, res->vector, carry_in);
        return res;
    }

    unique_B binary_adder(const BSharedVector<Share, EVector, Engine>& a,
                          const BSharedVector<Share, EVector, Engine>& b, bool carry_in) const {
#ifndef FORCE_PARALLEL_PREFIX_ADDER
        const double latency_sec = 1e-3 * this->engine.comm0()->getLatency();
        const double bandwidth_bps = 1e9 * this->engine.comm0()->getBandwidth();

        size_t n = a.size();

        auto t_rca = model_rca(latency_sec, bandwidth_bps, n);
        auto t_ppa = model_ppa(latency_sec, bandwidth_bps, n);

        // Output the model parameters
        // single_cout(t_rca << " RCA vs. " << t_ppa << " PPA");

        if (t_rca <= t_ppa) {
            return a.rca(b, carry_in);
        } else
#endif
        {
            return a.ppa(b, carry_in);
        }
    }

   public:
    static const size_t MAX_BITS_NUMBER = std::numeric_limits<std::make_unsigned_t<Share>>::digits;
    static const size_t LOG_MAX_BITS_NUMBER = std::bit_width(MAX_BITS_NUMBER - 1);

    using SharedVector_t = SharedVector<Share, EVector, Engine>;
    using EngineType = Engine;

    /**
     * @brief Bit packing. Operates in place by packing the bit at index `position`
     * from BSharedVector `source` into `this` BSharedVector.
     *
     * @param source
     * @param position
     */
    void pack_from(const BSharedVector& source, const int& position) {
        this->engine.pack_from(source.vector, this->vector, position);
    }

    /**
     * @brief Bit unpacking. Operates in place by unpacking the bit at index `position`
     * from BSharedVector `source` into `this`.
     *
     * @param source
     * @param position
     */
    void unpack_from(const BSharedVector& source, const int& position) {
        // TODO: check batching split elements, resulting in larger input
        this->engine.unpack_from(source.vector, this->vector, position);
    }

    /** @brief Decomposes each element of this BSharedVector into its bits, returning a vector of
     * BSharedVectors, each containing one bit position across all elements.
     *
     * @return A vector of BSharedVectors, where the i-th BSharedVector contains the i-th bits of
     * each element in `this`.
     */
    std::vector<BSharedVector> bit_decomposition() const {
        auto res = this->vector.bit_decomposition();
        std::vector<BSharedVector> result;
        for (auto& r : res) {
            result.push_back(BSharedVector(r, this->engine));
        }
        result->setPrecision(this->getPrecision());
        return result;
    }

    /**
     * Arithmetic right shift. Operates in place by putting the output into `this`. Respects the
     * sign bit.
     *
     * @param in input
     * @param shift_size The number of bits to right-shift each element of `this` vector.
     */
    void bit_arithmetic_right_shift(const BSharedVector& in, const int& shift_size) {
        this->engine.bit_arithmetic_right_shift(in.vector, this->vector, shift_size);
    }

    /**
     * Logical right shift. Operates in place. Does not respect the sign bit.
     *
     * @param in input
     * @param shift_size The number of bits to right-shift each element of `this` vector.
     */
    void bit_logical_right_shift(const BSharedVector& in, const int& shift_size) {
        this->engine.bit_logical_right_shift(in.vector, this->vector, shift_size);
    }

    /**
     * @brief Bit left shift. Operates in place. Does not respect the sign bit.
     *
     * @param in
     * @param shift_size
     */
    void bit_left_shift(const BSharedVector& in, const int& shift_size) {
        this->engine.bit_left_shift(in.vector, this->vector, shift_size);
    }

    /**
     * @brief Compute the parity of the input BSharedVector and place the result into `this`.
     *
     * @param in input shared vector
     */
    void bit_xor(const BSharedVector& in) { this->engine.bit_xor(in.vector, this->vector); }

    /**
     * Elementwise LSB extension. This method operates in place, copying the LSB of each `in`
     * element into `this`.
     *
     * @param in
     */
    void extend_lsb(const BSharedVector& in) { this->engine.extend_lsb(in.vector, this->vector); }

    /**
     * Returns the popcount (AKA Hamming weight) of `this` as a `std::unique_ptr` to a
     * `ASharedVector`. Uses a round-optimal concatenation approach, unpacking `this` into
     * the LSB of a new `BSharedVector` and then taking its `chunkedSum`.
     *
     * @return A pointer to a new ASharedVector with the popcount of `this`
     */
    auto popcount() const {
        auto res =
            std::make_unique<ASharedVector<Share, EVector, Engine>>(this->size(), this->engine);
        auto& engine = this->engine;
        size_t size = this->size(), bits = this->MAX_BITS_NUMBER;

        BSharedVector<Share, EVector, Engine> concat(size * bits, engine);
        concat.unpack_from(*this, 0);
        auto conv = concat.b2a_bit();

        *res = conv->chunkedSum(bits);
        res->setPrecision(this->getPrecision());
        return res;
    }

    /**
     * Creates a BSharedVector of size `_size` and initializes it with zeros.
     *
     * Note: This constructor is deprecated from public API [last resort].
     *
     * @param _size The size of the BSharedVector.
     */
    explicit BSharedVector(size_t _size, Engine& eng)
        : SharedVector<Share, EVector, Engine>(_size, Encoding::BShared, eng) {}

    /**
     * Creates a BSharedVector of size `_size` and initializes it with secret shares in the given
     * file.
     * @param _size The size of the BSharedVector.
     * @param _input_file The file that contains the secret shares.
     */
    explicit BSharedVector(size_t _size, const std::string& _input_file, Engine& eng)
        : SharedVector<Share, EVector, Engine>(_size, _input_file, Encoding::BShared, eng) {}

    /**
     * This is a shallow copy constructor from EVector.
     * @param _shares The EVector whose contents will be pointed by the BSharedVector.
     * @param eng The engine to be used.
     */
    explicit BSharedVector(EVector& _shares, Engine& eng)
        : SharedVector<Share, EVector, Engine>(_shares, Encoding::BShared, eng) {}

    /**
     * This is a move constructor from EVector.
     * @param _shares The EVector whose contents will be moved to the new BSharedVector.
     * @param eng The engine to be used.
     */
    BSharedVector(EVector&& _shares, Engine& eng)
        : SharedVector<Share, EVector, Engine>(_shares, Encoding::BShared, eng) {}

    /**
     * This is a move constructor from another BSharedVector.
     * @param other The BSharedVector whose contents will be moved to the new BSharedVector.
     */
    BSharedVector(BSharedVector&& other)
        : SharedVector<Share, EVector, Engine>(other.vector, Encoding::BShared, other.engine) {}

    /**
     * This is a copy constructor from another BSharedVector.
     * @param other The BSharedVector whose contents will be copied to the new BSharedVector.
     */
    BSharedVector(const BSharedVector& other)
        : SharedVector<Share, EVector, Engine>(other.vector, Encoding::BShared, other.engine) {}

    /**
     * Copy constructor from a SharedVector.
     * @param _shares The SharedVector object whose contents will be copied to the new
     * BSharedVector.
     */
    explicit BSharedVector(SharedVector<Share, EVector, Engine>& _shares)
        : SharedVector<Share, EVector, Engine>(_shares.vector, _shares.encoding, _shares.engine) {
        assert(_shares.encoding == Encoding::BShared);
        auto secretShares_ = reinterpret_cast<BSharedVector*>(&_shares);
    }

    /**
     * Move constructor that creates a BSharedVector from a unique pointer to a SharedVector object.
     * @param base The pointer to the SharedVector object whose contents will be moved to the new
     * BSharedVector.
     */
    BSharedVector(std::unique_ptr<BSharedVector>&& base)
        : BSharedVector((BSharedVector*)base.get()) {}

    /**
     * Shallow copy constructor that creates a BSharedVector from a unique pointer to a SharedVector
     * object.
     * @param base The SharedVector object whose contents will be pointed by the new
     * BSharedVector.
     */
    BSharedVector(std::unique_ptr<BSharedVector>& base)
        : BSharedVector((BSharedVector*)base.get()) {}

    /**
     * Move constructor that creates a BSharedVector from a pointer to another BSharedVector object.
     * @param _base The BSharedVector that will be moved as a whole (contents + state) to the new
     * BSharedVector.
     *
     * NOTE: This constructor is implicitly called by the two constructors above.
     */
    explicit BSharedVector(BSharedVector* _base) : BSharedVector(std::move(*_base)) {}

    /**
     * @brief Construct a new BSharedVector object like another BSharedVector but with a different
     * size.
     *
     * @param size The size of the new BSharedVector.
     * @return ASharedVector The newly constructed BSharedVector.
     */
    BSharedVector construct_like(std::optional<size_t> size = {}) const {
        auto new_size = size.value_or(this->size());
        return BSharedVector(this->vector.construct_like(new_size), this->engine);
    }

    /**
     * @brief Use `operator=` from the underlying `SharedVector`
     *
     */
    using SharedVector<Share, EVector, Engine>::operator=;

    BSharedVector& operator=(const BSharedVector&) = default;
    BSharedVector& operator=(BSharedVector&&) = default;

    virtual ~BSharedVector() {}

    /**
     * @brief Convert the LSB of each element of this BSharedVector to an arithmetic sharing. This
     * is substantially more efficient than a full-width conversion and suffices for most
     * applications.
     *
     * @return ASharedVector
     */
    auto b2a_bit() const {
        auto res =
            std::make_unique<ASharedVector<Share, EVector, Engine>>(this->size(), this->engine);
        res->setPrecision(this->getPrecision());
        this->engine.b2a_bit(this->vector, res->vector);
        return res;
    }

    /** @brief Full-width conversion from BSharedVector to ASharedVector. Generates an online
     * shared-bit correlation (random value shared in both domains), and then uses a boolean adder
     * to mask.
     *
     * The first loop is technically preprocessing, which would speed things up a lot.
     *
     * @return ASharedVector
     */
    std::unique_ptr<ASharedVector<Share, EVector, Engine>> b2a() const {
        using ret_t = ASharedVector<Share, EVector, Engine>;

        if constexpr (CONFIDENTIAL_1PC) {
            // No point doing conversion!
            return std::make_unique<ret_t>(this->asContainer(), this->engine);
        }

        if constexpr (SPDZ2k_NPC_PROTOCOL){
            // Same protocol.
            return std::make_unique<ret_t>(this->asContainer(), this->engine);
        }

        auto a_re = std::make_unique<ret_t>(this->size(), this->engine);
        BSharedVector<Share, EVector, Engine> b_re(this->size(), this->engine);

        auto me = this->engine.getPartyID();

        for (auto g : this->engine.getGroups()) {
            // Generate random vector (or maybe zero, if I'm not in the group)
            Vector<Share> myRandom(this->size());
            if (g.contains(me)) {
                this->engine.populateCommonRandom(myRandom, g);
            }

            *a_re += this->engine.public_share_a(myRandom, g);
            b_re += this->engine.public_share_b(myRandom, g);
        }

        // Now a_re === b_re but in different sharing schemes.

        // Add random value to mask (with binary adder), then open
        auto current_precision = this->getPrecision();
        b_re.setPrecision(this->getPrecision());
        auto z = (*this + b_re)->open();

        // Subtract secret-shared mask from public z
        a_re->inplace_invert();
        *a_re += this->engine.public_share_a(z);
        a_re->setPrecision(this->getPrecision());
        return a_re;
    }

    /**
     * This is a conversion from BSharedVector to ASharedVector.
     * It is insecure. It is only used in the generation of dummy permutations.
     */
    ASharedVector<Share, EVector, Engine> insecure_b2a() const {
        // It was meant to use communication not opening.
        auto opened = this->open();
        ASharedVector<Share, EVector, Engine> a(this->size(), this->engine);
        a.setPrecision(this->getPrecision());
        if (this->engine.getPartyID() == 0) {
            a.vector(0) = opened;
        }
        return a;
    }

    /**
     * @brief Compute the most common element in a vector of boolean shares. Used for multi-class
     * classification. Takes an optional parameter for the number of classes \f$k\f$. By default,
     * assumes \f$k=n\f$. Runtime is \f$O(kn)\$f. Classes are assumed to be
     * \f$[0,\dots,\mathtt{num_classes}-1\f$.
     *
     * @param num_classes number of classes.
     * @return The most common element as an ASharedVector of size 1.
     */
    auto most_common(std::optional<size_t> num_classes = {}) {
        auto N = this->size();
        auto K = num_classes.value_or(N);

        Vector<Share> plaintext_classes(K);
        std::iota(plaintext_classes.begin(), plaintext_classes.end(), 0);
        BSharedVector classes_shared = this->engine.public_share_b(plaintext_classes);

        auto x_repeated = this->cyclic_subset_reference(K);
        auto classes_repeated = classes_shared.repeated_subset_reference(N);

        // perform all equality checks. combined vector has size K*N.
        auto eq_a = (x_repeated == classes_repeated)->b2a_bit();

        // chunkedSum gives count per class. we want the max
        // TODO: figure out if just using a boolean adder here is better. (We don't have
        // chunkedSum_b)
        auto class_counts = eq_a->chunkedSum(N).a2b();

        auto next_pow_2 = 1 << std::bit_width(K - 1);

        class_counts->resize(next_pow_2);
        auto max_count = class_counts->construct_like();

        std::vector<BSharedVector> no_keys = {};

        cdough::aggregators::aggregate<Share, EVector, Engine>(
            no_keys, {{*class_counts, max_count, cdough::aggregators::max}}, {});

        // Get the last element (absolute maximum count)
        max_count.tail(1);
        class_counts->resize(K);
        auto m = max_count.repeated_subset_reference(K);

        auto mask = m == class_counts;
        auto empty = mask->construct_like();

        // If unique maximum, we could just take a dot product and be done. But that's not
        // necessarily true, so we need another aggregation. We'll always return the maximum as the
        // tiebreaker

        auto freq = multiplex(*mask, empty, classes_shared);

        freq.resize(next_pow_2);
        cdough::aggregators::aggregate<Share, EVector, Engine>(
            no_keys, {{freq, freq, cdough::aggregators::max}}, {});

        freq.tail(1);

        return freq;
    }

    // **************************************** //
    //            Boolean operators             //
    // **************************************** //

    /**
     * Elementwise secure bitwise XOR. Returns a unique_ptr.
     */
    binary_op(^, BSharedVector, xor_b, this, other);
    binary_element_op(^, xor_b, BSharedVector, Share);
    compound_assignment_op(^=, xor_b, BSharedVector);

    /**
     * Elementwise secure bitwise AND. Returns a unique_ptr.
     */
    binary_op(&, BSharedVector, and_b, this, other);
    binary_element_op(&, and_b, BSharedVector, Share);
    compound_assignment_op(&=, and_b, BSharedVector);

    /**
     * @brief Elementwise secure bitwise Or. Returns a unique_ptr.
     */
    binary_op(|, BSharedVector, or_b, this, other);
    binary_element_op(|, or_b, BSharedVector, Share);
    compound_assignment_op(|=, or_b, BSharedVector);

    /**
     * @brief Elementwise secure equality comparison. Returns a unique_ptr.
     */
    binary_op(==, BSharedVector, equal_b, this, other);
    binary_element_op(==, equal_b, BSharedVector, Share);

    /**
     * Elementwise secure boolean complement. Returns a unique_ptr.
     */
    unary_op(~, BSharedVector, not_b, this);

    /**
     * Elementwise secure boolean negation. Does not perform an equal-to-zero check: instead, just
     * considers the LSB. Returns a unique_ptr.
     */
    unary_op(!, BSharedVector, not_b_1, this);

    binary_op(&&, BSharedVector, and_b_1, this, other);
    binary_op(||, BSharedVector, or_b_1, this, other);

    std::unique_ptr<BSharedVector> xor_1(const BSharedVector& y) const {
        auto& x_vector = this->vector;
        auto& y_vector = reinterpret_cast<const BSharedVector*>(&y)->vector;
        assert(x_vector.size() == y_vector.size());
        auto res = std::make_unique<BSharedVector>(x_vector.size(), this->engine);
        this->engine.xor_b_1(x_vector, y_vector, res->vector);
        res->vector.matchPrecision(this->vector);
        return res;
    }

    /**
     * Elementwise secure less-than-zero comparison. Returns a unique_ptr.
     */
    fn_no_input(ltz, BSharedVector, this);

    /**
     * @brief Left shift. Returns a new BSharedVector.
     *
     * @param s
     * @return unique_B
     */
    unique_B operator<<(int s) const {
        auto out = std::make_unique<BSharedVector>(this->size(), this->engine);
        out->bit_left_shift(*this, s);
        return out;
    }

    /**
     * @brief Arithmetic right shift. Returns a new BSharedVector.
     *
     * @param s
     * @return unique_B
     */
    unique_B operator>>(int s) const {
        auto out = std::make_unique<BSharedVector>(this->size(), this->engine);
        out->bit_arithmetic_right_shift(*this, s);
        return out;
    }

    /**
     * @brief Left shift assignment operator.
     *
     * @param s
     */
    void operator<<=(int s) { this->bit_left_shift(*this, s); }

    /**
     * @brief Arithmetic right shift assignment operator.
     *
     * @param s
     */
    void operator>>=(int s) { this->bit_arithmetic_right_shift(*this, s); }

    /**
     * @brief Inherit access patterns from SharedVector.
     *
     */
    svector_reference(BSharedVector, simple_subset_reference);
    svector_reference(BSharedVector, alternating_subset_reference);
    svector_reference(BSharedVector, reversed_alternating_subset_reference);
    svector_reference(BSharedVector, repeated_subset_reference);
    svector_reference(BSharedVector, cyclic_subset_reference);
    svector_reference(BSharedVector, directed_subset_reference);
    svector_reference(BSharedVector, included_reference);
    svector_reference(BSharedVector, mapping_reference);
    svector_reference(BSharedVector, slice);

    /**
     * Masks each element in `this` vector by doing a bitwise logical AND with `n`.
     * @param n The mask.
     */
    // TODO: it should use the engine mask
    void mask(const Share& n) {
        if constexpr (SPDZ2k_NPC_PROTOCOL){
            std::cerr << "Warning: mask not supported." << std::endl;
        }else{
            this->engine.modify_parallel(this->vector, &EVector::mask, n);
        }
    }

    /**
     * @brief Invert the bits of this BSharedVector. Operates inplace.
     *
     */
    void inplace_invert() { this->engine.not_b(this->vector, this->vector); }

    // **************************************** //
    //           Comparison operators           //
    // **************************************** //

    /**
     * Elementwise secure inequality. This operator expects both input vectors (`this` and `other`)
     * to have the same size.
     *
     * @param other The second operand of inequality.
     * @return A unique pointer to a new shared vector that contains boolean shares of
     * the elementwise inequality comparisons.
     */
    std::unique_ptr<BSharedVector> operator!=(const BSharedVector& other) const {
        return !((*this) == other);
    }

    /**
     * @brief Greater-than or equals comparison circuit Primarily calls down to the `bit_same`
     * subroutine, with some additional handling for the sign of the inputs. The two additional
     * inputs are used as temporary storage. Results returned by reference to shared vectors passed
     * as arguments.
     *
     * @param other
     * @param eq_bits output containing the equality result
     * @param gt_bits output containing the greater-than result.
     */
    void _compare(const BSharedVector& other, BSharedVector& eq_bits, BSharedVector& gt_bits) const{
#ifdef USE_LINEAR_CIRCUIT_FOR_COMPARISON
        this->linear_compare(other, eq_bits, gt_bits);
#else
        this->log_compare(other, eq_bits, gt_bits);
#endif
    }

    std::unique_ptr<BSharedVector> rca_compare(const BSharedVector& other) const {
        auto res = std::make_unique<BSharedVector>(this->size(), this->engine);
        res->setPrecision(this->getPrecision());
        this->engine.rca_compare_b(this->vector, other.vector, res->vector);
        return res;
    }

    void linear_compare(const BSharedVector& other, BSharedVector& eq_bits, BSharedVector& gt_bits) const{
        this->engine.linear_compare_b(this->vector, other.vector, eq_bits.vector, gt_bits.vector);
    }

    void log_compare(const BSharedVector& other, BSharedVector& eq_bits, BSharedVector& gt_bits) const{
        this->engine.compare_b(this->vector, other.vector, eq_bits.vector, gt_bits.vector);
    }

    /**
     * Elementwise secure greater-than comparison.
     * This operator expects both input vectors (`this` and `other`) to have the same size. Call
     * down to the `_compare` subroutine.
     * @param other The second operand of greater-than.
     * @return A unique pointer to a new shared vector that contains boolean shares of
     * the elementwise greater-than comparisons.
     */
    std::unique_ptr<BSharedVector> operator>(const BSharedVector& other) const {
        // ignore here
        BSharedVector _eq(this->size(), this->engine);
        auto gt = std::make_unique<BSharedVector>(this->size(), this->engine);
        _eq.setPrecision(this->getPrecision());
        gt->setPrecision(this->getPrecision());

        _compare(other, _eq, *gt);
        return gt;
    }

    /**
     * Elementwise secure less-than comparison. Implement by calling greater-than with flipped
     * arguments.
     *
     * @param other The second operand of less-than.
     * @return A unique pointer to a new shared vector that contains boolean shares of
     * the elementwise less-than comparisons.
     */
    std::unique_ptr<BSharedVector> operator<(const BSharedVector& other) const {
        return other > (*this);
    }

    /**
     * Elementwise secure greater-or-equal comparison. Implement by inverting less-than.
     *
     * @param other The second operand of greater-or-equal.
     * @return A unique pointer to a new shared vector that contains boolean shares of
     * the elementwise greater-or-equal comparisons.
     */
    std::unique_ptr<BSharedVector> operator>=(const BSharedVector& other) const {
        return !((*this) < other);
    }

    /**
     * @brief Elementwise secure less-or-equal comparison. Implement by invert greater-than. This
     * operator expects both input vectors (`this` and `other`) to have the same size.
     *
     * @param other The second operand of less-or-equal.
     * @return A unique pointer to a new shared vector that contains boolean shares of
     * the elementwise less-or-equal comparisons.
     */
    std::unique_ptr<BSharedVector> operator<=(const BSharedVector& other) const {
        return !((*this) > other);
    }

    /**
     * Elementwise secure boolean addition. Call the compile-time-specified addition circuit.
     * @param other The second operand of boolean addition.
     * @return A unique pointer to a new shared vector that contains boolean shares of
     * the elementwise additions.
     */
    std::unique_ptr<BSharedVector> operator+(const BSharedVector& other) const {
        return binary_adder(*this, other, false);
    }

    /**
     * @brief Elementwise secure boolean compound assignment addition.
     *
     * @param other
     * @return BSharedVector&
     */
    BSharedVector& operator+=(const BSharedVector& other) {
        *this = *binary_adder(*this, other, false);
        return *this;
    }

    /**
     * @brief Unique pointer version of the above.
     *
     * @param other
     * @return BSharedVector&
     */
    BSharedVector& operator+=(const std::unique_ptr<BSharedVector> other) {
        *this = *binary_adder(*this, *other, false);
        return *this;
    }

    /**
     * @brief Binary subtraction. Calls `RCA(this, ~other) + 1` by setting
     * the carry-in bit of the RCA.
     *
     * @param other
     * @return std::unique_ptr<BSharedVector>
     */
    std::unique_ptr<BSharedVector> operator-(const BSharedVector& other) const {
        return binary_adder(*this, *(~other), true);
    }

    /**
     * @brief Binary subtraction compound assignment.
     *
     * @param other
     * @return BSharedVector&
     */
    BSharedVector& operator-=(const BSharedVector& other) {
        *this = *binary_adder(*this, *(~other), true);
        return *this;
    }

    /**
     * @brief Unique pointer version of the above.
     *
     * @param other
     * @return BSharedVector&
     */
    BSharedVector& operator-=(const std::unique_ptr<BSharedVector> other) {
        *this = *binary_adder(*this, *(~(*other)), true);
        return *this;
    }

    /**
     * @brief Negation, implemented via binary operator subtraction: \f$-x = 0-x\f$.
     *
     * @return std::unique_ptr<BSharedVector>
     */
    std::unique_ptr<BSharedVector> operator-() const {
        // TODO: use public sharing
        auto zero = BSharedVector(this->size(), this->engine);
        return zero - *this;
    }

    std::unique_ptr<BSharedVector> operator/(const BSharedVector& other) const;

    // friend class
    template <typename T, typename V, typename E>
    friend class ASharedVector;
};

}  // namespace cdough
