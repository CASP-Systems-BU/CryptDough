#pragma once

#include <vector>

namespace cdough::protocols::circuits {

template <typename Share, typename BContainer, typename Protocol>
static void ripple_carry_adder(const BContainer& a, const BContainer& b, BContainer& sum,
                               const bool carry_in, Protocol& protocol) {
    static const int MAX_BITS_NUMBER = std::numeric_limits<std::make_unsigned_t<Share>>::digits;
    size_t compressed_size = a.size() / MAX_BITS_NUMBER + ((a.size() % MAX_BITS_NUMBER) > 0);

    // Compressed vectors
    auto carry_i = a.construct_like(compressed_size);
    auto tmp_i = a.construct_like(compressed_size);

    auto a_xor_b = a.construct_like(a.size());
    protocol.xor_b(a, b, a_xor_b);

#ifndef USE_OPTIMIZED_BIT_DECOMPOSITION_FOR_RCA
    auto a_xor_b_i = a.construct_like(compressed_size);
    auto a_i = a.construct_like(compressed_size);
#else
    // Precompute bit decompositions
    auto a_xor_b_i_ = a_xor_b.bit_decomposition();
    auto a_i_ = a.bit_decomposition();
#endif

    if (carry_in) {
        protocol.inplace_invert_b(carry_i);
    }

    for (int i = 0; i < MAX_BITS_NUMBER; ++i) {
#ifndef USE_OPTIMIZED_BIT_DECOMPOSITION_FOR_RCA
        protocol.pack_from(a_xor_b, a_xor_b_i, i);
        protocol.pack_from(a, a_i, i);
#else
        auto& a_xor_b_i = a_xor_b_i_[i];
        a_xor_b_i.setPrecision(a.getPrecision());
        auto& a_i = a_i_[i];
        a_i.setPrecision(a.getPrecision());
#endif

        protocol.xor_b(a_xor_b_i, carry_i, tmp_i);
        protocol.unpack_from(tmp_i, sum, i);

        // don't compute carry on the last round
        if (i == MAX_BITS_NUMBER - 1) {
            break;
        }

        // This is not the canonical full-adder: you can do it with one AND
        // + 1 OR inside the loop, along with an extra AND up front (a & b).
        // However, while XORs are expensive in hardware, they are cheap in
        // MPC, so we should do it this way, instead (only 1 AND)

        // Effectively computing the recurrence
        //   c_i = ((c_{i-1} ^ a_i) & axb_i) ^ a_i
        protocol.xor_b(carry_i, a_i, carry_i);
        carry_i.setPrecision(0);
        a_xor_b_i.setPrecision(0);
        carry_i.setPrecision(0);
        protocol.and_b(carry_i, a_xor_b_i, carry_i);
        protocol.xor_b(carry_i, a_i, carry_i);
    }
}

template <typename Share, typename BContainer, typename Protocol>
static void ripple_carry_adder_packed_sign(const BContainer& a, const BContainer& b,
                                           BContainer& sum, const bool carry_in,
                                           Protocol& protocol) {
    static const int MAX_BITS_NUMBER = std::numeric_limits<std::make_unsigned_t<Share>>::digits;
    size_t compressed_size = a.size() / MAX_BITS_NUMBER + ((a.size() % MAX_BITS_NUMBER) > 0);

    // Compressed vectors
    auto carry_i = a.construct_like(compressed_size);

    auto a_xor_b = a.construct_like(a.size());
    protocol.xor_b(a, b, a_xor_b);

    // Precompute bit decompositions
    auto a_xor_b_i_ = a_xor_b.bit_decomposition();
    auto a_i_ = a.bit_decomposition();

    if (carry_in) {
        protocol.inplace_invert_b(carry_i);
    }

    for (int i = 0; i < MAX_BITS_NUMBER; ++i) {
        auto& a_xor_b_i = a_xor_b_i_[i];
        auto& a_i = a_i_[i];

        // don't compute carry on the last round
        if (i == MAX_BITS_NUMBER - 1) {
            protocol.xor_b(a_xor_b_i, carry_i, sum);
            break;
        }

        // This is not the canonical full-adder: you can do it with one AND
        // + 1 OR inside the loop, along with an extra AND up front (a & b).
        // However, while XORs are expensive in hardware, they are cheap in
        // MPC, so we should do it this way, instead (only 1 AND)

        // Effectively computing the recurrence
        //   c_i = ((c_{i-1} ^ a_i) & axb_i) ^ a_i
        protocol.xor_b(carry_i, a_i, carry_i);
        protocol.and_b(carry_i, a_xor_b_i, carry_i);
        protocol.xor_b(carry_i, a_i, carry_i);
    }
}

template <typename Share, typename BContainer, typename Protocol>
static void rca_compare(const BContainer& a, const BContainer& b, BContainer& sum,
                        Protocol& protocol) {
    static const int MAX_BITS_NUMBER = std::numeric_limits<std::make_unsigned_t<Share>>::digits;
    size_t size = a.size();
    size_t compressed_size = a.size() / MAX_BITS_NUMBER + ((a.size() % MAX_BITS_NUMBER) > 0);

    // compute a + (~b) + 1
    auto nb = a.construct_like(size);
    protocol.not_b(b, nb);

    // Compressed vectors
    auto carry_i = a.construct_like(compressed_size);
    auto tmp_i = a.construct_like(compressed_size);

    auto a_xor_b = a.construct_like(size);
    protocol.xor_b(a, nb, a_xor_b);

#ifndef USE_OPTIMIZED_BIT_DECOMPOSITION_FOR_RCA
    auto a_xor_b_i = a.construct_like(compressed_size);
    auto a_i = a.construct_like(compressed_size);
#else
    // Precompute bit decompositions
    auto a_xor_b_i_ = a_xor_b.bit_decomposition();
    auto a_i_ = a.bit_decomposition();
#endif

    // "+ 1" for subtraction
    protocol.inplace_invert_b(carry_i);

    int i = 0;
    for (; i < MAX_BITS_NUMBER; ++i) {
#ifndef USE_OPTIMIZED_BIT_DECOMPOSITION_FOR_RCA
        protocol.pack_from(a_xor_b, a_xor_b_i, i);
#else
        auto& a_xor_b_i = a_xor_b_i_[i];
        auto& a_i = a_i_[i];
#endif

        // don't care about intermediate sum bits - no unpack!
        if (i == MAX_BITS_NUMBER - 1) {
            break;
        }

#ifndef USE_OPTIMIZED_BIT_DECOMPOSITION_FOR_RCA
        protocol.pack_from(a, a_i, i);
#endif

        // Only compute carry bits
        protocol.xor_b(carry_i, a_i, carry_i);
        protocol.and_b(carry_i, a_xor_b_i, carry_i);
        protocol.xor_b(carry_i, a_i, carry_i);
    }

#ifdef USE_OPTIMIZED_BIT_DECOMPOSITION_FOR_RCA
    auto& a_xor_b_i = a_xor_b_i_.back();
#endif

    protocol.xor_b(carry_i, a_xor_b_i, carry_i);

    // i == MBN - 1
    // now, actually compute "sum" - just the carry bit
    // unfortunate need to unpack into full width datatype; TODO: 1-bit
    // types
    protocol.unpack_from(carry_i, sum, i);

    protocol.ltz(sum, sum);
}

template <typename Share, typename BContainer, typename Protocol>
static std::pair<BContainer, BContainer> prefix_sum(const std::pair<BContainer, BContainer>& x,
                                                    const std::pair<BContainer, BContainer>& y,
                                                    Protocol& protocol) {
    auto [g_x, p_x] = x;
    auto [g_y, p_y] = y;

    protocol.and_b(p_y, p_x, p_y);  // propagate
    protocol.and_b(g_y, p_x, g_y);

    // g_x & (g_y & p_x)
    auto gpg = g_x.construct_like(g_x.size());
    protocol.and_b(g_x, g_y, gpg);

    // This expression is equivalent to
    // g' := gx ^ gx.px ^ gx.gy.px
    // p' := py.px
    protocol.xor_b(gpg, g_x, gpg);
    protocol.xor_b(gpg, g_y, gpg);
    return {gpg, p_y};
}

template <typename Share, typename BContainer, typename Protocol>
static void parallel_prefix_adder(const BContainer& current, const BContainer& other,
                                  BContainer& sum, const bool carry_in, Protocol& protocol) {
    static const uint MAX_BITS_NUMBER = std::numeric_limits<std::make_unsigned_t<Share>>::digits;
    static const uint LOG_MAX_BITS_NUMBER = std::bit_width(MAX_BITS_NUMBER - 1);

    auto p = current.construct_like(current.size());
    auto g = current.construct_like(current.size());
    auto propagate = current.construct_like(current.size());

    protocol.xor_b(current, other, p);
    protocol.and_b(current, other, g);

    if (carry_in) {
        // Copy the LSB of p into a new variable.
        propagate = p;
        protocol.mask(propagate, 1);

        // if carry, flip LSB of p
        // ... p := (c ^ o ^ 1)
        // TODO: protocol.xor_b(p, 1, p);
        // TODO: is that just the ~ operator?
        p = p ^ 1;

        // Compute "dot product" between the 3 operands.
        // g := (c & 1) ^ (o & 1) ^ (c & o)
        // but this is just an XOR over the LSBs.
        protocol.xor_b(g, propagate, g);
    }

    propagate = p;

    auto g_shift = g.construct_like(g.size());
    auto p_shift = p.construct_like(p.size());

    for (int i = 0; i < LOG_MAX_BITS_NUMBER; ++i) {
        protocol.bit_left_shift(g, g_shift, 1 << i);
        protocol.bit_left_shift(p, p_shift, 1 << i);

        std::tie(g, p) =
            prefix_sum<Share, BContainer, Protocol>({g, p}, {g_shift, p_shift}, protocol);
    }

    // reuse temporary: p = g << 1
    protocol.bit_left_shift(g, p, 1);
    protocol.xor_b(propagate, p, sum);
}

template <typename Share, typename BContainer, typename Protocol>
static void parallel_prefix_adder_packed_sign(const BContainer& a, const BContainer& b,
                                              BContainer& sum, const bool carry_in,
                                              Protocol& protocol) {
    static const uint MAX_BITS_NUMBER = std::numeric_limits<std::make_unsigned_t<Share>>::digits;
    static const uint LOG_MAX_BITS_NUMBER = std::bit_width(MAX_BITS_NUMBER - 1);

    auto p = a.construct_like(a.size());
    auto g = a.construct_like(a.size());
    auto propagate = a.construct_like(a.size());

    protocol.xor_b(a, b, p);
    protocol.and_b(a, b, g);

    if (carry_in) {
        // Copy the LSB of p into a new variable.
        propagate = p;
        protocol.mask(propagate, 1);

        // if carry, flip LSB of p
        // ... p := (c ^ o ^ 1)
        // TODO: protocol.xor_b(p, 1, p);
        // TODO: is that just the ~ operator?
        p = p ^ 1;

        // Compute "dot product" between the 3 operands.
        // g := (c & 1) ^ (o & 1) ^ (c & o)
        // but this is just an XOR over the LSBs.
        protocol.xor_b(g, propagate, g);
    }

    propagate = p;

    auto g_shift = g.construct_like(g.size());
    auto p_shift = p.construct_like(p.size());

    for (int i = 0; i < LOG_MAX_BITS_NUMBER; ++i) {
        protocol.bit_left_shift(g, g_shift, 1 << i);
        protocol.bit_left_shift(p, p_shift, 1 << i);

        std::tie(g, p) =
            prefix_sum<Share, BContainer, Protocol>({g, p}, {g_shift, p_shift}, protocol);
    }

    // reuse temporary: p = g << 1
    protocol.bit_left_shift(g, p, 1);
    protocol.xor_b(propagate, p, propagate);

    protocol.pack_from(propagate, sum, MAX_BITS_NUMBER - 1);
}

template <typename Share, typename BContainer, typename Protocol>
BContainer bit_same(const BContainer& x, const BContainer& y, std::optional<BContainer> _temp,
                    Protocol& protocol) {
    static const uint MAX_BITS_NUMBER = std::numeric_limits<std::make_unsigned_t<Share>>::digits;
    static const uint LOG_MAX_BITS_NUMBER = std::bit_width(MAX_BITS_NUMBER - 1);

    // Initialize vector - initial compute
    auto sameBit = x.construct_like(x.size());
    protocol.xor_b(x, y, sameBit);

    protocol.inplace_invert_b(sameBit);

    // Reuse storage if provided. Otherwise, allocate new.
    BContainer temp = _temp.has_value() ? (*_temp) : y.construct_like(y.size());

    for (int level_size = 0; level_size < LOG_MAX_BITS_NUMBER; level_size++) {
        protocol.bit_arithmetic_right_shift(sameBit, temp, 1 << level_size);
        protocol.and_b(sameBit, temp, sameBit);
    }
    return sameBit;
}

template <typename Share, typename BContainer, typename Protocol>
static void linear_compare(const BContainer& current, const BContainer& other, BContainer& eq_bits,
                           BContainer& gt_bits, Protocol& protocol) {
    const size_t size = current.size();
    assert(size == other.size());

    // Number of bits in the share representation
    const int MAX_BITS_NUMBER = std::numeric_limits<std::make_unsigned_t<Share>>::digits;

    auto compressed_size = size / MAX_BITS_NUMBER + (size % MAX_BITS_NUMBER > 0);

    auto b1 = current.construct_like(compressed_size);
    auto b2 = current.construct_like(compressed_size);
    auto r = current.construct_like(compressed_size);

    auto t = current ^ other;
    for (int i = MAX_BITS_NUMBER - 1; i >= 0; i--) {
        protocol.pack_from(t, b1, i);
        protocol.inplace_invert_b(b1);

        if (i == MAX_BITS_NUMBER - 1) {
            r = b1;
        } else {
            protocol.and_b(r, b1, r);
        }

        protocol.unpack_from(r, eq_bits, i);
    }

    protocol.bit_arithmetic_right_shift(eq_bits, gt_bits, 1);
    protocol.xor_b(gt_bits, eq_bits, gt_bits);
    protocol.and_b(gt_bits, current, gt_bits);
    // inner expr is ((eq_bits >> 1) ^ eq_bits) & (*this))
    protocol.bit_xor(gt_bits, gt_bits);

    // If the shares are signed numbers, we need to treat the sign bits differently
    if (std::is_signed<Share>::value) {
        // Extract MSB (sign bit), compressed
        protocol.pack_from(current, b1, MAX_BITS_NUMBER - 1);
        protocol.pack_from(other, b2, MAX_BITS_NUMBER - 1);

        // Extract LSB from above
        protocol.pack_from(gt_bits, r, 0);

        // Update greater bits: `this` is greater than `other` iff the
        // signs are different and `other` is negative, otherwise keep
        // the existing greater bits
        protocol.xor_b(r, b2, r);
        protocol.xor_b(b2, b1, b2);
        protocol.or_b(b2, r, b2);
        protocol.xor_b(b1, b2, b1);  // result is actually in s1, not r

        // Decompress result
        protocol.unpack_from(b1, gt_bits, 0);
    }

    protocol.mask(eq_bits, 1);
    protocol.mask(gt_bits, 1);
}

template <typename Share, typename BContainer, typename Protocol>
static void compare(const BContainer& current, const BContainer& other, BContainer& eq_bits,
                    BContainer& gt_bits, Protocol& protocol) {
    const size_t size = current.size();
    assert(size == other.size());

    // Number of bits in the share representation
    const int MAX_BITS_NUMBER = std::numeric_limits<std::make_unsigned_t<Share>>::digits;

    auto compressed_size = size / MAX_BITS_NUMBER + (size % MAX_BITS_NUMBER > 0);

    // Compute same-bits prefix. Use eq_bits as temp storage, then copy
    // result in.
    eq_bits = bit_same<Share>(current, other, std::make_optional(eq_bits), protocol);

    // If MSBs are different, `this` is greater than `other` iff the
    // MSB of `this` is set, else if MSBs are the same and the second
    // MSBs are not, then `this` is greater than `other` iff the second
    // MSB of `this` is set, else...
    //
    // The plaintext value of xEdgeBit will either be all zeros, or have
    // a one somewhere, but the secret-shared values will be random.
    // This is basically a distributed point function.
    // XOR bits to get the single-bit result
    protocol.bit_arithmetic_right_shift(eq_bits, gt_bits, 1);
    protocol.xor_b(gt_bits, eq_bits, gt_bits);
    protocol.and_b(gt_bits, current, gt_bits);
    // inner expr is ((eq_bits >> 1) ^ eq_bits) & (*this))
    protocol.bit_xor(gt_bits, gt_bits);

    // If the shares are signed numbers, we need to treat the sign bits differently
    if (std::is_signed<Share>::value) {
        auto s1 = current.construct_like(compressed_size);
        auto s2 = current.construct_like(compressed_size);
        auto r = current.construct_like(compressed_size);

        // Extract MSB (sign bit), compressed
        protocol.pack_from(current, s1, MAX_BITS_NUMBER - 1);
        protocol.pack_from(other, s2, MAX_BITS_NUMBER - 1);

        // Extract LSB from above
        protocol.pack_from(gt_bits, r, 0);

        // Update greater bits: `this` is greater than `other` iff the
        // signs are different and `other` is negative, otherwise keep
        // the existing greater bits

        // Original expression:
        //   diffs = s1 ^ s2
        //   r = (diffs & s2) ^ (!diffs & r)
        // Equivalent expression with one fewer AND:
        //  r = s1 ^ ((s1 ^ s2) | (s2 ^ r));
        // Below: same thing, using compound assignment

        protocol.xor_b(r, s2, r);
        protocol.xor_b(s2, s1, s2);
        protocol.or_b(s2, r, s2);
        protocol.xor_b(s1, s2, s1);  // result is actually in s1, not r

        // Decompress result
        protocol.unpack_from(s1, gt_bits, 0);
    }

    protocol.mask(eq_bits, 1);
    protocol.mask(gt_bits, 1);
}

}  // namespace cdough::protocols::circuits