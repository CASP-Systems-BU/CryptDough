#pragma once

#include "core/protocols/spdz2k_npc.h"

namespace cdough::protocols {

template <typename Data, template <typename> class Share, template <typename> class Vector,
          template <typename, int, int> class EVector, typename AProtocol, typename BProtocol>
class edabits_protocol : public Protocol<Data, Share<typename AProtocol::ShareType>, Vector<Data>,
                                         typename AProtocol::ShareMultiVector> {
    std::unique_ptr<AProtocol> aProtocol;
    std::unique_ptr<BProtocol> bProtocol;

    using DataType = Data;
    using UnsignedDataType = std::make_unsigned_t<DataType>;

    using AShareType = typename AProtocol::ShareType;
    using ADataVector = typename AProtocol::DataVector;
    using AContainer = typename AProtocol::ShareMultiVector;

    using BShareType = typename BProtocol::ShareType;
    using BDataVector = typename BProtocol::DataVector;
    using BContainer = typename BProtocol::ShareMultiVector;

    static const size_t replication_number = 2;

   public:
    static int parties_num;

   public:
    edabits_protocol(const PartyID _partyID, const PartyID _numParties, Communicator* _communicator,
                     random::RandomnessManager* _randomnessManager,
                     std::unique_ptr<AProtocol> _protocol1, std::unique_ptr<BProtocol> _protocol2)
        : Protocol<Data, Share<typename AProtocol::ShareType>, Vector<DataType>,
                   typename AProtocol::ShareMultiVector>(_communicator, _randomnessManager,
                                                         _partyID, _numParties, replication_number),
          aProtocol(std::move(_protocol1)),
          bProtocol(std::move(_protocol2)) {
        // Initialize the static member with the number of parties
        parties_num = _numParties;
    }

    ////////////////////////////////////////////////////////////
    /////////////////////// A Primitives ///////////////////////
    ////////////////////////////////////////////////////////////
    void add_a(const AContainer& x, const AContainer& y, AContainer& z) {
        aProtocol->add_a(x, y, z);
    }

    void sub_a(const AContainer& x, const AContainer& y, AContainer& z) {
        aProtocol->sub_a(x, y, z);
    }

    void multiply_a(const AContainer& x, const AContainer& y, AContainer& z) {
        AContainer x_copy = x;
        AContainer y_copy = y;
        AContainer z_copy = z;

        auto xPrecision = x_copy.getPrecision();
        auto yPrecision = y_copy.getPrecision();
        auto zPrecision = z_copy.getPrecision();
        x_copy.setPrecision(0);
        y_copy.setPrecision(0);
        z_copy.setPrecision(0);

        aProtocol->multiply_a(x_copy, y_copy, z_copy);

        x_copy.setPrecision(xPrecision);
        y_copy.setPrecision(yPrecision);
        z_copy.setPrecision(zPrecision);

        this->handle_precision(x, y, z);
        this->truncate(z);
    }

    void neg_a(const AContainer& x, const AContainer& y, AContainer& z) {
        aProtocol->neg_a(x, y, z);
    }

    void dot_product_a(const AContainer& x, const AContainer& y, AContainer& z) {
        AContainer x_copy = x;
        AContainer y_copy = y;
        AContainer z_copy = z;

        auto xPrecision = x_copy.getPrecision();
        auto yPrecision = y_copy.getPrecision();
        auto zPrecision = z_copy.getPrecision();
        x_copy.setPrecision(0);
        y_copy.setPrecision(0);
        z_copy.setPrecision(0);

        aProtocol->dot_product_a(x_copy, y_copy, z_copy);

        x_copy.setPrecision(xPrecision);
        y_copy.setPrecision(yPrecision);
        z_copy.setPrecision(zPrecision);

        this->handle_precision(x, y, z);
        this->truncate(z);
    }

    void matrix_right_multiply_with_column_matrix_vectorized_a(
        const AContainer& x, const AContainer& y, AContainer& z, const size_t lhs_rows,
        const size_t lhs_cols, const size_t rhs_rows, const size_t rhs_cols) {
        AContainer x_copy = x;
        AContainer y_copy = y;
        AContainer z_copy = z;

        auto xPrecision = x_copy.getPrecision();
        auto yPrecision = y_copy.getPrecision();
        auto zPrecision = z_copy.getPrecision();
        x_copy.setPrecision(0);
        y_copy.setPrecision(0);
        z_copy.setPrecision(0);

        aProtocol->matrix_right_multiply_with_column_matrix_vectorized_a(
            x_copy, y_copy, z_copy, lhs_rows, lhs_cols, rhs_rows, rhs_cols);

        x_copy.setPrecision(xPrecision);
        y_copy.setPrecision(yPrecision);
        z_copy.setPrecision(zPrecision);

        this->handle_precision(x, y, z);
        this->truncate(z);
    }

    void conv_2d_vectorized_a(const AContainer& x, const AContainer& y, AContainer& z,
                              const size_t instancesCount, const size_t inputHeight,
                              const size_t inputWidth, const size_t filterHeight,
                              const size_t filterWidth, const size_t strideHeight,
                              const size_t strideWidth, const size_t paddingHeight,
                              const size_t paddingWidth) {
        AContainer x_copy = x;
        AContainer y_copy = y;
        AContainer z_copy = z;

        auto xPrecision = x_copy.getPrecision();
        auto yPrecision = y_copy.getPrecision();
        auto zPrecision = z_copy.getPrecision();
        x_copy.setPrecision(0);
        y_copy.setPrecision(0);
        z_copy.setPrecision(0);

        aProtocol->conv_2d_vectorized_a(x_copy, y_copy, z_copy, instancesCount, inputHeight, inputWidth,
                                        filterHeight, filterWidth, strideHeight, strideWidth,
                                        paddingHeight, paddingWidth);

        x_copy.setPrecision(xPrecision);
        y_copy.setPrecision(yPrecision);
        z_copy.setPrecision(zPrecision);

        this->handle_precision(x, y, z);
        this->truncate(z);
    }

    void sumPoolingVectorized(const AContainer& x, const AContainer& y, AContainer& z,
                              const size_t instancesCount, const size_t channels,
                              const size_t inputHeight, const size_t inputWidth,
                              const size_t poolHeight, const size_t poolWidth,
                              const size_t strideHeight, const size_t strideWidth,
                              const size_t paddingHeight, const size_t paddingWidth) {
        aProtocol->sumPoolingVectorized(x, y, z, instancesCount, channels, inputHeight, inputWidth,
                                        poolHeight, poolWidth, strideHeight, strideWidth,
                                        paddingHeight, paddingWidth);
    }

    std::pair<AContainer, AContainer> div_const_a(const AContainer& x, const DataType& divisor) {
        const size_t size = x.size();
        auto res = x.construct_like(size);
        auto err = x.construct_like(size);

        auto isPowerOfTwo = (divisor > 0) && ((divisor & (divisor - 1)) == 0);
        auto powerValue = isPowerOfTwo ? divisor : (1 << 20);

        // Get the power of 2 that powerValue is equal to
        int power = 0;
        DataType temp = powerValue;
        while (temp > 1) {
            temp = temp >> 1;
            power++;
        }

#if defined(USE_DIVISION_CORRECTION)
        // TODO: get actual eda-bits not dummy ones.
        auto [eda_bits_b, eda_bits_a] = getEdaBits(size);

        // let's get the difference between input and eda_bits_a
        auto diff_a = x.construct_like(size);

        if (isPowerOfTwo) {
            sub_a(x, eda_bits_a, diff_a);
        } else {
            // calculate the inverse of divisor.
            DataType divisor_inverse = powerValue / divisor;

            // multiply the inverse of divisor with input x
            auto divisor_inverse_x = x * divisor_inverse;

            // calculate the difference between divisor_inverse_x and eda_bits_a
            sub_a(divisor_inverse_x, eda_bits_a, diff_a);
        }

        // Open the diff then decompose it into bits
        auto diff_a_opened = open_shares_a(diff_a);

        // This truncates by ignoring bits up to power, so it effectively divides by powerValue.
        this->public_rca(eda_bits_b, diff_a_opened, res, std::nullopt, power);
        return {res, err};
#else
        // we are either truncating input or multiplying by inverse then truncating.
        auto toBeTruncated = isPowerOfTwo ? x : (x * ((DataType)powerValue / divisor));

        const size_t k = sizeof(DataType) * 8;
        const size_t l = k - 1;
        const size_t m = power;

        auto two_k_l_1 = (DataType(1) << (k - l - 1));
        auto two_l_m = (DataType(1) << (l - m));
        auto two_l = (DataType(1) << l);
        auto two_m = (DataType(1) << m);

        // We get (l-m) eda-bits
        auto [rb, ra] = getEdaBits(size, l - m);

        // We get m eda-bits
        auto [rb_, ra_] = getEdaBits(size, m);

        // we get a random bit
        auto [rbit_b, rbit_a] = getEdaBits(size, 1);

        // Step 1: we calculate c, masked values
        auto cShared = (toBeTruncated + rbit_a * two_l + ra * two_m + ra_) * two_k_l_1;
        auto c = open_shares_a(cShared);
        auto c_ = c / two_k_l_1;

        // Step 2: calculate v, overflow bit.
        // we extract bit l of c_
        ADataVector c_l(size);
        c_l.extract_bit_from_vector(c_, l);
        auto c_l_shared = aProtocol->public_share(c_l);

        // we calculate v = b ^ c_l
        auto v = rbit_a + c_l_shared - rbit_a * c_l * 2;

        // Step 3: calculate y, the truncated value with some error.
        auto truncated_c_ = (c_ % two_l) / two_m;
        auto shared_truncated_c_ = aProtocol->public_share(truncated_c_);
        res = shared_truncated_c_ - ra + v * two_l_m;

        return {res, err};
#endif
    }

    ////////////////////////////////////////////////////////////
    /////////////////////// B Primitives ///////////////////////
    ////////////////////////////////////////////////////////////

    void xor_b(const BContainer& x, const BContainer& y, BContainer& z) {
        aProtocol->xor_b(x, y, z);
    }

    void xor_b_1(const BContainer& x, const BContainer& y, BContainer& z) {
        // We need to calculate x + y - (xy * 2)
        BContainer xy = x.construct_like(x.size());
        this->multiply_a(x, y, xy);

        z(0) = x(0) + y(0) - xy(0) * 2;
        z(1) = x(1) + y(1) - xy(1) * 2;
    }

    void and_b(const BContainer& x, const BContainer& y, BContainer& z) {
        aProtocol->and_b(x, y, z);
    }

    void and_b_1(const BContainer& first, const BContainer& second, BContainer& result) {
        aProtocol->and_b_1(first, second, result);
    }

    void not_b(const BContainer& x, BContainer& y) { aProtocol->not_b(x, y); }

    void not_b_1(const BContainer& x, BContainer& y) { aProtocol->not_b_1(x, y); }

    void or_b(const BContainer& first, const BContainer& second, BContainer& result) {
        aProtocol->or_b(first, second, result);
    }

    void or_b_1(const BContainer& first, const BContainer& second, BContainer& result) {
        aProtocol->or_b_1(first, second, result);
    }

    void inplace_invert_b(BContainer& x) { aProtocol->inplace_invert_b(x); }

    ////////////////////////////////////////////////////////////
    //////////////////// B Bit Manipulation ////////////////////
    ////////////////////////////////////////////////////////////
    void pack_from(const BContainer& source, BContainer& destination, const int& pos) {
        aProtocol->pack_from(source, destination, pos);
    }

    void unpack_from(const BContainer& source, BContainer& destination, const int& pos) {
        aProtocol->unpack_from(source, destination, pos);
    }

    void bit_arithmetic_right_shift(const BContainer& in, BContainer& out, const int& shift_size) {
        aProtocol->bit_arithmetic_right_shift(in, out, shift_size);
    }

    void bit_logical_right_shift(const BContainer& in, BContainer& out, const int& shift_size) {
        aProtocol->bit_logical_right_shift(in, out, shift_size);
    }

    void bit_left_shift(const BContainer& in, BContainer& out, const int& shift_size) {
        aProtocol->bit_left_shift(in, out, shift_size);
    }

    void bit_xor(const BContainer& in, BContainer& out) { aProtocol->bit_xor(in, out); }

    void extend_lsb(const BContainer& in, BContainer& out) { aProtocol->extend_lsb(in, out); }

    void mask(BContainer& in, const DataType mask_value) { aProtocol->mask(in, mask_value); }

    ////////////////////////////////////////////////////////////
    ///////////////////////// Circuits /////////////////////////
    ////////////////////////////////////////////////////////////

    // TODO: operator::/

    // operator==
    virtual void equal_b(const BContainer& x, const BContainer& y, BContainer& result) {
        // gt_bits = (x > y)
        // gt_bits = ( 0 > y - x)
        // gt_bits = ( y - x < 0)
        auto sum1 = y - x;
        auto gt_bits = x.construct_like(x.size());
        this->ltz(sum1, gt_bits);

        auto sum2 = x - y;
        auto lt_bits = x.construct_like(x.size());
        this->ltz(sum2, lt_bits);

        // eq_bits = !(x > y) & !(y > x)
        // eq_bits = !(gt_bits) & !(lt_bits)
        // eq_bits = 1 - gt_bits - lt_bits + gt_bits * lt_bits
        BDataVector ones(x.size(), 1);
        auto ones_shared = aProtocol->public_share(ones);
        auto gt_and_lt = x.construct_like(x.size());
        this->multiply_a(gt_bits, lt_bits, gt_and_lt);
        result = ones_shared - gt_bits - lt_bits + gt_and_lt;
    }

    // compare
    virtual void compare_b(const BContainer& first, const BContainer& second, BContainer& eq_bits,
                           BContainer& gt_bits) {
        linear_compare_b(first, second, eq_bits, gt_bits);
    }

    // rca_compare
    virtual void rca_compare_b(const BContainer& first, const BContainer& second, BContainer& gt_bits) {
        linear_compare_b(first, second, gt_bits, gt_bits);
    }

    // linear_compare
    virtual void linear_compare_b(const BContainer& x, const BContainer& y, BContainer& eq_bits,
                                  BContainer& gt_bits) {
        // gt_bits = (x > y)
        // gt_bits = ( 0 > y - x)
        // gt_bits = ( y - x < 0)
        auto sum1 = y - x;
        this->ltz(sum1, gt_bits);

        auto sum2 = x - y;
        auto lt_bits = x.construct_like(x.size());
        this->ltz(sum2, lt_bits);

        // eq_bits = !(x > y) & !(y > x)
        // eq_bits = !(gt_bits) & !(lt_bits)
        // eq_bits = 1 - gt_bits - lt_bits + gt_bits * lt_bits
        BDataVector ones(x.size(), 1);
        auto ones_shared = aProtocol->public_share(ones);
        auto gt_and_lt = x.construct_like(x.size());
        this->multiply_a(gt_bits, lt_bits, gt_and_lt);
        eq_bits = ones_shared - gt_bits - lt_bits + gt_and_lt;

        // if x and y have different signs, then x > y is the same as y being negative or the sign of (y)
        // if signs are the same, then x > y is the same as (y - x) being negative or the sign of (y - x)
        // Otherwise, it is sign of (y)
        auto x_negative = x.construct_like(x.size());
        this->ltz(x, x_negative);

        auto y_negative = y.construct_like(y.size());
        this->ltz(y, y_negative);

        auto different_signs = x.construct_like(x.size());
        this->xor_b_1(x_negative, y_negative, different_signs);

        auto maskedYSign = y.construct_like(y.size());
        auto maskedGTSign = sum1.construct_like(sum1.size());
        this->multiply_a(different_signs, y_negative, maskedYSign);
        this->multiply_a(different_signs, gt_bits, maskedGTSign);

        gt_bits = gt_bits + maskedYSign - maskedGTSign;
    }

    // ripple_carry_adder
    virtual void ripple_carry_adder_b(const BContainer& x, const BContainer& y, BContainer& res,
                                      const bool carry_in) {
        res = x + y;
        if(carry_in) {
            BDataVector carry(res.size(), 1);
            auto carry_res = aProtocol->public_share(carry);
            res += carry_res;
        }
    }

    // ripple_carry_adder_packed_sign_b
    virtual void ripple_carry_adder_packed_sign_b(const BContainer& x, const BContainer& y, BContainer& res,
                                                  const bool carry_in) {
        // this->aProtocol->ripple_carry_adder_packed_sign_b(x, y, res, carry_in);
        std::cerr << "ripple_carry_adder_packed_sign_b is not implemented yet." << std::endl;
        exit(1);
    }

    // parallel_prefix_adder
    virtual void parallel_prefix_adder_b(const BContainer& x, const BContainer& y, BContainer& res,
                                         const bool& carry_in) {
        // this->aProtocol->parallel_prefix_adder_b(x, y, res, carry_in);
        this->ripple_carry_adder_b(x, y, res, carry_in);
    }

    virtual void parallel_prefix_adder_packed_sign_b(const BContainer& x, const BContainer& y, BContainer& res,
                                                  const bool carry_in) {
        // this->aProtocol->parallel_prefix_adder_packed_sign_b(x, y, res, carry_in);
        std::cerr << "parallel_prefix_adder_packed_sign_b is not implemented yet." << std::endl;
        exit(1);
    }

    std::pair<std::vector<BContainer>, AContainer> getEdaBits(
        const size_t& size, const size_t& bit_length = sizeof(Data) * 8) {
        // Eda-bits
        std::vector<BContainer> eda_bits_b;
        for (int i = 0; i < bit_length; i++) {
            eda_bits_b.push_back(BContainer(size));
        }
        AContainer eda_bits_a(size);

        return {eda_bits_b, eda_bits_a};
    }

    // TODO: generalize this
    void public_rca(const std::vector<BContainer>& xBits,
                             const ADataVector& yBits, std::optional<BContainer> res, 
                             std::optional<BContainer> signRes,
                                const int lsb = 0, int msb = -1, const bool signExtension = true) {
        static const int BITS_NUMBER = xBits.size();
        if (msb == -1) msb = BITS_NUMBER - 1;

        auto carry_i = xBits[0].construct_like(xBits[0].size());
        auto yBit = yBits.construct_like(xBits[0].size());
        auto xc_i = xBits[0].construct_like(xBits[0].size());

        BContainer sign = signRes.has_value()? signRes.value() : xBits[0].construct_like(xBits[0].size());

        // iterate over all bits and do full adder logic for the sign bit
        for (int i = 0; i < BITS_NUMBER; ++i) {
            auto& xBit = xBits[i];
            auto& carryBit = carry_i;
            auto& xcBit = xc_i;
            
            // extract the i-th bit of y
            yBit.extract_bit_from_vector(yBits, i);

            this->multiply_a(xBit, carryBit, xcBit);

            // don't compute carry on the last round
            // Instead compute the final sign bit
            if (((i == BITS_NUMBER - 1) && signRes.has_value()) || (res.has_value() && i >= lsb && i <= msb)) {
                auto yBitPublicShared = aProtocol->public_share(yBit);

                // r = x ^ y ^ c;
                // r = x + y + c - 2*(x*y + x*c + y*c) + 4*x*y*c;
                // r = x + y + c - 2*(x*y + c*y + x*c) + 4*x*y*c;
                sign = xBit + yBitPublicShared + carryBit - ((xBit * yBit + carryBit * yBit + xcBit) * 2) + xcBit * yBit * (4);
            }

            if(res.has_value() && ((i >= lsb && i <= msb))) {
                res.value() += (sign * ((UnsignedDataType)1 << (i-lsb)));
            }

            if (i == BITS_NUMBER - 1){
                if(signExtension) {
                    for(int i = msb + 1; i < BITS_NUMBER + lsb; i++) {
                        res.value() += (sign * ((UnsignedDataType)1 << (i-lsb)));
                    }
                }
                break;
            }

            // to calculate the ith bit of the carry, it is 1 if any 2 bits are 1 among the 3 bits (carry_i, xBits[i], yBits[i])
            // c = x&y || x&c || y&c;

            // and is just multiplication in arithmetic sharing,
            // c = x*y || x*c || y*c

            // Or is just addition - product
            // c = x*y + x*c + y*c - 2*(x*y*c)
            // c = x*y + y*c + x*c - 2*(x*y*c)
            // c = x*y + c*y + (1 - 2y) * x*c
            // c = x*y + c*y + (y*-2 + 1) * x*c

            // Note: y is public so only 1 secure multiplication is needed (x*c)
            auto temp = xBit * yBit + carryBit * yBit + xcBit + xcBit * yBit * (-2);
            carry_i = temp;
        }
    }

    // TODO: generalize this
    void gtez_a(const AContainer& in, AContainer& out) {
        const size_t size = in.size();
        auto sign_bit = out.construct_like(size);

        // TODO: works because ASharedVector and BSharedVector from same protocol
        this->ltz(in, sign_bit);

        // the sign is 1 when in it is negative, so we need to flip the bit to get gtez result
        this->not_b_1(sign_bit, out);
    }

    /**
     * @brief Defines vectorized less-than-zero comparison.
     *
     * @param in The input shared vector of size S.
     * @param out The output shared vector of size S.
     */
     // TODO: generalize this
    void ltz(const BContainer& in, BContainer& out) { 
        const size_t size = in.size();

        // TODO: get actual eda-bits not dummy ones.
        auto [eda_bits_b, eda_bits_a] = getEdaBits(size);

        // let's get the difference between input and eda_bits_a
        auto diff_a = in.construct_like(size);
        sub_a(in, eda_bits_a, diff_a);

        // Open the diff then decompose it into bits
        auto diff_a_opened = open_shares_a(diff_a); 

        // add eda_bits_b and diff_bits together to get the final result
        // We are interested in only the sign bit.
        
        this->public_rca(eda_bits_b, diff_a_opened, std::nullopt, out);
     }

    // // x is public and y is a bit decomposition of a public number
    // // y.size() = BITS_NUMBER * x.size()
    // BContainer less_than_public(const BDataVector& x, const BContainer& y) {
    //     // Prime Number
    //     const BShareType prime_number = 23;

    //     static const int MAX_BITS_NUMBER = std::numeric_limits<std::make_unsigned_t<Data>>::digits;
    //     // X: 0,1,2 ...
    //     // Y: 1[0],2[0],3[0], ..., n[0] , 1[1],2[1],3[1], ..., n[1], ...

    //     BContainer res(x.size());       // zero means that x < y
    //     BContainer prev_agg(x.size());  // zero means all bits are same so far

    //     for (int i = MAX_BITS_NUMBER - 1; i >= 0; i--) {
    //         // extract the i-th bit of x
    //         BDataVector x_i = (x >> i) & 1;

    //         // extract the i-th bit of y
    //         BContainer y_i(x.size());
    //         y_i = y.simple_subset(x.size() * i, 1, x.size() * (i + 1) - 1);

    //         // is the current two bits are different?
    //         BContainer x_i_XOR_y_i(x.size());
    //         BContainer x_i_XOR_y_i_01 =
    //             ((y_i) * ((((x_i * -1) % prime_number) + 1) % prime_number)) % prime_number;
    //         BContainer x_i_XOR_y_i_02 =
    //             (add_public_a(multiply_public_a(y_i, -1), 1) * x_i) % prime_number;
    //         add_a(x_i_XOR_y_i_01, x_i_XOR_y_i_02, x_i_XOR_y_i);

    //         if (i == MAX_BITS_NUMBER - 1) {
    //             prev_agg = x_i_XOR_y_i;

    //             // if current_agg/prev_agg is 1, it means first different bit is found ... assign
    //             // bit from y_i
    //             multiply_a(prev_agg, y_i, res);
    //         } else {
    //             BContainer current_res(x.size());

    //             // aggregate the bits
    //             BContainer current_agg = or_a(prev_agg, x_i_XOR_y_i);

    //             // first different bit
    //             sub_a(current_agg, prev_agg, current_res);

    //             // if we have a first different bit, assign bit from y_i
    //             multiply_a(current_res, y_i, current_res);

    //             // we want to detect only one such edge with y so we use addition
    //             add_a(res, current_res, res);

    //             // update aggregate for next iteration
    //             prev_agg = current_agg;
    //         }
    //     }
    //     return res;
    // }

    // BContainer least_siginificant_bit_a(const BContainer& x) {
    //     static const int MAX_BITS_NUMBER = std::numeric_limits<std::make_unsigned_t<Data>>::digits;
    //     const BShareType prime_number = 23;

    //     //////////// EDABITS /////////////
    //     BContainer r_a(x.size());  // arithmetic secret shares for random numbers
    //     BContainer r_b(
    //         MAX_BITS_NUMBER *
    //         x.size());  // equivalent boolean secret shares (decomposed) {0,0,0, .. 1,1,1, ...}

    //     // generate [r]_p and [r]_b
    //     // channel 6 for [r]_p and channel 7 for [r]_b
    //     this->randomGenerator->getMultipleNext(r_a.getContent(), 6, r_a.size());
    //     this->randomGenerator->getMultipleNext(r_b.getContent(), 7, r_b.size());
    //     ////////////////////////////////

    //     BContainer r_b_0(x.size());  // least significant bit of r_b
    //     r_b_0 = r_b;

    //     // [c]p = [x]p + [r]p
    //     auto c = x + r_a;

    //     // open c
    //     BDataVector c_opened = open_shares_a(c);
    //     BDataVector c_opened_0 = c_opened & 1;

    //     BContainer c_0_XOR_r_0(x.size());
    //     BContainer c_0_XOR_r_0_01 =
    //         ((r_b_0) * ((((c_opened_0 * -1) % prime_number) + 1) % prime_number)) % prime_number;
    //     BContainer c_0_XOR_r_0_02 =
    //         (add_public_a(multiply_public_a(r_b_0, -1), 1) * c_opened_0) % prime_number;
    //     add_a(c_0_XOR_r_0_01, c_0_XOR_r_0_02, c_0_XOR_r_0);

    //     // compute  [c <B r]p
    //     BContainer c_less_than_r = less_than_public(c_opened, r_b);

    //     // compute  c_less_than_r + c_0_XOR_r_0 - 2 * c_less_than_r * c_0_XOR_r_0;
    //     BContainer res = xor_a(c_less_than_r, c_0_XOR_r_0);

    //     return res;
    // }

    ////////////////////////////////////////////////////////////
    //////////////////////// Conversion ////////////////////////
    ////////////////////////////////////////////////////////////
    void b2a_bit(const AContainer& x, AContainer& y) {
        // TODO: change if not SPDZ2k
        y = x;
    }

    virtual void a2b_packed_sign_a_b(const AContainer& in, BContainer& out) {
        this->aProtocol->a2b_packed_sign_a_b(in, out);
    }

    ////////////////////////////////////////////////////////////
    ///////////////////// A Secret Sharing /////////////////////
    ////////////////////////////////////////////////////////////
    DataType reconstruct_from_a(const std::vector<Share<AShareType>>& shares) {
        return aProtocol->reconstruct_from_a(shares);
    }

    ADataVector reconstruct_from_a(const std::vector<AContainer>& shares) {
        return aProtocol->reconstruct_from_a(shares);
    }

    ADataVector open_shares_a(const AContainer& shares) { return aProtocol->open_shares_a(shares); }

    ADataVector unchecked_open_a(const AContainer& shares) {
        return aProtocol->unchecked_open_a(shares);
    }

    std::vector<AContainer> get_shares_a(const ADataVector& data) {
        return aProtocol->get_shares_a(data);
    }

    AContainer secret_share_a_internal(const ADataVector& data, const PartyID& data_party) {
        return aProtocol->secret_share_a_internal(data, data_party);
    }

    AContainer public_share(const ADataVector& x, const std::set<PartyID>& who_knows) {
        return aProtocol->public_share(x, who_knows);
    }


    AContainer public_share(const ADataVector& x) {
        return aProtocol->public_share(x);
    }

    // AContainer public_share(const ADataVector& x, const std::set<PartyID>& who_knows,
    //                              const int& fixed_point_precision) {
    //     return aProtocol->public_share(x, who_knows, fixed_point_precision);
    // }
    ////////////////////////////////////////////////////////////
    ///////////////////// B Secret Sharing /////////////////////
    ////////////////////////////////////////////////////////////
    DataType reconstruct_from_b(const std::vector<Share<BShareType>>& shares) {
        return aProtocol->reconstruct_from_b(shares);
    }

    BDataVector reconstruct_from_b(const std::vector<BContainer>& shares) {
        return aProtocol->reconstruct_from_b(shares);
    }

    BDataVector open_shares_b(const BContainer& shares) { return aProtocol->open_shares_b(shares); }

    BDataVector unchecked_open_b(const BContainer& shares) {
        return aProtocol->unchecked_open_b(shares);
    }

    std::vector<BContainer> get_shares_b(const BDataVector& data) {
        return aProtocol->get_shares_b(data);
    }

    BContainer secret_share_b_internal(const BDataVector& data, const PartyID& data_party) {
        return aProtocol->secret_share_b_internal(data, data_party);
    }

    std::pair<BContainer, BContainer> redistribute_shares_b(const AContainer& x) {
        return aProtocol->redistribute_shares_b(x);
    }

    bool malicious_check_internal() { return aProtocol->malicious_check_internal(); }

    void reset_malicious_state() { aProtocol->reset_malicious_state(); }
};

// Initialize the static member with a default value that will be overridden in constructor
template <typename Data, template <typename> class Share, template <typename> class Vector,
          template <typename, int, int> class EVector, typename AProtocol, typename BProtocol>
int edabits_protocol<Data, Share, Vector, EVector, AProtocol, BProtocol>::parties_num = 2;

template <template <template <typename> class, template <typename> class,
                    template <typename> class> class PFA,
          template <template <typename> class, template <typename> class,
                    template <typename> class> class PFB,
          template <typename> class S, template <typename> class V, template <typename> class E>
class EDA_BITS_Factory : public ProtocolFactory<EDA_BITS_Factory<PFA, PFB, S, V, E>> {
   public:
    using ProtocolFactoryA = PFA<S, V, E>;
    using ProtocolFactoryB = PFB<S, V, E>;

    template <typename T>
    using ProtocolInstance =
        edabits_protocol<T, S, V, cdough::EVector,
                         typename ProtocolFactoryA::template ProtocolInstance<T>,
                         typename ProtocolFactoryB::template ProtocolInstance<T>>;

    /**
     * @brief Open boolean shares to reveal plaintext.
     * Note: functionality not supported in SPDZ-2k.
     *
     * @param shares Input shared vector.
     * @return Opened plaintext vector.
     */

    template <typename T>
    using AInnerContainer = typename ProtocolFactoryA::template AInnerContainer<T>;

    template <typename T>
    using BInnerContainer = typename ProtocolFactoryB::template BInnerContainer<T>;

   public:
    /**
     * @brief Constructor for the EDA_BITS_Factory.
     *
     * @param partyID The unique identifier of the party using this factory.
     * @param partiesNumber The total number of parties in the protocol.
     */
    EDA_BITS_Factory(const int& partyID, const int& partiesNumber)
        : partyID_(partyID),
          partiesNumber_(partiesNumber),
          aFactory(partyID, partiesNumber),
          bFactory(partyID, partiesNumber) {
        if (partyID == 0) {
            std::cout << "EDA_BITS Factory created for " << partiesNumber_ << " parties."
                      << std::endl;
        }
    }

    /**
     * @brief Create an instance of the EDA_BITS protocol for a specific data type.
     *
     * @param communicator A pointer to the communicator.
     * @param randomnessManager A pointer to the randomness manager.
     * @return A unique pointer to the created protocol instance.
     */
    template <typename T>
    std::unique_ptr<ProtocolBase> create(int thread_id, Communicator* communicator,
                                         random::RandomnessManager* randomnessManager) {
        ProtocolInstance<T>::parties_num = partiesNumber_;

        auto aProtocol = aFactory.template create<T>(thread_id, communicator, randomnessManager);
        auto bProtocol = bFactory.template create<T>(thread_id, communicator, randomnessManager);

        // Cast base pointers to specific protocol types
        auto typedAProtocol =
            std::unique_ptr<typename ProtocolFactoryA::template ProtocolInstance<T>>(
                static_cast<typename ProtocolFactoryA::template ProtocolInstance<T>*>(
                    aProtocol.release()));
        auto typedBProtocol =
            std::unique_ptr<typename ProtocolFactoryB::template ProtocolInstance<T>>(
                static_cast<typename ProtocolFactoryB::template ProtocolInstance<T>*>(
                    bProtocol.release()));

        auto protocol = std::make_unique<ProtocolInstance<T>>(
            partyID_, partiesNumber_, communicator, randomnessManager, std::move(typedAProtocol),
            std::move(typedBProtocol));

        return protocol;
    }

   private:
    PartyID partyID_;
    int partiesNumber_;

    ProtocolFactoryA aFactory;
    ProtocolFactoryB bFactory;
};

template <template <typename> class S, template <typename> class V, template <typename> class E>
using EDA_BITS_SPDZ_Factory = EDA_BITS_Factory<SPDZ_2k_Factory, SPDZ_2k_Factory, S, V, E>;

}  // namespace cdough::protocols
