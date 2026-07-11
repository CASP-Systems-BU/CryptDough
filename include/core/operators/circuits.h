/**
 * @file circuits.h
 * @brief Implementation of various boolean circuits
 *
 * (moved out of BSharedVector.h)
 *
 */
#pragma once

#include "core/containers/b_shared_vector.h"

namespace cdough::operators {
/**
 * @brief Alias for a unique pointer to a BSharedVector.
 *
 * @tparam S
 * @tparam E
 */
template <typename S, typename E, typename Eng>
using unique_B = std::unique_ptr<BSharedVector<S, E, Eng>>;

}  // namespace cdough::operators

// Out-of-line definitions for BSharedVector member circuits.

/**
 * @brief Binary division using non-restoring algorithm.
 *
 * See: https://en.wikipedia.org/wiki/Division_algorithm for more.
 *
 * For now, only works for <= 64 bit ints, since we need twice as many
 * bits to implement the algorithm.
 *
 * Complexity is \f$\ell\times\f$ the complexity of boolean addition.
 *
 * @param other
 * @return std::unique_ptr<BSharedVector>
 */
template <typename T, typename E, typename Eng>
inline std::unique_ptr<cdough::BSharedVector<T, E, Eng>> cdough::BSharedVector<T, E, Eng>::operator/(
    const cdough::BSharedVector<T, E, Eng>& other) const {
    // Division is not supported for 128-bit types (no 256-bit type exists for intermediate values)
    if constexpr (std::is_same_v<T, __int128_t> || std::is_same_v<T, __uint128_t>) {
        std::cerr << "Error: Division operator is not supported for 128-bit integer types."
                  << std::endl;
        exit(-1);
    } else {
        auto& engine = this->engine;

        /* The data type of intermediate values. Eventually, this could be
         * dynamically generated (i.e., for int8_t BSharedVector, use
         * int16_t intermediates).
         */
        using T2 = DoubleWidth<T>::type;

        /* The EVector type (this should mirror the current EVector type,
         * just with a larger base datatype T.)
         */
        using E2 = typename Eng::template GenericInnerContainer<T2>;

        auto size = this->size();

#ifdef INSTRUMENT_TABLES
        single_cout("[PRIV_DIV] n=" << size);
#endif

        BSharedVector<T2, E2, Eng> r(size, engine);
        r = *this;

        // Must do this in 3 steps to force larger types
        BSharedVector<T2, E2, Eng> d(size, engine);
        d = other;
        d <<= BSharedVector<T, E, Eng>::MAX_BITS_NUMBER;

        BSharedVector<T2, E2, Eng> q(size, engine);

        BSharedVector<T2, E2, Eng> c(size, engine);

        BSharedVector<T2, E2, Eng> neg_d = -d;

        for (int i = BSharedVector<T, E, Eng>::MAX_BITS_NUMBER - 1; i >= 0; i--) {
            // c := r >= 0
            c = !r.ltz();

            /* 1 bit of the division, q(i). Instead of indexing, use
             * bitshift to get bit `c` to the `i`th location.
             *
             * If r >= 0 (c == 1), q(i) := 1
             * Otherwise (c == 0), q(i) := 0
             */
            q ^= c << i;

#ifdef DEBUG_DIVISION
            single_cout_nonl(VAR(i) << "r ");
            print(r.open());
            single_cout_nonl("q ");
            print(q.open());
            single_cout_nonl("c ");
            print(c.open());
#endif

            /* Update r. We multiply by 2 (`r << 1`) and then either add or
             * subtract d, based on the sign of r.
             *
             * If c == 1, subtract d.
             * If c == 0, add d.
             *
             * Implement the above with multiplex.
             */
            r = (r << 1) + operators::multiplex(c, d, neg_d);
        }

        /* Perform final correction. At this point, we don't actually have
         * a real binary string: `0` bits represent -1. The expression
         * `q -= ~q` corrects this.
         *
         * Then, adjust the parity of q: at this stage, q is always odd. If
         * r is negative (`r.ltz() == 1`), we subtract 1.
         *
         */
        q -= (~q) + r.ltz();

        // Reassign to the base type
        auto res = std::make_unique<BSharedVector<T, E, Eng>>(size, engine);
        *res = q;

        return res;
    }
}
