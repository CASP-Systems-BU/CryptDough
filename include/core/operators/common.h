#pragma once

#include "core/containers/a_shared_vector.h"
#include "core/containers/b_shared_vector.h"

namespace cdough {

namespace aggregators {

    /**
     * @brief Denotes the direction of an aggregation.
     *
     */
    enum class Direction { Forward, Reverse } Direction;
}  // namespace aggregators

namespace operators {

    /**
     * @brief Conditional selection between two arithmetic shared vectors.
     *
     * Performs oblivious selection: returns a if sel is 0, b if sel is 1.
     * Uses linear combination for arithmetic shares.
     *
     * @tparam Share The underlying data type of the shared vectors.
     *
     * @param sel Arithmetic selector vector.
     * @param a First input vector (returned when sel is 0).
     * @param b Second input vector (returned when sel is 1).
     * @return The multiplexed result vector.
     */
    template <typename Share, typename EVector, typename Engine>
    static ASharedVector<Share, EVector, Engine> multiplex(
        const ASharedVector<Share, EVector, Engine>& sel,
        const ASharedVector<Share, EVector, Engine>& a,
        const ASharedVector<Share, EVector, Engine>& b) {
        ASharedVector<Share, EVector, Engine> res = a + sel * (b - a);
        return res;
    }

    /**
     * @brief Conditional selection between two boolean shared vectors.
     *
     * Performs oblivious selection: returns a if sel is false, b if sel is true.
     *
     * @tparam Share The underlying data type of the shared vectors.
     * @tparam EVector The container type of the shared vectors.
     * @tparam Engine The secure computation engine type.
     *
     * @param sel Boolean selector vector (extended to full bit width).
     * @param a First input vector (returned when sel is false).
     * @param b Second input vector (returned when sel is true).
     * @return The multiplexed result vector.
     */
    template <typename Share, typename EVector, typename Engine>
    static BSharedVector<Share, EVector, Engine> multiplex(
        const BSharedVector<Share, EVector, Engine>& sel,
        const BSharedVector<Share, EVector, Engine>& a,
        const BSharedVector<Share, EVector, Engine>& b) {
        auto& engine = sel.engine;

        if constexpr (SPDZ2k_NPC_PROTOCOL) {
            EVector selVector = sel.vector;
            EVector aVector = a.vector;
            EVector bVector = b.vector;
            ASharedVector<Share, EVector, Engine> sel_a(selVector, sel.engine);
            ASharedVector<Share, EVector, Engine> a_a(aVector, a.engine);
            ASharedVector<Share, EVector, Engine> b_a(bVector, b.engine);
            ASharedVector<Share, EVector, Engine> res_a = multiplex(sel_a, a_a, b_a);
            BSharedVector<Share, EVector, Engine> res_b(res_a.vector, engine);
            return res_b;
        }

        BSharedVector<Share, EVector, Engine> sel_extended(sel.size(), engine);
        sel_extended.extend_lsb(sel);
        BSharedVector<Share, EVector, Engine> res = a ^ (sel_extended & (b ^ a));
        return res;
    }

    /**
     * @brief Identity conversion for boolean shared vectors.
     * @tparam Share The underlying data type of the shared vectors.
     * @param x Input boolean shared vector.
     * @param res Output boolean shared vector (copy of input).
     */
    template <typename Share, typename EVector, typename Engine>
    static void auto_conversion(BSharedVector<Share, EVector, Engine>& x,
                                BSharedVector<Share, EVector, Engine>& res) {
        res = x;
    }

    /**
     * @brief Identity conversion for arithmetic shared vectors.
     * @tparam Share The underlying data type of the shared vectors.
     * @param x Input arithmetic shared vector.
     * @param res Output arithmetic shared vector (copy of input).
     */
    template <typename Share, typename EVector, typename Engine>
    static void auto_conversion(ASharedVector<Share, EVector, Engine>& x,
                                ASharedVector<Share, EVector, Engine>& res) {
        res = x;
    }

    /**
     * @brief Converts boolean shared vector to arithmetic shared vector.
     * @tparam Share The underlying data type of the shared vectors.
     * @tparam EVector The container type of the shared vectors.
     * @tparam Engine The secure computation engine type.
     * @param x Input boolean shared vector.
     * @param res Output arithmetic shared vector.
     */
    // TODO: differentiate between "b2a_bit" and just "b2a"
    template <typename Share, typename EVector, typename Engine>
    static void auto_conversion(BSharedVector<Share, EVector, Engine>& x,
                                ASharedVector<Share, EVector, Engine>& res) {
        res = x.b2a_bit();
    }

    /**
     * @brief Converts arithmetic shared vector to boolean shared vector.
     * @tparam Share The underlying data type of the shared vectors.
     * @tparam EVector The container type of the shared vectors.
     * @tparam Engine The secure computation engine type.
     * @param x Input arithmetic shared vector.
     * @param res Output boolean shared vector.
     */
    template <typename Share, typename EVector, typename Engine>
    static void auto_conversion(ASharedVector<Share, EVector, Engine>& x,
                                BSharedVector<Share, EVector, Engine>& res) {
        res = x.a2b();
    }

    /**
     * @brief Creates boolean shared vector from plaintext data.
     * @tparam Share The underlying data type of the shared vectors.
     * @tparam EVector The container type of the shared vectors.
     * @tparam Engine The secure computation engine type.
     * @param data Plaintext vector to be secret-shared.
     * @param out Output boolean shared vector.
     * @param data_party The party that provides the input data.
     */
    template <typename Share, typename EVector, typename Engine>
    static void secret_share_vec(const Vector<Share>& data,
                                 BSharedVector<Share, EVector, Engine>& out,
                                 PartyID data_party = 0) {
        auto& engine = out.engine;
        out = engine.secret_share_b(data, data_party);
    }

    /**
     * @brief Creates arithmetic shared vector from plaintext data.
     * @tparam Share The underlying data type of the shared vectors.
     * @tparam EVector The container type of the shared vectors.
     * @tparam Engine The secure computation engine type.
     * @param data Plaintext vector to be secret-shared.
     * @param out Output arithmetic shared vector.
     * @param data_party The party that provides the input data.
     */
    template <typename Share, typename EVector, typename Engine>
    static void secret_share_vec(const Vector<Share>& data,
                                 ASharedVector<Share, EVector, Engine>& out,
                                 PartyID data_party = 0) {
        auto& engine = out.engine;
        out = engine.secret_share_a(data, data_party);
    }
}  // namespace operators
}  // namespace cdough
