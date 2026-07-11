#pragma once

#include <NTL/ZZ.h>
#include <NTL/ZZ_limbs.h>

#include <limits>
#include <type_traits>

namespace cdough::math {

static_assert(std::numeric_limits<unsigned long>::digits == 64, "unsigned long must be 64 bits");

const NTL::ZZ pow_128 = NTL::ZZ(1) << 128;

double log2(NTL::ZZ x) { return NTL::log(x) / log(2); }

template <typename T>
T div_ceil(T x, T y) {
    return (x + y - 1) / y;
}

/**
 * @brief Convert a `T` to an arbitrary-precision NTL integer.
 *
 * @tparam T
 * @param x
 * @return NTL::ZZ
 */
template <typename T>
NTL::ZZ to_ZZ(T x) {
    if constexpr (sizeof(T) <= sizeof(long)) {
        return NTL::ZZ(x);
    } else {
        // 128-bit
        using U = std::make_unsigned_t<T>;
        static_assert(std::numeric_limits<U>::digits == 128);

        U ux = static_cast<U>(x);
        auto lo = static_cast<unsigned long>(ux);
        auto hi = static_cast<unsigned long>(ux >> 64);
        NTL::ZZ res = (NTL::to_ZZ(hi) << 64) | NTL::to_ZZ(lo);

        if constexpr (std::is_signed_v<T>) {
            if (x < 0) {
                // handle 2s compl
                res -= pow_128;
            }
        }

        return res;
    }
}

/**
 * @brief Convert an NTL ZZ integer to a `T`, possibly destructively.
 *
 * @tparam T
 * @param x
 * @return T
 */
template <typename T>
T from_ZZ(NTL::ZZ x) {
    if constexpr (sizeof(T) <= sizeof(long)) {
        if constexpr (std::is_signed_v<T>) {
            return static_cast<T>(NTL::to_long(x));
        } else {
            return static_cast<T>(NTL::to_ulong(x));
        }
    } else {
        // 128-bit
        using U = std::make_unsigned_t<T>;
        static_assert(std::numeric_limits<U>::digits == 128);

        if constexpr (std::is_signed_v<T>) {
            if (x < 0) {
                // handle 2s compl
                x += pow_128;
            }
        }

        uint64_t lo = NTL::to_ulong(x);
        uint64_t hi = NTL::to_ulong(x >> 64);

        return static_cast<T>((static_cast<U>(hi) << 64) | lo);
    }
}
}  // namespace cdough::math