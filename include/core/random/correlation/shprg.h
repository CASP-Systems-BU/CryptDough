#pragma once

#include <vector>

#include "core/containers/vector.h"
#include "core/math/util.h"

namespace cdough::random {

/**
 * @brief Seed Homomorphic Pseudo-Random Generator.
 *
 * A PRG with the property that PRG(s1 + s2) = PRG(s1) + PRG(s2) + error,
 * enabling distributed seed generation and expansion.
 *
 * https://eprint.iacr.org/2015/220.pdf

 * This implementation uses both q and p as powers of 2.
 *
 * @tparam InT The input type.
 * @tparam OutT The output type.
 */
template <typename InT, typename OutT>
    requires(std::is_integral_v<InT> && std::is_integral_v<OutT> &&
             std::numeric_limits<OutT>::digits < std::numeric_limits<InT>::digits)
class SeedHomomorphicPRG {
    // once we sample an SHPRG / public parameter matrix, the seed length is fixed
    const size_t seed_length;

    // store the public parameter as integers of type InT
    using public_param_t = std::vector<std::vector<InT>>;
    public_param_t public_parameter;

    // a binary seed type where each element is represented with a single bit
    using seed_t = std::vector<InT>;
    using compressed_seed_t = std::vector<__int128_t>;

    /**
     * @brief Round an input value to an output value.
     *
     * Select the upper most bits of the input value.
     *
     * @param x The input value.
     * @return The rounded output value.
     */
    OutT round(InT x) const {
        constexpr size_t in_bits = sizeof(InT) * 8;
        constexpr size_t out_bits = sizeof(OutT) * 8;
        constexpr size_t shift = in_bits - out_bits;
        return static_cast<OutT>(x >> shift);
    }

   public:
    /**
     * @brief Constructor for the SHPRG.
     *
     * @tparam EngineType The engine type that provides populateLocalRandom.
     * @param engine The engine to use for random generation.
     * @param lambda The security parameter (seed size).
     * @param n The max expansion for a single expansion (output size).
     */
    template <typename EngineType>
    SeedHomomorphicPRG(EngineType& engine, const size_t seed_length, const size_t n)
        : seed_length(seed_length) {
        // sample the public parameter matrix
        // matrix is n x lambda: n rows (output size) by lambda cols (seed size)
        public_parameter = sample_public_parameter(engine, n, seed_length);
    }

    /**
     * @brief Sample a public parameter matrix for the SHPRG.
     *
     * Uniformly random over the field mod q (q is a power of 2).
     *
     * @tparam EngineType The engine type that provides populateLocalRandom.
     * @param engine The engine to use for random generation.
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @return A matrix of random elements.
     */
    template <typename EngineType>
    public_param_t sample_public_parameter(EngineType& engine, size_t rows, size_t cols) {
        public_param_t public_parameter(rows);
        for (size_t i = 0; i < rows; ++i) {
            cdough::Vector<InT> row_vec(cols);
            engine.populateLocalRandom(row_vec);
            public_parameter[i].assign(row_vec.begin(), row_vec.end());
        }
        return public_parameter;
    }

    /**
     * @brief Sample a (full domain) seed vector.
     *
     * @param size The size of the seed vector.
     * @return The seed vector.
     */
    template <typename EngineType>
    seed_t sample_seed(EngineType& engine) {
        cdough::Vector<InT> seed_vec(seed_length);
        engine.populateLocalRandom(seed_vec);
        return seed_vec.as_std_vector();
    }

    /**
     * @brief Sample a binary seed vector.
     *
     * @param size The size of the seed vector.
     * @return The seed vector.
     */
    template <typename EngineType>
    seed_t sample_binary_seed(EngineType& engine) {
        seed_t seed(seed_length);

        // generate all random bits at once
        size_t num_blocks = (seed_length + 127) / 128;
        cdough::Vector<__int128_t> seed_vec(num_blocks);
        engine.populateLocalRandom(seed_vec);
        for (size_t block = 0; block < num_blocks; ++block) {
            for (size_t bit = 0; bit < 128; ++bit) {
                seed[block * 128 + bit] = (seed_vec[block] >> bit) & 1;
            }
        }
        return seed;
    }

    /**
     * @brief Add two seed vectors.
     *
     * @param seed1 The first seed vector.
     * @param seed2 The second seed vector.
     * @return The sum of the two seed vectors.
     */
    seed_t add_seeds(const seed_t& seed1, const seed_t& seed2) {
        seed_t seed(seed_length);
        for (size_t i = 0; i < seed_length; ++i) {
            seed[i] = seed1[i] + seed2[i];
        }
        return seed;
    }

    /**
     * @brief Compress a binary seed vector into a densely packed vector.
     *
     * @param seed The seed vector.
     * @return The seed vector compressed into a densely packed vector.
     */
    compressed_seed_t compress_binary_seed(const seed_t& seed) {
        // determine the number of 128-bit blocks needed, round up to the nearest integer
        size_t num_blocks = (seed.size() + 127) / 128;
        compressed_seed_t compressed_seed(num_blocks);
        size_t i = 0;
        for (size_t block = 0; block < num_blocks; ++block) {
            for (int j = 0; j < 128; ++j) {
                compressed_seed[block] |= ((__int128_t)seed[i] << j);
                i++;
            }
        }
        return compressed_seed;
    }

    /**
     * @brief Expand a seed using the SHPRG.
     *
     * @param seed The input seed vector.
     * @return The expanded output vector.
     */
    std::vector<OutT> expand(const seed_t& seed) {
        std::vector<OutT> expanded(public_parameter.size());
        for (size_t i = 0; i < public_parameter.size(); ++i) {
            InT sum = 0;
            for (size_t j = 0; j < seed_length; ++j) {
                sum += public_parameter[i][j] * seed[j];
            }
            expanded[i] = round(sum);
        }
        return expanded;
    }

    /**
     * @brief Expand a binary seed vector.
     *
     * @param seed The input binary seed vector.
     * @return The expanded output vector.
     */
    std::vector<OutT> expand_binary_seed(const seed_t& seed) {
        std::vector<OutT> expanded(public_parameter.size());
        size_t i = 0;
        for (const auto& row : public_parameter) {
            InT sum = 0;
            // dot product, for binary seed this reduces to a sum of field elements
            for (size_t j = 0; j < row.size(); ++j) {
                if (seed[j] == 1) {
                    sum += row[j];
                }
            }
            expanded[i] = round(sum);
            i++;
        }
        return expanded;
    }

    /**
     * @brief Expand a compressed binary seed vector.
     *
     * @param compressed_seed The input compressed binary seed vector.
     * @return The expanded output vector.
     */
    std::vector<OutT> expand_compressed_binary_seed(const compressed_seed_t& compressed_seed) {
        std::vector<OutT> expanded(public_parameter.size());
        size_t i = 0;
        for (const auto& row : public_parameter) {
            InT sum = 0;
            // dot product, for binary seed this reduces to a sum of field elements
            for (size_t block = 0; block < compressed_seed.size(); ++block) {
                for (size_t j = 0; j < 128; ++j) {
                    // check the bit in the 128-bit block to determine whether to add
                    if ((compressed_seed[block] >> j) & 1) {
                        sum += row[block * 128 + j];
                    }
                }
            }
            expanded[i] = round(sum);
            i++;
        }
        return expanded;
    }

    /**
     * @brief Add two result vectors.
     *
     * @param result1 The first result vector.
     * @param result2 The second result vector.
     * @return The sum of the two result vectors.
     */
    std::vector<OutT> add_results(const std::vector<OutT>& result1,
                                  const std::vector<OutT>& result2) {
        std::vector<OutT> ret(result1.size());
        // vector addition
        for (size_t i = 0; i < result1.size(); ++i) {
            ret[i] = result1[i] + result2[i];
        }
        return ret;
    }
};

}  // namespace cdough::random
