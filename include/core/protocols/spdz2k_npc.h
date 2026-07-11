#pragma once

#include <algorithm>

#include "core/protocols/interface/interface.h"

// According to theorem 1, error probability < 2^(−s+log(s+1) )
// hence, we solve mapping s' = s-log(s+1)
// For S' = 42.4 >>>> s = 48
#define SPDZ2K_STAT_SECURITY_PARAMETER 48

// How many vectors needed to store the secret shares for each plaintext vector
#define SPDZ2K_MULTIVECTOR_SIZE 2

namespace cdough::protocols {

namespace {
    // Select share type based on data type
    template <typename T>
    struct ShareTypeSelector {
        using type = __int128_t;  // Default case for `int32_t` and `int64_t`
    };

    template <>
    struct ShareTypeSelector<int8_t> {
        using type = int64_t;
    };

    template <>
    struct ShareTypeSelector<int16_t> {
        using type = int64_t;
    };

    template <>
    struct ShareTypeSelector<int32_t> {
        using type = __int128_t;
    };

    template <>
    struct ShareTypeSelector<int64_t> {
        using type = __int128_t;
    };

    /* NOT SUPPORTED*/
    template <>
    struct ShareTypeSelector<__int128_t> {
        using type = __int128_t;
    };
}  // namespace

/**
 * @brief Implementation of the SPDZ-2k protocol for n-party.
 *
 * The SPDZ-2k protocol provides malicious security against dishonest majority,
 * operating over signed integers of k bits with s bits of statistical security.
 * The total share size used is (s+k) bits. We expect input data to be within k bits.
 *
 * Paper URL: https://eprint.iacr.org/2018/482.pdf
 *
 * Security Notes:
 * 1. For production use, ensure that you use secure authenticated preprocessing generators
 *    rather than the included dummy generators.
 * 2. Shares of data are exchanged between parties but shares of MACs are only locally aggregated
 *    and seen as an aggregate by other users. Do not expose your MACs to other parties.
 * 3. Fresh seeds are generated using a commitment protocol for each aggregation round
 *    to prevent malicious parties from pre-computing attack strategies. Seeds are committed
 *    in batches and opened on-demand to reduce communication overhead.
 *
 * @tparam Data - Plaintext data type.
 * @tparam Share - Share type.
 * @tparam Vector - Data container type.
 * @tparam EVector - Share container type.
 */
template <typename Data, template <typename> class Share, template <typename> class Vector,
          template <typename, int, int> class EVector>
class SPDZ_2k
    : public Protocol<Data, Share<typename ShareTypeSelector<Data>::type>, Vector<Data>,
                      EVector<typename ShareTypeSelector<Data>::type, SPDZ2K_MULTIVECTOR_SIZE,
                              std::numeric_limits<Data>::digits>> {
   public:
    using DataType = Data;
    using DataVector = Vector<DataType>;
    using ShareType = typename ShareTypeSelector<Data>::type;
    using ShareMultiVector =
        EVector<ShareType, SPDZ2K_MULTIVECTOR_SIZE, std::numeric_limits<DataType>::digits>;

   private:
    // Configuration parameters for data size and security.
    using UnsignedDataType = typename std::make_unsigned<DataType>::type;
    using UnsignedShareType = typename std::make_unsigned<ShareType>::type;

    const ShareType MAX_BITS_NUMBER = std::numeric_limits<UnsignedShareType>::digits;
    const ShareType k = std::numeric_limits<UnsignedDataType>::digits;  // Data size parameter
    const ShareType s = std::max((ShareType)SPDZ2K_STAT_SECURITY_PARAMETER,
                                 MAX_BITS_NUMBER - k);  // Statistical security parameter
    const ShareType r = MAX_BITS_NUMBER - s - k;        // Extra overflow bits; not needed

    // How many bytes for the hash result
    static const auto HASH_RES_BYTES = crypto_generichash_BYTES;
    // How many bytes for the hash key
    static const auto HASH_KEY_BYTES = crypto_generichash_KEYBYTES;
    // How many bytes for the hash message, which is the `aggregatedDiff`
    static const auto HASH_MESSAGE_BYTES = sizeof(ShareType);
    // How many bytes for the packed message (message + key), sent in same communication package
    static const auto HASH_PACKED_MESSAGE_BYTES = sizeof(ShareType) + crypto_generichash_BYTES;
    // How many ShareType elements are needed to store the packed message
    static const auto HASK_PACKED_MESSAGE_SIZE =
        (HASH_PACKED_MESSAGE_BYTES + sizeof(ShareType) - 1) / sizeof(ShareType);  // ceiling div

    // Derived parameters
    const ShareType s2 = (ShareType)1 << (s - 1);       // 2^(s - 1)
    const ShareType k2 = (ShareType)1 << (k - 1);       // 2^(k - 1)
    const ShareType ks2 = (ShareType)1 << (k + s - 1);  // 2^(s + k - 1)

    // Masks
    const ShareType mask_s = ((ShareType)1 << s) - 1;         // Mask for s bits
    const ShareType mask_k = ((ShareType)1 << k) - 1;         // Mask for k bits
    const ShareType mask_ks = ((ShareType)1 << (k + s)) - 1;  // Mask for (k+s) bits

    // Secret share of the key, it uses `s` bits but stored in `s+k` bits element.
    // In other words, the random value is mod 2^s but the alpha element is mod 2^(s+k).
    const ShareType alpha_i;

    // Back log for later verification of the `aggregatedDiff`.
    std::vector<ShareMultiVector> aggregationBackLog;
    size_t backLogElementSize = 0;

    // 16 million elements
    static constexpr size_t HEURISTIC_MAX_BACKLOG_ELEMENT_SIZE = 1 << 24;
    // 24 vectors in the back log (i.e. 12 secure batches)
    static constexpr size_t HEURISTIC_MAX_BACKLOG_SIZE = 24;

    // To aggregate the difference between (1) the openedShares * partyKeyShare and (2) the
    // partyMacSecretShares.
    ShareType aggregatedDiff = 0;

    // We have two preprocessing generators:
    //  1- Authanticated random numbers generator. It generates two types:
    //    - Random secret shared values of `s` bits: used in randomizing shares before opening.
    //    - Random secret shared values of `k` bits and their values in the clear: used in secret
    //      sharing inputs.
    std::shared_ptr<random::AuthRandomGeneratorBase<ShareType, ShareMultiVector>> authRGen;

    //  2- Authanticated Beaver triples: to evaluate the secure multiplication.
    std::shared_ptr<random::AuthTripleGeneratorBase<ShareType, ShareMultiVector>> authTGen;

    //  3- Common PRG Generator: to calculate the `aggregatedDiff`.
    std::shared_ptr<random::CommonPRGManager> rGen;

    //  4- LocalPRG: used to pad the aggregatedDiff to prevent reversal of the hash.
    std::shared_ptr<random::CommonPRG> localPRG;

    //  5- CommittedSeedsQueue: for fresh seeds in each aggregation round
    std::unique_ptr<random::CommittedSeedsQueue> seedQueue;

   public:
    // Configuration Parameters for number of parties.
    static int parties_num;

    /**
     * @brief Constructor for SPDZ-2k protocol.
     *
     * @param _partyID The (globally) unique identifier of the party that calls this constructor.
     * @param _numParties The total number of computing parties participating in the protocol.
     * @param _alpha_i The secret share of the MAC key for this party.
     * @param _communicator A pointer to the communicator.
     * @param _randomnessManager A pointer to the randomness manager.
     * @param _authRGen A pointer to the authenticated random generator.
     * @param _authTGen A pointer to the authenticated triple generator.
     */
    SPDZ_2k(const PartyID _partyID, const PartyID _numParties, const ShareType _alpha_i,
            Communicator* _communicator, random::RandomnessManager* _randomnessManager,
            const std::shared_ptr<random::AuthRandomGeneratorBase<ShareType, ShareMultiVector>>&
                _authRGen,
            const std::shared_ptr<random::AuthTripleGeneratorBase<ShareType, ShareMultiVector>>&
                _authTGen)
        : Protocol<Data, Share<ShareType>, Vector<Data>,
                   EVector<ShareType, SPDZ2K_MULTIVECTOR_SIZE, std::numeric_limits<Data>::digits>>(
              _communicator, _randomnessManager, _partyID, _numParties, SPDZ2K_MULTIVECTOR_SIZE),
          alpha_i(_alpha_i) {
        // Initialize the random generators
        authRGen = _authRGen;
        authTGen = _authTGen;
        rGen = _randomnessManager->commonPRGManager;
        localPRG = _randomnessManager->localPRG;

        // Other configuration parameters
        parties_num = _numParties;

        // Generate assisting members
        // 1- Generate vectors of party IDs for communication
        for (int i = 1; i < this->numParties; ++i) {
            toPartyIDs_.push_back(i);
            fromPartyIDs_.push_back(-i);
        }
        // 2- Generate a set that has all parties IDs.
        allPartiesSet_.insert(toPartyIDs_.begin(), toPartyIDs_.end());
        allPartiesSet_.insert(0);

        // 3- Initialize the committed seeds queue for fresh randomness
        seedQueue = std::make_unique<random::CommittedSeedsQueue>(
            *_communicator, localPRG, _partyID, _numParties, toPartyIDs_, fromPartyIDs_);

        // Initialize the first batch of committed seeds
        seedQueue->repopulateQueue();

        assert(s >= SPDZ2K_STAT_SECURITY_PARAMETER);
    }

    /**
     * @brief Secure arithmetic multiplication of two secret shared vectors.
     *
     * @param x The first secret shared vector.
     * @param y The second secret shared vector.
     * @param z The output secret shared vector to store the result.
     */
    void multiply_a(const ShareMultiVector& x, const ShareMultiVector& y, ShareMultiVector& z) {
        // If r < 0, the data type is not large enough to enforce the security parameter `s`.
        assert(r >= 0);

        // Get the next authenticated Beaver triples.
        auto [a, b, c] = authTGen->getNext(x.size());

        // Partial_open only opens the shares and add opened shares to back log. We do not need to
        // add random numbers during opening because the random beaver triples give the same effect.
        // However, opening secret shares outside the multiplication should check the
        // `aggregatedDiff` also by running `finalizeBatchCheck`.
        auto A = partial_open(x + a, false);
        auto B = partial_open(y + b, false);

        z = y * A - a * B + c;

        // It is only required before open,
        // but we do this optionally to reduce memory overhead.
        optionalAggregationBackLogOptimization();

        this->handle_precision(x, y, z);
        this->truncate(z);
    }

    void matrix_right_multiply_with_column_matrix_vectorized_a(
        const ShareMultiVector& x, const ShareMultiVector& y, ShareMultiVector& z,
        const size_t lhs_rows, const size_t lhs_cols, const size_t rhs_rows,
        const size_t rhs_cols) {
        // If r < 0, the data type is not large enough to enforce the security parameter `s`.
        assert(r >= 0);

        // Number of elements
        const size_t xSize = x.size();
        const size_t ySize = y.size();
        const size_t newSize = z.size();

        // TODO: generate matrix triples using a generator
        ShareMultiVector a(xSize), b(ySize), c(newSize);

        auto A = partial_open(x + a, false);
        auto B = partial_open(y + b, false);

        z(0) = A.matrixRightMultiplyWithColumnMatrixVectorized(y(0), lhs_rows, lhs_cols, rhs_rows,
                                                               rhs_cols) -
               a(0).matrixRightMultiplyWithColumnMatrixVectorized(B, lhs_rows, lhs_cols, rhs_rows,
                                                                  rhs_cols) +
               c(0);

        z(1) = A.matrixRightMultiplyWithColumnMatrixVectorized(y(1), lhs_rows, lhs_cols, rhs_rows,
                                                               rhs_cols) -
               a(1).matrixRightMultiplyWithColumnMatrixVectorized(B, lhs_rows, lhs_cols, rhs_rows,
                                                                  rhs_cols) +
               c(1);

        // TODO: we should do aggregation per round not batch.
        // Ensure all back log is aggregated
        processAggregationBackLog();

        this->handle_precision(x, y, z);
        this->truncate(z);
    }

    void conv_2d_vectorized_a(const ShareMultiVector& x, const ShareMultiVector& y,
                              ShareMultiVector& z, const size_t instancesCount,
                              const size_t inputHeight, const size_t inputWidth,
                              const size_t filterHeight, const size_t filterWidth,
                              const size_t strideHeight, const size_t strideWidth,
                              const size_t paddingHeight, const size_t paddingWidth) {
        // If r < 0, the data type is not large enough to enforce the security parameter `s`.
        assert(r >= 0);

        // Number of elements
        const size_t xSize = x.size();
        const size_t ySize = y.size();
        const size_t newSize = z.size();

        // TODO: generate convolution triples using a generator
        ShareMultiVector a(xSize), b(ySize), c(newSize);

        // Beaver triples open is optimized for communication
        auto A = partial_open(x + a, false);
        auto B = partial_open(y + b, false);

        // Convolution is optimized for local computation,
        auto A_ = A.conv2DLeftVectorization(instancesCount, inputHeight, inputWidth, filterHeight,
                                            filterWidth, strideHeight, strideWidth, paddingHeight,
                                            paddingWidth);

        auto a_ = a.conv2DLeftVectorization(instancesCount, inputHeight, inputWidth, filterHeight,
                                            filterWidth, strideHeight, strideWidth, paddingHeight,
                                            paddingWidth);

        auto xCols = filterHeight * filterWidth;
        auto xRows = A_.size() / xCols;
        auto yCols = B.size() / xCols;

        z(0) = A_.matrixRightMultiplyWithColumnMatrixVectorized(y(0), xRows, xCols, xCols, yCols) -
               a_(0).matrixRightMultiplyWithColumnMatrixVectorized(B, xRows, xCols, xCols, yCols) +
               c(0);

        z(1) = A_.matrixRightMultiplyWithColumnMatrixVectorized(y(1), xRows, xCols, xCols, yCols) -
               a_(1).matrixRightMultiplyWithColumnMatrixVectorized(B, xRows, xCols, xCols, yCols) +
               c(1);

        // TODO: we should do aggregation per round not batch.
        // Ensure all back log is aggregated
        processAggregationBackLog();

        this->handle_precision(x, y, z);
        this->truncate(z);
    }

    /**
     * @brief Secure division of an Arithmetic secret shared vector by a public constant.
     *  Note: This method is currently not supported in SPDZ-2k.
     *
     * @param x The input secret shared vector.
     * @param c The public constant to divide by.
     * @return A pair of secret shared vectors representing the (quotient and error correction).
     */
    std::pair<ShareMultiVector, ShareMultiVector> div_const_a(const ShareMultiVector& x,
                                                              const DataType& c) {
        // TODO: extend our public division for SPDZ2k.
        std::cerr << "Method 'div_const_a()' is not supported by SPDZ_2k." << std::endl;
        exit(-1);
    }

    /**
     * @brief Secure bitwise AND of two boolean secret shared vectors.
     *  Note: Not supported in SPDZ-2k.
     *
     * @param x The first secret shared vector.
     * @param y The second secret shared vector.
     * @param z The output secret shared vector to store the result.
     */
    void and_b(const ShareMultiVector& x, const ShareMultiVector& y, ShareMultiVector& z) {
        // Not supported by SPDZ_2k
        std::cerr << "Method 'and_b()' is not supported by SPDZ_2k." << std::endl;
        exit(-1);
    }

    void and_b_1(const ShareMultiVector& first, const ShareMultiVector& second,
                 ShareMultiVector& result) {
        this->multiply_a(first, second, result);
    }

    void or_b_1(const ShareMultiVector& first, const ShareMultiVector& second,
                ShareMultiVector& result) {
        // x + y - xy

        ShareMultiVector xy(first.size());
        this->multiply_a(first, second, xy);

        result(0) = first(0) + second(0) - xy(0);
        result(1) = first(1) + second(1) - xy(1);
    }

    /**
     * @brief Secure bitwise NOT of a boolean secret shared vector.
     *  Note: Not supported in SPDZ-2k.
     *
     * @param x The input secret shared vector.
     * @param y The output secret shared vector to store the result.
     */
    void not_b(const ShareMultiVector& x, ShareMultiVector& y) {
        // Not supported by SPDZ_2k
        std::cerr << "Method 'not_b()' is not supported by SPDZ_2k." << std::endl;
        exit(-1);
    }

    /**
     * @brief Secure NOT of a boolean secret shared vector (LSB only).
     * Note: Not supported in SPDZ-2k.
     *
     * @param x The input secret shared vector.
     * @param y The output secret shared vector to store the result.
     */
    void not_b_1(const ShareMultiVector& x, ShareMultiVector& y) {
        // We need the formula `NOT(x) = 1 - x`

        // We multiply current share by -1
        y(0) = -x(0);

        // We add 1 to only one party's secret share
        if (this->partyID == 0) {
            y(0) += 1;
        }

        // Every party adds their secret share of the key
        y(1) = -x(1);

        // TODO: add bin_op(element, evector) operator
        y(1) += alpha_i;
    }

    void xor_b(const ShareMultiVector& x, const ShareMultiVector& y, ShareMultiVector& z) {
        // Not supported by SPDZ_2k
        std::cerr << "Method 'xor_b()' is not supported by SPDZ_2k." << std::endl;
        exit(-1);
    }

    void xor_b_1(const ShareMultiVector& x, const ShareMultiVector& y, ShareMultiVector& z) {
        // We need to calculate x + y - (xy * 2)
        ShareMultiVector xy(x.size());
        this->multiply_a(x, y, xy);

        z(0) = x(0) + y(0) - xy(0) * 2;
        z(1) = x(1) + y(1) - xy(1) * 2;
    }

    void inplace_invert_b(ShareMultiVector& x) {
        // Not supported by SPDZ_2k
        std::cerr << "Method 'inplace_invert_b()' is not supported by SPDZ_2k." << std::endl;
        exit(-1);
    }

    ////////////////////////////////////////////////////////////
    //////////////////// B Bit Manipulation ////////////////////
    ////////////////////////////////////////////////////////////

     void pack_from(const ShareMultiVector& in, ShareMultiVector& out, const size_t pos) {
        // Not supported by SPDZ_2k
        std::cerr << "Method 'pack_from_b()' is not supported by SPDZ_2k." << std::endl;
        exit(-1);
     }

     void unpack_from(const ShareMultiVector& in, ShareMultiVector& out, const size_t pos) {
        // Not supported by SPDZ_2k
        std::cerr << "Method 'unpack_from_b()' is not supported by SPDZ_2k." << std::endl;
        exit(-1);
     }

     void bit_arithmetic_right_shift(const ShareMultiVector& in, ShareMultiVector& out,
                                    const int& shift_size) {
        // Not supported by SPDZ_2k
        std::cerr << "Method 'bit_arithmetic_right_shift()' is not supported by SPDZ_2k." << std::endl;
        exit(-1);
     }

     void bit_logical_right_shift(const ShareMultiVector& in, ShareMultiVector& out,
                                  const int& shift_size) {
        // Not supported by SPDZ_2k
        std::cerr << "Method 'bit_logical_right_shift()' is not supported by SPDZ_2k." << std::endl;
        exit(-1);
     }

     void bit_left_shift(const ShareMultiVector& in, ShareMultiVector& out, const int& shift_size) {
        // Not supported by SPDZ_2k
        std::cerr << "Method 'bit_left_shift()' is not supported by SPDZ_2k." << std::endl;
        exit(-1);
     }

     void bit_xor(const ShareMultiVector& in, ShareMultiVector& out) {
        // Not supported by SPDZ_2k
        std::cerr << "Method 'bit_xor()' is not supported by SPDZ_2k." << std::endl;
        exit(-1);
     }

     void extend_lsb(const ShareMultiVector& in, ShareMultiVector& out) {
        // Not supported by SPDZ_2k
        std::cerr << "Method 'extend_lsb()' is not supported by SPDZ_2k." << std::endl;
        exit(-1);
     }

     void mask(const ShareMultiVector& in, const DataType mask_value) {
        // Not supported by SPDZ_2k
        std::cerr << "Method 'mask()' is not supported by SPDZ_2k." << std::endl;
        exit(-1);
     }

    ////////////////////////////////////////////////////////////
    ///////////////////////// Circuits /////////////////////////
    ////////////////////////////////////////////////////////////

    void ltz(const ShareMultiVector& in, ShareMultiVector& out) { 
        // Not supported by SPDZ_2k
        std::cerr << "Method 'ltz()' is not supported by SPDZ_2k." << std::endl;
        exit(-1);
    }


    /**
     * @brief Convert boolean-shared bit to arithmetic sharing.
     * Note: Not supported in SPDZ-2k.
     *
     * @param x Input boolean shared vector.
     * @param y Output arithmetic shared vector.
     */
    void b2a_bit(const ShareMultiVector& x, ShareMultiVector& y) {
        std::cerr << "Method 'b2a_bit()' is not supported by SPDZ_2k." << std::endl;
        exit(-1);
    }

    /**
     * @brief Redistribute arithmetic shares among parties.
     * Note: Not supported in SPDZ-2k because it only supports arithmetic sharing.
     *
     * @param x Input vector.
     * @return Pair of redistributed shared vectors.
     */
    std::pair<ShareMultiVector, ShareMultiVector> redistribute_shares_b(const ShareMultiVector& x) {
        std::cerr << "Method 'redistribute_shares_b()' is not supported by SPDZ_nPC." << std::endl;
        exit(-1);
    }

    // Shares Opening without communication
    /**
     * @brief Reconstruct plaintext from arithmetic shares.
     * It opens shares without communication.
     * Note: This method is for protocols developer only.
     * Users should use `open_shares_a()` instead.
     *
     * @param shares Input shares from all parties.
     * @return Reconstructed plaintext value.
     */
    DataType reconstruct_from_a(const std::vector<Share<ShareType>>& shares) {
        DataType res = shares[0][0];
        for (int i = 1; i < shares.size(); ++i) {
            res += shares[i][0];
        }
        return res;
    }

    /**
     * @brief Reconstruct plaintext vector from arithmetic shares.
     * It opens shares without communication.
     * Note: This method is for protocols developer only.
     * Users should use `open_shares_a()` instead.
     *
     * @param shares Input shared vectors from all parties.
     * @return Reconstructed plaintext vector.
     */
    Vector<DataType> reconstruct_from_a(const std::vector<ShareMultiVector>& shares) {
        Vector<DataType> res(shares[0](0).size());
        res = shares[0](0);
        for (int i = 1; i < shares.size(); ++i) {
            res += shares[i](0);
        }
        return res;
    }

    /**
     * @brief Reconstruct plaintext from boolean shares.
     * It opens shares without communication.
     * Note: This method is for protocols developer only.
     * Users should use `open_shares_b()` instead.
     * Note: Not supported in SPDZ-2k.
     *
     * @param shares Input shares from all parties.
     * @return Reconstructed plaintext value.
     */
    DataType reconstruct_from_b(const std::vector<Share<ShareType>>& shares) {
        std::cerr << "Method 'reconstruct_from_b()' is not supported by SPDZ_nPC." << std::endl;
        exit(-1);
    }

    /**
     * @brief Reconstruct plaintext vector from boolean shares.
     * It opens shares without communication.
     * Note: This method is for protocols developer only.
     * Users should use `open_shares_b()` instead.
     * Note: Not supported in SPDZ-2k.
     *
     * @param shares Input shared vectors from all parties.
     * @return Reconstructed plaintext vector.
     */
    Vector<DataType> reconstruct_from_b(const std::vector<ShareMultiVector>& shares) {
        std::cerr << "Method 'reconstruct_from_b()' is not supported by SPDZ_nPC." << std::endl;
        exit(-1);
    }

    /**
     * @brief Finalize the malicious check after opening shares.
     *    It verifies the MACs of the opened shares so far.
     *
     * @return true if the verification succeeded, false otherwise.
     */
    bool malicious_check_internal() { return finalizeBatchCheck(false); }

    /**
     * @brief Open arithmetic shares to reveal plaintext.
     * It opens shares with communication. Protocol layer handles malicious checks.
     *
     * @param shares Input A shared vector.
     * @return Opened plaintext vector.
     */
    Vector<DataType> unchecked_open_a(const ShareMultiVector& shares) {
        assert(r >= 0);

        // Let's do the partial opening of the shares.
        auto res = partial_open(shares);

        // res = unPackIntoKBits(res);

        // Note: DataType is k-bits, we have implicit mod 2^k here.
        return res;
    }

    /**
     * @brief Open boolean shares to reveal plaintext.
     * It opens shares with communication.
     * Note: Not supported in SPDZ-2k.
     *
     * @param shares Input boolean shared vector.
     * @return Reconstructed plaintext vector.
     */
    Vector<DataType> unchecked_open_b(const ShareMultiVector& shares) {
        return unchecked_open_a(shares);
    }

    /**
     * @brief Secret share plaintext vector into arithmetic shares.
     * It should be generating shares without communication.
     * Note: functionality not supported in SPDZ-2k.
     *
     * @param data Input plaintext vector.
     * @return Vector of secret shared vectors for all parties.
     */
    std::vector<ShareMultiVector> get_shares_a(const Vector<DataType>& data) {
        std::cerr << "Method 'get_shares_a()' is not supported by SPDZ_nPC." << std::endl;
        exit(-1);
    }

    /**
     * @brief Secret share plaintext vector into boolean shares.
     * It should be generating shares without communication.
     * Note: functionality not supported in SPDZ-2k.
     *
     * @param data Input plaintext vector.
     * @return Vector of secret shared vectors for all parties.
     */
    std::vector<ShareMultiVector> get_shares_b(const Vector<DataType>& data) {
        std::cerr << "Method 'get_shares_b()' is not supported by SPDZ_nPC." << std::endl;
        exit(-1);
    }

    /**
     * @brief Secret share plaintext vector into boolean shares.
     * It uses communication to secret share the data.
     * Note: functionality not supported in SPDZ-2k.
     * Note: Can not be supported for 128-bit also.
     *
     * @param data Input plaintext vector.
     * @param data_party The party that provides the input data.
     * @return Secret shared vector.
     */
    ShareMultiVector secret_share_b_internal(const Vector<DataType>& data,
                                             const PartyID& data_party = 0) {
        return secret_share_a_internal(data, data_party);
    }

    /**
     * @brief Secret share plaintext vector into arithmetic shares.
     * It uses communication to secret share the data.
     * Note: does not support DataType of __int128_t.
     *
     * @param data Input plaintext vector.
     * @param data_party The party that provides the input data.
     * @return Secret shared vector.
     */
    ShareMultiVector secret_share_a_internal(const Vector<DataType>& data,
                                             const PartyID& data_party = 0) {
        // k must be greater than 0 for SPDZ_2k protocol. Otherwise, the data type is not large
        // enough to enforce the security parameter `s`.
        assert(r >= 0);

        // Check if DataType is __int128_t and throw runtime error
        if (std::is_same_v<DataType, __int128_t>) {
            std::cerr << "Error: DataType '__int128_t' is not supported by SPDZ_2k protocol."
                      << std::endl;
            exit(-1);
        }

        auto size = data.size();

        // auto data_shifted = packIntoKBits(data);
        Vector<ShareType> data_shifted(size);

        // We use mask in order to shift the expensive MAC generation part to the preprocessing
        // phase. Ideally, we want to get secret shares for random numbers of size `k` bits.
        // Only the secret sharing party knows the mask.
        auto [shares, mask] = authRGen->getNextMask(size, data_party);

        if (this->partyID == data_party) {
            // readjust the mask with input data
            data_shifted = data - mask;

            // broadcast the mask to all parties
            std::vector<Vector<ShareType>> toBeSentShares;
            for (int i = 1; i < this->numParties; ++i) {
                toBeSentShares.push_back(data_shifted);
            }
            this->communicator->sendShares(toBeSentShares, toPartyIDs_);

            // adjust the shares with the data shifted
            shares(0) += data_shifted;
        } else {
            // receive the mask from the input party
            this->communicator->receiveShares(data_shifted, data_party - this->partyID);
        }

        // Parties do secure public addition of the difference between data and mask
        shares(1) += data_shifted * alpha_i;

        return shares;
    }

    /**
     * @brief Create public shares from plaintext vector.
     * Note: does not support DataType of __int128_t.
     *
     * What happens if someone tries to open a public share?
     * They key share is still safe because MAC shares do not leave the computing parties.
     * Pay close attention to the BatchCheck protocol.
     *
     * @param x Input plaintext vector.
     * @return Public shared vector.
     */
    ShareMultiVector public_share(const Vector<DataType>& x, const std::set<PartyID>& who_knows) {
        // k must be greater than 0 for SPDZ_2k protocol. Otherwise, the data type is not large
        // enough to enforce the security parameter `s`.
        assert(r >= 0);

        // not supported with parties ID
        std::cerr << "Method 'public_share()' with parties set is not supported by SPDZ_2k." << std::endl;
        exit(-1);
    }

    ShareMultiVector public_share(const Vector<DataType>& x) {
        // k must be greater than 0 for SPDZ_2k protocol. Otherwise, the data type is not large
        // enough to enforce the security parameter `s`.
        assert(r >= 0);

        // Check if DataType is __int128_t and throw runtime error
        if (std::is_same_v<DataType, __int128_t>) {
            std::cerr << "Error: DataType '__int128_t' is not supported by SPDZ_2k protocol."
                      << std::endl;
            exit(-1);
        }

        ShareMultiVector res(x.size());

        if(this->partyID == 0) {
            res(0) += x;
        }
        res(1) += x;
        res(1) *= alpha_i;

        return res;
    }

   public:
    ShareMultiVector add_public_a(const ShareMultiVector& x, const Data val) {
        if (this->partyID == 0) {
            return {x(0) + val, x(1) + val * alpha_i};
        }

        return {x(0), x(1) + val * alpha_i};
    }

    ShareMultiVector sub_public_a(const ShareMultiVector& x, const Data val) {
        if (this->partyID == 0) {
            return {x(0) - val, x(1) - val * alpha_i};
        }

        return {x(0), x(1) - val * alpha_i};
    }

    ShareMultiVector multiply_public_a(const ShareMultiVector& x, const Vector<Data>& val) {
        return x * val;
    }

    ShareMultiVector multiply_public_a(const ShareMultiVector& x, const Data& val) {
        return x * val;
    }

    // ShareMultiVector and_1_public_b(const ShareMultiVector& x, const Vector<Data>& val) {
    //     ShareMultiVector zero(x.size());
    //     return this->multiply_a(x, {val, zero});
    // }


   private:
    // Asssiting generated members
    std::vector<cdough::PartyID> toPartyIDs_;
    std::vector<cdough::PartyID> fromPartyIDs_;
    std::set<cdough::PartyID> allPartiesSet_;

    /**
     * @brief Generate a hash of the given aggregate using libsodium.
     *
     * @param packedMessage The aggregate to hash + the key.
     * @return The hash of the aggregate as a Vector of int8_t.
     */
    static cdough::Vector<int8_t> generateHash(const Vector<ShareType> packedMessage) {
        // Configure the hash parameters
        cdough::Vector<int8_t> hash(HASH_RES_BYTES);

        crypto_generichash((uint8_t*)&hash[0], HASH_RES_BYTES, (const uint8_t*)(&packedMessage[0]),
                           sizeof(ShareType), (const uint8_t*)(&packedMessage[1]), HASH_KEY_BYTES);

        return hash;
    }

    /**
     * @brief Add the opened shares to the verification back log for later MAC checking.
     */
    void addToAggregationBackLog(const ShareMultiVector& shares) {
        aggregationBackLog.push_back(shares);
        backLogElementSize += shares(0).size();
    }

    /**
     * @brief Process the verification back log to calculate the aggregated difference for MAC
     * checking.
     */
    void processAggregationBackLog() {
        // Get fresh PRG from committed seeds queue
        auto freshPRG = seedQueue->getNextPRG();

        for (int i = 0; i < aggregationBackLog.size(); ++i) {
            auto& res = aggregationBackLog[i];
            auto& x = res(0);
            auto& m = res(1);

            // Generate random coefficients to calculate the aggregated MAC check
            // Using committed seeds (to prevent pre-computation attacks) ensures everyone gets
            // the same random coefficients for MAC verification.
            // TODO: we need only s random bits for each element. This would require modifying the
            // commonPRG implementation to support bit-limited generation
            Vector<ShareType> nums(x.size());

            freshPRG->getNext(nums);

            // Calculate the aggregate of the shares * keyShare - macShares.
            auto aggregate = x.dot_product(nums) * alpha_i - m.dot_product(nums);
            aggregatedDiff += aggregate[0];
        }
        aggregationBackLog.clear();
        backLogElementSize = 0;

        // Ensure enough committed seeds for future operations
        this->seedQueue->repopulateQueue();
    }

    /**
     * @brief Optional optimization to process the verification back log in batches to reduce memory
     * overhead. This can be called after each partial opening to ensure that the back log does
     * not grow too large, which can lead to high memory usage. The threshold for processing can be
     * adjusted based on the expected size of the opened shares and the available memory resources.
     */
    void optionalAggregationBackLogOptimization() {
        if (aggregationBackLog.size() > HEURISTIC_MAX_BACKLOG_SIZE ||
            backLogElementSize > HEURISTIC_MAX_BACKLOG_ELEMENT_SIZE) {
            processAggregationBackLog();
        }
    }

    /**
     * @brief Initiate batch opening and MAC checking of secret shared values.
     *
     * @param shares The secret shared values to open and verify.
     * @param addMask Whether to add random masks before opening. Use by default unless opening for
     * intermediate secret shares inside the multiplication protocol.
     */
    Vector<ShareType> initiateBatchCheck(const ShareMultiVector& shares,
                                         const bool addMask = true) {
        // Step 1: Open phase
        auto othersCount = this->numParties - 1;
        auto size = shares(0).size();

        // Ensure enough committed seeds for future operations
        this->seedQueue->repopulateQueue();

        // Mask the shares with authenticated random numbers
        // Optimally, this generates authenticated random numbers of size `s` bits for masking.
        // They are used for hiding the most significant `s` bits of the secret shares during
        // the opening phase. This hides information such as whether `x` + `y` overflowed when
        // creating secret shares for `z` for instance. To use them z_ = z + r_s << k; // where {z_,
        // z, r_s} are vectors of equal size.
        // TODO: generate only `s` random bits instead of `s+k` bits for better optimization.
        ShareMultiVector res(size);
        res = addMask ? shares + (authRGen->getNext(size) << k) : shares;

        // Create vector shares vectors for communication
        std::vector<Vector<ShareType>> sharesLocal;
        std::vector<Vector<ShareType>> sharesRemote;
        for (int i = 1; i <= othersCount; ++i) {
            sharesLocal.push_back(Vector<ShareType>(res(0)));
            sharesRemote.push_back(Vector<ShareType>(size));
        }

        // broadcast the share to all parties
        this->communicator->exchangeShares(sharesLocal, sharesRemote, toPartyIDs_, fromPartyIDs_);

        // Partial open result
        for (int i = 0; i < othersCount; ++i) {
            res(0) += sharesRemote[i];
        }

        // Step 2: MAC check phase - for verification:
        // 1- We need to compute the aggregate of the shares -  macShares.
        addToAggregationBackLog(res);

        // optional aggregation back log optimization to reduce memory overhead
        optionalAggregationBackLogOptimization();

        return res(0);
    }

    /**
     * @brief Finalize the batch opening and MAC checking of secret shared values.
     *
     * @param shouldAbort Whether to abort the protocol on verification failure.
     * @return true if the verification succeeded, false otherwise.
     */
    bool finalizeBatchCheck(bool shouldAbort = true) {
        // Step 2(continued): MAC check phase - for verification:
        // 2- Each party needs to commit to his aggregate.
        // 3- Each party reveals his aggregate to all other parties.
        // 4- Parties check that the aggregate was the commited.
        // 5- The parties check that the sum of the aggregates is zero.

        bool ok = true;

        // ensures all back log is aggregated
        // This is required not optional.
        processAggregationBackLog();

        // SUB_STEP 2: Commit to the aggregate, we can do this by:
        // 1) hashing the aggregate.
        // We pack the aggregate with the key in the same vector to do one communication round.
        Vector<ShareType> packedMessage(HASK_PACKED_MESSAGE_SIZE);
        localPRG->getNext(packedMessage);   // Generate random key
        packedMessage[0] = aggregatedDiff;  // Place the aggregate at the start
        auto hashLocal = generateHash(packedMessage);

        // 2) broadcasting the hash to all parties
        std::vector<cdough::Vector<int8_t>> hashesLocal;
        std::vector<cdough::Vector<int8_t>> hashesRemote;
        for (int i = 1; i < this->numParties; ++i) {
            hashesLocal.push_back(hashLocal);
            hashesRemote.push_back(cdough::Vector<int8_t>(HASH_RES_BYTES));
        }
        this->communicator->exchangeShares(hashesLocal, hashesRemote, toPartyIDs_, fromPartyIDs_);

        // SUB_STEP 3: Reveal the aggregate to all parties
        // We can do this by broadcasting the aggregate to all parties.
        std::vector<Vector<ShareType>> packedMessagesLocal;
        std::vector<Vector<ShareType>> packedMessagesRemote;
        for (int i = 1; i < this->numParties; ++i) {
            packedMessagesLocal.push_back(packedMessage);
            packedMessagesRemote.push_back(Vector<ShareType>(HASK_PACKED_MESSAGE_SIZE));
        }
        this->communicator->exchangeShares(packedMessagesLocal, packedMessagesRemote, toPartyIDs_,
                                           fromPartyIDs_);

        // SUB_STEP 4: Check that the aggregate was the committed.
        for (int i = 0; i < this->numParties - 1; ++i) {
            auto hashRemote = generateHash(packedMessagesRemote[i]);
            if (!hashRemote.same_as(hashesRemote[i], false)) {
                std::cerr << "Party " << this->partyID
                          << ": Commitment verification failed for aggregate from party "
                          << (this->partyID + i + 1) % this->parties_num << std::endl;
                ok = false;
                if (shouldAbort) {
                    abort();
                }
            }
        }

        // SUB_STEP 5: Check that the sum of the aggregates is zero.
        ShareType totalAggregate = aggregatedDiff;
        for (int i = 0; i < this->numParties - 1; ++i) {
            totalAggregate += packedMessagesRemote[i][0];
        }

        // If the total aggregate is not zero, then the verification failed.
        if (totalAggregate != 0) {
            std::cerr << "Party " << this->partyID
                      << ": Verification failed for the sum of the aggregates with sum="
                      << totalAggregate << std::endl;
            ok = false;
            if (shouldAbort) {
                abort();
            }
        }

        return ok;
    }

   public:
    /**
     * @brief Reset the malicious state.
     * It clears the aggregatedDiff and back log.
     */
    void reset_malicious_state() {
        aggregationBackLog.clear();
        backLogElementSize = 0;
        aggregatedDiff = 0;
    }

   private:
    /**
     * @brief Partial opening of secret shared values with MAC checking.
     *     This function initiates the batch opening and MAC checking.
     *
     * @param shares The secret shared values to open and verify.
     * @param addMask Whether to add random masks before opening. Use by default unless opening for
     * intermediate secret shares inside the multiplication protocol.
     * @return The opened values as a vector.
     */
    Vector<ShareType> partial_open(const ShareMultiVector& shares, const bool addMask = true) {
        return initiateBatchCheck(shares, addMask);
    }

    /**
     * @brief Pack data into k bits by moving the sign bit to the k-th position.
     *     Useful if need you use less number of bits than the data type size.
     *
     * @param data The input data to be packed.
     * @return The packed data with sign bit at k-th position.
     */
    Vector<DataType> packIntoKBits(const Vector<DataType>& data) {
        // k must be greater than 0 for SPDZ_2k protocol. Otherwise, the data type is not large
        // enough to enforce the security parameter `s`.
        assert(k > 0);
        assert(r >= 0);

        // we move the sign bit of the data to be at the `k`-th position.
        // Othwerwise, the sign bit will be removed during the opening phase.
        auto data_shifted = (((data >> s) & k2) | data) & mask_k;

        return data_shifted;
    }

    /**
     * @brief Unpack data from k bits by moving the sign bit from the k-th position to the most
     *    significant bit. This is the reverse operation of packIntoKBits.
     *
     * @param data The input data to be unpacked.
     * @return The unpacked data with sign bit at the most significant bit.
     */
    Vector<DataType> unPackIntoKBits(const Vector<DataType>& data) {
        // mask for data bits except for the sign bit
        const auto mask_k_ = mask_k >> 1;
        // Now we need to shift the sign bit in the k-th position to the most significant bit.
        // -(res & k2): does sign extension.
        // (res & mask_k_): keeps the data bits except for the sign bit.
        auto res = (-(data & k2)) | (data & mask_k_);

        return res;
    }
};

// Initialize the static member with a default value that will be overridden in constructor
template <typename Data, template <typename> class Share, template <typename> class Vector,
          template <typename, int, int> class EVector>
int SPDZ_2k<Data, Share, Vector, EVector>::parties_num = 2;

/**
 * @brief Factory class for creating SPDZ-2k protocol instances.
 *
 * This factory creates instances of the SPDZ-2k protocol for different data types.
 *
 * @tparam S Share type template.
 * @tparam V Data container type template (e.g. cdough::Vector).
 * @tparam E Share container type template (e.g. cdough::EVector).
 */
template <template <typename> class S, template <typename> class V, template <typename> class E>
class SPDZ_2k_Factory : public ProtocolFactory<SPDZ_2k_Factory<S, V, E>> {
   public:
    template <typename T>
    using DataToShareType = typename ShareTypeSelector<T>::type;

    template <typename T>
    using AInnerContainer =
        cdough::EVector<DataToShareType<T>, SPDZ2K_MULTIVECTOR_SIZE, std::numeric_limits<T>::digits>;

    template <typename T>
    using BInnerContainer =
        cdough::EVector<DataToShareType<T>, SPDZ2K_MULTIVECTOR_SIZE, std::numeric_limits<T>::digits>;

    template <typename T>
    using ProtocolInstance = SPDZ_2k<T, S, V, cdough::EVector>;

   public:
    /**
     * @brief Constructor for the SPDZ-2k Factory.
     *
     * @param partyID The unique identifier of the party using this factory.
     * @param partiesNumber The total number of parties in the protocol.
     */
    SPDZ_2k_Factory(int partyID, int partiesNumber)
        : partyID_(partyID), partiesNumber_(partiesNumber) {
        if (partyID == 0) {
            std::cout << "SPDZ_2k Factory created for " << partiesNumber_ << " parties."
                      << std::endl;
        }
    }

    /**
     * @brief Create an instance of the SPDZ-2k protocol for a specific data type.
     *
     * @param thread_id
     * @param communicator A pointer to the communicator.
     * @param randomnessManager A pointer to the randomness manager.
     * @return A unique pointer to the created protocol instance.
     */
    template <typename T>
    std::unique_ptr<ProtocolBase> create(int thread_id, Communicator* communicator,
                                         random::RandomnessManager* randomnessManager) {
        using UnsignedDataType = typename std::make_unsigned<T>::type;
        using UnsignedShareType = typename std::make_unsigned<DataToShareType<T>>::type;

        const auto MAX_BITS_NUMBER = std::numeric_limits<UnsignedShareType>::digits;
        const auto k = std::numeric_limits<UnsignedDataType>::digits;  // Data size parameter
        const auto s = std::max(SPDZ2K_STAT_SECURITY_PARAMETER,
                                MAX_BITS_NUMBER - k);  // Statistical security parameter

        ProtocolInstance<T>::parties_num = partiesNumber_;

        if (!keyShareInitialized) {
            // Initialize the authentication key share (alpha_i)
            randomnessManager->localPRG->getNext(alpha_i);

            // Ensure alpha_i is non-zero (Optional)
            assert(alpha_i != 0);
            keyShareInitialized = true;
        }

        auto zeroSharingGenerator = randomnessManager->zeroSharingGenerator;
        auto localPRG = randomnessManager->localPRG;

        // Note: We can use the same alpha_i for all protocol instantiations (64, 128, etc.)
        // because alpha_i is limited to `s` bits which is smaller than any used data type.
        // Create the authenticated random number generator.
        // TODO: if we need to store k < (max_bits - s), we need to revisit this.

        // Cast alpha_i to type T
        // We need to force alpha_i to be within `s` bits.
        const DataToShareType<T> t_alpha_i =
            static_cast<DataToShareType<T>>(alpha_i) & ((((__int128_t)1) << s) - 1);

#if defined USE_DUMMY_TRIPLES
        auto authRGen = std::make_shared<
            random::DummyAuthRandomGenerator<DataToShareType<T>, AInnerContainer<T>>>(
            partiesNumber_, t_alpha_i, partyID_, localPRG, zeroSharingGenerator, communicator);
        auto authTGen = std::make_shared<
            random::DummyAuthTripleGenerator<DataToShareType<T>, AInnerContainer<T>>>(
            partiesNumber_, t_alpha_i, partyID_, localPRG, zeroSharingGenerator, communicator);
#else
        auto authRGen = std::make_shared<
            random::ZeroAuthRandomGenerator<DataToShareType<T>, AInnerContainer<T>>>(partyID_);
        auto authTGen = std::make_shared<
            random::ZeroAuthTripleGenerator<DataToShareType<T>, AInnerContainer<T>>>(partyID_);
#endif

        // Create the SPDZ_2k protocol instance and pass the generators to it
        auto protocol =
            std::make_unique<ProtocolInstance<T>>(partyID_, partiesNumber_, t_alpha_i, communicator,
                                                  randomnessManager, authRGen, authTGen);

        return protocol;
    }

   private:
    // Party information
    const PartyID partyID_;
    const PartyID partiesNumber_;

    // Authentication key share - we use __int128_t to ensure it's large enough for all
    // instantiations This is valid because we only use s bits of alpha_i (the statistical security
    // parameter)
    __int128_t alpha_i;
    bool keyShareInitialized = false;
};

}  // namespace cdough::protocols
