#pragma once

namespace cdough::matrix::hybrid {

using HeightWidth = std::pair<size_t, size_t>;

#define define_binary_matrix_matrix_op(_op_)                            \
    template <typename T2>                                              \
    auto operator _op_(const ImplementedMatrix<T2, V>& y) const {       \
        assert((this->rows_ == y.rows_) && (this->cols_ == y.cols_));   \
        assert(this->columnWise_ == y.columnWise_);                     \
        auto res = this->data_ _op_ y.data_;                            \
        return ImplementedMatrix<T, V>(res, rows_, cols_, columnWise_); \
    }

#define define_binary_matrix_element_op(_op_)                           \
    template <typename T2>                                              \
    auto operator _op_(const T2& y) const {                             \
        auto res = this->data_ _op_ y;                                  \
        return ImplementedMatrix<T, V>(res, rows_, cols_, columnWise_); \
    }

#define define_binary_matrix_matrix_inplace_op(_op_)                  \
    template <typename T2>                                            \
    auto& operator _op_(const ImplementedMatrix<T2, V>& y) {          \
        assert((this->rows_ == y.rows_) && (this->cols_ == y.cols_)); \
        assert(this->columnWise_ == y.columnWise_);                   \
        this->data_ _op_ y.data_;                                     \
        return static_cast<ImplementedMatrix<T, V>&>(*this);          \
    }

#define define_binary_matrix_element_inplace_op(_op_)        \
    template <typename T2>                                   \
    auto& operator _op_(const T2 & y) {                      \
        this->data_ _op_ y;                                  \
        return static_cast<ImplementedMatrix<T, V>&>(*this); \
    }

/**
 * @brief Matrix class
 * @tparam T Data type
 * @tparam V Vector type
 * @tparam ImplementedMatrix The derived matrix class
 *
 * This class provides the common Matrix interface
 * between both secure and plaintext matrices.
 * The Matrix class is a wrapper around a 1D data abstraction.
 * The matrix utilizes its computational capabilities from the
 * underlying vector type.
 *
 * Note: we should not use for-loop functions in this class because the
 * secure matrix inherits from it.
 */
template <typename T, template <typename> class V,
          template <typename, template <typename> class> class ImplementedMatrix>
class Matrix {
   public:
    using ElementType = T;

    template <typename U>
    using VectorType = V<U>;

   public:
    ////////////////////////////////////////////////////////////
    ////////////////////// Constructors ////////////////////////
    ////////////////////////////////////////////////////////////
    /**
     * @brief Construct a new Matrix object
     * @param data 1D data vector
     * @param rows Number of rows
     * @param cols Number of columns
     * @param columnWise Whether the data is stored in column-wise order
     */
    Matrix(const V<T>& data, size_t rows, size_t cols, const bool& columnWise = false)
        : rows_(rows), cols_(cols), data_(data), columnWise_(columnWise) {
        assert(data.size() == rows * cols);
    }

    virtual ~Matrix() {}

    ////////////////////////////////////////////////////////////
    ///////////////////////// Getters //////////////////////////
    ////////////////////////////////////////////////////////////

    /**
     * @brief Get number of rows
     * @return size_t Number of rows
     */
    size_t rows() const { return rows_; }

    /**
     * @brief Get number of columns
     * @return size_t Number of columns
     */
    size_t cols() const { return cols_; }

    /**
     * @brief Get underlying data vector (const)
     * @return const V<T>& for the Underlying data vector
     */
    const V<T>& data() const { return data_; }

    /**
     * @brief Get underlying data vector
     * @return V<T>& for the Underlying data vector
     */
    V<T>& data() { return data_; }

    /**
     * @brief Check if the matrix is stored in column-wise order
     * @return true if the matrix is column-wise, false otherwise
     */
    bool isColumnWise() const { return columnWise_; }

    ////////////////////////////////////////////////////////////
    /////////////////////// ML operators ///////////////////////
    ////////////////////////////////////////////////////////////

    /**
     * @brief Vectorized Matrix right multiplication with a column matrix.
     * Expects the left-hand side matrix to be in row-major order.
     * Expects the right-hand side matrix to be in column-major order.
     *
     * @param rhs The right-hand side column matrix.
     * @return ImplementedMatrix<T, V> The resulting matrix
     */
    ImplementedMatrix<T, V> matrixRightMultiplyWithColumnMatrixVectorized(
        const ImplementedMatrix<T, V>& rhs) const {
        // We do dot product between each row in input and column in output.
        // But since rhs is in the transpose form, it's dot product between row in input
        // and row in output.
        assert(this->columnWise_ == false);
        assert(rhs.columnWise_ == true);
        size_t out_rows = this->rows();
        size_t out_cols = rhs.cols();

        auto result_data = this->data_.matrixRightMultiplyWithColumnMatrixVectorized(
            rhs.data(), this->rows(), this->cols(), rhs.rows(), rhs.cols());

        // return new Matrix
        return ImplementedMatrix<T, V>(result_data, out_rows, out_cols);
    }

    /**
     * @brief  Vectorized matrix right multiplication with a matrix.
     * Expects the left-hand side matrix to be in row-major order.
     *
     * @param rhs The right-hand side matrix.
     * @return ImplementedMatrix<T, V> The resulting matrix
     */
    ImplementedMatrix<T, V> matrixRightMultiplyVectorized(const ImplementedMatrix<T, V>& rhs) {
        assert(this->columnWise_ == false);
        if (rhs.columnWise_) {
            return this->matrixRightMultiplyWithColumnMatrixVectorized(rhs);
        } else {
            // TODO: implement physical conversion?
            throw std::runtime_error(
                "matrixRightMultiplyVectorized called with rhs not in column-wise format.");
        }
    }

    /**
     * @brief Secure 2D convolution.
     * Expects the input matrix to be in row-major order.
     * Expects the filter matrix to be in row-major order.
     * Assumes input to has 1 channel (equivalent to many but interleaved).
     * Output has multiple channels.
     *
     * Input layout:
     * The input consists of mutiple instances concatenated after each other.
     * Hence, the input size = instancesCount * inputHeight * inputWidth.
     * Each instance has multiple channels interleaved per spatial location.
     * For example, for 2x2 input with 2 channels:
     * [ch1(0,0), ch2(0,0), ch1(0,1), ch2(0,1),
     * ch1(1,0), ch2(1,0), ch1(1,1), ch2(1,1)]
     *
     * Filter layout: the filter is expected to have multiple channels.
     * Hence, the filter size = channels * filterHeight * filterWidth.
     * For example, the the physical layout for 2x2 filter with 2 channels
     * [f_ch1(0,0), f_ch2(0,0), f_ch1(1,0), f_ch2(1,0),
     * g_ch1(0,1), g_ch2(0,1), g_ch1(1,1), g_ch2(1,1)]
     *
     * Output layout: (same layout as input but different height and width).
     * The output consists of mutiple instances concatenated after each other.
     * Hence, the output size = instancesCount * outputHeight * outputWidth * channels.
     * Each instance has multiple channels interleaved per spatial location.
     * For example, for 2x2 output with 2 channels:
     * [ch1(0,0), ch2(0,0), ch1(0,1), ch2(0,1),
     * ch1(1,0), ch2(1,0), ch1(1,1), ch2(1,1)]
     *
     * @param rhs The filter matrix.
     * @param instancesCount Number of instances in the input batch.
     * @param filterSize Height and width of the filter.
     * @param stride Height and width of the stride.
     * @param padding Height and width of the padding.
     * @return ImplementedMatrix<T, V> The resulting matrix
     */
    ImplementedMatrix<T, V> conv2DVectorized(const ImplementedMatrix<T, V>& rhs,
                                             size_t instancesCount, const HeightWidth& filterSize,
                                             const HeightWidth& stride,
                                             const HeightWidth& padding) const {
        assert(this->columnWise_ == false);
        assert(rhs.columnWise_ == false);

        // Calculate output dimensions
        size_t out_rows =
            instancesCount *
            ((this->rows_ / instancesCount + 2 * padding.first - filterSize.first) / stride.first +
             1);
        size_t out_cols =
            (this->cols_ + 2 * padding.second - filterSize.second) / stride.second + 1;
        size_t out_channels = rhs.rows() / filterSize.first;

#ifdef FORCE_CONV2D_LEFT_VECTORIZATION_MATERIALIZATION
        // We can transform the conv2d operator into a matrix multiplication after applying
        // a physical transformation to the LHS.
        // physical transformation for LHS
        auto lhs = this->data_.conv2DLeftVectorization(
            instancesCount, this->rows_ / instancesCount, this->cols_, filterSize.first,
            filterSize.second, stride.first, stride.second, padding.first, padding.second);

        auto result_data = lhs.matrixRightMultiplyWithColumnMatrixVectorized(
            rhs.data(), out_rows * out_cols, filterSize.first * filterSize.second,
            filterSize.first * filterSize.second, rhs.rows() / filterSize.first);
#else
        // Perform conv2d directly
        // Delegated to Vector to reduce data views overhead.
        auto result_data = this->data_.conv2DVectorized(
            rhs.data(), instancesCount, this->rows_ / instancesCount, this->cols_, filterSize.first,
            filterSize.second, stride.first, stride.second, padding.first, padding.second);
#endif

        // return new Matrix
        return ImplementedMatrix<T, V>(result_data, out_rows, out_cols * out_channels);
    }

    /**
     * @brief Vectorized fully connected layer.
     * Expects the input matrix to be in row-major order.
     * Expects the weights matrix to be in column-major order.
     * Expects the bias matrix to be in row-major order.
     *
     * It performs matrix multiplication between input and weights,
     * then adds the bias to each instance in the output.
     * res = input.matMult(weights) + bias
     *
     * @param weights The weights matrix.
     * @param bias The bias matrix.
     * @return ImplementedMatrix<T, V> The resulting matrix
     */
    ImplementedMatrix<T, V> fullyConnectedVectorized(const ImplementedMatrix<T, V>& weights,
                                                     const ImplementedMatrix<T, V>& bias) const {
        assert(this->columnWise_ == false);
        assert(weights.columnWise_ == true);
        assert(bias.columnWise_ == false);

        // Matrix multiplication
        auto matmul_result = this->matrixRightMultiplyWithColumnMatrixVectorized(weights);

        // Calculate instances count
        const auto instancesCount = this->rows() / bias.rows();

        // TODO: we need to remove cyclic_subset_reference overhead.
        auto result_data = matmul_result.data_ + bias.data_.cyclic_subset_reference(instancesCount);

        // return new Matrix
        return ImplementedMatrix<T, V>(result_data, matmul_result.rows(), matmul_result.cols());
    }

    // TODO: MinPoolingVectorized - MaxPoolingVectorized

    /**
     * @brief Average Pooling, vectorized implementation.
     *
     * Expects the input matrix to be in row-major order.
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
     *
     * @param instancesCount Number of instances in the input batch.
     * @param channelsNum Number of channels in the input.
     * @param inputSize Height and width of each input channel.
     * @param filterSize Height and width of the pooling filter.
     * @param stride Height and width of the stride.
     * @param padding Height and width of the padding.
     * @return ImplementedMatrix<T, V> The resulting matrix
     */
    ImplementedMatrix<T, V> avgPoolingVectorized(size_t instancesCount, size_t channelsNum,
                                                 const HeightWidth& inputSize,
                                                 const HeightWidth& filterSize,
                                                 const HeightWidth& stride,
                                                 const HeightWidth& padding) const {
        assert(this->columnWise_ == false);

        // Calculate output dimensions
        size_t out_rows =
            instancesCount *
            ((inputSize.first + 2 * padding.first - filterSize.first) / stride.first + 1);
        size_t out_cols =
            ((inputSize.second + 2 * padding.second - filterSize.second) / stride.second + 1);

#ifdef FORCE_AVG_POOL_VECTORIZATION_MATERIALIZATION
        // Separate channels
        ImplementedMatrix<T, V> inputReshaped = this->separateChannels(channelsNum);

        // Perform sum pooling with 1 channel
        // Think of it as newInstances = instancesCount * channelsNum
        V<T> result_data = inputReshaped.data_.sumPoolingVectorized(
            instancesCount * channelsNum, 1, inputSize.first, inputSize.second, filterSize.first,
            filterSize.second, stride.first, stride.second, padding.first, padding.second);
        auto result_ = ImplementedMatrix<T, V>(result_data, out_rows * channelsNum, out_cols);

        // Interleave channels
        ImplementedMatrix<T, V> result = result_.interleaveChannels(channelsNum);
#else
        // Perform sum pooling with multiple channels
        V<T> result_data = this->data_.sumPoolingVectorized(
            instancesCount, channelsNum, inputSize.first, inputSize.second, filterSize.first,
            filterSize.second, stride.first, stride.second, padding.first, padding.second);
        auto result = ImplementedMatrix<T, V>(result_data, out_rows, out_cols * channelsNum);
#endif

        const T divisor = (T)(filterSize.first * filterSize.second);
        result.data_ = result.data_ / divisor;

        // return new Matrix
        return result;
    }

    /**
     * @brief Vectorized ReLU activation, vectorized implementation.
     *
     * @return ImplementedMatrix<T, V> The resulting matrix after ReLU
     */
    ImplementedMatrix<T, V> reLUVectorized() const {
        assert(this->columnWise_ == false);

        V<T> result_data = this->data_.gtez();

        // Turning off precision temporarily to perform a mask
        V<T> thisData = this->data_;
        auto precision = thisData.getPrecision();
        thisData.setPrecision(0);
        result_data.setPrecision(0);

        // Applying the ReLU mask
        result_data *= thisData;

        // Restoring precision
        thisData.setPrecision(precision);
        result_data.setPrecision(precision);

        return ImplementedMatrix<T, V>(result_data, this->rows_, this->cols_, this->columnWise_);
    }

    /**
     * @brief Reshape the matrix to new dimensions (rows, columns)
     * It does not change the underlying data or create a copy.
     * It changes the dimensions only.
     * It returns a new matrix object referencing the same data memory location.
     *
     * @param rows New number of rows
     * @param columns New number of columns
     * @return ImplementedMatrix<T, V> The reshaped matrix
     */
    const ImplementedMatrix<T, V> reshapeRef(size_t rows, size_t columns) const {
        assert(rows * columns == this->rows() * this->cols());
        return ImplementedMatrix<T, V>(this->data(), rows, columns, this->columnWise_);
    }

    /**
     * @brief separateChannels vectorized implementation.
     * It takes a matrix where channels are interleaved and make
     * it such that each channel is separated.
     *
     * Input layout:
     * The input consists of mutiple instances concatenated after each other.
     * {ch1(0,0), ch2(0,0), ... chN(0,0), ch1(0,1), ch2(0,1), ... chN(0,1), ..}
     *
     * Output layout:
     * {ch1(0,0), ch1(0,1), ... ch1(n,m)}, {ch2(0,0), ch2(0,1), ... ch2(n,m)},
     * ... {chN(0,0), chN(0,1), ... chN(n,m)}
     *
     * @param channels Number of channels to separate
     * @return ImplementedMatrix<T, V> The resulting matrix
     */
    ImplementedMatrix<T, V> separateChannels(size_t channels) const {
        V<T> data_new = this->data().matrixSeparateChannels(channels);
        return ImplementedMatrix<T, V>(std::move(data_new), rows() * channels, cols() / channels);
    }

    /**
     * @brief interleaveChannels vectorized implementation.
     * It takes a matrix where channels are separated and make
     * it such that channels are interleaved.
     *
     * Input layout:
     * {ch1(0,0), ch1(0,1), ... ch1(n,m)}, {ch2(0,0), ch2(0,1), ... ch2(n,m)},
     * ... {chN(0,0), chN(0,1), ... chN(n,m)}
     *
     * Output layout:
     * The output consists of mutiple instances concatenated after each other.
     * {ch1(0,0), ch2(0,0), ... chN(0,0), ch1(0,1), ch2(0,1), ... chN(0,1), ..}
     *
     * @param channels Number of channels to diffuse
     * @return ImplementedMatrix<T, V> The resulting matrix
     */
    ImplementedMatrix<T, V> interleaveChannels(size_t channels) const {
        V<T> data_new = this->data().matrixInterleaveChannels(channels);
        return ImplementedMatrix<T, V>(std::move(data_new), rows() / channels, cols() * channels);
    }

    ////////////////////////////////////////////////////////////
    ////////////////// Functional operators ////////////////////
    ////////////////////////////////////////////////////////////
    define_binary_matrix_matrix_op(+);
    define_binary_matrix_matrix_op(-);
    define_binary_matrix_matrix_op(*);
    // TODO: operator::/(matrix, matrix)
    // TODO: operator::%(matrix, matrix)

    // TODO: operator::^(matrix, matrix)
    // TODO: operator::&(matrix, matrix)
    // TODO: operator::|(matrix, matrix)

    // TODO: operator::>>(matrix, matrix)
    // TODO: operator::<<(matrix, matrix)

    define_binary_matrix_matrix_op(>);
    define_binary_matrix_matrix_op(<);
    define_binary_matrix_matrix_op(>=);
    define_binary_matrix_matrix_op(<=);
    define_binary_matrix_matrix_op(==);
    define_binary_matrix_matrix_op(!=);

    // TODO: operator::+(matrix, element)
    // TODO: operator::-(matrix, element)
    // TODO: operator::*(matrix, element)
    // TODO: operator::/(matrix, element)
    // TODO: operator::%(matrix, element)

    // TODO: operator::^(matrix, element)
    // TODO: operator::|(matrix, element)
    // TODO: operator::&(matrix, element)

    // TODO: operator::>(matrix, element)
    // TODO: operator::<(matrix, element)
    // TODO: operator::>=(matrix, element)
    // TODO: operator::<=(matrix, element)
    // TODO: operator::==(matrix, element)
    // TODO: operator::!=(matrix, element)

    // TODO: operator::>>(matrix, element)
    // TODO: operator::<<(matrix, element)

    // TODO: Unary Operators {~, !}

    ////////////////////////////////////////////////////////////
    //////////////////// Inplace operators /////////////////////
    ////////////////////////////////////////////////////////////
    define_binary_matrix_matrix_inplace_op(+=);
    define_binary_matrix_matrix_inplace_op(-=);
    define_binary_matrix_matrix_inplace_op(*=);
    // TODO: operator::/=(matrix, matrix)
    // TODO: operator::%=(matrix, matrix)

    // TODO: operator::^=(matrix, matrix)
    // TODO: operator::&=(matrix, matrix)
    // TODO: operator::|=(matrix, matrix)

    // TODO: operator::>>=(matrix, matrix)
    // TODO: operator::<<=(matrix, matrix)

    // TODO: operator::+=(matrix, element)
    // TODO: operator::-=(matrix, element)
    // TODO: operator::*=(matrix, element)
    // TODO: operator::/=(matrix, element)
    // TODO: operator::%=(matrix, element)

    // TODO: operator::^=(matrix, element)
    // TODO: operator::&=(matrix, element)
    // TODO: operator::|=(matrix, element)

   protected:
    V<T> data_;    // The matrix data stored in a 1D vector (row-major order)
    size_t rows_;  // Number of rows
    size_t cols_;  // Number of columns

    bool columnWise_ = false;  // Whether the data is stored in column-wise order
};

}  // namespace cdough::matrix::hybrid