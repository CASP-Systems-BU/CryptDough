#include <algorithm>
#include <cmath>

#include "cdough.h"

using namespace cdough::debug;
using namespace cdough::service;
using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

/*
 * Data format notes:
 * - For conv2d and avgpool with multiple input channels/instances:
 *   The format stores instances after each other. For each instance, elements are organized by
 * channels:
 *   - (0,0) element for channel 1, then (0,0) element for channel 2
 *   - Once all corresponding elements from each channel are stored, move to the next element (0,1)
 *   - Continue this pattern for all elements
 * - Matrix storage follows row-major order by default unless isColumnWise() is true
 */

/**
 * @brief Golden model for matrix multiplication.
 *
 * @param A Left-hand side matrix.
 * @param B Right-hand side matrix.
 * @return PlainMatrix<T> Resulting matrix after multiplication.
 */
template <typename T>
PlainMatrix<T> golden_matrix_multiply(const PlainMatrix<T>& A, const PlainMatrix<T>& B) {
    size_t A_rows = A.rows();
    size_t A_cols = A.cols();
    size_t B_rows = B.rows();
    size_t B_cols = B.cols();

    assert(A_cols == B_rows);

    cdough::Vector<T> result(A_rows * B_cols);
    const auto& A_data = A.data();
    const auto& B_data = B.data();

    for (size_t i = 0; i < A_rows; ++i) {
        for (size_t j = 0; j < B_cols; ++j) {
            T sum = 0;
            for (size_t k = 0; k < A_cols; ++k) {
                T A_val, B_val;
                // Handle row-major access for A
                if (A.isColumnWise()) {
                    A_val = A_data[k * A_rows + i];
                } else {
                    A_val = A_data[i * A_cols + k];
                }
                // Handle column-wise access for B (transposed)
                if (B.isColumnWise()) {
                    B_val = B_data[j * B_rows + k];
                } else {
                    B_val = B_data[k * B_cols + j];
                }
                sum += A_val * B_val;
            }
            result[i * B_cols + j] = sum;
        }
    }

    return PlainMatrix<T>(result, A_rows, B_cols);
}

/**
 * @brief Golden model for multi-channel 2D convolution.
 *
 * Input format: For each instance, channels are interleaved per spatial location E.g., for 2x2
 * input with 2 channels:
 * [ch1(0,0), ch2(0,0), ch1(0,1), ch2(0,1),
 * ch1(1,0), ch2(1,0), ch1(1,1), ch2(1,1)]
 *
 * @param input Input matrix containing multiple instances and channels.
 * @param filter Filter matrix containing multiple output channels.
 * @param instancesCount Number of instances in the input.
 * @param outputChannels Number of output channels (filters).
 * @param filterSize Size of the filter (height, width).
 * @param stride Stride for the convolution (height, width).
 * @param padding Padding for the convolution (height, width).
 * @return PlainMatrix<T> Resulting matrix after convolution.
 */
template <typename T>
PlainMatrix<T> golden_conv2d_multichannel(const PlainMatrix<T>& input, const PlainMatrix<T>& filter,
                                          size_t instancesCount, size_t outputChannels,
                                          cdough::matrix::HeightWidth filterSize,
                                          cdough::matrix::HeightWidth stride,
                                          cdough::matrix::HeightWidth padding) {
    size_t inputHeight = input.rows() / instancesCount;
    size_t inputWidth = input.cols();
    size_t filterHeight = filterSize.first;
    size_t filterWidth = filterSize.second;

    assert(padding.first == 0 &&
           padding.second == 0);  // For simplicity, only support no padding here
    assert(input.isColumnWise() == false);
    assert(filter.isColumnWise() == false);

    // Calculate output dimensions
    size_t outputHeight = (inputHeight + 2 * padding.first - filterHeight) / stride.first + 1;
    size_t outputWidth = (inputWidth + 2 * padding.second - filterWidth) / stride.second + 1;

    cdough::Vector<T> result(instancesCount * outputChannels * outputHeight * outputWidth);
    const auto& input_data = input.data();
    const auto& filter_data = filter.data();

    // Iterate over each instance
    for (int inst = 0; inst < instancesCount; ++inst) {
        // Iterate over each potential filter start
        for (int x = 0; x < inputHeight - filterHeight + 1; x += stride.first) {
            for (int y = 0; y < inputWidth - filterWidth + 1; y += stride.second) {
                // For each output channel
                for (int oc = 0; oc < outputChannels; ++oc) {
                    T sum = 0;
                    // Convolve over the filter area
                    for (int fh = 0; fh < filterHeight; ++fh) {
                        for (int fw = 0; fw < filterWidth; ++fw) {
                            // Calculate input index
                            int ih = x + fh;
                            int iw = y + fw;
                            int input_idx = inst * inputHeight * inputWidth + ih * inputWidth + iw;

                            auto lhs = input_data[input_idx];

                            // Calculate filter index
                            int filter_idx =
                                oc * (filterHeight * filterWidth) + fh * filterWidth + fw;
                            auto rhs = filter_data[filter_idx];

                            sum += lhs * rhs;
                        }
                    }
                    // Store result
                    int out_h = x / stride.first;
                    int out_w = y / stride.second;
                    int out_idx = inst * (outputChannels * outputHeight * outputWidth) +
                                  out_h * (outputChannels * outputWidth) + out_w * outputChannels +
                                  oc;

                    result[out_idx] = sum;
                }
            }
        }
    }

    return PlainMatrix<T>(result, instancesCount * outputHeight, outputWidth * outputChannels);
}

/**
 * @brief Golden model for multi-channel 2D average pooling.
 *
 * Input format: For each instance, channels are interleaved per spatial location
 * E.g., for 2x2 input with 2 channels:
 * [ch1(0,0), ch2(0,0), ch1(0,1), ch2(0,1),
 * ch1(1,0), ch2(1,0), ch1(1,1), ch2(1,1)]
 *
 * @param input Input matrix containing multiple instances and channels.
 * @param instancesCount Number of instances in the input.
 * @param channelsNum Number of channels in the input.
 * @param filterSize Size of the pooling filter (height, width).
 * @param stride Stride for the pooling (height, width).
 * @param padding Padding for the pooling (height, width).
 * @return PlainMatrix<T> Resulting matrix after average pooling.
 */
template <typename T>
PlainMatrix<T> golden_avgpool2d(const PlainMatrix<T>& input, size_t instancesCount,
                                size_t channelsNum, cdough::matrix::HeightWidth filterSize,
                                cdough::matrix::HeightWidth stride, cdough::matrix::HeightWidth padding) {
    size_t inputHeight = input.rows() / instancesCount;
    size_t inputWidth = input.cols() / channelsNum;  // Total width divided by channels
    size_t filterHeight = filterSize.first;
    size_t filterWidth = filterSize.second;

    // Calculate output dimensions
    size_t outputHeight = (inputHeight + 2 * padding.first - filterHeight) / stride.first + 1;
    size_t outputWidth = (inputWidth + 2 * padding.second - filterWidth) / stride.second + 1;

    // Result should have the same number of elements as output spatial locations
    cdough::Vector<T> result(instancesCount * outputHeight * outputWidth * channelsNum);
    const auto& input_data = input.data();

    for (int inst = 0; inst < instancesCount; ++inst) {
        for (int oh = 0; oh < outputHeight; ++oh) {
            for (int ow = 0; ow < outputWidth; ++ow) {
                // calculate the starting point in the input matrix
                int in_h_start = oh * stride.first;
                int in_w_start = ow * stride.second;

                // iterate for each channel
                for (int c = 0; c < channelsNum; ++c) {
                    T sum = 0;    // Reset sum for each channel
                    T count = 0;  // Reset count for each channel

                    // iterate over the filter window
                    for (int fh = 0; fh < filterHeight; ++fh) {
                        for (int fw = 0; fw < filterWidth; ++fw) {
                            int in_h = in_h_start + fh;
                            int in_w = in_w_start + fw;

                            // check for valid boundaries (no padding support in this version)
                            if (in_h >= inputHeight || in_w >= inputWidth) {
                                continue;  // skip out-of-bounds area
                            }

                            // calculate the actual input indices considering channels
                            int inputIndex = inst * (inputHeight * inputWidth * channelsNum) +
                                             in_h * (inputWidth * channelsNum) +
                                             in_w * channelsNum + c;

                            sum += input_data[inputIndex];
                            count++;
                        }
                    }

                    // calculate the output index considering channels
                    int outputIndex = inst * (outputHeight * outputWidth * channelsNum) +
                                      oh * (outputWidth * channelsNum) + ow * channelsNum + c;
                    if (count > 0) {
                        result[outputIndex] = sum / static_cast<T>(count);
                    } else {
                        result[outputIndex] =
                            static_cast<T>(0);  // Handle case with no valid inputs
                    }
                }
            }
        }
    }

    return PlainMatrix<T>(result, instancesCount * outputHeight, outputWidth * channelsNum);
}

/**
 * @brief Golden model for ReLU activation.
 *
 * @param input Input matrix.
 * @return PlainMatrix<T> Resulting matrix after ReLU activation.
 */
template <typename T>
PlainMatrix<T> golden_relu(const PlainMatrix<T>& input) {
    size_t rows = input.rows();
    size_t cols = input.cols();
    cdough::Vector<T> result(rows * cols);
    const auto& input_data = input.data();

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            T val;
            if (input.isColumnWise()) {
                val = input_data[j * rows + i];
            } else {
                val = input_data[i * cols + j];
            }
            result[i * cols + j] = val > 0 ? val : 0;
        }
    }

    return PlainMatrix<T>(result, rows, cols);
}

// Define a test for secret-vs-plaintext binary operators
#define DEFINE_TEST_BINARY_OP(_op_, name)                              \
    template <typename T, typename Engine>                             \
    void test_matrix_##name(int rows, int cols, Engine& engine) {      \
        auto L = std::numeric_limits<std::make_unsigned_t<T>>::digits; \
        single_cout_nonl(L << "-bit Matrix " #name "... ");            \
        auto test_size = rows * cols;                                  \
        cdough::Vector<T> x(test_size);                                   \
        cdough::Vector<T> y(test_size);                                   \
                                                                       \
        if (engine.getPartyID() == 0) {                                \
            engine.populateLocalRandom(x);                             \
            engine.populateLocalRandom(y);                             \
        }                                                              \
        PlainMatrix<T> mat_x(x, rows, cols);                           \
        PlainMatrix<T> mat_y(y, rows, cols);                           \
                                                                       \
        auto sx = engine.secret_share_matrix(mat_x, 0);                \
        auto sy = engine.secret_share_matrix(mat_y, 0);                \
                                                                       \
        auto sz = sx _op_ sy;                                          \
                                                                       \
        auto z1 = sz.open();                                           \
        auto z2 = mat_x _op_ mat_y;                                    \
                                                                       \
        if (engine.getPartyID() == 0) {                                \
            assert(z1.same_as(z2));                                    \
            single_cout("OK");                                         \
        }                                                              \
    }

#define DEFINE_TEST_BINARY_ELEMENT_OP(_op_, name)                      \
    template <typename T, typename Engine>                             \
    void test_matrix_##name(int rows, int cols, Engine& engine) {      \
        auto L = std::numeric_limits<std::make_unsigned_t<T>>::digits; \
        single_cout_nonl(L << "-bit Matrix " #name "... ");            \
        auto test_size = rows * cols;                                  \
        cdough::Vector<T> x(test_size);                                   \
        cdough::Vector<T> y_(1);                                          \
                                                                       \
        if (engine.getPartyID() == 0) {                                \
            engine.populateLocalRandom(x);                             \
            engine.populateLocalRandom(y_);                            \
        }                                                              \
        PlainMatrix<T> mat_x(x, rows, cols);                           \
        auto y = y_[0];                                                \
                                                                       \
        auto sx = engine.secret_share_matrix(mat_x, 0);                \
                                                                       \
        auto sz = sx _op_ y;                                           \
                                                                       \
        auto z1 = sz.open();                                           \
        auto z2 = mat_x _op_ y;                                        \
                                                                       \
        if (engine.getPartyID() == 0) {                                \
            assert(z1.same_as(z2));                                    \
            single_cout("OK");                                         \
        }                                                              \
    }

template <typename T, typename Engine>
void test_secure_matrix_multiplication(int rows, int cols, Engine& engine) {
    auto L = std::numeric_limits<std::make_unsigned_t<T>>::digits;
    single_cout_nonl(L << "-bit Secure Matrix Multiplication ... ");
    auto test_size = rows * cols;
    cdough::Vector<T> x(test_size);
    cdough::Vector<T> y(test_size);

    if (engine.getPartyID() == 0) {
        engine.populateLocalRandom(x);
        engine.populateLocalRandom(y);
    }
    PlainMatrix<T> mat_x(x, rows, cols);
    PlainMatrix<T> mat_y(y, rows, cols, true);  // y is column-wise

    auto sx = engine.secret_share_matrix(mat_x, 0);
    auto sy = engine.secret_share_matrix(mat_y, 0);

    auto sz = sx.matrixRightMultiplyVectorized(sy);

    auto z1 = sz.open();
    auto z2 = golden_matrix_multiply(mat_x, mat_y);

    if (engine.getPartyID() == 0) {
        assert(z1.same_as(z2));
        single_cout("OK");
    }
}

template <typename T, typename Engine>
void test_secure_matrix_conv2d(int instancesCount, int rows, int cols, int outChannels,
                               cdough::matrix::HeightWidth filterSize, cdough::matrix::HeightWidth stride,
                               cdough::matrix::HeightWidth padding, Engine& engine) {
    auto L = std::numeric_limits<std::make_unsigned_t<T>>::digits;
    single_cout_nonl(L << "-bit Secure CONV2D ... ");
    auto test_size = instancesCount * rows * cols;
    cdough::Vector<T> x(test_size);
    cdough::Vector<T> y(filterSize.first * filterSize.second * outChannels);

    if (engine.getPartyID() == 0) {
        engine.populateLocalRandom(x);
        engine.populateLocalRandom(y);
    }
    PlainMatrix<T> mat_x(x, instancesCount * rows, cols);
    PlainMatrix<T> mat_y(y, outChannels * filterSize.first, filterSize.second);

    auto sx = engine.secret_share_matrix(mat_x, 0);
    auto sy = engine.secret_share_matrix(mat_y, 0);

    auto sz = sx.conv2DVectorized(sy, instancesCount, filterSize, stride, padding);

    auto z1 = sz.open();
    auto z2 = golden_conv2d_multichannel(mat_x, mat_y, instancesCount, outChannels, filterSize,
                                         stride, padding);

    if (engine.getPartyID() == 0) {
        assert(z1.same_as(z2));
        single_cout("OK");
    }
}

template <typename T, typename Engine>
void test_secure_matrix_avg_Pool(int instancesCount, int rows, int cols, int channels,
                                 cdough::matrix::HeightWidth filterSize,
                                 cdough::matrix::HeightWidth stride, cdough::matrix::HeightWidth padding,
                                 Engine& engine) {
    auto L = std::numeric_limits<std::make_unsigned_t<T>>::digits;
    single_cout_nonl(L << "-bit Secure AVG_POOL ... ");
    auto test_size = instancesCount * rows * cols * channels;
    cdough::Vector<T> x(test_size);

    if (engine.getPartyID() == 0) {
        engine.populateLocalRandom(x);
        x = x % static_cast<T>(10000);  // To avoid catastrophic error
    }
    PlainMatrix<T> mat_x(x, instancesCount * rows, cols * channels);

    auto sx = engine.secret_share_matrix(mat_x, 0);
    auto sz =
        sx.avgPoolingVectorized(instancesCount, channels, cdough::matrix::HeightWidth{rows, cols},
                                filterSize, stride, padding);

    auto z1 = sz.open();
    auto z2 = golden_avgpool2d(mat_x, instancesCount, channels, filterSize, stride, padding);

    if (engine.getPartyID() == 0) {
        for (int i = 0; i < z1.data().size(); ++i) {
            // To pass by one errors
            assert(std::abs(static_cast<int64_t>(z1.data()[i]) -
                            static_cast<int64_t>(z2.data()[i])) <= 1);
        }
        single_cout("OK");
    }
}

template <typename T, typename Engine>
void test_secure_matrix_relu(int test_size, Engine& engine) {
    auto L = std::numeric_limits<std::make_unsigned_t<T>>::digits;
    single_cout_nonl(L << "-bit Secure ReLU ... ");
    cdough::Vector<T> x(test_size);

    if (engine.getPartyID() == 0) {
        engine.populateLocalRandom(x);
    }
    PlainMatrix<T> mat_x(x, 1, test_size);
    auto sx = engine.secret_share_matrix(mat_x, 0);
    auto sz = sx.reLUVectorized();

    auto z1 = sz.open();
    auto z2 = golden_relu(mat_x);

    if (engine.getPartyID() == 0) {
        assert(z1.same_as(z2));
        single_cout("OK");
    }
}

DEFINE_TEST_BINARY_OP(+, addition);
DEFINE_TEST_BINARY_OP(-, subtraction);
DEFINE_TEST_BINARY_OP(*, multiplication);
DEFINE_TEST_BINARY_OP(>, greater_than);
DEFINE_TEST_BINARY_OP(<, less_than);
DEFINE_TEST_BINARY_OP(>=, greater_equal);
DEFINE_TEST_BINARY_OP(<=, less_equal);
DEFINE_TEST_BINARY_OP(==, equal);
DEFINE_TEST_BINARY_OP(!=, not_equal);

DEFINE_TEST_BINARY_ELEMENT_OP(+, element_addition);
DEFINE_TEST_BINARY_ELEMENT_OP(-, element_subtraction);

// Note that inplace operators modify the left operand and return a reference to it
DEFINE_TEST_BINARY_OP(+=, inplace_addition);
DEFINE_TEST_BINARY_OP(-=, inplace_subtraction);
DEFINE_TEST_BINARY_OP(*=, inplace_multiplication);

DEFINE_TEST_BINARY_ELEMENT_OP(+=, inplace_element_addition);
DEFINE_TEST_BINARY_ELEMENT_OP(-=, inplace_element_subtraction);

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);

    // The party's unique id
    auto pID = engine.getPartyID();

    const size_t test_size = 10;
    const size_t channels = 3;
    const size_t rows = 128;
    const size_t cols = 128;

    const cdough::matrix::HeightWidth filterSize{3, 3};
    const cdough::matrix::HeightWidth stride{1, 1};
    const cdough::matrix::HeightWidth padding{0, 0};

    ////////////////////////////////////////////////////////////
    /////////////////// Test ML operators //////////////////////
    ////////////////////////////////////////////////////////////
    {
        // test matrix multiplication
        PlainMatrix<int> mat_a(cdough::Vector<int>{1, 2, 3, 4, 5, 6}, 2, 3);
        PlainMatrix<int> mat_b(cdough::Vector<int>{7, 8, 9, 10, 11, 12}, 3, 2, true);
        PlainMatrix<int> groundTruth(cdough::Vector<int>{50, 68, 122, 167}, 2, 2);
        auto res = mat_a.matrixRightMultiplyVectorized(mat_b);

        auto shared_mat_a = engine.secret_share_matrix(mat_a, 0);
        auto shared_mat_b = engine.secret_share_matrix(mat_b, 0);
        auto shared_res = shared_mat_a.matrixRightMultiplyVectorized(shared_mat_b);
        auto opened_res = shared_res.open();

        assert(res.same_as(groundTruth));
        assert(opened_res.same_as(groundTruth));

        // if (pID == 0) mat_a.print();
        // if (pID == 0) mat_b.print();
        // if (pID == 0) res.print();
        // if (pID == 0) opened_res.print();
    }
    test_secure_matrix_multiplication<int8_t>(rows, cols, engine);
    test_secure_matrix_multiplication<int16_t>(rows, cols, engine);
    test_secure_matrix_multiplication<int32_t>(rows, cols, engine);
    test_secure_matrix_multiplication<int64_t>(rows, cols, engine);
#if !defined(MPC_PROTOCOL_SPDZ_2K_NPC)
    test_secure_matrix_multiplication<__int128_t>(rows, cols, engine);
#endif

    {
        // Test matrix conv2DVectorized
        size_t instancesCount = 3;
        cdough::matrix::HeightWidth inputSize{3, 3};
        cdough::matrix::HeightWidth filterSize{2, 2};
        cdough::matrix::HeightWidth stride{1, 1};
        PlainMatrix<int> mat(cdough::Vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5,
                                              4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                             inputSize.first * instancesCount, inputSize.second);

        PlainMatrix<int> filter(cdough::Vector<int>{1, 0, 0, 1}, filterSize.first, filterSize.second);

        PlainMatrix<int> groundTruth(cdough::Vector<int>{6, 8, 12, 14, 14, 12, 8, 6, 2, 2, 2, 2}, 6,
                                     2);

        auto res = mat.conv2DVectorized(filter, instancesCount, filterSize, stride, {0, 0});

        auto shared_mat = engine.secret_share_matrix(mat, 0);
        auto shared_filter = engine.secret_share_matrix(filter, 0);
        auto shared_res =
            shared_mat.conv2DVectorized(shared_filter, instancesCount, filterSize, stride, {0, 0});
        auto opened_res = shared_res.open();

        assert(res.same_as(groundTruth));
        assert(opened_res.same_as(groundTruth));

        // if (pID == 0) mat.print();
        // if (pID == 0) filter.print();
        // if (pID == 0) res.print();
        // if (pID == 0) opened_res.print();
    }
    test_secure_matrix_conv2d<int8_t>(test_size, rows, cols, channels, filterSize, stride, padding,
                                      engine);
    test_secure_matrix_conv2d<int16_t>(test_size, rows, cols, channels, filterSize, stride, padding,
                                       engine);
    test_secure_matrix_conv2d<int32_t>(test_size, rows, cols, channels, filterSize, stride, padding,
                                       engine);
    test_secure_matrix_conv2d<int64_t>(test_size, rows, cols, channels, filterSize, stride, padding,
                                       engine);
#if !defined(MPC_PROTOCOL_SPDZ_2K_NPC)
    test_secure_matrix_conv2d<__int128_t>(test_size, rows, cols, channels, filterSize, stride,
                                          padding, engine);
#endif

    {
        // test matrix ReLU
        PlainMatrix<int> mat_a(cdough::Vector<int>{1, 2, 3, 4, 0, -1}, 2, 3);
        PlainMatrix<int> groundTruth(cdough::Vector<int>{1, 2, 3, 4, 0, 0}, 2, 3);

        auto shared_mat_a = engine.secret_share_matrix(mat_a, 0);
        auto shared_res = shared_mat_a.reLUVectorized();
        auto opened_res = shared_res.open();

        assert(opened_res.same_as(groundTruth));

        // if (pID == 0) mat_a.print();
        // if (pID == 0) opened_res.print();
    }
    test_secure_matrix_relu<int8_t>(test_size * rows * cols, engine);
    test_secure_matrix_relu<int16_t>(test_size * rows * cols, engine);
    test_secure_matrix_relu<int32_t>(test_size * rows * cols, engine);
    test_secure_matrix_relu<int64_t>(test_size * rows * cols, engine);
#if !defined(MPC_PROTOCOL_SPDZ_2K_NPC)
    test_secure_matrix_relu<__int128_t>(test_size * rows * cols, engine);
#endif

    // TODO: support smaller sizes after trunction
    test_secure_matrix_avg_Pool<int64_t>(test_size, rows, cols, channels, filterSize, stride,
        padding, engine);
#if !defined(MPC_PROTOCOL_SPDZ_2K_NPC)
    test_secure_matrix_avg_Pool<__int128_t>(test_size, rows, cols, channels, filterSize, stride,
                                            padding, engine);
#endif

    {
        // Test matrix secret sharing
        PlainMatrix<int> mat(
            cdough::Vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, 4, 4);

        auto shared_mat = engine.secret_share_matrix(mat, 0);
        auto opened_mat = shared_mat.open();

        assert(opened_mat.same_as(mat));

        if (pID == 0) std::cout << "Matrix secret sharing...OK" << std::endl;
    }

    ////////////////////////////////////////////////////////////
    /////////////// Test Functional operators //////////////////
    ////////////////////////////////////////////////////////////
    {
        // Test matrix addition
        test_matrix_addition<int8_t>(rows, cols, engine);
        test_matrix_addition<int16_t>(rows, cols, engine);
        test_matrix_addition<int32_t>(rows, cols, engine);
        test_matrix_addition<int64_t>(rows, cols, engine);
#if !defined(MPC_PROTOCOL_SPDZ_2K_NPC)
        test_matrix_addition<__int128_t>(rows, cols, engine);
#endif

        // Test matrix subtraction
        test_matrix_subtraction<int8_t>(rows, cols, engine);
        test_matrix_subtraction<int16_t>(rows, cols, engine);
        test_matrix_subtraction<int32_t>(rows, cols, engine);
        test_matrix_subtraction<int64_t>(rows, cols, engine);
#if !defined(MPC_PROTOCOL_SPDZ_2K_NPC)
        test_matrix_subtraction<__int128_t>(rows, cols, engine);
#endif

        // Test matrix multiplication
        test_matrix_multiplication<int8_t>(rows, cols, engine);
        test_matrix_multiplication<int16_t>(rows, cols, engine);
        test_matrix_multiplication<int32_t>(rows, cols, engine);
        test_matrix_multiplication<int64_t>(rows, cols, engine);
#if !defined(MPC_PROTOCOL_SPDZ_2K_NPC)
        test_matrix_multiplication<__int128_t>(rows, cols, engine);
#endif

        // Test matrix Greater Than
        test_matrix_greater_than<int8_t>(rows, cols, engine);
        test_matrix_greater_than<int16_t>(rows, cols, engine);
        test_matrix_greater_than<int32_t>(rows, cols, engine);
        test_matrix_greater_than<int64_t>(rows, cols, engine);
#if !defined(MPC_PROTOCOL_SPDZ_2K_NPC)
        test_matrix_greater_than<__int128_t>(rows, cols, engine);
#endif

        // Test matrix Less Than
        test_matrix_less_than<int8_t>(rows, cols, engine);
        test_matrix_less_than<int16_t>(rows, cols, engine);
        test_matrix_less_than<int32_t>(rows, cols, engine);
        test_matrix_less_than<int64_t>(rows, cols, engine);
#if !defined(MPC_PROTOCOL_SPDZ_2K_NPC)
        test_matrix_less_than<__int128_t>(rows, cols, engine);
#endif
        // Test matrix Greater Equal
        test_matrix_greater_equal<int8_t>(rows, cols, engine);
        test_matrix_greater_equal<int16_t>(rows, cols, engine);
        test_matrix_greater_equal<int32_t>(rows, cols, engine);
        test_matrix_greater_equal<int64_t>(rows, cols, engine);
#if !defined(MPC_PROTOCOL_SPDZ_2K_NPC)
        test_matrix_greater_equal<__int128_t>(rows, cols, engine);
    #endif

        // Test matrix Less Equal
        test_matrix_less_equal<int8_t>(rows, cols, engine);
        test_matrix_less_equal<int16_t>(rows, cols, engine);
        test_matrix_less_equal<int32_t>(rows, cols, engine);
        test_matrix_less_equal<int64_t>(rows, cols, engine);
#if !defined(MPC_PROTOCOL_SPDZ_2K_NPC)
        test_matrix_less_equal<__int128_t>(rows, cols, engine);
#endif

        // Test matrix Equal
        test_matrix_equal<int8_t>(rows, cols, engine);
        test_matrix_equal<int16_t>(rows, cols, engine);
        test_matrix_equal<int32_t>(rows, cols, engine);
        test_matrix_equal<int64_t>(rows, cols, engine);
#if !defined(MPC_PROTOCOL_SPDZ_2K_NPC)
        test_matrix_equal<__int128_t>(rows, cols, engine);
#endif

        // Test matrix Not Equal
        test_matrix_not_equal<int8_t>(rows, cols, engine);
        test_matrix_not_equal<int16_t>(rows, cols, engine);
        test_matrix_not_equal<int32_t>(rows, cols, engine);
        test_matrix_not_equal<int64_t>(rows, cols, engine);
#if !defined(MPC_PROTOCOL_SPDZ_2K_NPC)
        test_matrix_not_equal<__int128_t>(rows, cols, engine);
#endif
    }

    ////////////////////////////////////////////////////////////
    //////////////// Test Inplace operators ////////////////////
    ////////////////////////////////////////////////////////////
    {
        // Test matrix inplace addition
        test_matrix_inplace_addition<int8_t>(rows, cols, engine);
        test_matrix_inplace_addition<int16_t>(rows, cols, engine);
        test_matrix_inplace_addition<int32_t>(rows, cols, engine);
        test_matrix_inplace_addition<int64_t>(rows, cols, engine);
#if !defined(MPC_PROTOCOL_SPDZ_2K_NPC)
        test_matrix_inplace_addition<__int128_t>(rows, cols, engine);
#endif

        // Test matrix inplace subtraction
        test_matrix_inplace_subtraction<int8_t>(rows, cols, engine);
        test_matrix_inplace_subtraction<int16_t>(rows, cols, engine);
        test_matrix_inplace_subtraction<int32_t>(rows, cols, engine);
        test_matrix_inplace_subtraction<int64_t>(rows, cols, engine);
#if !defined(MPC_PROTOCOL_SPDZ_2K_NPC)
        test_matrix_inplace_subtraction<__int128_t>(rows, cols, engine);
#endif

        // Test matrix inplace multiplication
        test_matrix_inplace_multiplication<int8_t>(rows, cols, engine);
        test_matrix_inplace_multiplication<int16_t>(rows, cols, engine);
        test_matrix_inplace_multiplication<int32_t>(rows, cols, engine);
        test_matrix_inplace_multiplication<int64_t>(rows, cols, engine);
#if !defined(MPC_PROTOCOL_SPDZ_2K_NPC)
        test_matrix_inplace_multiplication<__int128_t>(rows, cols, engine);
#endif
    }

    return 0;
}