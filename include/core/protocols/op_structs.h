#pragma once

namespace cdough {
/**
 * @brief Operator functors for generic binary operations.
 *
 * These structs define the operators used in binary operations like MULT and AND,
 * allowing the control flow to be templated and shared between implementations.
 */

struct ArithmeticOps {
    template <typename T>
    void accumulateOp(T& a, const T& b) const {
        a += b;
    }

    template <typename T>
    T multiplyOp(const T& a, const T& b) const {
        return a * b;
    }

    template <typename T>
    T addOp(const T& a, const T& b) const {
        return a + b;
    }

    // Whether to call truncate() after computation
    static constexpr bool do_truncate = true;
};

struct DotProductOps {
    size_t agg = 0;

    DotProductOps(size_t agg = 0) : agg(agg) {}

    template <typename T>
    void accumulateOp(T& a, const T& b) const {
        a += b;
    }

    template <typename T>
    T multiplyOp(const T& a, const T& b) const {
        return a.dot_product(b, agg);
    }

    template <typename T>
    T addOp(const T& a, const T& b) const {
        return a + b;
    }

    // Whether to call truncate() after computation
    static constexpr bool do_truncate = true;
};

struct MatMulOps {
    size_t lhs_rows = 0;
    size_t lhs_cols = 0;
    size_t rhs_rows = 0;
    size_t rhs_cols = 0;

    MatMulOps() = default;

    MatMulOps(size_t lhs_rows, size_t lhs_cols, size_t rhs_rows, size_t rhs_cols)
        : lhs_rows(lhs_rows), lhs_cols(lhs_cols), rhs_rows(rhs_rows), rhs_cols(rhs_cols) {}

    template <typename T>
    void accumulateOp(T& a, const T& b) const {
        a += b;
    }

    template <typename T>
    T multiplyOp(const T& a, const T& b) const {
        return a.matrixRightMultiplyWithColumnMatrixVectorized(b, lhs_rows, lhs_cols, rhs_rows,
                                                               rhs_cols);
    }

    template <typename T>
    T addOp(const T& a, const T& b) const {
        return a + b;
    }

    static constexpr bool do_truncate = true;
};

struct MatConvOps {
    size_t instancesCount = 0;
    size_t inputHeight = 0;
    size_t inputWidth = 0;
    size_t filterHeight = 0;
    size_t filterWidth = 0;
    size_t strideHeight = 0;
    size_t strideWidth = 0;
    size_t paddingHeight = 0;
    size_t paddingWidth = 0;

    MatConvOps() = default;

    MatConvOps(size_t instancesCount, size_t inputHeight, size_t inputWidth, size_t filterHeight,
               size_t filterWidth, size_t strideHeight, size_t strideWidth, size_t paddingHeight,
               size_t paddingWidth)
        : instancesCount(instancesCount),
          inputHeight(inputHeight),
          inputWidth(inputWidth),
          filterHeight(filterHeight),
          filterWidth(filterWidth),
          strideHeight(strideHeight),
          strideWidth(strideWidth),
          paddingHeight(paddingHeight),
          paddingWidth(paddingWidth) {}

    template <typename T>
    void accumulateOp(T& a, const T& b) const {
        a += b;
    }

    template <typename T>
    T multiplyOp(const T& a, const T& b) const {
        return a.conv2DVectorized(b, instancesCount, inputHeight, inputWidth, filterHeight,
                                  filterWidth, strideHeight, strideWidth, paddingHeight,
                                  paddingWidth);
    }

    template <typename T>
    T addOp(const T& a, const T& b) const {
        return a + b;
    }

    static constexpr bool do_truncate = true;
};

struct BooleanOps {
    // Equivalent of addition
    template <typename T>
    void accumulateOp(T& a, const T& b) const {
        a ^= b;
    }

    // Equivalent of multiplication
    template <typename T>
    T multiplyOp(const T& a, const T& b) const {
        return a & b;
    }

    template <typename T>
    T addOp(const T& a, const T& b) const {
        return a ^ b;
    }

    // No fixed point ops for boolean
    static constexpr bool do_truncate = false;
};
}  // namespace cdough