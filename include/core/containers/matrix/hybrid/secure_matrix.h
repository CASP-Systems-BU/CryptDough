#pragma once

#include "core/containers/matrix/hybrid/matrix.h"

namespace cdough::matrix::hybrid {

/**
 * @brief Secure Matrix class
 * @tparam T Data type
 * @tparam V Vector type
 *
 * This class implements a secure matrix.
 */
template <typename T, template <typename> class V>
class SecureMatrix : public Matrix<T, V, SecureMatrix> {
   public:
    ////////////////////////////////////////////////////////////
    ////////////////////// Constructors ////////////////////////
    ////////////////////////////////////////////////////////////

    /**
     * @brief Construct a new Secure Matrix object
     * @param data 1D data vector
     * @param rows Number of rows
     * @param cols Number of columns
     * @param columnWise Whether the data is stored in column-wise order
     */
    SecureMatrix(const V<T>& data, size_t rows, size_t cols, const bool& columnWise = false)
        : Matrix<T, V, SecureMatrix>(data, rows, cols, columnWise) {
        assert(data.size() == rows * cols);
    }

    virtual ~SecureMatrix() {}

    /**
     * @brief Set the precision for fixed-point representation
     * @param precision Number of bits for the fractional part
     */
    void setPrecision(size_t precision) { this->data_.setPrecision(precision); }

    /**
     * @brief Open the secure matrix and return a plaintext matrix
     * @return PlainMatrix<T, V> The opened plaintext matrix
     */
    auto open() const {
        auto opened_data = this->data_.open();
        return typename V<T>::EngineType::template PlainMatrix<T>(opened_data, this->rows_,
                                                                  this->cols_);
    }

    /**
     * @brief Get the contents of the secure matrix as a vector
     * @return V<T> The data vector of the secure matrix
     */
    V<T> getContents() const { return this->data_; }
};

}  // namespace cdough::matrix::hybrid