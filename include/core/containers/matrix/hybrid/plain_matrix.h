#pragma once

#include "core/containers/matrix/hybrid/matrix.h"

namespace cdough::matrix::hybrid {

/**
 * @brief Plain Matrix class
 * @tparam T Data type
 * @tparam V Vector type
 *
 * This class implements a plaintext matrix.
 */
template <typename T, template <typename> class V>
class PlainMatrix : public Matrix<T, V, PlainMatrix> {
   public:
    /**
     * @brief Generate a random plain matrix,
     * output can be modulo `mod` if specified.
     *
     * @tparam Generator Randomness generator type
     *
     * @param generator Randomness generator
     * @param rows Number of rows
     * @param cols Number of columns
     * @param mod Modulus for the random values
     * @return PlainMatrix<T, V> The generated random matrix
     */
    template <typename Generator>
    static PlainMatrix RandomMatrix(Generator& generator, size_t rows, size_t cols,
                                    const T& mod = std::numeric_limits<T>::max()) {
        V<T> data(rows * cols);
        generator.template populateLocalRandom<T>(data);
        data = data % mod;
        return PlainMatrix(data, rows, cols);
    }

    /**
     * @brief Generate a random plain column-wise matrix
     * @tparam Generator Randomness generator type
     * @param generator Randomness generator
     * @param rows Number of rows
     * @param cols Number of columns
     * @param mod Modulus for the random values
     * @return PlainMatrix<T, V> The generated random column-wise matrix
     */
    template <typename Generator>
    static PlainMatrix RandomColumnMatrix(Generator& generator, size_t rows, size_t cols,
                                          const T& mod = std::numeric_limits<T>::max()) {
        auto mat = RandomMatrix(generator, rows, cols, mod);
        mat.columnWise_ = true;
        return mat;
    }

   public:
    ////////////////////////////////////////////////////////////
    ////////////////////// Constructors ////////////////////////
    ////////////////////////////////////////////////////////////

    /**
     * @brief Construct a new Plain Matrix object
     * @param data 1D data vector
     * @param rows Number of rows
     * @param cols Number of columns
     * @param columnWise Whether the data is stored in column-wise order
     */
    PlainMatrix(const V<T>& data, size_t rows, size_t cols, const bool& columnWise = false)
        : Matrix<T, V, PlainMatrix>(data, rows, cols, columnWise) {
        assert(data.size() == rows * cols);
    }

    /**
     * @brief Construct a new Plain Matrix object
     * @param rows Number of rows
     * @param cols Number of columns
     * @param columnWise Whether the data is stored in column-wise order
     */
    PlainMatrix(size_t rows, size_t cols, const bool& columnWise = false)
        : Matrix<T, V, PlainMatrix>(V<T>(rows * cols), rows, cols, columnWise) {}

    virtual ~PlainMatrix() {}

    ////////////////////////////////////////////////////////////
    ////////////////////////// Other ///////////////////////////
    ////////////////////////////////////////////////////////////

    /**
     * @brief Check if two matrices are the same
     * @param other The other matrix to compare with
     * @return true if the matrices are the same, false otherwise
     */
    auto same_as(const PlainMatrix& other) const {
        if ((this->rows_ != other.rows_) ||
            (this->cols_ != other.cols_ || (this->columnWise_ != other.columnWise_))) {
            return false;
        }
        return this->data_.same_as(other.data_);
    }

    /**
     * @brief Prints the matrix
     */
    void print() const {
        std::cout << "Matrix " << this->rows_ << "x" << this->cols_ << ":" << std::endl;
        std::cout << "[" << std::endl;

        for (size_t i = 0; i < this->rows_; i++) {
            std::cout << "  [";
            for (size_t j = 0; j < this->cols_; j++) {
                if (this->columnWise_) {
                    std::cout << std::setw(8) << this->data_[j * this->rows_ + i];
                } else {
                    std::cout << std::setw(8) << this->data_[i * this->cols_ + j];
                }
                if (j < this->cols_ - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]";
            if (i < this->rows_ - 1) {
                std::cout << ",";
            }
            std::cout << std::endl;
        }

        std::cout << "]" << std::endl;
    }
};

}  // namespace cdough::matrix::hybrid