#pragma once

#include "FloatMatrix.hpp"

namespace nnn {

  /**
   * @brief Common interface for all weight initialization algorithms.
   */
  class IWeightInitializer {
   public:
    virtual ~IWeightInitializer() = 0;

    /**
     * @brief Creates a matrix of the given size assuming it to be a matrix
     * of weight in between layers of corresponding sizes.
     * 
     * @param row the number of rows in the matrix.
     * @param column the number of columns in the matrix.
     * 
     * @return A new matrix with the initialized weight values.
     */
    virtual FloatMatrix Initialize(size_t row, size_t col) = 0;
  };

  inline IWeightInitializer::~IWeightInitializer() = default;
}  // namespace nnn
