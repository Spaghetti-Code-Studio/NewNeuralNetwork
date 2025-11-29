#include "TestDataSoftmaxEvaluator.hpp"

#include <limits>

#include "FloatMatrixInvalidDimensionException.hpp"

namespace nnn::TestDataSoftmaxEvaluator {

  Result Evaluate(const FloatMatrix& result, const FloatMatrix& testingLabels) {  //

    if (result.GetColCount() != testingLabels.GetColCount() || result.GetRowCount() != testingLabels.GetRowCount()) {
      throw FloatMatrixInvalidDimensionException("Dimensions of matrices for their evaluation have to be equal!");
    }

    size_t totalExamplesCount = result.GetColCount();
    size_t classes = result.GetRowCount();
    size_t correctlyClassifiedCount = 0;

    for (int col = 0; col < totalExamplesCount; ++col) {  //

      float max = -std::numeric_limits<float>::infinity();
      int max_index = -1;

      for (int row = 0; row < classes; ++row) {
        float current_probability = result(row, col);
        if (current_probability > max) {
          max = current_probability;
          max_index = row;
        }
      }

      if (testingLabels(max_index, col) == 1.0f) {
        correctlyClassifiedCount++;
      }
    }

    return {.totalExamplesCount = totalExamplesCount, .correctlyClassifiedCount = correctlyClassifiedCount};
  }
}  // namespace nnn::TestDataSoftmaxEvaluator
