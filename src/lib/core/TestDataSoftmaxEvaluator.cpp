#include "TestDataSoftmaxEvaluator.hpp"

#include <iostream>
#include <limits>

#include "FloatMatrixInvalidDimensionException.hpp"

namespace nnn::TestDataSoftmaxEvaluator {

  EvaluationResult Evaluate(const FloatMatrix& result, const FloatMatrix& testingLabels) {  //

    if (result.GetColCount() != testingLabels.GetColCount() || result.GetRowCount() != testingLabels.GetRowCount()) {
      throw FloatMatrixInvalidDimensionException("Dimensions of matrices for their evaluation have to be equal!");
    }

    size_t totalExamplesCount = result.GetColCount();
    size_t classes = result.GetRowCount();
    size_t correctlyClassifiedCount = 0;

    for (int col = 0; col < totalExamplesCount; ++col) {  //

      // TODO: code duplication with the logic in CSVLabelsWriter::Write() method.
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
  void EvaluationResult::Print() const {
    std::cout << "Percentage of correctly classified examples: "
              << (static_cast<float>(correctlyClassifiedCount) / static_cast<float>(totalExamplesCount) * 100.0f)
              << "%." << std::endl;
  }
}  // namespace nnn::TestDataSoftmaxEvaluator
