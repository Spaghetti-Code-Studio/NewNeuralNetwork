#pragma once

#include <iostream>

#include "FloatMatrix.hpp"

namespace nnn::TestDataSoftmaxEvaluator {

  struct Result {
    size_t totalExamplesCount;
    size_t correctlyClassifiedCount;

    void Print() const {
      std::cout << "Percentage of correctly classified examples: "
                << (static_cast<float>(correctlyClassifiedCount) / static_cast<float>(totalExamplesCount) * 100.0f)
                << "%." << std::endl;
    }
  };

  Result Evaluate(const FloatMatrix& result, const FloatMatrix& testingLabels);
}  // namespace nnn::TestDataSoftmaxEvaluator
