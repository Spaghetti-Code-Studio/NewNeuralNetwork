#pragma once

#include "FloatMatrix.hpp"

namespace nnn::TestDataSoftmaxEvaluator {

  struct EvaluationResult {
    size_t totalExamplesCount;
    size_t correctlyClassifiedCount;

    void Print() const;
  };

  EvaluationResult Evaluate(const FloatMatrix& result, const FloatMatrix& testingLabels);
}  // namespace nnn::TestDataSoftmaxEvaluator
