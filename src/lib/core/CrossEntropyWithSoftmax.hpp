#pragma once

#include "FloatMatrix.hpp"
#include "ILossFunction.hpp"

namespace nnn {
  /**
   * @brief It is expected that this loss function is used together with softmax activation function in the last
   * (output) layer.
   */
  class CrossEntropyWithSoftmax : public ILossFunction {
   public:
    CrossEntropyWithSoftmax() = default;
    /**
     * @note Implemented using
     * https://www.geeksforgeeks.org/machine-learning/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss/.
     */
    FloatMatrix Loss(const FloatMatrix& actual, const FloatMatrix& expected) override;
  };
}  // namespace nnn
