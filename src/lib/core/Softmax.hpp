#pragma once

#include "FloatMatrix.hpp"
#include "IActivationFunction.hpp"

namespace nnn {

  class Softmax : public IActivationFunction {
   public:
    Softmax() = default;

    /**
     * @note Implementation inspired by https://www.aussieai.com/book/ch25-softmax-cpp-optimizations and
     * https://learncplusplus.org/what-is-the-softmax-function-in-neural-networks/.
     */
    void Evaluate(FloatMatrix& input) const override;

    /**
     * @todo Not implemented yet.
     */
    void Derivative(FloatMatrix& input) const override;
  };
}  // namespace nnn
