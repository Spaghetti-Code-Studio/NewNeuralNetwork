#pragma once

#include "FloatMatrix.hpp"
#include "IActivationFunction.hpp"

namespace nnn {

  class ReLU : public IActivationFunction {
   public:
    ReLU() = default;
    void Evaluate(FloatMatrix& input) const override;
  };
}  // namespace nnn
