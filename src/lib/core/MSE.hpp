#pragma once

#include "FloatMatrix.hpp"
#include "ILossFunction.hpp"

namespace nnn {
  class MSE : public ILossFunction {
   public:
    MSE() = default;
    FloatMatrix Loss(const FloatMatrix& actual, const FloatMatrix& expected) override;
  };
}  // namespace nnn
