#pragma once

#include "FloatMatrix.hpp"
#include "IActivationFunction.hpp"

namespace nnn {

  class LeakyReLU : public IActivationFunction {
   public:
    LeakyReLU() = default;
    LeakyReLU(float alpha);
    void Evaluate(FloatMatrix& input) const override;
    void Derivative(FloatMatrix& input) const override;
   private:
    float m_alpha = 0.05;
  };
}  // namespace nnn
