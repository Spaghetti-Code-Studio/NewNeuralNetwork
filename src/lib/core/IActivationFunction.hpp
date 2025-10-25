#pragma once

#include "FloatMatrix.hpp"

namespace nnn {

  class IActivationFunction {
   public:
    virtual ~IActivationFunction() = 0;
    virtual void Evaluate(FloatMatrix& input) const = 0;
  };

  inline IActivationFunction::~IActivationFunction() = default;
}  // namespace nnn
