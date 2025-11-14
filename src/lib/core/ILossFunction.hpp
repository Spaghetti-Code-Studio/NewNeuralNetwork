#pragma once

#include "FloatMatrix.hpp"

namespace nnn {

  class ILossFunction {
   public:
    virtual ~ILossFunction() = 0;
    virtual FloatMatrix Loss(const FloatMatrix& actual, const FloatMatrix& expected) = 0;
  };

  inline ILossFunction::~ILossFunction() = default;
}  // namespace nnn
