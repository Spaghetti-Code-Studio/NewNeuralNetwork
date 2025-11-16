#pragma once

#include "FloatMatrix.hpp"
#include "ILayer.hpp"

namespace nnn {

  class IOutputLayer : public ILayer {
   public:
    virtual ~IOutputLayer() = 0;
    virtual FloatMatrix ComputeOutputGradient(const FloatMatrix& actual, const FloatMatrix& expected) = 0;
  };

  inline IOutputLayer::~IOutputLayer() = default;
}  // namespace nnn
