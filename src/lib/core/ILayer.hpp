#pragma once

#include "FloatMatrix.hpp"

namespace nnn {

  class ILayer {
   public:
    virtual ~ILayer() = 0;
    virtual FloatMatrix Forward(const FloatMatrix& inputVector) const = 0;
    virtual const FloatMatrix& GetBiases() const = 0;
    virtual const FloatMatrix& GetWeights() const = 0;
    virtual void Update(const FloatMatrix& weights, const FloatMatrix& biases) = 0;
  };

  inline ILayer::~ILayer() = default;
}  // namespace nnn
