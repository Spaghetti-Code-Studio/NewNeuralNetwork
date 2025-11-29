#pragma once

#include "FloatMatrix.hpp"
#include "ILayer.hpp"

namespace nnn {

  class IOutputLayer : public virtual ILayer {
   public:
    virtual ~IOutputLayer() = 0;

    /**
     * @brief Computes output gradient by using loss function.
     *
     * @param actual (...)
     * @param expected (...)
     *
     * @returns (...)
     */
    virtual FloatMatrix ComputeOutputGradient(const FloatMatrix& actual, const FloatMatrix& expected) = 0;
  };

  inline IOutputLayer::~IOutputLayer() = default;
}  // namespace nnn
