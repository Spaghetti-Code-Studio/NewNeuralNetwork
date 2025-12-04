#pragma once

#include "FloatMatrix.hpp"
#include "ILayer.hpp"

namespace nnn {

  /**
   * @brief The abstraction last layer of the network, reports the gradient for training.
   * This is done either pseudo-randomly or deterministically based on some seed. 
   */
  class IOutputLayer : public virtual ILayer {
   public:
    virtual ~IOutputLayer() = 0;

    /**
     * @brief Computes output gradient by using some cost function.
     *
     * @param actual the output of the network on the input.
     * @param expected the correct labels for the same input.
     *
     * @returns Gradient vector (averaged if given a batch).
     */
    virtual FloatMatrix ComputeOutputGradient(const FloatMatrix& actual, const FloatMatrix& expected) = 0;
  };

  inline IOutputLayer::~IOutputLayer() = default;
}  // namespace nnn
