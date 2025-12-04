#pragma once

#include "FloatMatrix.hpp"

namespace nnn {

  /**
   * @brief The core abstraction of the network, the common interface for all layer types.
   */
  class ILayer {
   public:
    virtual ~ILayer() = 0;

    /**
     * @brief Computes the forward pass through the network for the given vector (or a batch).
     * @return The resulting values given by the last layer of the vector.
     */
    virtual FloatMatrix Forward(const FloatMatrix& input) = 0;

    /**
     * @brief Performs backpropagation provided the gradient of the previous layer (in the backward direction).
     * @return The gradient for the next layer (in the backward direction).
     */
    virtual FloatMatrix Backward(const FloatMatrix& gradient) = 0;

    virtual const FloatMatrix& GetBiases() const = 0;
    virtual const FloatMatrix& GetWeights() const = 0;
    virtual const FloatMatrix& GetWeightsGradient() const = 0;
    virtual const FloatMatrix& GetBiasesGradient() const = 0;

    /**
     * @brief Forcefully updates the weights and biases of the layer ignoring any internal state.
     * @param weight the new values for the weights matrix.
     * @param biases the new values for the biases vector.
     */
    virtual void Update(const FloatMatrix& weights, const FloatMatrix& biases) = 0;

    virtual FloatMatrix& GetWeightsVelocity() = 0;
    virtual FloatMatrix& GetBiasesVelocity() = 0;
  };

  inline ILayer::~ILayer() = default;
}  // namespace nnn
