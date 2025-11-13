#pragma once

#include <memory>

#include "FloatMatrix.hpp"
#include "IActivationFunction.hpp"
#include "ILayer.hpp"

namespace nnn {

  class DenseLayer : public ILayer {
   public:
    DenseLayer(size_t batchSize,
        size_t inputSize,
        size_t outputSize,
        std::unique_ptr<IActivationFunction>&& activationFunction);

    DenseLayer(size_t inputSize, size_t outputSize, std::unique_ptr<IActivationFunction>&& activationFunction);
    FloatMatrix Forward(const FloatMatrix& inputVector) const override;

    /**
     * @param gradient how much does the loss change when my outputs change (dE/dy)
     * @returns how much the loss changes when next layer outputs change (with respect to backward pass)
     */
    FloatMatrix Backward(const FloatMatrix& gradient) override;
    void Update(const FloatMatrix& weights, const FloatMatrix& biases) override;

    const FloatMatrix& GetWeights() const override;
    const FloatMatrix& GetBiases() const override;
    inline FloatMatrix GetGradientWeights() const { return m_gradientWeigths; }
    inline FloatMatrix GetGradientBiases() const { return m_gradientBias; }

   private:
    size_t m_inputSize;
    size_t m_outputSize;
    FloatMatrix m_weights;
    FloatMatrix m_biases;
    std::unique_ptr<IActivationFunction> m_activationFunction;
    FloatMatrix m_lastInnerPotential;
    FloatMatrix m_lastInput;
    FloatMatrix m_gradientWeigths;
    FloatMatrix m_gradientBias;
  };
}  // namespace nnn
