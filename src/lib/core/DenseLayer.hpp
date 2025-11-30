#pragma once

#include <memory>

#include "FloatMatrix.hpp"
#include "IActivationFunction.hpp"
#include "ILayer.hpp"
#include "IWeightInitializer.hpp"

namespace nnn {

  class DenseLayer : public virtual ILayer {
   public:
    DenseLayer(size_t batchSize,
        size_t inputSize,
        size_t outputSize,
        std::unique_ptr<IActivationFunction>&& activationFunction,
        IWeightInitializer& initializer);

    DenseLayer(size_t batchSize,
        size_t inputSize,
        size_t outputSize,
        std::unique_ptr<IActivationFunction>&& activationFunction);

    DenseLayer(size_t inputSize, size_t outputSize, std::unique_ptr<IActivationFunction>&& activationFunction);

    DenseLayer(size_t inputSize,
        size_t outputSize,
        std::unique_ptr<IActivationFunction>&& activationFunction,
        IWeightInitializer& initializer);

    FloatMatrix Forward(const FloatMatrix& inputVector) override;

    /**
     * @param gradient how much does the loss change when my outputs change (dE/dy)
     * @returns how much the loss changes when next layer outputs change (with respect to backward pass)
     */
    FloatMatrix Backward(const FloatMatrix& gradient) override;
    void Update(const FloatMatrix& weights, const FloatMatrix& biases) override;

    const FloatMatrix& GetWeights() const override;
    const FloatMatrix& GetBiases() const override;
    inline const FloatMatrix& GetWeightsGradient() const override { return m_gradientWeights; }
    inline const FloatMatrix& GetBiasesGradient() const override { return m_gradientBias; }
    inline FloatMatrix& GetWeightsVelocity() override { return m_weightVelocity;}
    inline FloatMatrix& GetBiasesVelocity() override { return m_biasesVelocity;}


   protected:
    size_t m_inputSize;
    size_t m_outputSize;
    FloatMatrix m_weights;
    FloatMatrix m_biases;
    std::unique_ptr<IActivationFunction> m_activationFunction;
    FloatMatrix m_lastInnerPotential;
    FloatMatrix m_lastInput;
    FloatMatrix m_gradientWeights;
    FloatMatrix m_gradientBias;
    FloatMatrix m_weightVelocity;
    FloatMatrix m_biasesVelocity;
  };
}  // namespace nnn
