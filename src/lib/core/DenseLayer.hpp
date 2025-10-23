#pragma once

#include "ILayer.hpp"
#include <FloatMatrix.hpp>
#include "IActivationFunction.hpp"

namespace nnn {

  class DenseLayer : public ILayer {
   public:
    DenseLayer(size_t neuronCount, size_t outputSize, const IActivationFunction& activationFunction);
    FloatMatrix Forward(FloatMatrix& inputVector) override;
    void Update(const FloatMatrix& weights, const FloatMatrix& biases) override;

    inline const FloatMatrix& GetWeights() const { return m_weights; }
    inline const FloatMatrix& GetBiases() const { return m_biases; }

   private:
    size_t m_inputSize;
    size_t m_outputSize;
    FloatMatrix m_weights;
    FloatMatrix m_biases;
    const IActivationFunction& m_activationFunction;
  };
}  // namespace nnn
