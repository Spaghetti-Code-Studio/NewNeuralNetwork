#pragma once

#include <memory>

#include "FloatMatrix.hpp"
#include "IActivationFunction.hpp"
#include "ILayer.hpp"

namespace nnn {

  class DenseLayer : public ILayer {
   public:
    DenseLayer(size_t inputSize, size_t outputSize, std::unique_ptr<IActivationFunction>&& activationFunction);
    FloatMatrix Forward(FloatMatrix& inputVector) override;
    void Update(const FloatMatrix& weights, const FloatMatrix& biases) override;

    inline const FloatMatrix& GetWeights() const { return m_weights; }
    inline const FloatMatrix& GetBiases() const { return m_biases; }

   private:
    size_t m_inputSize;
    size_t m_outputSize;
    FloatMatrix m_weights;
    FloatMatrix m_biases;
    std::unique_ptr<IActivationFunction> m_activationFunction;
  };
}  // namespace nnn
