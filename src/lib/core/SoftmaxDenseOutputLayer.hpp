#pragma once

#include "CrossEntropyWithSoftmax.hpp"
#include "DenseLayer.hpp"
#include "FloatMatrix.hpp"
#include "ILossFunction.hpp"
#include "IOutputLayer.hpp"

namespace nnn {

  class SoftmaxDenseOutputLayer : public DenseLayer, public IOutputLayer {
   public:
    SoftmaxDenseOutputLayer(size_t batchSize, size_t inputSize, size_t outputSize, IWeightInitializer& initializer);
    SoftmaxDenseOutputLayer(size_t inputSize, size_t outputSize, IWeightInitializer& initializer);
    SoftmaxDenseOutputLayer(size_t batchSize, size_t inputSize, size_t outputSize);
    SoftmaxDenseOutputLayer(size_t inputSize, size_t outputSize);

    FloatMatrix ComputeOutputGradient(const FloatMatrix& actual, const FloatMatrix& expected) override;

    FloatMatrix Backward(const FloatMatrix& gradient) override;

   private:
    std::unique_ptr<ILossFunction> m_crossEntropyLossFunction = std::make_unique<CrossEntropyWithSoftmax>();
  };
}  // namespace nnn
