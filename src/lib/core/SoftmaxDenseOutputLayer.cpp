#include "Softmax.hpp"
#include "SoftmaxDenseOutputLayer.hpp"

namespace nnn {

  SoftmaxDenseOutputLayer::SoftmaxDenseOutputLayer(size_t batchSize, size_t inputSize, size_t outputSize)
      : DenseLayer(batchSize, inputSize, outputSize, std::make_unique<Softmax>()) {}

  SoftmaxDenseOutputLayer::SoftmaxDenseOutputLayer(size_t inputSize, size_t outputSize)
      : DenseLayer(inputSize, outputSize, std::make_unique<Softmax>()) {}

  FloatMatrix SoftmaxDenseOutputLayer::ComputeOutputGradient(const FloatMatrix& actual, const FloatMatrix& expected) {
    return m_crossEntropyLossFunction->Loss(actual, expected);
  }

  FloatMatrix SoftmaxDenseOutputLayer::Backward(const FloatMatrix& gradient) {  //

    // Gradient here is already (actual - expecetd) from cross-entropy loss function.
    // No need to call m_activationFunction->Derivative().

    auto hnc = gradient;

    m_lastInput.Transpose();
    m_gradientWeigths = hnc * m_lastInput;
    m_gradientBias = FloatMatrix::SumColumns(hnc);

    hnc.Transpose();
    auto nextGradient = hnc * m_weights;
    nextGradient.Transpose();

    return nextGradient;
  }
}  // namespace nnn