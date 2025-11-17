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
    // TODO: check this code
    m_lastInput.Transpose();
    m_gradientWeigths = gradient * m_lastInput;
    m_gradientBias = FloatMatrix::SumColumns(gradient);

    m_weights.Transpose();  // cannot transpose gradient const gradient reference here so I transpose weights
    auto nextGradient = gradient * m_weights;
    nextGradient.Transpose();
    m_weights.Transpose();
    return nextGradient;
  }
}  // namespace nnn