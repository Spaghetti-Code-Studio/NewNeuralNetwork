#include "DenseLayer.hpp"
#include "FloatMatrix.hpp"

namespace nnn {

  DenseLayer::DenseLayer(size_t inputSize, size_t outputSize, std::unique_ptr<IActivationFunction>&& activationFunction)
      : m_inputSize(inputSize),
        m_outputSize(outputSize),
        m_biases(FloatMatrix::Zeroes(outputSize, 1)),
        m_weights(FloatMatrix::Random(outputSize, inputSize)),
        m_activationFunction(std::move(activationFunction)) {}

  FloatMatrix DenseLayer::Forward(const FloatMatrix& inputVector) const {
    auto result = m_weights * inputVector;
    result.AddToAllRows(m_biases);
    m_activationFunction->Evaluate(result);
    return result;
  }

  void DenseLayer::Update(const FloatMatrix& weights, const FloatMatrix& biases) {
    m_weights = weights;
    m_biases = biases;
  }
}  // namespace nnn