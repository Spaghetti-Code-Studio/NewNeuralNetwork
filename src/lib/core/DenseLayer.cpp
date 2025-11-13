#include "DenseLayer.hpp"
#include "FloatMatrix.hpp"

namespace nnn {

  DenseLayer::DenseLayer(
      size_t batchSize, size_t inputSize, size_t outputSize, std::unique_ptr<IActivationFunction>&& activationFunction)
      : m_inputSize(inputSize),
        m_outputSize(outputSize),
        m_biases(FloatMatrix::Zeroes(outputSize, 1)),
        m_weights(FloatMatrix::Random(outputSize, inputSize)),
        m_activationFunction(std::move(activationFunction)),
        m_lastInnerPotential(FloatMatrix::Zeroes(outputSize, batchSize)),
        m_lastInput(FloatMatrix::Zeroes(inputSize, batchSize)),
        m_gradientWeigths(outputSize, inputSize),
        m_gradientBias(outputSize, 1)

  {}

  DenseLayer::DenseLayer(size_t inputSize, size_t outputSize, std::unique_ptr<IActivationFunction>&& activationFunction)
      : m_inputSize(inputSize),
        m_outputSize(outputSize),
        m_biases(FloatMatrix::Zeroes(outputSize, 1)),
        m_weights(FloatMatrix::Random(outputSize, inputSize)),
        m_activationFunction(std::move(activationFunction)),
        m_lastInnerPotential(FloatMatrix::Zeroes(outputSize, 1)),
        m_lastInput(FloatMatrix::Zeroes(inputSize, 1)),
        m_gradientWeigths(outputSize, inputSize),
        m_gradientBias(outputSize, 1)

  {}

  FloatMatrix DenseLayer::Forward(const FloatMatrix& inputVector) {
    m_lastInput = inputVector;
    auto result = m_weights * inputVector;
    result.AddToAllCols(m_biases);
    m_lastInnerPotential = result;
    m_activationFunction->Evaluate(result);
    return result;
  }

  FloatMatrix DenseLayer::Backward(const FloatMatrix& gradient) {
    // slide 213
    m_activationFunction->Derivative(m_lastInnerPotential);  // sigma'(inner potential)
    auto hnc = m_lastInnerPotential.Hadamard(gradient);      // dE/dy * sigma'(inner potential)
    m_lastInput.Transpose();
    m_gradientWeigths = hnc * m_lastInput;  // dE/dw

    m_gradientBias = FloatMatrix::SumColumns(hnc);

    hnc.Transpose();
    auto nextGradient = hnc * m_weights;  // dE/dy+1
    nextGradient.Transpose();
    return nextGradient;  // here the final dimensions are cols = batch, rows = input, where input is acutally same size
                          // as output of next
  }

  void DenseLayer::Update(const FloatMatrix& weights, const FloatMatrix& biases) {
    m_weights = weights;
    m_biases = biases;
  }
  const FloatMatrix& DenseLayer::GetWeights() const { return m_weights; }
  const FloatMatrix& DenseLayer::GetBiases() const { return m_biases; }
}  // namespace nnn