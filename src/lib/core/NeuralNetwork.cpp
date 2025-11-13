#include "NeuralNetwork.hpp"

namespace nnn {

  size_t NeuralNetwork::AddLayer(std::unique_ptr<ILayer>&& layer) {
    m_layers.push_back(std::move(layer));
    return m_layers.size() - 1;
  }

  FloatMatrix NeuralNetwork::RunForwardPass(FloatMatrix input) {  //

    for (const auto& layer : m_layers) {
      input = layer->Forward(input);
    }

    return input;
  }

  void NeuralNetwork::Train(const FloatMatrix& input, const FloatMatrix& expected, HyperParameters params) {
    auto actual = RunForwardPass(input);

    // TODO: compute correct loss function
    FloatMatrix loss = (actual - expected) * 2.0f;

    for (int i = m_layers.size() - 1; i >= 0; --i) {
      loss = m_layers[i]->Backward(loss);
    }

    for (const auto& layer : m_layers) {
      // TODO: make this cleaner and nicer
      auto copyOfWeights = layer->GetWeights();
      copyOfWeights *= params.learningRate;
      auto copyOfBiases = layer->GetBiases();
      copyOfBiases *= params.learningRate;
      layer->Update(copyOfWeights, copyOfBiases);
    }
  }

  ILayer* NeuralNetwork::GetLayer(size_t index) {  //

    if (index >= m_layers.size()) {
      return nullptr;
    }

    return m_layers[index].get();
  }
}  // namespace nnn
