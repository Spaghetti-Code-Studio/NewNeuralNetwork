#include "NeuralNetwork.hpp"

namespace nnn {
  void NeuralNetwork::AddLayer(std::unique_ptr<ILayer>&& layer) { m_layers.push_back(std::move(layer)); }

  FloatMatrix NeuralNetwork::RunForwardPass(FloatMatrix& input) {
    for (auto& layer : m_layers) {
      input = layer->Forward(input);
    }
    return input;
  }
  ILayer* NeuralNetwork::GetLayer(size_t index) {
    if (index >= m_layers.size()) {
      return nullptr;
    }
    return m_layers[index].get();
  }
}  // namespace nnn
