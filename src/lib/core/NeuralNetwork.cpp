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

  ILayer* NeuralNetwork::GetLayer(size_t index) {  //

    if (index >= m_layers.size()) {
      return nullptr;
    }

    return m_layers[index].get();
  }
}  // namespace nnn
