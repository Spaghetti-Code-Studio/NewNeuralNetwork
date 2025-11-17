#include "NeuralNetwork.hpp"

namespace nnn {

  size_t NeuralNetwork::AddHiddenLayer(std::unique_ptr<ILayer>&& layer) {
    m_hiddenLayers.push_back(std::move(layer));
    return m_hiddenLayers.size() - 1;
  }

  size_t NeuralNetwork::SetOutputLayer(std::unique_ptr<IOutputLayer>&& layer) {
    m_outputLayer = std::move(layer);
    return m_hiddenLayers.size();
  }

  FloatMatrix NeuralNetwork::RunForwardPass(FloatMatrix input) {
    ForEachLayerForward([&](ILayer& layer) { input = layer.Forward(input); });
    return input;
  }

  void NeuralNetwork::RunBackwardPass(FloatMatrix gradient) {
    ForEachLayerBackward([&](ILayer& layer) { gradient = layer.Backward(gradient); });
  }

  void NeuralNetwork::UpdateWeights() {
    ForEachLayerForward([&](ILayer& layer) {
      FloatMatrix newWeights = layer.GetWeights() - (layer.GetWightsGradient() * m_params.learningRate);
      FloatMatrix newBiases = layer.GetBiases() - (layer.GetBiasesGradient() * m_params.learningRate);
      layer.Update(newWeights, newBiases);
    });
  }

  void NeuralNetwork::Train(const FloatMatrix& input, const FloatMatrix& expected) {  //

    for (size_t epoch = 0; epoch < m_params.epochs; ++epoch) {
      // TODO: handle batches
      FloatMatrix actual = RunForwardPass(input);
      FloatMatrix gradient = m_outputLayer->ComputeOutputGradient(actual, expected);
      RunBackwardPass(gradient);
      UpdateWeights();
    }
  }

  ILayer* NeuralNetwork::GetLayer(size_t index) {  //

    if (index > m_hiddenLayers.size()) {
      return nullptr;
    }

    if (index == m_hiddenLayers.size()) {
      return m_outputLayer.get();
    }

    return m_hiddenLayers[index].get();
  }
}  // namespace nnn
