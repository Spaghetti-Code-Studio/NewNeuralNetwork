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
    ForEachLayer([&](ILayer& layer) { input = layer.Forward(input); });
    return input;
  }

  // TODO: implement this
  FloatMatrix NeuralNetwork::RunBackwardPass(FloatMatrix input) {
    return FloatMatrix::Identity(4);  // tmp
  }

  void NeuralNetwork::Train(const FloatMatrix& input, const FloatMatrix& expected, HyperParameters params) {  //

    auto actual = RunForwardPass(input);

    FloatMatrix gradient = m_outputLayer->ComputeOutputGradient(actual, expected);

    // TODO: use backward ForEachLayer here
    gradient = m_outputLayer->Backward(gradient);
    for (int i = (int)m_hiddenLayers.size() - 1; i >= 0; --i) {
      gradient = m_hiddenLayers[i]->Backward(gradient);
    }

    // TODO: use backward ForEachLayer here
    {
      FloatMatrix newWeights = m_outputLayer->GetWeights() - (m_outputLayer->GetWightsGradient() * params.learningRate);
      FloatMatrix newBiases = m_outputLayer->GetBiases() - (m_outputLayer->GetBiasesGradient() * params.learningRate);
      m_outputLayer->Update(newWeights, newBiases);
    }
    ForEachLayer([&](ILayer& layer) {
      FloatMatrix newWeights = layer.GetWeights() - (layer.GetWightsGradient() * params.learningRate);
      FloatMatrix newBiases = layer.GetBiases() - (layer.GetBiasesGradient() * params.learningRate);
      layer.Update(newWeights, newBiases);
    });
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
