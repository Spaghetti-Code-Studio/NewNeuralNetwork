#include "NeuralNetwork.hpp"
#include <cmath>

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

  NeuralNetwork::Statistics NeuralNetwork::Train(TrainingDataset& trainingDataset) {  //

    auto losses = std::vector<float>();
    losses.reserve(m_params.epochs);

    for (size_t epoch = 0; epoch < m_params.epochs; ++epoch) {
      while (trainingDataset.HasNextBatch()) {
        auto input = trainingDataset.GetNextBatch();
        FloatMatrix actual = RunForwardPass(input.features);
        FloatMatrix gradient = m_outputLayer->ComputeOutputGradient(actual, input.labels);

        RunBackwardPass(gradient);
        UpdateWeights();
      }

      if (trainingDataset.HasValidationDataset()) {
        auto actual = RunForwardPass(trainingDataset.GetValidationFeatures());

        actual.MapInPlace([](float x) { return std::max(1e-10f, std::min(1.0f - 1e-10f, x)); });  // clip just in case
        actual.MapInPlace([](float x) { return std::log(x); });

        auto loss = trainingDataset.GetValidationLabels().Hadamard(actual);
        loss.Transpose();
        auto flat = FloatMatrix::SumColumns(loss);
        flat.Transpose();
        auto total = FloatMatrix::SumColumns(flat);
        losses.push_back(-total(0, 0) / loss.GetRowCount());
      }

      trainingDataset.Reset();
    }
    return NeuralNetwork::Statistics(losses);
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
