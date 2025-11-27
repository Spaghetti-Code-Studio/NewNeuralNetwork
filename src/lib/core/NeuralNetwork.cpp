#include "NeuralNetwork.hpp"
#include <cmath>
#include <assert.h>
#include <iostream>

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

      for (size_t i = 0; i < newWeights.GetRowCount(); i++) {
        for (size_t j = 0; j < newWeights.GetColCount(); j++) {
          assert(std::abs(newWeights(i, j)) < 99999 && "The weight seems to be exploding!");  // TODO: remove this later
        }
      }
      layer.Update(newWeights, newBiases);
    });
  }

  NeuralNetwork::Statistics NeuralNetwork::Train(TrainingDataset& trainingDataset, bool reportProgress) {  //

    auto lossesValidation = std::vector<float>();
    auto lossesTraining = std::vector<float>();
    lossesValidation.reserve(m_params.epochs);
    lossesTraining.reserve(m_params.epochs);

    FloatMatrix allTrainFeatures = trainingDataset.GetTrainingFeatures();
    FloatMatrix allTrainLabels = trainingDataset.GetTrainingLabels();
    FloatMatrix allValidationFeatures = trainingDataset.GetValidationFeatures();
    FloatMatrix allValidationLabels = trainingDataset.GetValidationLabels();

    for (size_t epoch = 0; epoch < m_params.epochs; ++epoch) {
      while (trainingDataset.HasNextBatch()) {
        auto input = trainingDataset.GetNextBatch();
        FloatMatrix actual = RunForwardPass(input.features);
        FloatMatrix gradient = m_outputLayer->ComputeOutputGradient(actual, input.labels);
        gradient.MapInPlace([input](float x) { return x / input.features.GetColCount(); });
        RunBackwardPass(gradient);
        UpdateWeights();
      }
      trainingDataset.Reset();

      // compute loss for training dataset
      FloatMatrix trainPredictions = RunForwardPass(allTrainFeatures);
      trainPredictions.MapInPlace([](float x) { return std::log(x); });
      auto trainLoss = allTrainLabels.Hadamard(trainPredictions);
      trainLoss.Transpose();
      auto trainFlat = FloatMatrix::SumColumns(trainLoss);
      trainFlat.Transpose();
      auto trainTotal = FloatMatrix::SumColumns(trainFlat);
      lossesTraining.push_back(-trainTotal(0, 0) / trainLoss.GetRowCount());

      // compute loss for validation dataset
      // TODO: separate this into a function when doing final code cleanups
      if (trainingDataset.HasValidationDataset()) {
        auto actual = RunForwardPass(allValidationFeatures);
        actual.MapInPlace([](float x) { return std::log(x); });
        auto loss = allValidationLabels.Hadamard(actual);
        loss.Transpose();
        auto flat = FloatMatrix::SumColumns(loss);
        flat.Transpose();
        auto total = FloatMatrix::SumColumns(flat);
        lossesValidation.push_back(-total(0, 0) / loss.GetRowCount());
      }

      if (reportProgress) {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Epoch " << epoch + 1 << "/" << m_params.epochs << " - training loss: " << lossesTraining.back();
        if (trainingDataset.HasValidationDataset()) {
          std::cout << ", validation loss: " << lossesValidation.back();
          std::cout << std::setprecision(2) << " (aprox. " << std::exp(-lossesValidation.back()) * 100 << "%)";
        }
        std::cout << "." << std::endl;
      }

      m_params.learningRate *= m_params.learningRateDecay;
    }

    return {lossesTraining, lossesValidation};
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
